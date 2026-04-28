from __future__ import annotations

"""
evaluate_gen.py
===============

Offline evaluator for the generative multi-cylinder modular-DT demo.

The evaluator loads:
1. either a stage-1 residual-AE checkpoint or a stage-2 rectified-flow checkpoint,
2. for stage 2, the frozen deterministic modular-DT checkpoint referenced by
   that checkpoint,
3. one canonical-cycle case from the packed HDF5 dataset.

It then reconstructs the target grid (stage 1) or generates one or more
stochastic residual samples (stage 2) at a chosen phase and saves:
- quicklook figures,
- compressed NPZ arrays,
- a JSON metrics summary.

E.g.:
python src/evaluate_gen.py  --stage 1 --case-id gen001 --dataset-case-id 0150 --split train
python src/evaluate_gen.py  --stage 2 --case-id gen001 --dataset-case-id 0150 --split train --mode cycle --n-steps 4 --cycle-noise-mode shared

The visualization is intentionally close to the deterministic evaluator style so results can be compared side by side.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Optional

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from model_gen import (
    ConvResidualAE,
    GridStats,
    LatentEMA,
    LatentRectifiedFlow,
    LatentVelocityUNet,
    build_dense_condition_grid,
    build_global_condition_vector,
    denormalize_grid,
    normalize_grid,
)
from train_gen import (
    deterministic_grid_forward,
    load_deterministic_model,
    resolve_config_path,
    resolve_demo_path,
    safe_torch_load,
)
from organizer_viz import extract_organization_arrays, render_soft_organization


# -----------------------------------------------------------------------------
# CLI and small helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stage-1 or stage-2 generative checkpoints for multi-cylinder demo.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to best_model.pt or latest_model.pt. If omitted, the newest Saved_Model_Gen run is selected by --case-id and --stage.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config used for checkpoint discovery and optional overrides. Defaults to train_gen_config_template.json for discovery.",
    )
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2], help="Generative training stage to evaluate. Defaults to config/checkpoint stage.")
    parser.add_argument("--latest", action="store_true", help="Load latest_model.pt instead of best_model.pt from the selected run directory.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to read.")
    parser.add_argument("--case-id", type=str, default=None, help="Generative training run case_id, e.g. gen001. Defaults to config case_id.")
    parser.add_argument("--dataset-case-id", type=str, default=None, help="Specific HDF5 case id. Defaults to first matching split case.")
    parser.add_argument("--mode", type=str, default="snapshot", choices=["snapshot", "cycle"], help="Evaluate one phase snapshot or a full canonical cycle.")
    parser.add_argument("--cycle", action="store_true", help="Shortcut for --mode cycle.")
    parser.add_argument("--phase-index", type=int, default=0, help="Canonical phase-bin index.")
    parser.add_argument("--phase-start", type=int, default=0, help="First phase bin for cycle mode.")
    parser.add_argument("--phase-stop", type=int, default=None, help="Exclusive final phase bin for cycle mode. Defaults to all bins.")
    parser.add_argument("--phase-stride", type=int, default=1, help="Stride through canonical phase bins for cycle mode.")
    parser.add_argument("--n-samples", type=int, default=4, help="Number of stochastic samples to draw for stage 2.")
    parser.add_argument("--n-steps", type=int, default=None, help="Override rectified-flow ODE steps.")
    parser.add_argument("--ode-solver", type=str, default=None, choices=["euler", "heun"])
    parser.add_argument("--cycle-noise-mode", type=str, default="independent", choices=["independent", "shared", "harmonic"], help="Latent noise coupling across phases in cycle mode.")
    parser.add_argument("--phase-chunk-size", type=int, default=8, help="Number of phase bins processed at once in cycle mode.")
    parser.add_argument("--sample-chunk-size", type=int, default=1, help="Number of stochastic samples processed together in cycle mode.")
    parser.add_argument("--gif-fps", type=float, default=6.0, help="Frames per second for cycle GIF output.")
    parser.add_argument("--viz-channel", type=int, default=None, help="Field channel used for quicklooks and cycle GIFs; omega is channel 3.")
    parser.add_argument("--viz-field", type=str, default=None, help="Field name used for quicklooks/GIFs, e.g. omega or temperature. Overrides --viz-channel.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory.")
    parser.add_argument("--organization-threshold", type=float, default=0.15, help="Minimum soft weight used when drawing organization edges.")
    parser.add_argument("--topk-me-links", type=int, default=3, help="Reserved for deterministic compatibility; env-token links are suppressed in the refined organizer overlay.")
    parser.add_argument("--organization-view", choices=["all", "physical", "matrices", "sankey", "schematic"], default="all", help="Which deterministic-backbone organizer diagnostic view to render in stage-2 snapshot mode.")
    parser.add_argument("--organization-topk-cylinders", type=int, default=3, help="Number of top cylinder memberships to list for each hyperedge.")
    parser.add_argument("--organization-topk-env", type=int, default=5, help="Number of top environment tokens to list for each hyperedge.")
    parser.add_argument("--organization-min-gap", type=float, default=0.08, help="Minimum normalized vertical gap for Sankey node layout.")
    parser.add_argument("--organization-table", action=argparse.BooleanOptionalAction, default=True, help="Show the hyperedge summary table in the physical organization view.")
    parser.add_argument("--disable-edge", dest="disable_edge", action="store_true", default=None, help="Enable deterministic active-edge masking for this evaluation run only.")
    parser.add_argument("--no-disable-edge", dest="disable_edge", action="store_false", help="Disable deterministic active-edge masking for this evaluation run only.")
    parser.add_argument("--show-disabled-edges", action="store_true", help="Draw disabled deterministic hyperedges in grey dashed style instead of hiding them.")
    args = parser.parse_args()
    if args.cycle:
        args.mode = "cycle"
    return args


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def select_device(device_arg: str) -> torch.device:
    return torch.device(device_arg if device_arg else ("cuda:0" if torch.cuda.is_available() else "cpu"))


def sort_case_ids(case_ids):
    def key_fn(case_id):
        try:
            return (0, int(case_id))
        except Exception:
            return (1, str(case_id))
    return sorted(case_ids, key=key_fn)


INERT_CHANNEL_ORDER = ("u", "v", "p", "omega")
ACTIVE_CHANNEL_ORDER = ("u", "v", "p", "omega", "temperature")


def decode_string_array(values) -> list[str]:
    arr = np.asarray(values)
    out = []
    for item in arr.reshape(-1):
        out.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return out


def channel_order_from_attr(value) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "mixed":
            return None
        try:
            payload = json.loads(text)
            if isinstance(payload, (list, tuple)):
                return [str(v) for v in payload]
        except json.JSONDecodeError:
            pass
        return [piece.strip() for piece in text.split(",") if piece.strip()]
    return decode_string_array(value)


def get_case_channel_order(grp: h5py.Group, h5_file: h5py.File) -> list[str]:
    if "channel_order" in grp:
        return decode_string_array(grp["channel_order"][...])
    root_order = channel_order_from_attr(h5_file.attrs.get("channel_order"))
    if root_order is not None:
        return root_order
    field_dim = int(grp.attrs.get("field_dim", grp["canonical_cycle"].shape[-1] if "canonical_cycle" in grp else 4))
    return list(ACTIVE_CHANNEL_ORDER if field_dim == 5 else INERT_CHANNEL_ORDER)


def resolve_viz_channel(args: argparse.Namespace, channel_order: list[str], field_dim: int) -> tuple[int, str]:
    if args.viz_field is not None:
        field_name = str(args.viz_field).strip()
        if field_name not in channel_order:
            raise ValueError(f"--viz-field={field_name!r} is not in channel_order={channel_order}.")
        return int(channel_order.index(field_name)), field_name
    if args.viz_channel is None:
        channel = channel_order.index("omega") if "omega" in channel_order else 0
        return channel, channel_order[channel] if channel < len(channel_order) else f"channel {channel}"
    channel = int(args.viz_channel)
    if channel < 0 or channel >= int(field_dim):
        raise ValueError(f"--viz-channel={channel} is out of range for field_dim={field_dim}.")
    return channel, channel_order[channel] if channel < len(channel_order) else f"channel {channel}"


def default_cycle_viz_channels(args: argparse.Namespace, channel_order: list[str], field_dim: int) -> list[tuple[int, str]]:
    if args.viz_field is not None or args.viz_channel is not None:
        return [resolve_viz_channel(args, channel_order, field_dim)]
    if "temperature" in channel_order and "omega" in channel_order:
        return [(channel_order.index("temperature"), "temperature"), (channel_order.index("omega"), "omega")]
    return [resolve_viz_channel(args, channel_order, field_dim)]


def add_extra_module_if_needed(sample: Dict, future_module_feature_dim: int, heat_power_scale: float = 1.0) -> None:
    future_dim = int(future_module_feature_dim)
    if future_dim <= 0:
        return
    structure = sample["structure"]
    centers = structure["centers"]
    max_num_cylinders = centers.shape[1]
    powers = np.asarray(sample.get("heat_powers", np.zeros((int(sample.get("num_cylinders", 0)),), dtype=np.float32)), dtype=np.float32).reshape(-1)
    padded = np.zeros((1, max_num_cylinders, future_dim), dtype=np.float32)
    padded[0, : min(powers.shape[0], max_num_cylinders), 0] = powers[:max_num_cylinders] / max(float(heat_power_scale), 1e-12)
    structure["extra_module"] = torch.from_numpy(padded)


def load_config(config_arg: Optional[str]) -> Dict:
    if config_arg is None:
        return {}
    config_path = resolve_config_path(config_arg)
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _legacy_safe_pool(values: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
    if values is None:
        return []
    if values.ndim == 2:
        return [values]
    if mask is None:
        return [values.mean(dim=1), values.max(dim=1).values]
    m = mask.to(dtype=values.dtype, device=values.device).unsqueeze(-1)
    raw_count = m.sum(dim=1)
    denom = raw_count.clamp_min(1.0)
    mean = (values * m).sum(dim=1) / denom
    masked = values.masked_fill(m <= 0, -1e9)
    max_val = masked.max(dim=1).values
    valid_any = raw_count > 0
    max_val = torch.where(valid_any, max_val, torch.zeros_like(max_val))
    return [mean, max_val]


def _build_legacy_global_condition_vector(det_outputs: Dict[str, torch.Tensor], structure: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Recreate the global-condition layout used by older stage-2 checkpoints."""
    pieces: list[torch.Tensor] = []

    for key in ["behavior_latent", "mean_latent", "dynamic_global_token", "freq_pred"]:
        val = det_outputs.get(key)
        if val is None:
            continue
        if val.ndim == 3 and val.shape[1] == 1:
            val = val[:, 0]
        pieces.append(val.reshape(val.shape[0], -1))

    cyl_mask = structure.get("cyl_mask")
    for key in ["module_state", "env_state", "hyper_state", "dynamic_hyper_base", "dynamic_hyper_tokens"]:
        val = det_outputs.get(key)
        if val is None:
            continue
        mask = cyl_mask if key == "module_state" else None
        pieces.extend(_legacy_safe_pool(val, mask=mask))

    for key in [
        "hyper_module_mass",
        "hyper_env_mass",
        "hyper_strength",
        "hyper_source_coords",
        "hyper_wake_coords",
        "hyper_wake_axis",
        "hyper_wake_extent",
    ]:
        val = det_outputs.get(key)
        if val is not None:
            pieces.extend(_legacy_safe_pool(val, mask=None))

    for key in ["re_values", "num_cylinders"]:
        if key in structure:
            pieces.append(structure[key].reshape(structure[key].shape[0], -1))

    if not pieces:
        raise RuntimeError("No deterministic-condition features were available.")
    return torch.cat(pieces, dim=-1)


def _build_checkpoint_global_condition_vector(
    det_outputs: Dict[str, torch.Tensor],
    structure: Dict[str, torch.Tensor],
    expected_dim: int,
) -> torch.Tensor:
    global_cond = build_global_condition_vector(det_outputs, structure)
    if int(global_cond.shape[-1]) == int(expected_dim):
        return global_cond

    legacy_global_cond = _build_legacy_global_condition_vector(det_outputs, structure)
    if int(legacy_global_cond.shape[-1]) == int(expected_dim):
        return legacy_global_cond

    raise ValueError(
        "Global condition width does not match the stage-2 checkpoint: "
        f"current={int(global_cond.shape[-1])}, legacy={int(legacy_global_cond.shape[-1])}, "
        f"checkpoint={int(expected_dim)}. Re-train stage 2 or update the checkpoint condition layout."
    )


def find_latest_gen_run(save_root: Path, case_id: str, stage: int) -> Path:
    prefix = f"Gen_Case{case_id}_Stage{stage}_"
    candidates = [p for p in save_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No generative run directory found in {save_root} for case_id={case_id!r}, stage={stage}.")
    return sorted(candidates, key=lambda p: p.name)[-1]


def resolve_checkpoint_for_args(args: argparse.Namespace, cfg: Dict) -> tuple[Path, int]:
    if args.checkpoint is not None:
        checkpoint_path = resolve_demo_path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if args.stage is not None:
            return checkpoint_path, int(args.stage)
        ckpt = safe_torch_load(checkpoint_path, map_location="cpu")
        stage = int(ckpt.get("stage", cfg.get("generation", {}).get("training_stage", 2)))
        return checkpoint_path, stage

    stage = int(args.stage if args.stage is not None else cfg.get("generation", {}).get("training_stage", 1))
    case_id = str(args.case_id if args.case_id is not None else cfg.get("case_id", "gen"))
    save_root = resolve_demo_path(cfg.get("paths", {}).get("saved_model_dir", "./Saved_Model_Gen"))
    run_dir = find_latest_gen_run(save_root, case_id, stage)
    checkpoint_name = "latest_model.pt" if args.latest else "best_model.pt"
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists() and not args.latest:
        fallback = run_dir / "latest_model.pt"
        if fallback.exists():
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(f"Neither best_model.pt nor latest_model.pt exists in {run_dir}")
    elif not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path, stage


# -----------------------------------------------------------------------------
# Dataset case loading
# -----------------------------------------------------------------------------


def load_case_snapshot(packed_h5_path: Path, split: str, case_id: Optional[str], phase_index: int, max_num_cylinders: int) -> Dict:
    """Load one canonical-cycle snapshot and structure tensors from HDF5."""
    with h5py.File(packed_h5_path, "r") as h5:
        cases = h5["cases"]
        if case_id is None:
            matched = []
            for cid in sort_case_ids(cases.keys()):
                grp = cases[cid]
                if split in {"all", grp.attrs.get("split", "all")} and "canonical_cycle" in grp:
                    matched.append(cid)
            if not matched:
                raise RuntimeError(f"No canonical cases found for split={split!r}.")
            case_id = matched[0]
        grp = cases[str(case_id)]
        phase_bins = np.asarray(grp["phase_bin_centers"], dtype=np.float32)
        phase_tau_bins = np.asarray(grp["phase_tau_centers"], dtype=np.float32) if "phase_tau_centers" in grp else phase_bins
        query_time_bins = (
            np.asarray(grp["thermal_time_centers"], dtype=np.float32)
            if "thermal_time_centers" in grp
            else (np.asarray(grp["tau_abs_centers"], dtype=np.float32) if "tau_abs_centers" in grp else phase_bins)
        )
        cycle_index_bins = np.asarray(grp["cycle_index_centers"], dtype=np.float32) if "cycle_index_centers" in grp else np.floor(phase_bins)
        if phase_index < 0 or phase_index >= len(phase_bins):
            raise IndexError(f"phase_index={phase_index} out of range for {len(phase_bins)} bins.")

        field = np.asarray(grp["canonical_cycle"][phase_index], dtype=np.float32)  # [H,W,C]
        field_dim = int(grp.attrs.get("field_dim", field.shape[-1]))
        channel_order = get_case_channel_order(grp, h5)[:field_dim]
        mean = np.asarray(grp["mean_field"], dtype=np.float32)
        residual = field - mean
        x_grid = np.asarray(grp["x_grid"], dtype=np.float32)
        y_grid = np.asarray(grp["y_grid"], dtype=np.float32)
        centers = np.asarray(grp["cylinder_centers"], dtype=np.float32)
        if centers.shape[0] > max_num_cylinders:
            raise ValueError(f"Case {case_id} has {centers.shape[0]} cylinders but max_num_cylinders={max_num_cylinders}.")
        heat_powers = (
            np.asarray(grp["heat_powers"], dtype=np.float32)
            if "heat_powers" in grp
            else np.zeros((centers.shape[0],), dtype=np.float32)
        )
        heat_power_scale = max(
            float(np.max(np.abs(heat_powers))) if heat_powers.size else 0.0,
            abs(float(grp.attrs.get("thermal_power_min", 0.0))),
            abs(float(grp.attrs.get("thermal_power_max", 0.0))),
            1.0,
        )

        padded = np.zeros((max_num_cylinders, 2), dtype=np.float32)
        mask = np.zeros((max_num_cylinders,), dtype=np.float32)
        padded[: centers.shape[0]] = centers
        mask[: centers.shape[0]] = 1.0

        return {
            "case_id": str(case_id),
            "phase_index": int(phase_index),
            "tau": float(phase_tau_bins[phase_index]),
            "tau_abs": float(phase_bins[phase_index]),
            "query_time": float(query_time_bins[phase_index]),
            "cycle_index": float(cycle_index_bins[phase_index]),
            "field_dim": field_dim,
            "channel_order": channel_order,
            "field_grid": torch.from_numpy(np.moveaxis(field, -1, 0)).unsqueeze(0),
            "mean_grid": torch.from_numpy(np.moveaxis(mean, -1, 0)).unsqueeze(0),
            "residual_grid": torch.from_numpy(np.moveaxis(residual, -1, 0)).unsqueeze(0),
            "x_grid": torch.from_numpy(x_grid).unsqueeze(0),
            "y_grid": torch.from_numpy(y_grid).unsqueeze(0),
            "structure": {
                "re_values": torch.tensor([[float(grp.attrs["re"])]], dtype=torch.float32),
                "num_cylinders": torch.tensor([[int(grp.attrs["num_cylinders"])]], dtype=torch.float32),
                "centers": torch.from_numpy(padded).unsqueeze(0),
                "cyl_mask": torch.from_numpy(mask).unsqueeze(0),
            },
            "centers_np": centers,
            "heat_powers": heat_powers,
            "heat_power_scale": heat_power_scale,
            "re": float(grp.attrs["re"]),
            "num_cylinders": int(grp.attrs["num_cylinders"]),
        }


def _select_case_group(h5: h5py.File, split: str, case_id: Optional[str]):
    cases = h5["cases"]
    if case_id is None:
        matched = []
        for cid in sort_case_ids(cases.keys()):
            grp = cases[cid]
            if split in {"all", grp.attrs.get("split", "all")} and "canonical_cycle" in grp:
                matched.append(cid)
        if not matched:
            raise RuntimeError(f"No canonical cases found for split={split!r}.")
        case_id = matched[0]
    return str(case_id), cases[str(case_id)]


def _phase_indices_from_args(num_bins: int, args: argparse.Namespace) -> np.ndarray:
    start = max(0, int(args.phase_start))
    stop = num_bins if args.phase_stop is None else min(num_bins, int(args.phase_stop))
    stride = max(1, int(args.phase_stride))
    indices = np.arange(start, stop, stride, dtype=np.int64)
    if indices.size == 0:
        raise ValueError(f"No phase bins selected from start={start}, stop={stop}, stride={stride}, num_bins={num_bins}.")
    return indices


def load_case_cycle(
    packed_h5_path: Path,
    split: str,
    case_id: Optional[str],
    phase_indices: np.ndarray,
    max_num_cylinders: int,
) -> Dict:
    """Load selected canonical-cycle bins as [T,C,H,W] tensors."""
    with h5py.File(packed_h5_path, "r") as h5:
        case_id, grp = _select_case_group(h5, split, case_id)
        phase_bins = np.asarray(grp["phase_bin_centers"], dtype=np.float32)
        phase_tau_bins = np.asarray(grp["phase_tau_centers"], dtype=np.float32) if "phase_tau_centers" in grp else phase_bins
        query_time_bins = (
            np.asarray(grp["thermal_time_centers"], dtype=np.float32)
            if "thermal_time_centers" in grp
            else (np.asarray(grp["tau_abs_centers"], dtype=np.float32) if "tau_abs_centers" in grp else phase_bins)
        )
        cycle_index_bins = np.asarray(grp["cycle_index_centers"], dtype=np.float32) if "cycle_index_centers" in grp else np.floor(phase_bins)
        if phase_indices.min() < 0 or phase_indices.max() >= len(phase_bins):
            raise IndexError(f"Selected phase index is out of range for {len(phase_bins)} bins.")

        cycle = np.asarray(grp["canonical_cycle"][phase_indices], dtype=np.float32)  # [T,H,W,C]
        field_dim = int(grp.attrs.get("field_dim", cycle.shape[-1]))
        channel_order = get_case_channel_order(grp, h5)[:field_dim]
        mean = np.asarray(grp["mean_field"], dtype=np.float32)
        residual = cycle - mean[None, ...]
        x_grid = np.asarray(grp["x_grid"], dtype=np.float32)
        y_grid = np.asarray(grp["y_grid"], dtype=np.float32)
        centers = np.asarray(grp["cylinder_centers"], dtype=np.float32)
        if centers.shape[0] > max_num_cylinders:
            raise ValueError(f"Case {case_id} has {centers.shape[0]} cylinders but max_num_cylinders={max_num_cylinders}.")
        heat_powers = (
            np.asarray(grp["heat_powers"], dtype=np.float32)
            if "heat_powers" in grp
            else np.zeros((centers.shape[0],), dtype=np.float32)
        )
        heat_power_scale = max(
            float(np.max(np.abs(heat_powers))) if heat_powers.size else 0.0,
            abs(float(grp.attrs.get("thermal_power_min", 0.0))),
            abs(float(grp.attrs.get("thermal_power_max", 0.0))),
            1.0,
        )

        padded = np.zeros((max_num_cylinders, 2), dtype=np.float32)
        mask = np.zeros((max_num_cylinders,), dtype=np.float32)
        padded[: centers.shape[0]] = centers
        mask[: centers.shape[0]] = 1.0

        return {
            "case_id": str(case_id),
            "phase_indices": phase_indices.astype(np.int64),
            "tau_values": phase_tau_bins[phase_indices].astype(np.float32),
            "tau_abs_values": phase_bins[phase_indices].astype(np.float32),
            "query_time_values": query_time_bins[phase_indices].astype(np.float32),
            "cycle_index_values": cycle_index_bins[phase_indices].astype(np.float32),
            "field_dim": field_dim,
            "channel_order": channel_order,
            "field_cycle": torch.from_numpy(np.moveaxis(cycle, -1, 1)),  # [T,C,H,W]
            "mean_grid": torch.from_numpy(np.moveaxis(mean, -1, 0)).unsqueeze(0),
            "residual_cycle": torch.from_numpy(np.moveaxis(residual, -1, 1)),
            "x_grid": torch.from_numpy(x_grid).unsqueeze(0),
            "y_grid": torch.from_numpy(y_grid).unsqueeze(0),
            "structure": {
                "re_values": torch.tensor([[float(grp.attrs["re"])]], dtype=torch.float32),
                "num_cylinders": torch.tensor([[int(grp.attrs["num_cylinders"])]], dtype=torch.float32),
                "centers": torch.from_numpy(padded).unsqueeze(0),
                "cyl_mask": torch.from_numpy(mask).unsqueeze(0),
            },
            "centers_np": centers,
            "heat_powers": heat_powers,
            "heat_power_scale": heat_power_scale,
            "re": float(grp.attrs["re"]),
            "num_cylinders": int(grp.attrs["num_cylinders"]),
        }


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def build_ae_from_checkpoint(ckpt: Dict) -> ConvResidualAE:
    cfg = ckpt.get("ae_config", {})
    return ConvResidualAE(
        n_fields=int(ckpt.get("n_fields", 4)),
        base_ch=int(cfg.get("base_ch", ckpt.get("config", {}).get("stage1", {}).get("architecture", {}).get("base_ch", 48))),
        latent_ch=int(cfg.get("latent_ch", ckpt.get("config", {}).get("stage1", {}).get("architecture", {}).get("latent_ch", 96))),
        n_levels=int(cfg.get("n_levels", ckpt.get("config", {}).get("stage1", {}).get("architecture", {}).get("n_levels", 3))),
        num_res_blocks=int(cfg.get("num_res_blocks", ckpt.get("config", {}).get("stage1", {}).get("architecture", {}).get("num_res_blocks", 1))),
        num_y=int(ckpt["num_y"]),
        num_x=int(ckpt["num_x"]),
    )


def load_ae(checkpoint_path: Path, device: torch.device) -> tuple[ConvResidualAE, GridStats, Dict]:
    """Load a stage-1 AE checkpoint, or the embedded AE from stage 2."""
    ckpt = safe_torch_load(checkpoint_path, map_location=device)
    stage = int(ckpt.get("stage", 0))
    if stage not in {1, 2}:
        raise ValueError(f"Expected a stage-1 or stage-2 generative checkpoint, got stage={ckpt.get('stage')}.")
    stats = GridStats(mean=ckpt["stats"]["mean"].to(device), std=ckpt["stats"]["std"].to(device))
    ae = build_ae_from_checkpoint(ckpt).to(device)
    ae.load_state_dict(ckpt["ae_state_dict"])
    ae.eval().requires_grad_(False)
    return ae, stats, ckpt


def load_generator(checkpoint_path: Path, device: torch.device) -> tuple[LatentRectifiedFlow, Optional[LatentEMA], GridStats, Dict]:
    """Load stage-2 latent rectified-flow generator."""
    ckpt = safe_torch_load(checkpoint_path, map_location=device)
    if int(ckpt.get("stage", 0)) != 2:
        raise ValueError(f"Expected a stage-2 generative checkpoint, got stage={ckpt.get('stage')}.")
    cfg = ckpt["config"]
    stats = GridStats(mean=ckpt["stats"]["mean"].to(device), std=ckpt["stats"]["std"].to(device))
    ae = build_ae_from_checkpoint(ckpt).to(device)
    ae.load_state_dict(ckpt["ae_state_dict"])
    ae.eval().requires_grad_(False)

    arch = cfg["stage2"]["architecture"]
    velocity = LatentVelocityUNet(
        latent_ch=ae.latent_ch,
        cond_ch=int(ckpt["cond_ch"]),
        global_cond_dim=int(ckpt["global_cond_dim"]),
        base_ch=int(arch.get("base_ch", ckpt.get("fm_base_ch", 192))),
        ch_mult=tuple(arch.get("ch_mult", [1, 2])),
        num_res_blocks=int(arch.get("num_res_blocks", 2)),
        num_heads=int(arch.get("num_heads", 4)),
        dropout=float(arch.get("dropout", 0.0)),
    ).to(device)
    flow = LatentRectifiedFlow(ae=ae, velocity_net=velocity, cond_downsample_mode=arch.get("cond_downsample_mode", "area")).to(device)
    flow.velocity_net.load_state_dict(ckpt["velocity_state_dict"])

    ema = None
    if ckpt.get("ema_state_dict") is not None:
        ema = LatentEMA(flow.velocity_net, decay=float(arch.get("ema_decay", 0.999)))
        ema.load_state_dict(ckpt["ema_state_dict"])
    flow.eval()
    return flow, ema, stats, ckpt


# -----------------------------------------------------------------------------
# Metrics and plots
# -----------------------------------------------------------------------------


def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def rel_l2_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.reshape(-1) - b.reshape(-1)) / (np.linalg.norm(b.reshape(-1)) + 1e-8))


def _channel_cmap(channel: int) -> str:
    """Match deterministic evaluate.py channel colormaps."""
    cmaps = ["coolwarm", "coolwarm", "magma", "RdBu_r", "inferno"]
    return cmaps[int(channel)] if 0 <= int(channel) < len(cmaps) else "coolwarm"


def _robust_symmetric_limits(values: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    vmax = float(np.percentile(np.abs(np.asarray(values, dtype=np.float32)), percentile))
    vmax = max(vmax, 1e-8)
    return -vmax, vmax


def _grid_extent(x_grid: np.ndarray, y_grid: np.ndarray) -> tuple[float, float, float, float]:
    return (float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max()))


def _format_heat_power(value: float) -> str:
    return f"{value:+.2f}" if abs(value) < 100.0 else f"{value:+.2e}"


def _overlay_cylinders_with_heat(
    ax,
    centers: np.ndarray,
    heat_powers: Optional[np.ndarray] = None,
    *,
    cylinder_radius: float = 0.5,
    linewidth: float = 1.0,
    fontsize: float = 7.0,
) -> None:
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    powers = (
        np.asarray(heat_powers, dtype=np.float32).reshape(-1)
        if heat_powers is not None
        else np.zeros((centers.shape[0],), dtype=np.float32)
    )
    max_abs_power = float(np.max(np.abs(powers))) if powers.size else 0.0
    show_heat = powers.size >= centers.shape[0] and max_abs_power > 0.0

    for idx, (cx, cy) in enumerate(centers):
        qdot = float(powers[idx]) if idx < powers.size else 0.0
        color = "#b2182b" if show_heat and qdot > 0.0 else "#2166ac" if show_heat and qdot < 0.0 else "k"
        ax.add_patch(plt.Circle((cx, cy), cylinder_radius, fill=False, color=color, linewidth=linewidth + (0.7 if show_heat else 0.0)))
        if show_heat:
            ax.text(
                cx,
                cy + 1.35 * cylinder_radius,
                f"C{idx} q={_format_heat_power(qdot)}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=color,
                weight="bold",
                bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.82, "boxstyle": "round,pad=0.18", "linewidth": 0.7},
                zorder=20,
            )


def _plot_field(
    ax,
    data: np.ndarray,
    title: str,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    centers: np.ndarray,
    vmin=None,
    vmax=None,
    cmap: str = "coolwarm",
    cylinder_radius: float = 0.5,
    show_ticks: bool = False,
    heat_powers: Optional[np.ndarray] = None,
):
    im = ax.imshow(
        data,
        origin="lower",
        extent=_grid_extent(x_grid, y_grid),
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _overlay_cylinders_with_heat(ax, centers, heat_powers, cylinder_radius=cylinder_radius, linewidth=1.0, fontsize=7.0)
    return im


def save_quicklook(
    out_path: Path,
    gt_field: np.ndarray,
    det_field: np.ndarray,
    samples: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    centers: np.ndarray,
    channel: int = 3,
    channel_name: str = "omega",
    heat_powers: Optional[np.ndarray] = None,
) -> None:
    """Save a 2x3 figure comparing GT, deterministic, sample, and ensemble."""
    sample0 = samples[0]
    ens_mean = samples.mean(axis=0)
    ens_std = samples.std(axis=0)
    gt = gt_field[channel]
    det = det_field[channel]
    gen = sample0[channel]
    gen_mean = ens_mean[channel]
    gen_std = ens_std[channel]
    err = gen_mean - gt
    cmap = _channel_cmap(channel)
    vmin, vmax = _robust_symmetric_limits(gt) if channel == 3 else (None, None)
    err_abs = float(max(np.percentile(np.abs(err), 99.0), 1e-8))

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=150, constrained_layout=True)
    ims = []
    ims.append(_plot_field(axes[0, 0], gt, f"GT {channel_name}", x_grid, y_grid, centers, vmin, vmax, cmap=cmap, heat_powers=heat_powers))
    ims.append(_plot_field(axes[0, 1], det, f"Deterministic {channel_name}", x_grid, y_grid, centers, vmin, vmax, cmap=cmap, heat_powers=heat_powers))
    ims.append(_plot_field(axes[0, 2], gen, f"Generated sample {channel_name}", x_grid, y_grid, centers, vmin, vmax, cmap=cmap, heat_powers=heat_powers))
    ims.append(_plot_field(axes[1, 0], gen_mean, f"Generated ensemble mean {channel_name}", x_grid, y_grid, centers, vmin, vmax, cmap=cmap, heat_powers=heat_powers))
    ims.append(_plot_field(axes[1, 1], gen_std, f"Generated ensemble std {channel_name}", x_grid, y_grid, centers, cmap="magma", heat_powers=heat_powers))
    ims.append(_plot_field(axes[1, 2], err, f"Ensemble mean - GT {channel_name}", x_grid, y_grid, centers, -err_abs, err_abs, cmap="coolwarm", heat_powers=heat_powers))
    for ax, im in zip(axes.reshape(-1), ims):
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out_path)
    plt.close(fig)


def save_stage1_quicklook(
    out_path: Path,
    gt_field: np.ndarray,
    mean_field: np.ndarray,
    target_grid: np.ndarray,
    recon_target: np.ndarray,
    recon_field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    centers: np.ndarray,
    target_mode: str,
    channel: int = 3,
    channel_name: str = "omega",
    heat_powers: Optional[np.ndarray] = None,
) -> None:
    """Save a 2x3 figure for stage-1 AE reconstruction."""
    gt = gt_field[channel]
    baseline = mean_field[channel] if target_mode == "residual" else np.zeros_like(gt)
    recon = recon_field[channel]
    target = target_grid[channel]
    recon_t = recon_target[channel]
    err = recon - gt
    cmap = _channel_cmap(channel)
    field_vmin, field_vmax = _robust_symmetric_limits(gt) if channel == 3 else (None, None)
    target_vmin, target_vmax = _robust_symmetric_limits(target) if channel == 3 else (None, None)
    err_abs = float(max(np.percentile(np.abs(err), 99.0), 1e-8))

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=150, constrained_layout=True)
    ims = []
    ims.append(_plot_field(axes[0, 0], gt, f"GT {channel_name}", x_grid, y_grid, centers, field_vmin, field_vmax, cmap=cmap, heat_powers=heat_powers))
    baseline_title = f"Mean {channel_name}" if target_mode == "residual" else "Zero baseline"
    ims.append(_plot_field(axes[0, 1], baseline, baseline_title, x_grid, y_grid, centers, field_vmin, field_vmax, cmap=cmap, heat_powers=heat_powers))
    ims.append(_plot_field(axes[0, 2], recon, f"AE reconstruction {channel_name}", x_grid, y_grid, centers, field_vmin, field_vmax, cmap=cmap, heat_powers=heat_powers))
    target_title = f"Target residual {channel_name}" if target_mode == "residual" else f"Target field {channel_name}"
    recon_title = f"Reconstructed residual {channel_name}" if target_mode == "residual" else f"Reconstructed field {channel_name}"
    ims.append(_plot_field(axes[1, 0], target, target_title, x_grid, y_grid, centers, target_vmin, target_vmax, cmap=cmap, heat_powers=heat_powers))
    ims.append(_plot_field(axes[1, 1], recon_t, recon_title, x_grid, y_grid, centers, target_vmin, target_vmax, cmap=cmap, heat_powers=heat_powers))
    ims.append(_plot_field(axes[1, 2], err, f"AE reconstruction - GT {channel_name}", x_grid, y_grid, centers, -err_abs, err_abs, cmap="coolwarm", heat_powers=heat_powers))
    for ax, im in zip(axes.reshape(-1), ims):
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out_path)
    plt.close(fig)


def _mean_sq(a: np.ndarray) -> float:
    return float(np.mean(np.square(a)))


def temporal_smoothness_np(cycle: np.ndarray) -> float:
    """mean_t ||field[t+1] - field[t]||^2 for [T,C,H,W]."""
    if cycle.shape[0] < 2:
        return 0.0
    return _mean_sq(cycle[1:] - cycle[:-1])


def cycle_closure_np(cycle: np.ndarray) -> float:
    return _mean_sq(cycle[0] - cycle[-1])


def compute_per_phase_metrics(
    gt_cycle: np.ndarray,
    det_cycle: np.ndarray,
    generated_samples: np.ndarray,
    tau_values: np.ndarray,
    phase_indices: np.ndarray,
    omega_channel: int,
) -> list[Dict[str, float]]:
    gen_mean = generated_samples.mean(axis=0)
    gen_std = generated_samples.std(axis=0)
    rows = []
    for t in range(gt_cycle.shape[0]):
        rows.append(
            {
                "phase_order": int(t),
                "phase_index": int(phase_indices[t]),
                "tau": float(tau_values[t]),
                "det_mse_per_phase": mse_np(det_cycle[t], gt_cycle[t]),
                "gen_mean_mse_per_phase": mse_np(gen_mean[t], gt_cycle[t]),
                "det_omega_mse_per_phase": mse_np(det_cycle[t, omega_channel], gt_cycle[t, omega_channel]),
                "gen_mean_omega_mse_per_phase": mse_np(gen_mean[t, omega_channel], gt_cycle[t, omega_channel]),
                "gen_sample_diversity_per_phase": float(gen_std[t].mean()),
            }
        )
    return rows


def compute_cycle_metrics(
    gt_cycle: np.ndarray,
    mean_grid: np.ndarray,
    det_cycle: np.ndarray,
    generated_samples: np.ndarray,
    omega_channel: int,
) -> Dict[str, float]:
    gen_mean = generated_samples.mean(axis=0)
    gen_std = generated_samples.std(axis=0)
    eps = 1e-8
    gt_residual = gt_cycle - mean_grid[None, ...]
    det_residual = det_cycle - mean_grid[None, ...]
    gen_mean_residual = gen_mean - mean_grid[None, ...]
    gt_res_energy = float(np.mean(gt_residual ** 2)) + eps
    gt_enstrophy = float(np.mean(gt_cycle[:, omega_channel] ** 2)) + eps
    return {
        "det_cycle_mse": mse_np(det_cycle, gt_cycle),
        "gen_mean_cycle_mse": mse_np(gen_mean, gt_cycle),
        "det_cycle_rel_l2": rel_l2_np(det_cycle, gt_cycle),
        "gen_mean_cycle_rel_l2": rel_l2_np(gen_mean, gt_cycle),
        "det_omega_cycle_mse": mse_np(det_cycle[:, omega_channel], gt_cycle[:, omega_channel]),
        "gen_mean_omega_cycle_mse": mse_np(gen_mean[:, omega_channel], gt_cycle[:, omega_channel]),
        "generated_diversity_mean_std": float(gen_std.mean()),
        "generated_omega_diversity_mean_std": float(gen_std[:, omega_channel].mean()),
        "residual_energy_ratio_gen_mean": float(np.mean(gen_mean_residual ** 2) / gt_res_energy),
        "residual_energy_ratio_det": float(np.mean(det_residual ** 2) / gt_res_energy),
        "enstrophy_ratio_gen_mean": float(np.mean(gen_mean[:, omega_channel] ** 2) / gt_enstrophy),
        "enstrophy_ratio_det": float(np.mean(det_cycle[:, omega_channel] ** 2) / gt_enstrophy),
        "temporal_smoothness_gen": temporal_smoothness_np(gen_mean),
        "temporal_smoothness_gt": temporal_smoothness_np(gt_cycle),
        "temporal_smoothness_det": temporal_smoothness_np(det_cycle),
        "cycle_closure_error_gen": cycle_closure_np(gen_mean),
        "cycle_closure_error_gt": cycle_closure_np(gt_cycle),
        "cycle_closure_error_det": cycle_closure_np(det_cycle),
    }


def write_per_phase_csv(out_path: Path, rows: list[Dict[str, float]]) -> None:
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _imshow_with_cylinders(
    ax,
    data: np.ndarray,
    title: str,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    centers: np.ndarray,
    vmin=None,
    vmax=None,
    cmap: str = "coolwarm",
    cylinder_radius: float = 0.5,
    show_ticks: bool = False,
    heat_powers: Optional[np.ndarray] = None,
):
    im = ax.imshow(
        data,
        origin="lower",
        extent=_grid_extent(x_grid, y_grid),
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    _overlay_cylinders_with_heat(ax, centers, heat_powers, cylinder_radius=cylinder_radius, linewidth=1.0, fontsize=7.0)
    return im


def save_cycle_gif(
    out_path: Path,
    gt_cycle: np.ndarray,
    det_cycle: np.ndarray,
    generated_samples: np.ndarray,
    generated_mean: np.ndarray,
    generated_std: np.ndarray,
    tau_values: np.ndarray,
    phase_indices: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    centers: np.ndarray,
    *,
    channel: int = 3,
    channel_name: str = "omega",
    fps: float = 6.0,
    heat_powers: Optional[np.ndarray] = None,
) -> None:
    """Write a 2x3 animated cycle diagnostic for one channel."""
    sample0 = generated_samples[0]
    field_vmin, field_vmax = _robust_symmetric_limits(gt_cycle[:, channel]) if channel == 3 else (None, None)
    field_cmap = _channel_cmap(channel)
    std_vmax = float(max(generated_std[:, channel].max(), 1e-8))
    err = generated_mean[:, channel] - gt_cycle[:, channel]
    err_abs = float(max(np.percentile(np.abs(err), 99.0), 1e-8))
    frame_l2 = np.sqrt(np.mean((generated_mean - gt_cycle) ** 2, axis=(1, 2, 3)))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=110, constrained_layout=True)
    axes_flat = axes.reshape(-1)
    titles = [
        f"GT {channel_name}",
        f"Deterministic {channel_name}",
        f"Generated sample {channel_name}",
        f"Generated mean {channel_name}",
        f"Generated std {channel_name}",
        f"Generated mean - GT {channel_name}",
    ]
    initial = [
        gt_cycle[0, channel],
        det_cycle[0, channel],
        sample0[0, channel],
        generated_mean[0, channel],
        generated_std[0, channel],
        err[0],
    ]
    limits = [
        (field_vmin, field_vmax),
        (field_vmin, field_vmax),
        (field_vmin, field_vmax),
        (field_vmin, field_vmax),
        (0.0, std_vmax),
        (-err_abs, err_abs),
    ]
    ims = []
    panel_cmaps = [field_cmap, field_cmap, field_cmap, field_cmap, "magma", "coolwarm"]
    for ax, data, title, (vmin, vmax), cmap in zip(axes_flat, initial, titles, limits, panel_cmaps):
        ims.append(_imshow_with_cylinders(ax, data, title, x_grid, y_grid, centers, vmin, vmax, cmap=cmap, heat_powers=heat_powers))
    for ax, im in zip(axes_flat, ims):
        fig.colorbar(im, ax=ax, shrink=0.78)

    def update(frame_idx: int):
        frame_data = [
            gt_cycle[frame_idx, channel],
            det_cycle[frame_idx, channel],
            sample0[frame_idx, channel],
            generated_mean[frame_idx, channel],
            generated_std[frame_idx, channel],
            err[frame_idx],
        ]
        for im, data in zip(ims, frame_data):
            im.set_data(data)
        fig.suptitle(
            f"phase {int(phase_indices[frame_idx])} | tau={tau_values[frame_idx]:.4f} | avg L2 err={frame_l2[frame_idx]:.4e}",
            fontsize=12,
        )
        return ims

    update(0)
    ani = animation.FuncAnimation(fig, update, frames=gt_cycle.shape[0], interval=1000.0 / max(float(fps), 1e-6), blit=False)
    ani.save(out_path, writer=animation.PillowWriter(fps=max(1, int(round(fps)))))
    plt.close(fig)


def save_cycle_montage(
    out_path: Path,
    gt_cycle: np.ndarray,
    det_cycle: np.ndarray,
    generated_samples: np.ndarray,
    generated_mean: np.ndarray,
    generated_std: np.ndarray,
    tau_values: np.ndarray,
    phase_indices: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    centers: np.ndarray,
    *,
    channel: int = 3,
    channel_name: str = "omega",
    heat_powers: Optional[np.ndarray] = None,
) -> None:
    T = gt_cycle.shape[0]
    selected = np.unique(np.clip(np.round(np.linspace(0, T - 1, num=min(4, T))).astype(int), 0, T - 1))
    sample0 = generated_samples[0]
    field_vmin, field_vmax = _robust_symmetric_limits(gt_cycle[:, channel]) if channel == 3 else (None, None)
    field_cmap = _channel_cmap(channel)
    std_vmax = float(max(generated_std[:, channel].max(), 1e-8))
    err = generated_mean[:, channel] - gt_cycle[:, channel]
    err_abs = float(max(np.percentile(np.abs(err), 99.0), 1e-8))

    fig, axes = plt.subplots(len(selected), 6, figsize=(18, 3.2 * len(selected)), dpi=140, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, t in enumerate(selected):
        panels = [
            (gt_cycle[t, channel], f"GT {channel_name}", field_vmin, field_vmax),
            (det_cycle[t, channel], f"Det {channel_name}", field_vmin, field_vmax),
            (sample0[t, channel], f"Sample {channel_name}", field_vmin, field_vmax),
            (generated_mean[t, channel], f"Mean {channel_name}", field_vmin, field_vmax),
            (generated_std[t, channel], f"Std {channel_name}", 0.0, std_vmax),
            (err[t], "Mean - GT", -err_abs, err_abs),
        ]
        for col, (data, title, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            cmap = "magma" if col == 4 else ("coolwarm" if col == 5 else field_cmap)
            im = _imshow_with_cylinders(ax, data, f"{title}\nphase {int(phase_indices[t])}, tau={tau_values[t]:.3f}", x_grid, y_grid, centers, vmin, vmax, cmap=cmap, heat_powers=heat_powers)
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(out_path)
    plt.close(fig)


def save_gt_generated_cycle_gif(
    out_path: Path,
    gt_cycle: np.ndarray,
    generated_cycle: np.ndarray,
    tau_values: np.ndarray,
    phase_indices: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    centers: np.ndarray,
    *,
    channel: int = 3,
    channel_name: str = "omega",
    fps: float = 10.0,
    heat_powers: Optional[np.ndarray] = None,
) -> None:
    """Simple deterministic-evaluator-style two-panel GT vs generated GIF."""
    gt = gt_cycle[:, channel]
    gen = generated_cycle[:, channel]
    vmin, vmax = _robust_symmetric_limits(gt) if channel == 3 else (None, None)
    cmap = _channel_cmap(channel)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=120, constrained_layout=True)
    im_gt = _imshow_with_cylinders(axes[0], gt[0], f"GT {channel_name}", x_grid, y_grid, centers, vmin, vmax, cmap=cmap, show_ticks=True, heat_powers=heat_powers)
    im_gen = _imshow_with_cylinders(axes[1], gen[0], f"Generated {channel_name}", x_grid, y_grid, centers, vmin, vmax, cmap=cmap, show_ticks=True, heat_powers=heat_powers)

    def update(frame_idx: int):
        im_gt.set_data(gt[frame_idx])
        im_gen.set_data(gen[frame_idx])
        fig.suptitle(f"Phase {int(phase_indices[frame_idx])} | Tau: {tau_values[frame_idx]:.3f}")
        return [im_gt, im_gen]

    update(0)
    ani = animation.FuncAnimation(fig, update, frames=gt.shape[0], interval=1000.0 / max(float(fps), 1e-6), blit=False)
    ani.save(out_path, writer=animation.PillowWriter(fps=max(1, int(round(fps)))))
    plt.close(fig)


def _repeat_structure(structure: Dict[str, torch.Tensor], count: int, device: torch.device) -> Dict[str, torch.Tensor]:
    repeated = {}
    for key, value in structure.items():
        tensor = value.to(device)
        if tensor.shape[0] == 1:
            repeated[key] = tensor.expand((count,) + tuple(tensor.shape[1:])).contiguous()
        else:
            repeated[key] = tensor
    return repeated


def _latent_shape(flow: LatentRectifiedFlow, batch: int, device: torch.device, dtype: torch.dtype) -> tuple[int, int, int, int]:
    latent_h = flow.ae.H_pad // (2 ** flow.ae.n_levels)
    latent_w = flow.ae.W_pad // (2 ** flow.ae.n_levels)
    return (batch, flow.ae.latent_ch, latent_h, latent_w)


def _cycle_initial_latent(
    flow: LatentRectifiedFlow,
    mode: str,
    tau_values: np.ndarray,
    sample_indices: range,
    phase_slice: slice,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Build correlated latent starts for [S_chunk, T_chunk] sampling."""
    if mode == "independent":
        return None

    latents = []
    tau = torch.from_numpy(tau_values[phase_slice]).to(device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    for s in sample_indices:
        gen = torch.Generator(device=device)
        gen.manual_seed(1234 + int(s) * 100003)
        if mode == "shared":
            base = torch.randn(_latent_shape(flow, 1, device, dtype), generator=gen, device=device, dtype=dtype)
            z = base[:, None].expand(1, tau.shape[1], -1, -1, -1)
        elif mode == "harmonic":
            a = torch.randn(_latent_shape(flow, 1, device, dtype), generator=gen, device=device, dtype=dtype)
            b = torch.randn(_latent_shape(flow, 1, device, dtype), generator=gen, device=device, dtype=dtype)
            z = torch.cos(2.0 * np.pi * tau) * a[:, None] + torch.sin(2.0 * np.pi * tau) * b[:, None]
        else:
            raise ValueError(f"Unknown cycle noise mode: {mode}")
        latents.append(z)
    return torch.cat(latents, dim=0).reshape(-1, *_latent_shape(flow, 1, device, dtype)[1:])


def _run_deterministic_cycle(
    det_model,
    det_model_cfg: Dict,
    sample: Dict,
    cfg: Dict,
    stats: GridStats,
    device: torch.device,
    phase_chunk_size: int,
    global_cond_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic mean/residual/field and global conditions as CPU tensors."""
    tau_values = sample["tau_values"]
    query_time_values = sample.get("query_time_values", tau_values)
    T = len(tau_values)
    x_base = sample["x_grid"].to(device)
    y_base = sample["y_grid"].to(device)
    det_means, det_residuals, det_fields, global_conds = [], [], [], []
    for start in range(0, T, phase_chunk_size):
        end = min(start + phase_chunk_size, T)
        tc = end - start
        structure = _repeat_structure(sample["structure"], tc, device)
        x_grid = x_base.expand(tc, -1, -1).contiguous()
        y_grid = y_base.expand(tc, -1, -1).contiguous()
        tau = torch.from_numpy(tau_values[start:end]).to(device=device, dtype=torch.float32).view(tc, 1)
        query_time = torch.from_numpy(query_time_values[start:end]).to(device=device, dtype=torch.float32).view(tc, 1)
        det_out = deterministic_grid_forward(
            det_model,
            structure,
            x_grid,
            y_grid,
            tau,
            query_time=query_time,
            query_batch_size=int(cfg["generation"].get("det_query_batch_size", 32768)),
        )
        global_cond = _build_checkpoint_global_condition_vector(det_out, structure, expected_dim=global_cond_dim)
        det_means.append(det_out["pred_mean"].detach().cpu())
        det_residuals.append(det_out["pred_residual"].detach().cpu())
        det_fields.append(det_out["pred_field"].detach().cpu())
        global_conds.append(global_cond.detach().cpu())
    return torch.cat(det_means, dim=0), torch.cat(det_residuals, dim=0), torch.cat(det_fields, dim=0), torch.cat(global_conds, dim=0)


def run_stage2_cycle(args: argparse.Namespace, cfg: Dict, checkpoint_path: Path, packed_path: Path, device: torch.device) -> None:
    """Generate one stochastic tau-conditioned sample per selected phase bin."""
    if int(args.phase_stride) < 1:
        raise ValueError("--phase-stride must be >= 1.")
    if int(args.phase_chunk_size) < 1 or int(args.sample_chunk_size) < 1:
        raise ValueError("--phase-chunk-size and --sample-chunk-size must be >= 1.")

    with h5py.File(packed_path, "r") as h5:
        _, grp = _select_case_group(h5, args.split, args.dataset_case_id)
        num_bins = len(grp["phase_bin_centers"])
    phase_indices = _phase_indices_from_args(num_bins, args)
    sample = load_case_cycle(
        packed_path,
        split=args.split,
        case_id=args.dataset_case_id,
        phase_indices=phase_indices,
        max_num_cylinders=int(cfg["dataset"].get("max_num_cylinders", 8)),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        out_dir = checkpoint_path.parent / "Evaluation_Gen" / f"stage2_case_{sample['case_id']}_cycle_{timestamp}"
    else:
        out_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(out_dir)

    flow, ema, stats, ckpt = load_generator(checkpoint_path, device)
    deterministic_checkpoint_path = ckpt.get("deterministic_checkpoint_path") or cfg.get("deterministic_model", {}).get("checkpoint_path")
    if not deterministic_checkpoint_path:
        raise KeyError("No deterministic checkpoint path was found in the stage-2 checkpoint or config.")
    det_model, det_model_cfg, det_ckpt_path = load_deterministic_model({"checkpoint_path": deterministic_checkpoint_path}, device)
    if args.disable_edge is not None and hasattr(det_model, "set_edge_disable_runtime"):
        det_model.set_edge_disable_runtime(bool(args.disable_edge))
    det_field_dim = int(det_model_cfg.get("field_dim", getattr(det_model, "cfg", object()).field_dim if hasattr(getattr(det_model, "cfg", None), "field_dim") else sample["field_dim"]))
    if det_field_dim != int(sample["field_dim"]):
        raise ValueError(
            f"Deterministic checkpoint field_dim={det_field_dim} does not match dataset field_dim={sample['field_dim']} "
            f"for channel_order={sample['channel_order']}."
        )
    add_extra_module_if_needed(
        sample,
        future_module_feature_dim=int(det_model_cfg.get("future_module_feature_dim", 0)),
        heat_power_scale=float(sample.get("heat_power_scale", 1.0)),
    )

    phase_chunk_size = int(args.phase_chunk_size)
    sample_chunk_size = int(args.sample_chunk_size)
    n_steps = int(args.n_steps if args.n_steps is not None else cfg["stage2"].get("sampling", {}).get("n_steps", 16))
    ode_solver = str(args.ode_solver if args.ode_solver is not None else cfg["stage2"].get("sampling", {}).get("ode_solver", "euler"))
    T, C, H, W = sample["field_cycle"].shape
    cycle_viz_channels = default_cycle_viz_channels(args, list(sample["channel_order"]), C)
    omega_channel, channel_name = cycle_viz_channels[-1]

    with torch.no_grad():
        det_mean_t, det_res_t, det_field_t, global_cond_t = _run_deterministic_cycle(
            det_model,
            det_model_cfg,
            sample,
            cfg,
            stats,
            device,
            phase_chunk_size,
            int(ckpt["global_cond_dim"]),
        )

        generated = torch.empty((int(args.n_samples), T, C, H, W), dtype=torch.float32)
        context = ema.average_parameters(flow.velocity_net) if ema is not None else torch.no_grad()
        with context:
            for s0 in range(0, int(args.n_samples), sample_chunk_size):
                s1 = min(s0 + sample_chunk_size, int(args.n_samples))
                s_range = range(s0, s1)
                ns = s1 - s0
                for start in range(0, T, phase_chunk_size):
                    end = min(start + phase_chunk_size, T)
                    tc = end - start
                    det_mean = det_mean_t[start:end].to(device)
                    det_res = det_res_t[start:end].to(device)
                    det_field = det_field_t[start:end].to(device)
                    x_grid = sample["x_grid"].to(device).expand(tc, -1, -1).contiguous()
                    y_grid = sample["y_grid"].to(device).expand(tc, -1, -1).contiguous()
                    tau = torch.from_numpy(sample["tau_values"][start:end]).to(device=device, dtype=torch.float32).view(tc, 1)
                    query_time = torch.from_numpy(sample.get("query_time_values", sample["tau_values"])[start:end]).to(device=device, dtype=torch.float32).view(tc, 1)
                    structure = _repeat_structure(sample["structure"], tc, device)
                    cond_grid = build_dense_condition_grid(
                        det_mean=det_mean,
                        det_residual=det_res,
                        det_field=det_field,
                        x_grid=x_grid,
                        y_grid=y_grid,
                        tau=tau,
                        thermal_time=query_time,
                        re_values=structure["re_values"],
                        stats=stats.to(device, dtype=det_mean.dtype),
                        domain_length_x=float(det_model_cfg.get("domain_length_x", 24.0)),
                        domain_length_y=float(det_model_cfg.get("domain_length_y", 12.0)),
                        re_scale=float(det_model_cfg.get("re_scale", 200.0)),
                        include_field=bool(cfg["stage2"]["conditioning"].get("include_pred_field", True)),
                    )
                    cond_grid = cond_grid.repeat(ns, 1, 1, 1)
                    global_cond = global_cond_t[start:end].to(device).repeat(ns, 1)
                    det_mean_rep = det_mean.repeat(ns, 1, 1, 1)
                    initial_latent = _cycle_initial_latent(
                        flow,
                        str(args.cycle_noise_mode),
                        sample["tau_values"],
                        s_range,
                        slice(start, end),
                        device,
                        cond_grid.dtype,
                    )
                    seed = None if initial_latent is not None else 1234 + s0 * 100003 + start
                    gen_res_norm = flow.sample(
                        cond_grid,
                        global_cond,
                        n_steps=n_steps,
                        ode_solver=ode_solver,
                        seed=seed,
                        initial_latent=initial_latent,
                    )
                    gen_res = denormalize_grid(gen_res_norm, stats.to(device, dtype=gen_res_norm.dtype))
                    gen_field = det_mean_rep + gen_res
                    generated[s0:s1, start:end] = gen_field.detach().cpu().reshape(ns, tc, C, H, W)

    gt_cycle = sample["field_cycle"].numpy()
    mean_grid = sample["mean_grid"].numpy()[0]
    det_cycle = det_field_t.numpy()
    generated_samples = generated.numpy()
    generated_mean = generated_samples.mean(axis=0)
    generated_std = generated_samples.std(axis=0)
    x_np = sample["x_grid"].numpy()[0]
    y_np = sample["y_grid"].numpy()[0]

    cycle_metrics = compute_cycle_metrics(gt_cycle, mean_grid, det_cycle, generated_samples, omega_channel)
    cycle_metrics.update(
        {
            "stage": 2,
            "mode": "cycle",
            "case_id": sample["case_id"],
            "split": args.split,
            "re": float(sample["re"]),
            "num_cylinders": int(sample["num_cylinders"]),
            "n_samples": int(args.n_samples),
            "n_phase_bins": int(T),
            "n_steps": int(n_steps),
            "ode_solver": ode_solver,
            "cycle_noise_mode": str(args.cycle_noise_mode),
            "phase_chunk_size": phase_chunk_size,
            "sample_chunk_size": sample_chunk_size,
            "checkpoint": str(checkpoint_path),
            "deterministic_checkpoint": str(det_ckpt_path),
            "field_dim": int(C),
            "channel_order": list(sample["channel_order"]),
            "viz_channel": int(omega_channel),
            "viz_field": channel_name,
            "cycle_viz_fields": [name for _, name in cycle_viz_channels],
            "cycle_mode_note": "Stage-2 was trained on phase snapshots. Cycle mode samples each tau-conditioned phase; independent noise is not temporally coherent, while shared/harmonic modes correlate the initial latent noise across phases.",
        }
    )
    per_phase_rows = compute_per_phase_metrics(gt_cycle, det_cycle, generated_samples, sample["tau_values"], sample["phase_indices"], omega_channel)

    with (out_dir / "cycle_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(cycle_metrics, f, indent=2)
    with (out_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(cycle_metrics, f, indent=2)
    write_per_phase_csv(out_dir / "per_phase_metrics.csv", per_phase_rows)

    np.savez_compressed(
        out_dir / "cycle_reconstruction.npz",
        gt_cycle=gt_cycle,
        deterministic_cycle=det_cycle,
        generated_samples=generated_samples,
        generated_mean=generated_mean,
        generated_std=generated_std,
        tau_values=sample["tau_values"],
        phase_indices=sample["phase_indices"],
        x_grid=x_np,
        y_grid=y_np,
        centers=sample["centers_np"],
        channel_order=np.asarray(sample["channel_order"]),
        re=np.array(sample["re"], dtype=np.float32),
        num_cylinders=np.array(sample["num_cylinders"], dtype=np.int64),
        heat_powers=np.asarray(sample["heat_powers"], dtype=np.float32),
        n_steps=np.array(n_steps, dtype=np.int64),
        ode_solver=np.array(ode_solver),
        cycle_noise_mode=np.array(str(args.cycle_noise_mode)),
    )

    for viz_channel, viz_name in cycle_viz_channels:
        save_cycle_gif(
            out_dir / f"cycle_{viz_name}.gif",
            gt_cycle,
            det_cycle,
            generated_samples,
            generated_mean,
            generated_std,
            sample["tau_values"],
            sample["phase_indices"],
            x_np,
            y_np,
            sample["centers_np"],
            channel=viz_channel,
            channel_name=viz_name,
            fps=float(args.gif_fps),
            heat_powers=sample["heat_powers"],
        )
        save_gt_generated_cycle_gif(
            out_dir / f"cycle_{viz_name}_gt_generated.gif",
            gt_cycle,
            generated_mean,
            sample["tau_values"],
            sample["phase_indices"],
            x_np,
            y_np,
            sample["centers_np"],
            channel=viz_channel,
            channel_name=viz_name,
            fps=float(args.gif_fps),
            heat_powers=sample["heat_powers"],
        )
        save_cycle_montage(
            out_dir / f"cycle_montage_{viz_name}.png",
            gt_cycle,
            det_cycle,
            generated_samples,
            generated_mean,
            generated_std,
            sample["tau_values"],
            sample["phase_indices"],
            x_np,
            y_np,
            sample["centers_np"],
            channel=viz_channel,
            channel_name=viz_name,
            heat_powers=sample["heat_powers"],
        )

    print(f"Output directory: {out_dir}")
    for k, v in cycle_metrics.items():
        print(f"{k}: {v}")


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if int(args.n_samples) < 1:
        raise ValueError("--n-samples must be >= 1.")
    device = select_device(args.device)
    discovery_cfg = load_config(args.config or "train_gen_config_template.json")
    checkpoint_path, requested_stage = resolve_checkpoint_for_args(args, discovery_cfg)
    ckpt_probe = safe_torch_load(checkpoint_path, map_location="cpu")
    checkpoint_stage = int(ckpt_probe.get("stage", requested_stage))
    if requested_stage != checkpoint_stage:
        raise ValueError(f"Requested stage {requested_stage}, but checkpoint has stage={checkpoint_stage}: {checkpoint_path}")
    cfg = ckpt_probe.get("config", discovery_cfg)
    if args.config is not None:
        cfg = discovery_cfg

    stage = checkpoint_stage
    target_mode = str(cfg.get("generation", {}).get("target_mode", "residual"))
    if target_mode not in {"residual", "field"}:
        raise ValueError(f"Unsupported generation.target_mode={target_mode!r}; expected 'residual' or 'field'.")
    if stage == 2 and target_mode != "residual":
        raise ValueError("Stage 2 currently expects generation.target_mode='residual' for deterministic mean + generated residual.")

    packed_path = resolve_demo_path(cfg["dataset"]["packed_h5_path"])
    if args.mode == "cycle":
        if stage != 2:
            raise ValueError("Cycle mode is implemented for stage-2 latent rectified-flow checkpoints.")
        run_stage2_cycle(args, cfg, checkpoint_path, packed_path, device)
        return

    sample = load_case_snapshot(
        packed_path,
        split=args.split,
        case_id=args.dataset_case_id,
        phase_index=int(args.phase_index),
        max_num_cylinders=int(cfg["dataset"].get("max_num_cylinders", 8)),
    )
    viz_channel, viz_field = resolve_viz_channel(args, list(sample["channel_order"]), int(sample["field_dim"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        out_dir = checkpoint_path.parent / "Evaluation_Gen" / f"stage{stage}_case_{sample['case_id']}_phase_{args.phase_index:03d}_{timestamp}"
    else:
        out_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(out_dir)

    if stage == 1:
        ae, stats, ckpt = load_ae(checkpoint_path, device)
        target_t = sample["residual_grid"] if target_mode == "residual" else sample["field_grid"]
        with torch.no_grad():
            target = target_t.to(device=device)
            target_norm = normalize_grid(target, stats.to(device, dtype=target.dtype))
            recon_norm, latent = ae(target_norm)
            recon_target = denormalize_grid(recon_norm, stats.to(device, dtype=recon_norm.dtype))
            if target_mode == "residual":
                recon_field = sample["mean_grid"].to(device=device) + recon_target
            else:
                recon_field = recon_target

        gt_field = sample["field_grid"].numpy()[0]
        mean_field = sample["mean_grid"].numpy()[0]
        target_np = target_t.numpy()[0]
        recon_target_np = recon_target.detach().cpu().numpy()[0]
        recon_field_np = recon_field.detach().cpu().numpy()[0]
        x_np = sample["x_grid"].numpy()[0]
        y_np = sample["y_grid"].numpy()[0]

        save_stage1_quicklook(
            out_dir / f"quicklook_stage1_{viz_field}.png",
            gt_field=gt_field,
            mean_field=mean_field,
            target_grid=target_np,
            recon_target=recon_target_np,
            recon_field=recon_field_np,
            x_grid=x_np,
            y_grid=y_np,
            centers=sample["centers_np"],
            target_mode=target_mode,
            channel=viz_channel,
            channel_name=viz_field,
            heat_powers=sample["heat_powers"],
        )

        np.savez_compressed(
            out_dir / "stage1_reconstruction.npz",
            gt_field=gt_field,
            mean_field=mean_field,
            target_grid=target_np,
            reconstructed_target=recon_target_np,
            reconstructed_field=recon_field_np,
            x_grid=x_np,
            y_grid=y_np,
            centers=sample["centers_np"],
            heat_powers=np.asarray(sample["heat_powers"], dtype=np.float32),
            tau=np.array([sample["tau"]], dtype=np.float32),
            channel_order=np.asarray(sample["channel_order"]),
        )

        metrics = {
            "stage": 1,
            "case_id": sample["case_id"],
            "phase_index": int(sample["phase_index"]),
            "tau": float(sample["tau"]),
            "re": float(sample["re"]),
            "num_cylinders": int(sample["num_cylinders"]),
            "field_dim": int(sample["field_dim"]),
            "channel_order": list(sample["channel_order"]),
            "viz_channel": int(viz_channel),
            "viz_field": viz_field,
            "target_mode": target_mode,
            "latent_shape": list(latent.shape),
            "field_mse": mse_np(recon_field_np, gt_field),
            "field_rel_l2": rel_l2_np(recon_field_np, gt_field),
            "selected_channel_mse": mse_np(recon_field_np[viz_channel], gt_field[viz_channel]),
            "target_mse": mse_np(recon_target_np, target_np),
            "target_rel_l2": rel_l2_np(recon_target_np, target_np),
            "checkpoint": str(checkpoint_path),
        }
    else:
        flow, ema, stats, ckpt = load_generator(checkpoint_path, device)

        deterministic_checkpoint_path = ckpt.get("deterministic_checkpoint_path") or cfg.get("deterministic_model", {}).get("checkpoint_path")
        if not deterministic_checkpoint_path:
            raise KeyError("No deterministic checkpoint path was found in the stage-2 checkpoint or config.")
        det_model, det_model_cfg, det_ckpt_path = load_deterministic_model(
            {"checkpoint_path": deterministic_checkpoint_path},
            device,
        )
        if args.disable_edge is not None and hasattr(det_model, "set_edge_disable_runtime"):
            det_model.set_edge_disable_runtime(bool(args.disable_edge))
        det_field_dim = int(det_model_cfg.get("field_dim", getattr(det_model, "cfg", object()).field_dim if hasattr(getattr(det_model, "cfg", None), "field_dim") else sample["field_dim"]))
        if det_field_dim != int(sample["field_dim"]):
            raise ValueError(
                f"Deterministic checkpoint field_dim={det_field_dim} does not match dataset field_dim={sample['field_dim']} "
                f"for channel_order={sample['channel_order']}."
            )
        add_extra_module_if_needed(
            sample,
            future_module_feature_dim=int(det_model_cfg.get("future_module_feature_dim", 0)),
            heat_power_scale=float(sample.get("heat_power_scale", 1.0)),
        )

        # Move structure and grids to device.
        structure = {k: v.to(device) for k, v in sample["structure"].items()}
        x_grid = sample["x_grid"].to(device)
        y_grid = sample["y_grid"].to(device)
        tau = torch.tensor([[sample["tau"]]], dtype=torch.float32, device=device)
        query_time = torch.tensor([[sample.get("query_time", sample["tau"])]], dtype=torch.float32, device=device)

        with torch.no_grad():
            det_out = deterministic_grid_forward(
                det_model,
                structure,
                x_grid,
                y_grid,
                tau,
                query_time=query_time,
                query_batch_size=int(cfg["generation"].get("det_query_batch_size", 32768)),
            )
            cond_grid = build_dense_condition_grid(
                det_mean=det_out["pred_mean"],
                det_residual=det_out["pred_residual"],
                det_field=det_out["pred_field"],
                x_grid=x_grid,
                y_grid=y_grid,
                tau=tau,
                thermal_time=query_time,
                re_values=structure["re_values"],
                stats=stats.to(device, dtype=det_out["pred_mean"].dtype),
                domain_length_x=float(det_model_cfg.get("domain_length_x", 24.0)),
                domain_length_y=float(det_model_cfg.get("domain_length_y", 12.0)),
                re_scale=float(det_model_cfg.get("re_scale", 200.0)),
                include_field=bool(cfg["stage2"]["conditioning"].get("include_pred_field", True)),
            )
            global_cond = _build_checkpoint_global_condition_vector(det_out, structure, expected_dim=int(ckpt["global_cond_dim"]))

            n_steps = int(args.n_steps if args.n_steps is not None else cfg["stage2"].get("sampling", {}).get("n_steps", 16))
            ode_solver = str(args.ode_solver if args.ode_solver is not None else cfg["stage2"].get("sampling", {}).get("ode_solver", "euler"))
            samples = []
            # Use EMA weights for sampling when present.
            context = ema.average_parameters(flow.velocity_net) if ema is not None else torch.no_grad()
            with context:
                for s in range(int(args.n_samples)):
                    gen_res_norm = flow.sample(cond_grid, global_cond, n_steps=n_steps, ode_solver=ode_solver, seed=1234 + s)
                    gen_res = denormalize_grid(gen_res_norm, stats.to(device, dtype=gen_res_norm.dtype))
                    # Final generated physical field uses deterministic mean + generated residual.
                    gen_field = det_out["pred_mean"] + gen_res
                    samples.append(gen_field.detach().cpu())
            samples_t = torch.cat(samples, dim=0)  # [S,C,H,W]

        gt_field = sample["field_grid"].numpy()[0]
        det_field = det_out["pred_field"].detach().cpu().numpy()[0]
        gen_samples = samples_t.numpy()
        x_np = sample["x_grid"].numpy()[0]
        y_np = sample["y_grid"].numpy()[0]

        save_quicklook(
            out_dir / f"quicklook_{viz_field}.png",
            gt_field=gt_field,
            det_field=det_field,
            samples=gen_samples,
            x_grid=x_np,
            y_grid=y_np,
            centers=sample["centers_np"],
            channel=viz_channel,
            channel_name=viz_field,
            heat_powers=sample["heat_powers"],
        )
        organization_paths = render_soft_organization(
            out_dir,
            det_out,
            sample,
            tau_value=float(sample["tau"]),
            phase_idx=int(sample["phase_index"]),
            threshold=float(args.organization_threshold),
            topk_me_links=int(args.topk_me_links),
            organization_view=str(args.organization_view),
            topk_cylinders=int(args.organization_topk_cylinders),
            topk_env=int(args.organization_topk_env),
            min_gap=float(args.organization_min_gap),
            show_table=bool(args.organization_table),
            show_disabled_edges=bool(args.show_disabled_edges),
            visualize_disabled_edges=bool(getattr(det_model, "cfg", None) is not None and det_model.cfg.DISABLE_EDGE and det_model.cfg.disable_edge_apply_to_visualization),
        )
        org_arrays = extract_organization_arrays(det_out, sample)

        np.savez_compressed(
            out_dir / "gen_reconstruction.npz",
            gt_field=gt_field,
            deterministic_field=det_field,
            generated_samples=gen_samples,
            x_grid=x_np,
            y_grid=y_np,
            centers=sample["centers_np"],
            heat_powers=np.asarray(sample["heat_powers"], dtype=np.float32),
            tau=np.array([sample["tau"]], dtype=np.float32),
            channel_order=np.asarray(sample["channel_order"]),
            hyper_parent_index=org_arrays["hyper_parent_index"].astype(np.int64),
            A_mh=org_arrays["A_mh_raw"].astype(np.float32),
            A_eh=org_arrays["A_eh_raw"].astype(np.float32),
            A_mh_effective=org_arrays["A_mh_effective"].astype(np.float32),
            A_eh_effective=org_arrays["A_eh_effective"].astype(np.float32),
            hyper_effective_env_token_count=org_arrays["hyper_effective_env_token_count"].astype(np.float32),
        )

        ens_mean = gen_samples.mean(axis=0)
        metrics = {
            "stage": 2,
            "case_id": sample["case_id"],
            "phase_index": int(sample["phase_index"]),
            "tau": float(sample["tau"]),
            "re": float(sample["re"]),
            "num_cylinders": int(sample["num_cylinders"]),
            "field_dim": int(sample["field_dim"]),
            "channel_order": list(sample["channel_order"]),
            "viz_channel": int(viz_channel),
            "viz_field": viz_field,
            "n_samples": int(args.n_samples),
            "n_steps": int(n_steps),
            "ode_solver": ode_solver,
            "det_mse": mse_np(det_field, gt_field),
            "gen_mean_mse": mse_np(ens_mean, gt_field),
            "det_rel_l2": rel_l2_np(det_field, gt_field),
            "gen_mean_rel_l2": rel_l2_np(ens_mean, gt_field),
            "det_selected_channel_mse": mse_np(det_field[viz_channel], gt_field[viz_channel]),
            "gen_mean_selected_channel_mse": mse_np(ens_mean[viz_channel], gt_field[viz_channel]),
            "sample_diversity_mean_std": float(gen_samples.std(axis=0).mean()),
            "checkpoint": str(checkpoint_path),
            "deterministic_checkpoint": str(det_ckpt_path),
            "hyper_parent_index": org_arrays["hyper_parent_index"].astype(int).tolist(),
            "hyper_effective_env_token_count": org_arrays["hyper_effective_env_token_count"].astype(float).tolist(),
            "effective_env_coverage": float(np.mean(np.max(org_arrays["A_eh_effective"], axis=1) > 1e-6)),
            "organization_paths": organization_paths,
        }

    with (out_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Output directory: {out_dir}")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
