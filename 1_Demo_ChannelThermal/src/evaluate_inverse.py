from __future__ import annotations

"""Evaluate a trained ChannelThermal inverse generator on target JSON specs."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-inverse")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from channelthermal_datasets import CHANNEL_ORDER
from channelthermal_model_utils import current_timestamp, load_trusted_checkpoint, resolve_demo_path, select_device, strip_module_prefix, write_json
from model_inverse import ThermalInverseDesignFlow, channel_clearance_diagnostics
from thermal_inverse_kpi import DEFAULT_KPI_NAMES, build_target_spec_vector, score_candidate_kpis, compute_steady_thermal_kpis
from train_inverse import (
    ThermalInverseDesignDataset,
    load_forward_model,
    predict_candidate_with_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a ChannelThermal inverse-design checkpoint.")
    parser.add_argument("--inverse-run", type=str, default="auto", help="Inverse run directory, checkpoint path, or auto.")
    parser.add_argument("--checkpoint-name", type=str, default="best_verified_model.pt", help="Inverse checkpoint filename or fallback selector.")
    parser.add_argument("--target", type=str, required=True, help="Target JSON path.")
    parser.add_argument("--dataset", type=str, default=None, help="Packed HDF5 override for fixed conditions/reference grid.")
    parser.add_argument("--reference-split", type=str, default="test", help="Dataset split used for fixed conditions.")
    parser.add_argument("--reference-case-index", type=int, default=0, help="Reference case index in split.")
    parser.add_argument("--n-samples", type=int, default=64, help="Number of inverse candidates to sample.")
    parser.add_argument("--n-steps", type=int, default=32, help="Rectified-flow ODE steps.")
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    parser.add_argument("--output-dir", type=str, default=None, help="Evaluation output directory.")
    parser.add_argument("--query-batch-size", type=int, default=32768, help="Forward verifier grid query batch size.")
    parser.add_argument("--forward-run-dir", type=str, default=None, help="Override forward_model.run_dir.")
    parser.add_argument("--forward-checkpoint-name", type=str, default=None, help="Override forward_model.checkpoint_name.")
    parser.add_argument("--local-surrogate-checkpoint-path", type=str, default=None, help="Override local surrogate checkpoint path.")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().numpy())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def latest_inverse_run(root: Path) -> Path:
    runs = sorted([path for path in root.glob("Run_*") if path.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No inverse Run_* directories found under {root}.")
    return runs[-1]


def resolve_inverse_checkpoint(inverse_run: str, checkpoint_name: str) -> Path:
    raw = str(inverse_run)
    if raw.lower() == "auto":
        run_dir = latest_inverse_run(resolve_demo_path("./Saved_Model_Inverse"))
    else:
        path = resolve_demo_path(raw)
        if path.suffix == ".pt":
            return path
        run_dir = path
    requested = run_dir / checkpoint_name
    if requested.exists():
        return requested.resolve()
    for name in ("best_verified_model.pt", "best_model.pt", "latest_model.pt"):
        candidate = run_dir / name
        if candidate.exists():
            print(f"[warning] {requested.name} not found; using {candidate.name}.")
            return candidate.resolve()
    raise FileNotFoundError(f"No inverse checkpoint found in {run_dir}.")


def load_inverse_checkpoint(path: Path, device: torch.device) -> Tuple[ThermalInverseDesignFlow, Dict[str, Any]]:
    checkpoint = load_trusted_checkpoint(path, map_location=device)
    model = ThermalInverseDesignFlow(checkpoint["model_config"]).to(device)
    model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
    model.eval()
    return model, checkpoint


def load_target_payload(path: str | Path) -> Dict[str, Any]:
    target_path = resolve_demo_path(path)
    with target_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "kpis" not in payload:
        raise ValueError(f"Target JSON must contain a 'kpis' block: {target_path}")
    payload["_path"] = str(target_path)
    return payload


def target_spec_from_payload(
    payload: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    model: ThermalInverseDesignFlow,
) -> Dict[str, Any]:
    kpi_names = tuple(checkpoint.get("kpi_names", DEFAULT_KPI_NAMES))
    kpi_stats = checkpoint.get("kpi_stats")
    train_cfg = checkpoint.get("train_config", {})
    target_cfg = train_cfg.get("target_kpis", {}) if isinstance(train_cfg, Mapping) else {}
    prefs = payload.get("preferences", {}) if isinstance(payload.get("preferences", {}), Mapping) else {}
    constraints = {
        "num_modules_min": payload.get("num_modules_min", payload.get("num_cylinders_min")),
        "num_modules_max": payload.get("num_modules_max", payload.get("num_cylinders_max")),
        "min_center_distance": payload.get("min_center_distance", prefs.get("min_center_distance")),
        "wall_clearance": payload.get("wall_clearance", prefs.get("wall_clearance")),
        "inlet_clearance": payload.get("inlet_clearance", prefs.get("inlet_clearance")),
        "outlet_clearance": payload.get("outlet_clearance", prefs.get("outlet_clearance")),
        "heat_power_total": payload.get("heat_power_total"),
    }
    vector = build_target_spec_vector(
        kpi_targets=payload.get("kpis", {}),
        kpi_names=kpi_names,
        stats=kpi_stats,
        normalize=bool(target_cfg.get("normalize", True)),
        num_modules_min=constraints["num_modules_min"],
        num_modules_max=constraints["num_modules_max"],
        min_center_distance=constraints["min_center_distance"],
        wall_clearance=constraints["wall_clearance"],
        inlet_clearance=constraints["inlet_clearance"],
        outlet_clearance=constraints["outlet_clearance"],
        heat_power_total=constraints["heat_power_total"],
        max_num_modules=model.max_num_modules,
        domain_length_scale=max(float(model.cfg.domain_length_x), float(model.cfg.domain_length_y)),
        heat_power_scale=float(model.cfg.heat_power_scale),
        return_spec=False,
    )
    return {
        "name": payload.get("name", "inverse_target"),
        "vector": vector,
        "kpi_names": list(kpi_names),
        "kpi_targets": dict(payload.get("kpis", {})),
        "kpi_stats": kpi_stats,
        "constraints": constraints,
        "preferences": dict(prefs),
        "target_payload": dict(payload),
    }


def _candidate_kpi_payload(
    record: Any,
    prediction: Mapping[str, Any],
    candidate: Mapping[str, Any],
    model: ThermalInverseDesignFlow,
) -> Dict[str, Any]:
    kpis = compute_steady_thermal_kpis(
        prediction["pred_field_grid"],
        x_grid=record.x_grid,
        y_grid=record.y_grid,
        channel_order=CHANNEL_ORDER,
        module_centers=prediction["centers_padded"],
        module_present=prediction["module_present"],
        heat_powers=prediction.get("heat_powers", record.heat_powers),
        module_internal_temperature=prediction.get("pred_internal_temperature"),
        module_internal_mask=record.module_internal_mask,
        interface_target=prediction.get("pred_interface"),
        interface_condition=None,
        domain={"domain_length_x": record.domain_length_x, "domain_length_y": record.domain_length_y, "module_radius": record.module_radius},
        material_params=record.material_params,
    )
    centers = np.asarray(candidate.get("centers", []), dtype=np.float32).reshape(-1, 2)
    kpis.update(channel_clearance_diagnostics(centers, domain_length_x=record.domain_length_x, domain_length_y=record.domain_length_y, module_radius=record.module_radius))
    kpis["num_modules"] = int(candidate.get("count", centers.shape[0]))
    heat_used = np.asarray(prediction.get("heat_powers", record.heat_powers), dtype=np.float32).reshape(-1)
    kpis["heat_power_total"] = float(np.sum(heat_used[: kpis["num_modules"]])) if heat_used.size else 0.0
    kpis["valid"] = bool(candidate.get("validity", {}).get("valid", False))
    return kpis


def write_candidates_csv(candidates: Sequence[Mapping[str, Any]], path: Path) -> None:
    keys = [
        "rank",
        "sample_index",
        "count",
        "valid",
        "total_score",
        "kpi_score",
        "constraint_penalty",
        "min_center_distance",
        "wall_clearance",
        "inlet_clearance",
        "outlet_clearance",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in candidates:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_kpi_scores_csv(candidates: Sequence[Mapping[str, Any]], kpi_names: Sequence[str], path: Path) -> None:
    keys = ["rank", "sample_index", "total_score", *kpi_names]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in candidates:
            kpis = row.get("verified_kpis", {})
            writer.writerow({key: row.get(key, kpis.get(key, "")) for key in keys})


def _extent(record: Any) -> Tuple[float, float, float, float]:
    return (float(np.min(record.x_grid)), float(np.max(record.x_grid)), float(np.min(record.y_grid)), float(np.max(record.y_grid)))


def _draw_layout(ax: Any, record: Any, centers: np.ndarray, *, title: str = "") -> None:
    ax.set_xlim(0.0, float(record.domain_length_x))
    ax.set_ylim(0.0, float(record.domain_length_y))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    for cx, cy in np.asarray(centers, dtype=np.float32).reshape(-1, 2):
        ax.add_patch(plt.Circle((float(cx), float(cy)), float(record.module_radius), fill=False, lw=1.3, color="#1f77b4"))


def plot_candidate_layouts(candidates: Sequence[Mapping[str, Any]], record: Any, out_path: Path, *, max_panels: int = 8) -> None:
    n = min(len(candidates), int(max_panels))
    if n <= 0:
        return
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.4 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, cand in zip(axes_arr, candidates[:n]):
        _draw_layout(ax, record, cand["centers"], title=f"#{cand['rank']} score={cand['total_score']:.3f}")
    for ax in axes_arr[n:]:
        ax.axis("off")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_temperature_field(candidate: Mapping[str, Any], record: Any, out_path: Path) -> None:
    pred = candidate["prediction"]["pred_field_grid"]
    names = list(CHANNEL_ORDER)
    t_idx = names.index("temperature") if "temperature" in names else pred.shape[-1] - 1
    fig, ax = plt.subplots(figsize=(9.5, 3.4), constrained_layout=True)
    im = ax.imshow(pred[..., t_idx], origin="lower", extent=_extent(record), cmap="inferno", aspect="equal")
    _draw_layout(ax, record, candidate["centers"], title=f"Best verified temperature, score={candidate['total_score']:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _local_disk_image(values: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
    image = np.full(local_mask.shape, np.nan, dtype=np.float32)
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    image[np.asarray(local_mask, dtype=bool)] = flat[: int(np.sum(local_mask))]
    return image


def plot_composite_internal(candidate: Mapping[str, Any], record: Any, out_path: Path) -> None:
    internal = np.asarray(candidate["prediction"].get("pred_internal_temperature"), dtype=np.float32)
    if internal.size == 0 or record.module_internal_mask is None:
        return
    pred = candidate["prediction"]["pred_field_grid"]
    t_idx = list(CHANNEL_ORDER).index("temperature") if "temperature" in CHANNEL_ORDER else pred.shape[-1] - 1
    composite = pred[..., t_idx].copy()
    centers = np.asarray(candidate["centers"], dtype=np.float32).reshape(-1, 2)
    for m, (cx, cy) in enumerate(centers):
        if m >= internal.shape[0]:
            continue
        local = _local_disk_image(internal[m, :, 0] if internal.ndim == 4 else internal[m], record.module_internal_mask)
        inside = np.hypot(record.x_grid - float(cx), record.y_grid - float(cy)) <= float(record.module_radius)
        n = local.shape[0]
        xi = np.clip((record.x_grid[inside] - float(cx)) / max(float(record.module_radius), 1.0e-12), -1.0, 1.0)
        eta = np.clip((record.y_grid[inside] - float(cy)) / max(float(record.module_radius), 1.0e-12), -1.0, 1.0)
        ii = np.rint((xi + 1.0) * 0.5 * (n - 1)).astype(int)
        jj = np.rint((eta + 1.0) * 0.5 * (n - 1)).astype(int)
        values = local[jj, ii]
        valid = np.isfinite(values)
        indices = np.flatnonzero(inside.reshape(-1))
        composite.reshape(-1)[indices[valid]] = values[valid]
    fig, ax = plt.subplots(figsize=(9.5, 3.4), constrained_layout=True)
    im = ax.imshow(composite, origin="lower", extent=_extent(record), cmap="inferno", aspect="equal")
    _draw_layout(ax, record, centers, title="Best layout composite internal temperature")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_internal_bars(candidate: Mapping[str, Any], out_path: Path) -> None:
    internal = np.asarray(candidate["prediction"].get("pred_internal_temperature"), dtype=np.float32)
    if internal.size == 0:
        return
    values = internal[..., 0] if internal.shape[-1] == 1 else internal
    count = int(candidate.get("count", values.shape[0]))
    means = [float(np.nanmean(values[i])) for i in range(min(count, values.shape[0]))]
    peaks = [float(np.nanmax(values[i])) for i in range(min(count, values.shape[0]))]
    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    x = np.arange(len(means))
    ax.bar(x - 0.18, means, width=0.36, label="mean")
    ax.bar(x + 0.18, peaks, width=0.36, label="peak")
    ax.set_xlabel("module")
    ax.set_ylabel("temperature")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_interface_curves(candidate: Mapping[str, Any], out_path: Path) -> None:
    interface = np.asarray(candidate["prediction"].get("pred_interface"), dtype=np.float32)
    if interface.size == 0 or interface.ndim < 3:
        return
    count = min(int(candidate.get("count", interface.shape[0])), interface.shape[0], 3)
    if count <= 0:
        return
    theta = np.linspace(0.0, 2.0 * math.pi, interface.shape[1], endpoint=False)
    fig, axes = plt.subplots(count, 2, figsize=(10.0, 2.8 * count), constrained_layout=True)
    if count == 1:
        axes = axes[None, :]
    for row in range(count):
        axes[row, 0].plot(theta, interface[row, :, 0])
        axes[row, 0].set_title(f"M{row} T_surface")
        axes[row, 1].plot(theta, interface[row, :, 1])
        axes[row, 1].set_title(f"M{row} q_normal")
        for col in range(2):
            axes[row, col].set_xlabel("theta")
            axes[row, col].grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_target_vs_verified(candidate: Mapping[str, Any], target_spec: Mapping[str, Any], out_path: Path) -> None:
    targets = target_spec.get("kpi_targets", {})
    names = [name for name in target_spec.get("kpi_names", []) if name in targets]
    if not names:
        return
    values = [float(candidate["verified_kpis"].get(name, np.nan)) for name in names]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.45 * len(names)), 4.2), constrained_layout=True)
    x = np.arange(len(names))
    ax.bar(x, values, color="#4c78a8", alpha=0.85, label="verified")
    for i, name in enumerate(names):
        entry = targets[name]
        if not isinstance(entry, Mapping):
            ax.scatter([i], [float(entry)], color="black", s=18)
            continue
        mode = str(entry.get("mode", "exact"))
        if mode in {"range", "between"}:
            lo = entry.get("low", entry.get("lower"))
            hi = entry.get("high", entry.get("upper"))
            if lo is not None and hi is not None:
                ax.vlines(i, float(lo), float(hi), color="black", lw=2.0)
        elif mode in {"max", "upper", "at_most"}:
            hi = entry.get("high", entry.get("upper"))
            if hi is not None:
                ax.scatter([i], [float(hi)], marker="v", color="black", s=30)
        elif mode in {"min", "lower", "at_least"}:
            lo = entry.get("low", entry.get("lower"))
            if lo is not None:
                ax.scatter([i], [float(lo)], marker="^", color="black", s=30)
        else:
            val = entry.get("value", entry.get("target"))
            if val is not None:
                ax.scatter([i], [float(val)], color="black", s=22)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("KPI value")
    ax.set_title("Target vs best verified KPIs")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_diversity(candidates: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    if not candidates:
        return
    scores = [float(cand["total_score"]) for cand in candidates]
    counts = [int(cand["count"]) for cand in candidates]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.scatter(counts, scores, c=np.arange(len(candidates)), cmap="viridis", s=26)
    ax.set_xlabel("module count")
    ax.set_ylabel("verified score")
    ax.set_title("Candidate diversity")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def try_plot_organization(candidate: Mapping[str, Any], record: Any, out_path: Path) -> None:
    aux = candidate.get("prediction", {}).get("organizer_aux", {})
    if not isinstance(aux, Mapping) or "A_mh" not in aux or "A_eh" not in aux:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    im0 = axes[0].imshow(np.asarray(aux["A_mh"], dtype=np.float32), aspect="auto", cmap="viridis")
    axes[0].set_title("A_mh")
    axes[0].set_xlabel("hyperedge")
    axes[0].set_ylabel("module")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(np.asarray(aux["A_eh"], dtype=np.float32), aspect="auto", cmap="magma")
    axes[1].set_title("A_eh")
    axes[1].set_xlabel("hyperedge")
    axes[1].set_ylabel("env token")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_top_npz(candidates: Sequence[Mapping[str, Any]], path: Path, *, top_k: int = 16, max_num_modules: int = 12) -> None:
    top = list(candidates[: min(top_k, len(candidates))])
    centers = np.zeros((len(top), max_num_modules, 2), dtype=np.float32)
    masks = np.zeros((len(top), max_num_modules), dtype=np.float32)
    scores = np.zeros((len(top),), dtype=np.float32)
    for i, cand in enumerate(top):
        arr = np.asarray(cand["centers"], dtype=np.float32).reshape(-1, 2)
        n = min(arr.shape[0], max_num_modules)
        centers[i, :n] = arr[:n]
        masks[i, :n] = 1.0
        scores[i] = float(cand["total_score"])
    np.savez_compressed(path, centers=centers, masks=masks, scores=scores)


def apply_forward_overrides(forward_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(forward_cfg)
    if args.forward_run_dir is not None:
        cfg["run_dir"] = args.forward_run_dir
    if args.forward_checkpoint_name is not None:
        cfg["checkpoint_name"] = args.forward_checkpoint_name
    if args.local_surrogate_checkpoint_path is not None:
        cfg["local_surrogate_checkpoint_path"] = args.local_surrogate_checkpoint_path
    cfg.setdefault("enabled", True)
    return cfg


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    inverse_path = resolve_inverse_checkpoint(args.inverse_run, args.checkpoint_name)
    inverse_model, checkpoint = load_inverse_checkpoint(inverse_path, device)
    target_payload = load_target_payload(args.target)
    target_spec = target_spec_from_payload(target_payload, checkpoint, inverse_model)
    train_cfg = checkpoint.get("train_config", {})
    dataset_cfg = train_cfg.get("dataset", {}) if isinstance(train_cfg, Mapping) else {}
    packed_path = args.dataset or dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5")
    dataset = ThermalInverseDesignDataset(
        packed_path,
        split=args.reference_split,
        kpi_names=target_spec["kpi_names"],
        kpi_stats=checkpoint.get("kpi_stats"),
        normalize_targets=False,
        target_augmentation={},
        max_num_modules=inverse_model.max_num_modules,
        generate_heat_power=bool(inverse_model.cfg.generate_heat_power),
        heat_power_scale=float(inverse_model.cfg.heat_power_scale),
        max_cases=max(args.reference_case_index + 1, 1),
        use_all_if_split_missing=True,
        seed=int(args.seed),
        behavior_latent_dim=int(inverse_model.cfg.behavior_latent_dim),
        organization_latent_dim=int(inverse_model.cfg.organization_latent_dim),
    )
    record = dataset.records[min(max(int(args.reference_case_index), 0), len(dataset.records) - 1)]
    forward_cfg = apply_forward_overrides(train_cfg.get("forward_model", {}) if isinstance(train_cfg, Mapping) else {}, args)
    forward_model, forward_metadata, _ = load_forward_model(forward_cfg, device)
    base_out = Path(args.output_dir) if args.output_dir else inverse_path.parent / "evaluation" / f"inverse_eval_{current_timestamp()}"
    out_dir = resolve_demo_path(base_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_vec = np.asarray(target_spec["vector"], dtype=np.float32)
    sampled = inverse_model.sample_designs(target_vec, n_samples=int(args.n_samples), n_steps=int(args.n_steps), seed=int(args.seed), device=device)
    candidates: List[Dict[str, Any]] = []
    for sample_idx, cand in enumerate(tqdm(sampled, desc="verify", unit="candidate", dynamic_ncols=True)):
        prediction = predict_candidate_with_forward(
            forward_model,
            forward_metadata,
            record,
            cand,
            device,
            max_num_modules=inverse_model.max_num_modules,
            generate_heat_power=bool(inverse_model.cfg.generate_heat_power),
            query_batch_size=int(args.query_batch_size),
        )
        verified_kpis = _candidate_kpi_payload(record, prediction, cand, inverse_model)
        score = score_candidate_kpis(verified_kpis, target_spec)
        row = {
            "sample_index": int(sample_idx),
            "count": int(cand.get("count", 0)),
            "centers": np.asarray(cand["centers"], dtype=np.float32),
            "valid": bool(cand.get("validity", {}).get("valid", False)),
            "validity": cand.get("validity", {}),
            "verified_kpis": verified_kpis,
            "score_detail": score,
            "total_score": float(score["total_score"]),
            "kpi_score": float(score.get("kpi_score", score["total_score"])),
            "constraint_penalty": float(score.get("constraint_penalty", 0.0)),
            "prediction": prediction,
        }
        for key in ("min_center_distance", "wall_clearance", "inlet_clearance", "outlet_clearance"):
            row[key] = float(verified_kpis.get(key, float("nan")))
        candidates.append(row)
    candidates.sort(key=lambda row: (0 if row["valid"] else 1, float(row["total_score"]), -int(row["count"])))
    for rank, row in enumerate(candidates):
        row["rank"] = int(rank)

    serializable = []
    for row in candidates:
        lite = {key: value for key, value in row.items() if key != "prediction"}
        serializable.append(json_safe(lite))
    write_json(out_dir / "candidates.json", {"target": json_safe(target_spec), "candidates": serializable})
    write_candidates_csv(candidates, out_dir / "candidates.csv")
    write_kpi_scores_csv(candidates, target_spec["kpi_names"], out_dir / "kpi_scores.csv")
    write_json(out_dir / "target_spec_resolved.json", json_safe(target_spec))
    save_top_npz(candidates, out_dir / "top_candidates.npz", max_num_modules=inverse_model.max_num_modules)

    best = candidates[0] if candidates else None
    summary = {
        "inverse_checkpoint": str(inverse_path),
        "target_path": target_payload.get("_path"),
        "reference_case_id": record.case_id,
        "n_samples": int(args.n_samples),
        "best_score": best["total_score"] if best else None,
        "best_valid": bool(best["valid"]) if best else None,
        "validity_rate": float(np.mean([float(c["valid"]) for c in candidates])) if candidates else 0.0,
        "forward_checkpoint": forward_metadata.get("checkpoint_path"),
        "local_surrogate_checkpoint": forward_metadata.get("local_surrogate_checkpoint_path"),
    }
    write_json(out_dir / "verification_summary.json", json_safe(summary))

    if best is not None:
        plot_target_vs_verified(best, target_spec, out_dir / "target_vs_verified_kpis.png")
        plot_candidate_layouts(candidates, record, out_dir / "candidate_layouts_ranked.png")
        plot_temperature_field(best, record, out_dir / "best_layout_temperature_field.png")
        plot_composite_internal(best, record, out_dir / "best_layout_composite_internal_temperature.png")
        plot_internal_bars(best, out_dir / "best_layout_module_temperature_bars.png")
        plot_interface_curves(best, out_dir / "best_layout_interface_curves.png")
        try_plot_organization(best, record, out_dir / "best_layout_organization_overview.png")
        plot_diversity(candidates, out_dir / "candidate_diversity.png")
    print(f"[done] inverse evaluation saved to {out_dir}")
    if best is not None:
        print(f"[best] score={best['total_score']:.6f}, valid={best['valid']}, count={best['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
