from __future__ import annotations

"""Train the ChannelThermal hypergraph mechanism prior and layout realizer."""

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-design-prior")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from tqdm.auto import tqdm

import _bootstrap_imports
from channelthermal_model_utils import current_timestamp, ensure_dir, resolve_demo_path, select_device, set_seed, write_json
from design_prior_dataset import DesignPriorDataset, collate_fn
from hypergraph_mechanism import HypergraphMechanismAtlas, MechanismFeatureConfig
from model_design_prior import HypergraphConditionedLayoutRealizer, MechanismLayoutRealizerConfig

DEFAULT_CONFIG_PATH = "./Configs/train_design_prior_config_template.json"
CHECKPOINT_STAGE = "channelthermal_hypergraph_mechanism_design_prior"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ChannelThermal hypergraph mechanism design prior.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="JSON config path.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--Run_ID", dest="run_id", type=str, default=None, help="Numeric run serial, e.g. 0001.")
    parser.add_argument("--dry-run", action="store_true", help="Load data, fit a small atlas, run one batch loss, then exit.")
    parser.add_argument("--dry-run-max-samples", type=int, default=256, help="Maximum samples used for dry-run atlas fitting.")
    return parser.parse_args()


def normalize_run_id(value: Any, fallback: str = "0001") -> str:
    raw = str(value if value is not None and str(value).strip() else fallback).strip()
    if not raw.isdigit():
        raise ValueError(f"Run_ID must be a numeric serial such as '0001'; got {raw!r}.")
    return raw.zfill(4)


def resolve_run_id(args: argparse.Namespace, cfg: Mapping[str, Any]) -> str:
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), Mapping) else {}
    return normalize_run_id(args.run_id or cfg.get("Run_ID") or cfg.get("run_id") or training_cfg.get("Run_ID") or training_cfg.get("run_id"), "0001")


def make_run_dir(cfg: Mapping[str, Any], run_id: str) -> Path:
    output_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), Mapping) else {}
    saved_root = resolve_demo_path(output_cfg.get("saved_root", "Saved_Model_Prior"))
    run_name = f"Run_{run_id}_{current_timestamp()}"
    if not re.match(r"^Run_\d{4}_\d{8}_\d{6}$", run_name):
        raise ValueError(f"Unexpected design-prior run directory name: {run_name}")
    return ensure_dir(saved_root / run_name)


def read_json(path: str | Path) -> Dict[str, Any]:
    with resolve_demo_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def mean_rows(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out = {}
    for key in keys:
        vals = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        out[key] = float(np.mean(finite)) if finite.size else float("nan")
    return out


class MechanismFeatureDataset(Dataset):
    """Attach precomputed atlas features while preserving raw dataset items."""

    def __init__(self, base: Dataset, mechanism_features: np.ndarray) -> None:
        self.base = base
        self.mechanism_features = np.asarray(mechanism_features, dtype=np.float32)
        if len(base) != self.mechanism_features.shape[0]:
            raise ValueError("mechanism_features must have one row per dataset sample.")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.base[int(idx)])
        item["mechanism_feature"] = self.mechanism_features[int(idx)]
        return item


def _subset_arrays(arrays: Mapping[str, np.ndarray], max_samples: Optional[int]) -> Dict[str, np.ndarray]:
    if max_samples is None:
        return {key: np.asarray(value) for key, value in arrays.items()}
    n = min(int(max_samples), int(np.asarray(arrays["design_vec"]).shape[0]))
    return {key: np.asarray(value)[:n] for key, value in arrays.items()}


def build_mechanism_atlas(
    arrays: Mapping[str, np.ndarray],
    mechanism_cfg: Mapping[str, Any],
) -> Tuple[HypergraphMechanismAtlas, np.ndarray]:
    feature_cfg = MechanismFeatureConfig(**{key: value for key, value in dict(mechanism_cfg).items() if key in MechanismFeatureConfig.__dataclass_fields__})
    atlas = HypergraphMechanismAtlas(feature_cfg).fit(
        arrays.get("hypergraph_vec"),
        arrays.get("behavior_vec"),
        arrays.get("count_vec", arrays.get("true_count")),
    )
    mechanism_features = atlas.encode(
        arrays.get("hypergraph_vec"),
        arrays.get("behavior_vec"),
        arrays.get("count_vec", arrays.get("true_count")),
    )
    return atlas, mechanism_features.astype(np.float32)


def build_model_config(dataset: DesignPriorDataset, mechanism_features: np.ndarray, cfg: Mapping[str, Any]) -> MechanismLayoutRealizerConfig:
    model_cfg = dict(cfg.get("layout_realizer", {}) if isinstance(cfg.get("layout_realizer"), Mapping) else {})
    model_cfg["design_dim"] = int(dataset.design_vec.shape[1])
    model_cfg["mechanism_dim"] = int(mechanism_features.shape[1])
    model_cfg["hypergraph_dim"] = int(dataset.hypergraph_vec.shape[1])
    model_cfg["behavior_dim"] = int(dataset.behavior_vec.shape[1])
    model_cfg["context_dim"] = int(dataset.context_vec.shape[1])
    return MechanismLayoutRealizerConfig.from_dict(model_cfg)


def run_epoch(
    model: HypergraphConditionedLayoutRealizer,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip_norm: float = 1.0,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows = []
    for batch in tqdm(loader, desc="train" if training else "val", leave=False, dynamic_ncols=True):
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            metrics = model.training_loss(batch)
            loss = metrics["loss_total"]
        if training:
            loss.backward()
            if float(grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
        rows.append({key: scalar(value) for key, value in metrics.items()})
    return mean_rows(rows)


def write_history(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not history:
        return
    keys = sorted({key for row in history for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in keys})


def plot_loss_curves(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not history:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    epochs = [int(row["epoch"]) for row in history]
    positive_values = []
    keys = (
        "train_loss_total",
        "val_loss_total",
        "train_layout_flow_loss",
        "val_layout_flow_loss",
        "train_geometry_loss",
        "val_geometry_loss",
        "train_count_loss",
        "val_count_loss",
    )
    for key in keys:
        vals = [float(row.get(key, np.nan)) for row in history]
        if any(math.isfinite(v) for v in vals):
            ax.plot(epochs, vals, marker="o", markersize=2.5, label=key)
            positive_values.extend(v for v in vals if math.isfinite(v) and v > 0.0)
    if positive_values:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def _pca_2d(features: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], 2), dtype=np.float32)
    x = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[: min(2, vt.shape[0])].T
    coords = x @ comps
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    return coords[:, :2].astype(np.float32)


def plot_mechanism_cluster_hist(atlas: HypergraphMechanismAtlas, path: Path) -> None:
    counts = np.asarray(atlas.cluster_counts, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    ax.bar(np.arange(counts.size), counts)
    ax.set_xlabel("cluster id")
    ax.set_ylabel("count")
    ax.set_title("Mechanism cluster sizes")
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def plot_mechanism_feature_pca(features: np.ndarray, assignments: np.ndarray, path: Path) -> None:
    coords = _pca_2d(features)
    fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=assignments, s=12, cmap="tab20", alpha=0.85)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Mechanism features")
    fig.colorbar(scatter, ax=ax, label="cluster id")
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def split_design_vec(vec: np.ndarray, max_num_modules: int) -> Tuple[np.ndarray, np.ndarray]:
    m = int(max_num_modules)
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size < 3 * m:
        arr = np.pad(arr, (0, 3 * m - arr.size))
    centers = arr[: 2 * m].reshape(m, 2)
    mask = arr[2 * m : 3 * m] > 0.5
    return centers, mask


def draw_layout_axis(ax: Any, design_vec: np.ndarray, cfg: MechanismLayoutRealizerConfig, title: str = "") -> None:
    centers, mask = split_design_vec(design_vec, int(cfg.max_num_modules))
    centers = centers.copy()
    centers[:, 0] *= float(cfg.domain_length_x)
    centers[:, 1] *= float(cfg.domain_length_y)
    ax.set_xlim(0.0, float(cfg.domain_length_x))
    ax.set_ylim(0.0, float(cfg.domain_length_y))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(plt.Rectangle((0.0, 0.0), float(cfg.domain_length_x), float(cfg.domain_length_y), fill=False, lw=1.0, color="0.25"))
    for cx, cy in centers[mask]:
        ax.add_patch(Circle((float(cx), float(cy)), float(cfg.module_radius), fill=False, lw=1.0, color="tab:red"))
    if title:
        ax.set_title(title, fontsize=8)


def write_mechanism_representatives(
    run_dir: Path,
    dataset: DesignPriorDataset,
    features: np.ndarray,
    assignments: np.ndarray,
    atlas: HypergraphMechanismAtlas,
    model_cfg: MechanismLayoutRealizerConfig,
    *,
    per_cluster: int = 4,
) -> None:
    rep_dir = ensure_dir(run_dir / "mechanism_representatives")
    centers = np.asarray(atlas.cluster_centers, dtype=np.float32)
    for cid in range(centers.shape[0]):
        idx = np.where(assignments == cid)[0]
        if idx.size == 0:
            continue
        dist = np.sum((features[idx] - centers[cid][None, :]) ** 2, axis=1)
        order = idx[np.argsort(dist)[:per_cluster]]
        write_json(rep_dir / f"cluster_{cid:03d}_representatives.json", {"cluster_id": int(cid), "sample_indices": [int(i) for i in order]})
        cols = max(int(order.size), 1)
        fig, axes = plt.subplots(1, cols, figsize=(3.0 * cols, 2.2), constrained_layout=True)
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, sample_idx in zip(axes_arr, order):
            draw_layout_axis(ax, dataset.design_vec[int(sample_idx)], model_cfg, title=f"c{cid} i{int(sample_idx)}")
        fig.savefig(str(rep_dir / f"cluster_{cid:03d}_representatives.png"), dpi=150)
        plt.close(fig)


def build_mechanism_cluster_summary(
    arrays: Mapping[str, np.ndarray],
    features: np.ndarray,
    atlas: HypergraphMechanismAtlas,
) -> Dict[str, Any]:
    assignments = np.asarray(atlas.assignments, dtype=np.int64)
    centers = np.asarray(atlas.cluster_centers, dtype=np.float32)
    true_count = np.asarray(arrays.get("true_count", []), dtype=np.float32).reshape(-1)
    behavior = np.asarray(arrays.get("behavior_vec", np.zeros((features.shape[0], 0))), dtype=np.float32)
    hg_dim = int(atlas.stats.get("hypergraph_dim", 0))
    summary: Dict[str, Any] = {
        "num_clusters": int(centers.shape[0]),
        "feature_dim": int(features.shape[1]),
        "hypergraph_dim": int(hg_dim),
        "behavior_dim": int(behavior.shape[1]) if behavior.ndim == 2 else 0,
        "clusters": [],
    }
    for cid in range(centers.shape[0]):
        idx = np.where(assignments == cid)[0]
        if idx.size == 0:
            summary["clusters"].append({"cluster_id": int(cid), "count": 0})
            continue
        dist = np.sqrt(np.sum((features[idx] - centers[cid][None, :]) ** 2, axis=1))
        rep_order = idx[np.argsort(dist)[: min(8, idx.size)]]
        hg_dist = np.sqrt(np.sum((features[idx, :hg_dim] - centers[cid][None, :hg_dim]) ** 2, axis=1)) if hg_dim > 0 else np.zeros((idx.size,), dtype=np.float32)
        behavior_means = []
        if behavior.ndim == 2 and behavior.shape[1] > 0:
            behavior_means = np.mean(behavior[idx, : min(8, behavior.shape[1])], axis=0).astype(float).tolist()
        summary["clusters"].append(
            {
                "cluster_id": int(cid),
                "count": int(idx.size),
                "mean_true_count": float(np.mean(true_count[idx])) if true_count.size else 0.0,
                "mean_selected_behavior_dims": behavior_means,
                "nearest_representative_sample_indices": [int(i) for i in rep_order],
                "mean_hypergraph_feature_distance": float(np.mean(hg_dist)) if hg_dist.size else 0.0,
                "mean_feature_distance": float(np.mean(dist)) if dist.size else 0.0,
            }
        )
    return summary


@torch.no_grad()
def plot_generated_layout_examples_by_mechanism(
    model: HypergraphConditionedLayoutRealizer,
    atlas: HypergraphMechanismAtlas,
    device: torch.device,
    path: Path,
    *,
    max_clusters: int = 8,
    samples_per_cluster: int = 3,
) -> None:
    model.eval()
    centers = np.asarray(atlas.cluster_centers, dtype=np.float32)
    if centers.size == 0:
        return
    counts = np.asarray(atlas.cluster_counts, dtype=np.int64)
    cluster_ids = np.argsort(-counts)[: min(max_clusters, centers.shape[0])]
    rows = int(cluster_ids.size)
    cols = int(samples_per_cluster)
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.25 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(rows, cols)
    for r, cid in enumerate(cluster_ids):
        mech = torch.as_tensor(centers[int(cid)][None, :], dtype=torch.float32, device=device)
        samples = model.sample_layout(mech, num_samples=cols, steps=max(2, min(int(model.cfg.flow_steps_default), 16)))["design_vec"].detach().cpu().numpy()
        for c in range(cols):
            draw_layout_axis(axes_arr[r, c], samples[c], model.cfg, title=f"cluster {int(cid)}")
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def save_checkpoint(
    path: Path,
    *,
    model: HypergraphConditionedLayoutRealizer,
    atlas: HypergraphMechanismAtlas,
    cfg: Mapping[str, Any],
    dataset_stats: Mapping[str, Any],
    epoch: int,
    best_metric: float,
) -> None:
    torch.save(
        {
            "stage": CHECKPOINT_STAGE,
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "mechanism_atlas_state": atlas.to_state(),
            "mechanism_feature_config": atlas.to_state().get("feature_config", {}),
            "model_config": model.cfg.to_dict(),
            "model_state_dict": model.state_dict(),
            "dataset_stats": dict(dataset_stats),
            "train_config": dict(cfg),
        },
        path,
    )


def make_cluster_balanced_sampler(dataset: Dataset, assignments: np.ndarray) -> WeightedRandomSampler:
    indices = np.asarray(getattr(dataset, "indices", np.arange(len(dataset))), dtype=np.int64)
    cluster_ids = np.asarray(assignments, dtype=np.int64)[indices]
    max_cluster = int(np.max(cluster_ids)) + 1 if cluster_ids.size else 1
    counts = np.bincount(cluster_ids, minlength=max_cluster).astype(np.float64)
    weights = 1.0 / np.maximum(counts[cluster_ids], 1.0)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=int(indices.size), replacement=True)


def write_run_diagnostics(
    run_dir: Path,
    *,
    dataset: DesignPriorDataset,
    arrays: Mapping[str, np.ndarray],
    features: np.ndarray,
    atlas: HypergraphMechanismAtlas,
    model: HypergraphConditionedLayoutRealizer,
    device: torch.device,
) -> Dict[str, Any]:
    assignments = np.asarray(atlas.assignments, dtype=np.int64)
    summary = build_mechanism_cluster_summary(arrays, features, atlas)
    write_json(run_dir / "mechanism_cluster_summary.json", summary)
    plot_mechanism_cluster_hist(atlas, run_dir / "mechanism_cluster_hist.png")
    plot_mechanism_feature_pca(features, assignments, run_dir / "mechanism_feature_pca.png")
    plot_mechanism_feature_pca(features, assignments, run_dir / "mechanism_feature_umap_or_pca.png")
    write_mechanism_representatives(run_dir, dataset, features, assignments, atlas, model.cfg)
    plot_generated_layout_examples_by_mechanism(model, atlas, device, run_dir / "generated_layout_examples_by_mechanism.png")
    return summary


def _load_dataset(data_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> DesignPriorDataset:
    library_path = data_cfg.get("library_path", "Data_Saved/DesignPrior_Library/design_library.h5")
    try:
        return DesignPriorDataset(
            library_path,
            behavior_dim=model_cfg.get("behavior_dim"),
            generate_heat_power=bool(model_cfg.get("generate_heat_power", False)),
        )
    except FileNotFoundError as exc:
        resolved = resolve_demo_path(library_path)
        raise FileNotFoundError(f"Design library not found: {resolved}") from exc


def dry_run(args: argparse.Namespace, cfg: Mapping[str, Any]) -> int:
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), Mapping) else {}
    mechanism_cfg = cfg.get("mechanism", {}) if isinstance(cfg.get("mechanism"), Mapping) else {}
    realizer_cfg = cfg.get("layout_realizer", {}) if isinstance(cfg.get("layout_realizer"), Mapping) else {}
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), Mapping) else {}
    try:
        dataset = _load_dataset(data_cfg, realizer_cfg)
    except FileNotFoundError as exc:
        print(f"[dry-run] {exc}")
        return 2
    arrays = dataset.get_arrays_for_mechanism_fit()
    fit_arrays = _subset_arrays(arrays, int(args.dry_run_max_samples))
    atlas, fit_features = build_mechanism_atlas(fit_arrays, mechanism_cfg)
    features = atlas.encode(arrays.get("hypergraph_vec"), arrays.get("behavior_vec"), arrays.get("count_vec", arrays.get("true_count")))
    model_config = build_model_config(dataset, features, cfg)
    device_arg = training_cfg.get("device", "auto")
    device = select_device(None if str(device_arg).lower() == "auto" else str(device_arg))
    model = HypergraphConditionedLayoutRealizer(model_config).to(device)
    dry_dataset = MechanismFeatureDataset(dataset, features)
    loader = DataLoader(dry_dataset, batch_size=min(int(training_cfg.get("batch_size", 128)), len(dry_dataset)), shuffle=False, num_workers=0, collate_fn=collate_fn)
    batch = move_batch(next(iter(loader)), device)
    metrics = model.training_loss(batch)
    print("[dry-run] loaded samples:", len(dataset))
    print("[dry-run] atlas fit samples:", int(fit_features.shape[0]))
    print("[dry-run] mechanism_dim:", int(features.shape[1]))
    print("[dry-run] hypergraph_dim:", int(dataset.hypergraph_vec.shape[1]))
    print("[dry-run] behavior_dim:", int(dataset.behavior_vec.shape[1]))
    print("[dry-run] num_clusters:", int(atlas.cluster_centers.shape[0]))
    print("[dry-run] one batch metrics:", {key: scalar(value) for key, value in metrics.items()})
    return 0


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    if args.dry_run:
        return dry_run(args, cfg)

    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), Mapping) else {}
    mechanism_cfg = cfg.get("mechanism", {}) if isinstance(cfg.get("mechanism"), Mapping) else {}
    realizer_cfg = cfg.get("layout_realizer", {}) if isinstance(cfg.get("layout_realizer"), Mapping) else {}
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), Mapping) else {}
    run_id = resolve_run_id(args, cfg)
    cfg["Run_ID"] = run_id
    set_seed(int(training_cfg.get("seed", 0)))
    device_arg = training_cfg.get("device", "auto")
    device = select_device(None if str(device_arg).lower() == "auto" else str(device_arg))

    dataset = _load_dataset(data_cfg, realizer_cfg)
    arrays = dataset.get_arrays_for_mechanism_fit()
    atlas, mechanism_features = build_mechanism_atlas(arrays, mechanism_cfg)
    model_config = build_model_config(dataset, mechanism_features, cfg)
    model = HypergraphConditionedLayoutRealizer(model_config).to(device)
    if args.resume:
        checkpoint = torch.load(resolve_demo_path(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    wrapped_dataset = MechanismFeatureDataset(dataset, mechanism_features)
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    val_len = int(round(len(wrapped_dataset) * val_fraction)) if len(wrapped_dataset) > 1 else 0
    val_len = min(max(val_len, 0), max(len(wrapped_dataset) - 1, 0))
    train_len = len(wrapped_dataset) - val_len
    generator = torch.Generator().manual_seed(int(training_cfg.get("seed", 0)))
    if val_len > 0:
        train_dataset, val_dataset = random_split(wrapped_dataset, [train_len, val_len], generator=generator)
    else:
        train_dataset, val_dataset = wrapped_dataset, wrapped_dataset

    train_sampler = make_cluster_balanced_sampler(train_dataset, atlas.assignments) if bool(mechanism_cfg.get("sample_cluster_balance", False)) else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 128)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate_fn,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-5)),
    )

    run_dir = make_run_dir(cfg, run_id)
    write_json(run_dir / "resolved_train_design_prior_config.json", cfg)
    write_json(run_dir / "dataset_stats.json", dataset.stats)
    write_json(run_dir / "mechanism_atlas_state.json", atlas.to_state())
    epochs = int(args.epochs or training_cfg.get("epochs", 2000))
    history = []
    best = float("inf")
    save_every = max(int(training_cfg.get("save_every", 10)), 1)
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer, grad_clip_norm=float(training_cfg.get("grad_clip_norm", 1.0)))
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        row: Dict[str, Any] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        metric = float(row.get("val_loss_total", row.get("train_loss_total", float("inf"))))
        if metric < best:
            best = metric
            save_checkpoint(run_dir / "best_model.pt", model=model, atlas=atlas, cfg=cfg, dataset_stats=dataset.stats, epoch=epoch, best_metric=best)
        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(run_dir / "latest_model.pt", model=model, atlas=atlas, cfg=cfg, dataset_stats=dataset.stats, epoch=epoch, best_metric=best)
        write_history(history, run_dir / "loss_history.csv")
        plot_loss_curves(history, run_dir / "loss_curves.png")
        print(f"[epoch {epoch:04d}] train={row.get('train_loss_total', float('nan')):.6g} val={row.get('val_loss_total', float('nan')):.6g} best={best:.6g}")

    cluster_summary = write_run_diagnostics(run_dir, dataset=dataset, arrays=arrays, features=mechanism_features, atlas=atlas, model=model, device=device)
    write_json(
        run_dir / "summary.json",
        {
            "stage": CHECKPOINT_STAGE,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "num_samples": int(len(dataset)),
            "mechanism_dim": int(mechanism_features.shape[1]),
            "hypergraph_dim": int(dataset.hypergraph_vec.shape[1]),
            "behavior_dim": int(dataset.behavior_vec.shape[1]),
            "num_clusters": int(atlas.cluster_centers.shape[0]),
            "best_metric": float(best),
            "best_epoch": int(min(history, key=lambda row: float(row.get("val_loss_total", row.get("train_loss_total", float("inf")))))["epoch"]) if history else 0,
            "model_config": model.cfg.to_dict(),
            "mechanism_feature_config": atlas.to_state().get("feature_config", {}),
            "cluster_summary_path": "mechanism_cluster_summary.json",
            "nonempty_clusters": int(sum(1 for row in cluster_summary.get("clusters", []) if int(row.get("count", 0)) > 0)),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
