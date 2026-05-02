from __future__ import annotations

"""Sample, verify, and rank inverse-design candidates."""

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import torch

from inverse_kpi import (
    DEFAULT_KPI_NAMES,
    build_target_spec_vector,
    compute_cycle_kpis,
    score_candidate_kpis,
)
from model_inverse import HypergraphInverseDesignFlow, InverseModelConfig, periodic_min_distance
from train_inverse import (
    DEMO_ROOT,
    extract_forward_latent_targets,
    load_forward_model,
    make_eval_grid,
    predict_cycle_for_centers,
    read_json,
    resolve_demo_path,
    safe_torch_load,
    select_device,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the inverse-design generator.")
    parser.add_argument("--inverse-run", type=str, required=True, help="Inverse run directory containing best_model.pt/latest_model.pt.")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Inverse checkpoint filename.")
    parser.add_argument("--target-json", type=str, default=None, help="Target KPI JSON file.")
    parser.add_argument("--n-samples", type=int, default=64, help="Number of inverse candidates to sample.")
    parser.add_argument("--verify-top-k", type=int, default=16, help="Number of sampled candidates to forward-verify.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed.")
    parser.add_argument("--n-steps", type=int, default=32, help="Inverse ODE integration steps.")
    parser.add_argument("--phase-bins", type=int, default=None, help="Forward verification phase bins.")
    parser.add_argument("--nx", type=int, default=None, help="Forward verification grid x cells.")
    parser.add_argument("--ny", type=int, default=None, help="Forward verification grid y cells.")
    parser.add_argument("--re", type=float, default=None, help="Simple target Re if --target-json is omitted.")
    parser.add_argument("--num-cylinders-min", type=int, default=None, help="Simple count lower bound.")
    parser.add_argument("--num-cylinders-max", type=int, default=None, help="Simple count upper bound.")
    parser.add_argument("--min-center-distance", type=float, default=None, help="Simple geometry preference.")
    parser.add_argument(
        "--kpi",
        action="append",
        default=[],
        help="Simple KPI target, e.g. enstrophy:range:0.08:0.16 or pressure_range:max:0.08.",
    )
    parser.add_argument("--refine-top-k", type=int, default=0, help="Reserved for optional post-sampling refinement; default disabled.")
    parser.add_argument("--refine-steps", type=int, default=0, help="Reserved for optional post-sampling refinement; default disabled.")
    return parser.parse_args()


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def parse_simple_kpi(entries: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for entry in entries:
        pieces = entry.split(":")
        if len(pieces) < 2:
            raise ValueError(f"Invalid --kpi entry {entry!r}; expected name:mode:values.")
        name, mode = pieces[0], pieces[1].lower()
        if mode == "range":
            if len(pieces) < 4:
                raise ValueError(f"Range KPI {entry!r} needs low and high.")
            out[name] = {"mode": "range", "low": float(pieces[2]), "high": float(pieces[3]), "weight": 1.0}
        elif mode in {"max", "upper", "at_most"}:
            if len(pieces) < 3:
                raise ValueError(f"Max KPI {entry!r} needs high.")
            out[name] = {"mode": "max", "high": float(pieces[2]), "weight": 1.0}
        elif mode in {"min", "lower", "at_least"}:
            if len(pieces) < 3:
                raise ValueError(f"Min KPI {entry!r} needs low.")
            out[name] = {"mode": "min", "low": float(pieces[2]), "weight": 1.0}
        elif mode in {"minimize", "maximize"}:
            out[name] = {"mode": mode, "weight": float(pieces[2]) if len(pieces) >= 3 else 1.0}
        else:
            if len(pieces) < 3:
                raise ValueError(f"Exact KPI {entry!r} needs value.")
            out[name] = {"mode": "exact", "value": float(pieces[2]), "weight": 1.0}
    return out


def load_target_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if args.target_json:
        path = Path(args.target_json).expanduser()
        if not path.is_absolute():
            local = Path.cwd() / path
            path = local if local.exists() else DEMO_ROOT / path
        return read_json(path.resolve())
    return {
        "Re": 100.0 if args.re is None else float(args.re),
        "num_cylinders_min": args.num_cylinders_min,
        "num_cylinders_max": args.num_cylinders_max,
        "kpis": parse_simple_kpi(args.kpi),
        "preferences": {"min_center_distance": args.min_center_distance},
    }


def load_inverse_checkpoint(inverse_run: Path, checkpoint_name: str, device: torch.device) -> Tuple[HypergraphInverseDesignFlow, Dict[str, Any], Dict[str, Any], Path]:
    ckpt_path = inverse_run / checkpoint_name
    if not ckpt_path.exists() and checkpoint_name == "best_model.pt":
        ckpt_path = inverse_run / "latest_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Inverse checkpoint not found: {ckpt_path}")
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    model_cfg = ckpt.get("inverse_model_config")
    if model_cfg is None:
        model_cfg = ckpt.get("config", {}).get("inverse_model")
    if model_cfg is None:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain inverse_model_config.")
    model = HypergraphInverseDesignFlow(InverseModelConfig.from_dict(model_cfg))
    state = ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain model_state_dict.")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, ckpt, dict(model_cfg), ckpt_path


def target_spec_from_payload(
    payload: Mapping[str, Any],
    *,
    kpi_names: Sequence[str],
    kpi_stats: Optional[Mapping[str, Any]],
    normalize: bool,
    max_num_cylinders: int,
    re_scale: float,
    domain_length_scale: float,
) -> Dict[str, Any]:
    preferences = payload.get("preferences", {}) if isinstance(payload.get("preferences", {}), Mapping) else {}
    min_center_distance = payload.get("min_center_distance", preferences.get("min_center_distance"))
    return build_target_spec_vector(
        kpi_names=kpi_names,
        kpi_targets=payload.get("kpis", {}),
        stats=kpi_stats,
        normalize=normalize,
        re_value=payload.get("Re", payload.get("re")),
        num_cylinders_min=payload.get("num_cylinders_min"),
        num_cylinders_max=payload.get("num_cylinders_max"),
        min_center_distance=min_center_distance,
        max_num_cylinders=max_num_cylinders,
        re_scale=re_scale,
        domain_length_scale=domain_length_scale,
        return_spec=True,
    )


def candidate_prefilter_key(candidate: Mapping[str, Any]) -> Tuple[int, float, int]:
    validity = candidate.get("validity", {})
    valid = bool(validity.get("valid", False)) if isinstance(validity, Mapping) else False
    min_dist = float(validity.get("min_pair_distance", 0.0)) if isinstance(validity, Mapping) else 0.0
    return (0 if valid else 1, -min_dist, int(candidate.get("count", 0)))


def plot_candidate_flow(
    cycle: np.ndarray,
    centers: np.ndarray,
    out_path: Path,
    *,
    channel_order: Sequence[str],
    lx: float,
    ly: float,
) -> None:
    frame = np.asarray(cycle[0], dtype=np.float32)
    names = list(channel_order)[: frame.shape[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5), dpi=150, constrained_layout=True)
    cmaps = {"u": "coolwarm", "v": "coolwarm", "p": "magma", "omega": "RdBu_r"}
    for ax, idx in zip(axes.reshape(-1), range(min(4, frame.shape[-1]))):
        name = names[idx] if idx < len(names) else f"ch{idx}"
        im = ax.imshow(frame[..., idx], origin="lower", extent=[0, lx, 0, ly], cmap=cmaps.get(name, "viridis"), aspect="auto")
        ax.scatter(centers[:, 0], centers[:, 1], s=24, c="white", edgecolors="black", linewidths=0.7)
        ax.set_title(name)
        ax.set_xlim(0, lx)
        ax.set_ylim(0, ly)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(out_path)
    plt.close(fig)


def plot_organization(aux: Mapping[str, torch.Tensor], centers: np.ndarray, out_path: Path, *, lx: float, ly: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.scatter(centers[:, 0], centers[:, 1], s=40, c="#1f77b4", edgecolors="white", linewidths=0.8, label="cylinders")
    source = aux.get("hyper_source_coords")
    wake = aux.get("hyper_wake_coords")
    strength = aux.get("hyper_strength")
    if source is not None and wake is not None:
        src = source.detach().cpu().numpy().reshape(-1, 2)
        wk = wake.detach().cpu().numpy().reshape(-1, 2)
        src_phys = np.column_stack([src[:, 0] * lx, src[:, 1] * ly])
        wk_phys = np.column_stack([wk[:, 0] * lx, wk[:, 1] * ly])
        if strength is not None:
            weights = strength.detach().cpu().numpy().reshape(-1)
            weights = weights / max(float(np.max(np.abs(weights))), 1.0e-8)
        else:
            weights = np.ones(src_phys.shape[0], dtype=np.float32)
        for i in range(src_phys.shape[0]):
            ax.arrow(
                src_phys[i, 0],
                src_phys[i, 1],
                wk_phys[i, 0] - src_phys[i, 0],
                wk_phys[i, 1] - src_phys[i, 1],
                width=0.015 * max(lx, ly),
                head_width=0.18,
                alpha=0.25 + 0.6 * float(weights[i]),
                color="#d62728",
                length_includes_head=True,
            )
        ax.scatter(src_phys[:, 0], src_phys[:, 1], s=18, c="#d62728", label="hyper source")
        ax.scatter(wk_phys[:, 0], wk_phys[:, 1], s=18, c="#2ca02c", label="wake center")
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_sampled_layouts(candidates: Sequence[Mapping[str, Any]], out_path: Path, *, lx: float, ly: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    scores = [float(c.get("score", float("nan"))) for c in candidates]
    finite_scores = [s for s in scores if math.isfinite(s)]
    fallback = max(finite_scores) if finite_scores else 1.0
    for idx, candidate in enumerate(candidates):
        centers = np.asarray(candidate["centers"], dtype=np.float32).reshape(-1, 2)
        score = scores[idx] if math.isfinite(scores[idx]) else fallback
        color = plt.cm.viridis(1.0 - min(score / max(fallback, 1.0e-8), 1.0))
        ax.scatter(centers[:, 0], centers[:, 1], s=14, color=color, alpha=0.65)
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("Sampled layouts colored by verified score")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_kpi_target_vs_achieved(
    verified: Sequence[Mapping[str, Any]],
    target_payload: Mapping[str, Any],
    out_path: Path,
) -> None:
    target_kpis = list((target_payload.get("kpis") or {}).keys())
    if not target_kpis or not verified:
        return
    top = verified[: min(5, len(verified))]
    x = np.arange(len(target_kpis))
    width = 0.8 / max(len(top), 1)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(target_kpis)), 4.5), dpi=150)
    for i, candidate in enumerate(top):
        vals = [float(candidate.get("kpis", {}).get(name, 0.0)) for name in target_kpis]
        ax.bar(x + i * width, vals, width=width, label=f"rank {i}")
    for idx, name in enumerate(target_kpis):
        spec = target_payload["kpis"][name]
        if not isinstance(spec, Mapping):
            ax.axhline(float(spec), color="black", lw=0.7, alpha=0.25)
            continue
        mode = str(spec.get("mode", "exact"))
        if mode == "range":
            ax.vlines(idx + 0.4, float(spec.get("low", 0.0)), float(spec.get("high", 0.0)), color="black", lw=2.0)
        elif "value" in spec:
            ax.scatter([idx + 0.4], [float(spec["value"])], color="black", s=18, zorder=5)
        elif "high" in spec:
            ax.scatter([idx + 0.4], [float(spec["high"])], color="black", marker="v", s=18, zorder=5)
        elif "low" in spec:
            ax.scatter([idx + 0.4], [float(spec["low"])], color="black", marker="^", s=18, zorder=5)
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(target_kpis, rotation=30, ha="right")
    ax.set_ylabel("KPI value")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def padded_layout_vector(centers: np.ndarray, max_num_cylinders: int) -> np.ndarray:
    arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    order = np.lexsort((arr[:, 1], arr[:, 0])) if arr.shape[0] else []
    arr = arr[order] if arr.shape[0] else arr
    padded = np.zeros((max_num_cylinders, 2), dtype=np.float32)
    padded[: min(max_num_cylinders, arr.shape[0])] = arr[:max_num_cylinders]
    return padded.reshape(-1)


def plot_diversity(candidates: Sequence[Mapping[str, Any]], out_path: Path, *, max_num_cylinders: int) -> None:
    if len(candidates) < 2:
        return
    vecs = np.stack([padded_layout_vector(np.asarray(c["centers"]), max_num_cylinders) for c in candidates], axis=0)
    dists: List[float] = []
    for i in range(vecs.shape[0]):
        for j in range(i + 1, vecs.shape[0]):
            dists.append(float(np.linalg.norm(vecs[i] - vecs[j])))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(dists, bins=min(24, max(6, len(dists) // 2)), color="#4c78a8", alpha=0.85)
    ax.set_xlabel("Padded layout distance")
    ax.set_ylabel("Pair count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def try_save_cycle_gif(cycle: np.ndarray, out_path: Path, channel_order: Sequence[str]) -> None:
    try:
        import imageio.v2 as imageio
    except Exception:
        return
    names = [str(name).lower() for name in channel_order]
    omega_idx = names.index("omega") if "omega" in names else min(3, cycle.shape[-1] - 1)
    omega = cycle[..., omega_idx]
    vmax = max(float(np.max(np.abs(omega))), 1.0e-8)
    frames = []
    for frame in omega:
        normalized = np.clip(0.5 + 0.5 * frame / vmax, 0.0, 1.0)
        rgba = plt.cm.RdBu_r(normalized)
        frames.append((rgba[..., :3] * 255).astype(np.uint8))
    imageio.mimsave(out_path, frames, duration=0.12)


def write_candidates_csv(candidates: Sequence[Mapping[str, Any]], path: Path) -> None:
    keys = [
        "rank",
        "verified",
        "score",
        "constraint_penalty",
        "latent_consistency",
        "Re",
        "num_cylinders",
        "centers_json",
        "valid",
        "min_pair_distance",
        "per_kpi_errors_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for candidate in candidates:
            validity = candidate.get("validity", {})
            writer.writerow(
                {
                    "rank": candidate.get("rank", ""),
                    "verified": bool(candidate.get("verified", False)),
                    "score": candidate.get("score", ""),
                    "constraint_penalty": candidate.get("constraint_penalty", ""),
                    "latent_consistency": candidate.get("latent_consistency", ""),
                    "Re": candidate.get("Re", ""),
                    "num_cylinders": candidate.get("num_cylinders", candidate.get("count", "")),
                    "centers_json": json.dumps(json_safe(candidate.get("centers", []))),
                    "valid": validity.get("valid", ""),
                    "min_pair_distance": validity.get("min_pair_distance", ""),
                    "per_kpi_errors_json": json.dumps(json_safe(candidate.get("per_kpi_errors", {}))),
                }
            )


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    inverse_run = resolve_demo_path(args.inverse_run)
    model, ckpt, inv_model_cfg, ckpt_path = load_inverse_checkpoint(inverse_run, args.checkpoint, device)
    cfg = ckpt.get("config", {})
    kpi_names = ckpt.get("kpi_names", cfg.get("target_kpis", {}).get("names", DEFAULT_KPI_NAMES))
    kpi_stats = ckpt.get("kpi_stats", cfg.get("target_kpis", {}).get("stats"))
    normalize = bool(cfg.get("target_kpis", {}).get("normalize", True))
    target_payload = load_target_payload(args)
    re_value = float(target_payload.get("Re", target_payload.get("re", 100.0)))
    preferences = target_payload.get("preferences", {}) if isinstance(target_payload.get("preferences", {}), Mapping) else {}
    min_center_distance = target_payload.get("min_center_distance", preferences.get("min_center_distance", 1.1))
    min_center_distance = 1.1 if min_center_distance is None else float(min_center_distance)

    target_spec = target_spec_from_payload(
        target_payload,
        kpi_names=kpi_names,
        kpi_stats=kpi_stats,
        normalize=normalize,
        max_num_cylinders=int(inv_model_cfg.get("max_num_cylinders", 8)),
        re_scale=float(inv_model_cfg.get("re_scale", 200.0)),
        domain_length_scale=max(float(inv_model_cfg.get("domain_length_x", 24.0)), float(inv_model_cfg.get("domain_length_y", 12.0))),
    )
    target_vec = torch.from_numpy(np.asarray(target_spec["vector"], dtype=np.float32)).to(device=device)
    samples = model.sample_designs(
        target_vec,
        n_samples=int(args.n_samples),
        n_steps=int(args.n_steps),
        seed=int(args.seed),
        min_center_distance=min_center_distance,
        device=device,
    )

    forward_model, forward_model_cfg, forward_ckpt_path = load_forward_model(cfg["forward_model"], device)
    validation_cfg = cfg.get("validation", {})
    phase_bins = int(args.phase_bins or validation_cfg.get("forward_verify_phase_bins", 12))
    nx = int(args.nx or validation_cfg.get("forward_verify_nx", 96))
    ny = int(args.ny or validation_cfg.get("forward_verify_ny", 48))
    query_batch_size = int(cfg.get("forward_model", {}).get("query_batch_size", 32768))
    lx = float(inv_model_cfg.get("domain_length_x", 24.0))
    ly = float(inv_model_cfg.get("domain_length_y", 12.0))
    channel_order = list(INERT_CHANNEL_ORDER) if "INERT_CHANNEL_ORDER" in globals() else ["u", "v", "p", "omega"]

    out_dir = inverse_run / "evaluation" / f"inverse_eval_{current_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "target_spec.json", json_safe({"payload": target_payload, "target_spec": target_spec}))

    candidates: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        candidate = dict(sample)
        candidate["sample_index"] = idx
        candidate["Re"] = re_value
        candidate["num_cylinders"] = int(sample["count"])
        candidate["verified"] = False
        candidate["score"] = float("inf")
        candidates.append(candidate)

    verify_k = min(max(int(args.verify_top_k), 0), len(candidates))
    verify_indices = [c["sample_index"] for c in sorted(candidates, key=candidate_prefilter_key)[:verify_k]]
    verified_candidates: List[Dict[str, Any]] = []
    for rank_idx, sample_idx in enumerate(verify_indices):
        candidate = candidates[sample_idx]
        centers = np.asarray(candidate["centers"], dtype=np.float32).reshape(-1, 2)
        cycle, aux = predict_cycle_for_centers(
            forward_model,
            forward_model_cfg,
            centers,
            re_value=re_value,
            max_num_cylinders=int(inv_model_cfg.get("max_num_cylinders", 8)),
            phase_bins=phase_bins,
            nx=nx,
            ny=ny,
            lx=lx,
            ly=ly,
            query_batch_size=query_batch_size,
            device=device,
        )
        kpis = compute_cycle_kpis(cycle, x_grid=None, y_grid=None, channel_order=channel_order, domain={"lx": lx, "ly": ly})
        kpis["num_cylinders"] = int(candidate["num_cylinders"])
        kpis["min_center_distance"] = float(periodic_min_distance(centers, lx, ly))
        kpis["valid"] = bool(candidate.get("validity", {}).get("valid", True))
        score = score_candidate_kpis(kpis, target_spec)
        behavior_forward, org_forward = extract_forward_latent_targets(aux)
        behavior_hat = torch.from_numpy(np.asarray(candidate["behavior_latent_hat"], dtype=np.float32))
        org_hat = torch.from_numpy(np.asarray(candidate["organization_latent_hat"], dtype=np.float32))
        behavior_mse = torch.mean((behavior_forward[: behavior_hat.numel()] - behavior_hat[: behavior_forward.numel()]) ** 2).item()
        org_dim = min(org_forward.numel(), org_hat.numel())
        org_mse = torch.mean((org_forward[:org_dim] - org_hat[:org_dim]) ** 2).item() if org_dim > 0 else float("nan")
        latent_consistency = float(behavior_mse + (0.0 if not math.isfinite(org_mse) else org_mse))

        candidate.update(
            {
                "verified": True,
                "kpis": kpis,
                "score": float(score["total_score"]),
                "per_kpi_errors": score["per_kpi_errors"],
                "constraint_penalty": float(score["constraint_penalty"]),
                "latent_consistency": latent_consistency,
                "behavior_consistency_mse": float(behavior_mse),
                "organization_consistency_mse": float(org_mse),
                "cycle_shape": list(cycle.shape),
            }
        )
        verified_candidates.append(candidate)
        if rank_idx < 5:
            plot_candidate_flow(cycle, centers, out_dir / f"candidate_{rank_idx:03d}_flow.png", channel_order=channel_order, lx=lx, ly=ly)
            plot_organization(aux, centers, out_dir / f"candidate_{rank_idx:03d}_organization.png", lx=lx, ly=ly)
            try_save_cycle_gif(cycle, out_dir / f"candidate_{rank_idx:03d}_cycle.gif", channel_order)

    ranked = sorted(
        candidates,
        key=lambda c: (
            0 if bool(c.get("verified", False)) else 1,
            0 if bool(c.get("validity", {}).get("valid", False)) else 1,
            float(c.get("score", float("inf"))),
            float(c.get("latent_consistency", float("inf"))),
        ),
    )
    for rank, candidate in enumerate(ranked):
        candidate["rank"] = rank if candidate.get("verified", False) else ""

    verified_ranked = [c for c in ranked if c.get("verified", False)]
    write_candidates_csv(ranked, out_dir / "inverse_candidates.csv")
    write_json(
        out_dir / "inverse_candidates.json",
        json_safe(
            {
                "inverse_run": str(inverse_run),
                "checkpoint": str(ckpt_path),
                "forward_checkpoint": str(forward_ckpt_path),
                "target": target_payload,
                "candidates": ranked,
            }
        ),
    )
    plot_sampled_layouts(ranked, out_dir / "sampled_layouts_by_score.png", lx=lx, ly=ly)
    plot_kpi_target_vs_achieved(verified_ranked, target_payload, out_dir / "kpi_target_vs_achieved.png")
    plot_diversity(ranked, out_dir / "layout_diversity.png", max_num_cylinders=int(inv_model_cfg.get("max_num_cylinders", 8)))

    if args.refine_top_k or args.refine_steps:
        print("[refine] optional refinement is reserved in this demo build; sampled candidates were evaluated directly.")

    best = verified_ranked[0] if verified_ranked else None
    print(f"[done] wrote inverse evaluation to {out_dir}")
    if best is not None:
        print(f"[best] score={best['score']:.4e} count={best['num_cylinders']} centers={json.dumps(json_safe(best['centers']))}")


INERT_CHANNEL_ORDER = ("u", "v", "p", "omega")


if __name__ == "__main__":
    main()
