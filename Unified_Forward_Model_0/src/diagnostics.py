"""Diagnostics and plotting helpers for unified forward-model runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


EPS = 1e-8


def compute_basic_field_metrics(pred: torch.Tensor, target: Optional[torch.Tensor]) -> Dict[str, Any]:
    """Compute basic prediction metrics when targets are available."""
    metrics: Dict[str, Any] = {"pred_shape": list(pred.shape), "target_shape": None, "shape_mismatch": False}
    if target is None:
        metrics["field_mse"] = float("nan")
        return metrics
    pred = pred.detach()
    target = target.detach().to(device=pred.device, dtype=pred.dtype)
    metrics["target_shape"] = list(target.shape)
    if pred.shape != target.shape:
        metrics["field_mse"] = float("nan")
        metrics["shape_mismatch"] = True
        return metrics
    metrics["field_mse"] = float(torch.mean((pred - target) ** 2).cpu())
    if pred.shape[-1] >= 5 and target.shape[-1] >= 5:
        metrics["temperature_mse"] = float(torch.mean((pred[..., 4] - target[..., 4]) ** 2).cpu())
    return metrics


def compute_hypergraph_diagnostics(org: Dict[str, Any]) -> Dict[str, float]:
    """Summarize organizer incidence, mass, and shortcut diagnostics."""
    out: Dict[str, float] = {}
    strength = org.get("hyper_strength")
    if torch.is_tensor(strength):
        strength_detached = strength.detach()
        out["active_edge_count"] = float((strength_detached > 0.05).float().sum(dim=-1).mean().cpu())
        out["hyper_strength_mean"] = float(strength_detached.mean().cpu())

    A_mh = org.get("A_mh")
    if torch.is_tensor(A_mh):
        out["A_mh_entropy"] = _entropy(A_mh.detach(), dim=-1)
    module_mass = org.get("hyper_module_mass")
    if torch.is_tensor(module_mass):
        out["module_mass_max"] = float(module_mass.detach().amax().cpu())
    module_mass_raw = org.get("hyper_module_mass_raw")
    if torch.is_tensor(module_mass_raw):
        out["module_mass_raw_max"] = float(module_mass_raw.detach().amax().cpu())

    A_eh = org.get("A_eh")
    if torch.is_tensor(A_eh):
        out["A_eh_entropy"] = _entropy(A_eh.detach(), dim=-1)
    env_mass = org.get("hyper_env_mass")
    if torch.is_tensor(env_mass):
        out["env_mass_max"] = float(env_mass.detach().amax().cpu())
    env_mass_raw = org.get("hyper_env_mass_raw")
    if torch.is_tensor(env_mass_raw):
        out["env_mass_raw_max"] = float(env_mass_raw.detach().amax().cpu())

    gate = org.get("direct_residual_gate")
    if torch.is_tensor(gate):
        out["direct_residual_gate"] = float(gate.detach().mean().cpu())
    elif gate is not None:
        out["direct_residual_gate"] = float(gate)

    geom_mean = org.get("hyper_geometry_bias_mean")
    if torch.is_tensor(geom_mean):
        out["hyper_geometry_bias_mean"] = float(geom_mean.detach().mean().cpu())
    geom_std = org.get("hyper_geometry_bias_std")
    if torch.is_tensor(geom_std):
        out["hyper_geometry_bias_std"] = float(geom_std.detach().mean().cpu())

    return out


def save_diagnostics_json(payload: Dict[str, Any], path: str | Path) -> None:
    """Save diagnostics payload as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def plot_organization_overview(org: Dict[str, Any], path: str | Path) -> None:
    """Plot module centers, environment tokens, and hyperedge source/region centers."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[diagnostics] matplotlib is not installed; skipping organization plot.")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module_centers = org["module_centers"][0].detach().cpu()
    module_present = org["module_present"][0].detach().cpu() > 0
    env_coords = org["env_coords"][0].detach().cpu()
    source = org["hyper_source_coords"][0].detach().cpu()
    region = org["hyper_region_coords"][0].detach().cpu()
    strength = org.get("hyper_strength")
    strength_cpu = strength[0].detach().cpu() if torch.is_tensor(strength) else None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(env_coords[:, 0], env_coords[:, 1], s=8, alpha=0.35, label="env")
    ax.scatter(module_centers[module_present, 0], module_centers[module_present, 1], s=55, label="modules")
    sizes = 40 if strength_cpu is None else 30 + 40 * strength_cpu / strength_cpu.max().clamp_min(EPS)
    ax.scatter(source[:, 0], source[:, 1], s=sizes, marker="x", label="hyper source")
    ax.scatter(region[:, 0], region[:, 1], s=sizes, marker="s", facecolors="none", label="hyper region")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    ax.set_title("Organization Overview")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ablation_summary(rows: list[Dict[str, Any]], path: str | Path) -> None:
    """Plot simple ablation MSE bars."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[diagnostics] matplotlib is not installed; skipping ablation plot.")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [str(row.get("name", idx)) for idx, row in enumerate(rows)]
    mses = [float(row.get("field_mse", float("nan"))) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, mses)
    ax.set_ylabel("field MSE")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _entropy(weights: torch.Tensor, dim: int) -> float:
    probs = weights / weights.sum(dim=dim, keepdim=True).clamp_min(EPS)
    entropy = -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=dim)
    return float(entropy.mean().cpu())


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return {
            "shape": list(value.shape),
            "mean": float(value.detach().float().mean().cpu()),
            "std": float(value.detach().float().std(unbiased=False).cpu()) if value.numel() > 1 else 0.0,
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
