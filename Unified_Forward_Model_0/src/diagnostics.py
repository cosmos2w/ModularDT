"""Diagnostics and plotting helpers for unified forward-model runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


EPS = 1e-8


def compute_basic_field_metrics(pred: torch.Tensor, target: Optional[torch.Tensor]) -> Dict[str, Any]:
    """Compute basic prediction metrics when targets are available."""
    metrics: Dict[str, Any] = {"pred_shape": list(pred.shape), "target_shape": None, "shape_mismatch": 0.0}
    if target is None:
        metrics["field_mse"] = float("nan")
        return metrics
    pred = pred.detach()
    target = target.detach().to(device=pred.device, dtype=pred.dtype)
    metrics["target_shape"] = list(target.shape)
    if pred.shape != target.shape:
        metrics["field_mse"] = float("nan")
        metrics["shape_mismatch"] = 1.0
        return metrics
    metrics["field_mse"] = float(torch.mean((pred - target) ** 2).cpu())
    channel_mse = torch.mean((pred - target) ** 2, dim=tuple(range(pred.ndim - 1)))
    for idx, value in enumerate(channel_mse):
        metrics[f"mse_channel_{idx}"] = float(value.cpu())
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

    for key in (
        "uses_hyper_context",
        "uses_global_context",
        "uses_direct_context",
        "uses_near_module_context",
        "hyper_context_norm",
        "nonhyper_context_norm",
    ):
        value = org.get(key)
        if torch.is_tensor(value):
            out[key] = float(value.detach().mean().cpu())
        elif value is not None:
            out[key] = float(value)

    return out


def compute_organizer_regularization(
    output: Dict[str, Any],
    model_config: Any,
    training_cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Weak generic anti-collapse regularization for learned hypergraph usage."""
    cfg = _organizer_reg_cfg(training_cfg)
    reference = output.get("hyper_strength")
    if not torch.is_tensor(reference):
        ref = output["pred_field"] if torch.is_tensor(output.get("pred_field")) else None
        if ref is None:
            raise KeyError("compute_organizer_regularization requires hyper_strength or pred_field for device inference.")
        zero = ref.new_tensor(0.0)
        return _zero_org_reg_payload(zero)

    strength = reference
    zero = strength.new_tensor(0.0)
    num_edges = max(int(getattr(model_config, "num_hyperedges", strength.shape[-1])), 1)
    threshold = float(cfg["edge_strength_threshold"])
    temperature = max(float(cfg["edge_strength_temperature"]), EPS)
    soft_active = torch.sigmoid((strength - threshold) / temperature)
    soft_active_edge_count = soft_active.sum(dim=-1).mean()
    active_loss = (soft_active_edge_count - float(cfg["target_active_edges"])).square()

    env_mass = output.get("hyper_env_mass")
    module_mass = output.get("hyper_module_mass")
    env_entropy_norm = _mass_entropy_norm(env_mass, num_edges) if torch.is_tensor(env_mass) else zero
    module_entropy_norm = _mass_entropy_norm(module_mass, num_edges) if torch.is_tensor(module_mass) else zero
    min_entropy = float(cfg["min_mass_entropy_fraction"])
    env_entropy_loss = F.relu(min_entropy - env_entropy_norm).square()
    module_entropy_loss = F.relu(min_entropy - module_entropy_norm).square()

    env_mass_max = env_mass.amax(dim=-1).mean() if torch.is_tensor(env_mass) else zero
    module_mass_max = module_mass.amax(dim=-1).mean() if torch.is_tensor(module_mass) else zero
    max_mass = float(cfg["max_mass_fraction"])
    mass_max_loss = F.relu(env_mass_max - max_mass).square() + F.relu(module_mass_max - max_mass).square()

    duplicate_loss = zero
    A_eh = output.get("A_eh")
    A_mh = output.get("A_mh")
    duplicate_threshold = float(cfg["duplicate_similarity_threshold"])
    if torch.is_tensor(A_eh):
        duplicate_loss = duplicate_loss + _duplicate_column_loss(A_eh, duplicate_threshold)
    if torch.is_tensor(A_mh):
        duplicate_loss = duplicate_loss + _duplicate_column_loss(A_mh, duplicate_threshold)

    enabled = bool(cfg["enabled"])
    org_reg_loss = zero
    if enabled:
        org_reg_loss = (
            float(cfg["active_edge_weight"]) * active_loss
            + float(cfg["env_mass_entropy_weight"]) * env_entropy_loss
            + float(cfg["module_mass_entropy_weight"]) * module_entropy_loss
            + float(cfg["mass_max_weight"]) * mass_max_loss
            + float(cfg["duplicate_weight"]) * duplicate_loss
        )

    return {
        "org_reg_loss": org_reg_loss,
        "org_active_loss": active_loss,
        "org_env_entropy_loss": env_entropy_loss,
        "org_module_entropy_loss": module_entropy_loss,
        "org_mass_max_loss": mass_max_loss,
        "org_duplicate_loss": duplicate_loss,
        "soft_active_edge_count": soft_active_edge_count,
        "env_mass_entropy_norm": env_entropy_norm,
        "module_mass_entropy_norm": module_entropy_norm,
        "env_mass_max": env_mass_max,
        "module_mass_max": module_mass_max,
    }


def compute_channelthermal_region_metrics(
    pred: torch.Tensor,
    target: Optional[torch.Tensor],
    batch: Any,
    model_config: Any,
) -> Dict[str, Any]:
    """Compute ChannelThermal MSE diagnostics on simple geometric regions."""
    metrics: Dict[str, Any] = {}
    if target is None:
        return metrics
    pred = pred.detach()
    target = target.detach().to(device=pred.device, dtype=pred.dtype)
    if pred.shape != target.shape:
        return metrics

    query_xy = batch.query_xy.to(device=pred.device, dtype=pred.dtype)
    module_centers = batch.module_centers.to(device=pred.device, dtype=pred.dtype)
    module_present = batch.module_present.to(device=pred.device, dtype=pred.dtype)
    radius = float(getattr(model_config, "module_radius", 0.45))
    lx = float(getattr(model_config, "domain_length_x", 12.0))
    ly = float(getattr(model_config, "domain_length_y", 4.0))

    delta = query_xy[:, :, None, :] - module_centers[:, None, :, :]
    dist = torch.sqrt(delta.square().sum(dim=-1).clamp_min(EPS))
    large = torch.full_like(dist, 1.0e6)
    nearest = torch.where(module_present[:, None, :] > 0, dist, large).amin(dim=-1)

    x = query_xy[..., 0]
    y = query_xy[..., 1]
    masks = {
        "near_module": nearest <= radius + 0.25,
        "far_field": nearest > radius + 1.0,
        "wall_band": (y < 0.4) | (y > ly - 0.4),
        "outlet_band": x > lx - 0.75,
        "downstream_mid_box": (x >= 0.55 * lx) & (x <= 0.95 * lx) & (y >= 0.25 * ly) & (y <= 0.75 * ly),
    }
    diff2 = (pred - target).square()
    for name, mask in masks.items():
        metrics[f"field_mse_{name}"] = _masked_mse(diff2, mask)
        if pred.shape[-1] >= 5:
            metrics[f"temperature_mse_{name}"] = _masked_mse(diff2[..., 4], mask)
    return metrics


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


def plot_ablation_summary(rows: list[Dict[str, Any]], path: str | Path, metric_key: str = "best_val_field_mse_physical") -> None:
    """Plot simple ablation MSE bars."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[diagnostics] matplotlib is not installed; skipping ablation plot.")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [str(row.get("name", idx)) for idx, row in enumerate(rows)]
    mses = [float(row.get(metric_key, row.get("field_mse", row.get("best_val_field_mse", float("nan"))))) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, mses)
    ax.set_ylabel(metric_key)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _entropy(weights: torch.Tensor, dim: int) -> float:
    probs = weights / weights.sum(dim=dim, keepdim=True).clamp_min(EPS)
    entropy = -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=dim)
    return float(entropy.mean().cpu())


def _organizer_reg_cfg(training_cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "enabled": False,
        "active_edge_weight": 0.005,
        "target_active_edges": 3.0,
        "edge_strength_threshold": 0.05,
        "edge_strength_temperature": 0.02,
        "env_mass_entropy_weight": 0.002,
        "module_mass_entropy_weight": 0.001,
        "min_mass_entropy_fraction": 0.65,
        "mass_max_weight": 0.001,
        "max_mass_fraction": 0.75,
        "duplicate_weight": 0.001,
        "duplicate_similarity_threshold": 0.95,
    }
    payload = dict(training_cfg.get("organizer_regularization", {}) or {})
    defaults.update(payload)
    return defaults


def _zero_org_reg_payload(zero: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "org_reg_loss": zero,
        "org_active_loss": zero,
        "org_env_entropy_loss": zero,
        "org_module_entropy_loss": zero,
        "org_mass_max_loss": zero,
        "org_duplicate_loss": zero,
        "soft_active_edge_count": zero,
        "env_mass_entropy_norm": zero,
        "module_mass_entropy_norm": zero,
        "env_mass_max": zero,
        "module_mass_max": zero,
    }


def _mass_entropy_norm(mass: torch.Tensor, num_edges: int) -> torch.Tensor:
    probs = mass / mass.sum(dim=-1, keepdim=True).clamp_min(EPS)
    entropy = -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=-1)
    return (entropy / max(math.log(float(max(num_edges, 2))), EPS)).mean()


def _duplicate_column_loss(weights: torch.Tensor, threshold: float) -> torch.Tensor:
    if weights.shape[-1] <= 1:
        return weights.new_tensor(0.0)
    columns = F.normalize(weights.float(), p=2, dim=1, eps=EPS)
    similarity = torch.einsum("bnk,bnl->bkl", columns, columns)
    eye = torch.eye(similarity.shape[-1], device=similarity.device, dtype=torch.bool).unsqueeze(0)
    off_diag = similarity.masked_fill(eye, 0.0)
    denom = float(max(similarity.shape[-1] * (similarity.shape[-1] - 1), 1))
    return F.relu(off_diag - threshold).square().sum(dim=(-1, -2)).mean() / denom


def _masked_mse(diff2: torch.Tensor, mask: torch.Tensor) -> float:
    while mask.ndim < diff2.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=diff2.device, dtype=diff2.dtype)
    denom = mask.expand_as(diff2).sum().clamp_min(1.0)
    return float((diff2 * mask).sum().div(denom).detach().cpu())


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
