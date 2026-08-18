"""CORE HONF scalar diagnostics and generic organizer regularization.

Inputs are the legacy-compatible model output dictionary containing
`organizer_aux` and `routing_aux`. Outputs are scalar diagnostics for training
logs plus optional generic anti-collapse regularization. The metrics are
domain-reusable HONF quantities: assignment entropy, mass concentration,
query-to-H attention summaries, pairwise/context norms, and feature flags.

This helper does not retain dense query-routing maps; it consumes only scalar
decoder summaries and static organizer tensors already returned by the model.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch


EPS = 1.0e-8

HONF_DIAGNOSTIC_KEYS = [
    "candidate_edge_count",
    "selected_edge_count",
    "viable_selected_edge_count",
    "functional_edge_count",
    "soft_functional_edge_count",
    "empty_selected_edge_count",
    "effective_query_edge_count",
    "selection_module_coverage",
    "selection_environment_coverage",
    "candidate_module_nonzero_fraction",
    "selected_module_nonzero_fraction",
    "candidate_environment_nonzero_fraction",
    "selected_environment_nonzero_fraction",
    "selected_module_probability_mass_min",
    "selected_module_probability_mass_p05",
    "selected_module_probability_mass_mean",
    "selected_environment_probability_mass_min",
    "selected_environment_probability_mass_p05",
    "selected_environment_probability_mass_mean",
    "query_edge_retained_probability_mass_min",
    "query_edge_retained_probability_mass_p05",
    "query_edge_retained_probability_mass_mean",
    "retained_module_incidence_mass_min",
    "retained_module_incidence_mass_p05",
    "retained_module_incidence_mass_mean",
    "A_mh_entropy",
    "A_eh_entropy",
    "module_mass_entropy_norm",
    "env_mass_entropy_norm",
    "module_mass_max",
    "env_mass_max",
    "query_attention_entropy",
    "query_attention_effective_edges",
    "query_attention_max",
    "pairwise_kernel_gate",
    "pairwise_context_norm",
    "hyper_value_context_norm",
    "total_hyper_context_norm",
    "nonhyper_context_norm",
    "uses_hyper_value_context",
    "uses_pairwise_kernel",
]


def _scalar(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Reduce a tensor/value to one scalar on the requested device and dtype."""

    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype).mean()
    if value is None:
        return torch.zeros((), device=device, dtype=dtype)
    return torch.as_tensor(float(value), device=device, dtype=dtype)


def _entropy(prob: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute categorical entropy along one tensor dimension."""

    p = prob.clamp_min(EPS)
    return -(p * torch.log(p)).sum(dim=dim)


def _entropy_norm(prob: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize categorical entropy by its maximum ``log(category_count)``."""

    count = max(int(prob.shape[dim]), 2)
    return _entropy(prob, dim=dim) / math.log(float(count))


def compute_honf_diagnostics(
    output: Dict[str, Any],
    *,
    edge_strength_threshold: float = 0.05,
    edge_strength_temperature: float = 0.05,
) -> Dict[str, float]:
    """Return scalar HONF diagnostics suitable for `metrics.csv`."""

    pred = output["pred_field"]
    device, dtype = pred.device, pred.dtype
    org = output.get("organizer_aux", {})
    routing = output.get("routing_aux", {})
    strength = org.get("hyper_strength")
    if not torch.is_tensor(strength):
        strength = pred.new_zeros(pred.shape[0], 0)
    strength = strength.to(device=device, dtype=dtype)
    threshold = float(edge_strength_threshold)
    temperature = max(float(edge_strength_temperature), EPS)
    functional = (strength > threshold).to(dtype=dtype).sum(dim=-1).mean() if strength.numel() else pred.new_zeros(())
    soft_functional = torch.sigmoid((strength - threshold) / temperature).sum(dim=-1).mean() if strength.numel() else pred.new_zeros(())
    candidate_count = _scalar(org.get("candidate_edge_count"), device, dtype)
    if org.get("candidate_edge_count") is None:
        candidate_count = pred.new_tensor(float(strength.shape[-1]))
    selected_mask = org.get("edge_active_mask")
    if torch.is_tensor(selected_mask):
        selected_mask = selected_mask.to(device=device, dtype=dtype)
        selected_count = selected_mask.sum(dim=-1).mean()
    else:
        selected_mask = torch.ones_like(strength)
        selected_count = _scalar(org.get("selected_edge_count"), device, dtype)
        if org.get("selected_edge_count") is None:
            selected_count = candidate_count
    effective_mask = org.get("effective_edge_mask")
    if torch.is_tensor(effective_mask):
        effective_mask = effective_mask.to(device=device, dtype=dtype)
        viable_selected = effective_mask.sum(dim=-1).mean()
    else:
        viable_selected = _scalar(org.get("viable_selected_edge_count"), device, dtype)
        if org.get("viable_selected_edge_count") is None:
            viable_selected = selected_count

    A_mh = org.get("A_mh")
    A_eh = org.get("A_eh")
    module_mass = org.get("hyper_module_mass")
    env_mass = org.get("hyper_env_mass")
    if torch.is_tensor(A_mh):
        A_mh = A_mh.to(device=device, dtype=dtype)
        module_present = org.get("module_present")
        if torch.is_tensor(module_present):
            weights = module_present.to(device=device, dtype=dtype)
            A_mh_entropy = (_entropy(A_mh, dim=-1) * weights).sum() / weights.sum().clamp_min(EPS)
        else:
            A_mh_entropy = _entropy(A_mh, dim=-1).mean()
    else:
        A_mh_entropy = pred.new_zeros(())
    A_eh_entropy = _entropy(A_eh.to(device=device, dtype=dtype), dim=-1).mean() if torch.is_tensor(A_eh) else pred.new_zeros(())
    module_mass = module_mass.to(device=device, dtype=dtype) if torch.is_tensor(module_mass) else pred.new_zeros(*strength.shape)
    env_mass = env_mass.to(device=device, dtype=dtype) if torch.is_tensor(env_mass) else pred.new_zeros(*strength.shape)

    empty_selected = org.get("empty_selected_edge_count")
    if empty_selected is None and strength.numel():
        empty_mask = selected_mask.to(dtype=torch.bool) & (
            (module_mass <= EPS) | (env_mass <= EPS)
        )
        empty_selected_value = empty_mask.to(dtype=dtype).sum(dim=-1).mean()
    else:
        empty_selected_value = _scalar(empty_selected, device, dtype)

    values = {
        "candidate_edge_count": candidate_count,
        "selected_edge_count": selected_count,
        "viable_selected_edge_count": viable_selected,
        "functional_edge_count": functional,
        "soft_functional_edge_count": soft_functional,
        "empty_selected_edge_count": empty_selected_value,
        "effective_query_edge_count": _scalar(
            routing.get("effective_query_edge_count", routing.get("mean_query_nonzero_edges")),
            device,
            dtype,
        ),
        "selection_module_coverage": _scalar(org.get("selection_module_coverage"), device, dtype),
        "selection_environment_coverage": _scalar(org.get("selection_environment_coverage"), device, dtype),
        "candidate_module_nonzero_fraction": _scalar(org.get("candidate_module_nonzero_fraction"), device, dtype),
        "selected_module_nonzero_fraction": _scalar(org.get("selected_module_nonzero_fraction"), device, dtype),
        "candidate_environment_nonzero_fraction": _scalar(org.get("candidate_environment_nonzero_fraction"), device, dtype),
        "selected_environment_nonzero_fraction": _scalar(org.get("selected_environment_nonzero_fraction"), device, dtype),
        "selected_module_probability_mass_min": _scalar(org.get("selected_module_probability_mass_min"), device, dtype),
        "selected_module_probability_mass_p05": _scalar(org.get("selected_module_probability_mass_p05"), device, dtype),
        "selected_module_probability_mass_mean": _scalar(org.get("selected_module_probability_mass_mean"), device, dtype),
        "selected_environment_probability_mass_min": _scalar(org.get("selected_environment_probability_mass_min"), device, dtype),
        "selected_environment_probability_mass_p05": _scalar(org.get("selected_environment_probability_mass_p05"), device, dtype),
        "selected_environment_probability_mass_mean": _scalar(org.get("selected_environment_probability_mass_mean"), device, dtype),
        "query_edge_retained_probability_mass_min": _scalar(routing.get("query_edge_retained_probability_mass_min"), device, dtype),
        "query_edge_retained_probability_mass_p05": _scalar(routing.get("query_edge_retained_probability_mass_p05"), device, dtype),
        "query_edge_retained_probability_mass_mean": _scalar(routing.get("query_edge_retained_probability_mass_mean"), device, dtype),
        "retained_module_incidence_mass_min": _scalar(routing.get("retained_module_incidence_mass_min"), device, dtype),
        "retained_module_incidence_mass_p05": _scalar(routing.get("retained_module_incidence_mass_p05"), device, dtype),
        "retained_module_incidence_mass_mean": _scalar(routing.get("retained_module_incidence_mass_mean"), device, dtype),
        "A_mh_entropy": A_mh_entropy,
        "A_eh_entropy": A_eh_entropy,
        "module_mass_entropy_norm": _entropy_norm(module_mass, dim=-1).mean() if module_mass.numel() else pred.new_zeros(()),
        "env_mass_entropy_norm": _entropy_norm(env_mass, dim=-1).mean() if env_mass.numel() else pred.new_zeros(()),
        "module_mass_max": module_mass.amax(dim=-1).mean() if module_mass.numel() else pred.new_zeros(()),
        "env_mass_max": env_mass.amax(dim=-1).mean() if env_mass.numel() else pred.new_zeros(()),
        "query_attention_entropy": _scalar(routing.get("hyper_attention_entropy"), device, dtype),
        "query_attention_effective_edges": _scalar(routing.get("hyper_attention_effective_edges"), device, dtype),
        "query_attention_max": _scalar(routing.get("hyper_attention_max"), device, dtype),
        "pairwise_kernel_gate": _scalar(routing.get("pairwise_kernel_gate"), device, dtype),
        "pairwise_context_norm": _scalar(routing.get("pairwise_context_norm"), device, dtype),
        "hyper_value_context_norm": _scalar(routing.get("hyper_value_context_norm"), device, dtype),
        "total_hyper_context_norm": _scalar(routing.get("total_hyper_context_norm", routing.get("hyper_context_norm")), device, dtype),
        "nonhyper_context_norm": _scalar(routing.get("nonhyper_context_norm"), device, dtype),
        "uses_hyper_value_context": _scalar(routing.get("uses_hyper_value_context"), device, dtype),
        "uses_pairwise_kernel": _scalar(routing.get("pairwise_kernel_enabled"), device, dtype),
    }
    return {key: float(values[key].detach().cpu()) for key in HONF_DIAGNOSTIC_KEYS}


def organizer_regularization_loss(output: Dict[str, Any], config: Dict[str, Any] | None) -> torch.Tensor:
    """Generic HONF anti-collapse regularization; disabled unless configured."""

    cfg = dict(config or {})
    pred = output["pred_field"]
    if not bool(cfg.get("enabled", False)):
        return pred.new_zeros(())
    org = output.get("organizer_aux", {})
    strength = org.get("hyper_strength")
    if not torch.is_tensor(strength):
        return pred.new_zeros(())
    strength = strength.to(device=pred.device, dtype=pred.dtype)
    threshold = float(cfg.get("edge_strength_threshold", 0.05))
    temperature = max(float(cfg.get("edge_strength_temperature", 0.05)), EPS)
    soft_active = torch.sigmoid((strength - threshold) / temperature).sum(dim=-1)
    target_active = float(cfg.get("target_active_edges", max(1, strength.shape[-1])))
    loss = pred.new_zeros(())
    active_weight = float(cfg.get("active_edge_weight", 0.0))
    if active_weight:
        loss = loss + pred.new_tensor(active_weight) * (soft_active - target_active).square().mean()

    min_fraction = float(cfg.get("min_mass_entropy_fraction", 0.0))
    for key, weight_key in (
        ("hyper_env_mass", "env_mass_entropy_floor_weight"),
        ("hyper_module_mass", "module_mass_entropy_floor_weight"),
    ):
        weight = float(cfg.get(weight_key, 0.0))
        mass = org.get(key)
        if weight and torch.is_tensor(mass):
            entropy_norm = _entropy_norm(mass.to(device=pred.device, dtype=pred.dtype), dim=-1)
            loss = loss + pred.new_tensor(weight) * torch.relu(pred.new_tensor(min_fraction) - entropy_norm).square().mean()

    max_weight = float(cfg.get("max_mass_weight", 0.0))
    max_fraction = float(cfg.get("max_mass_fraction", 1.0))
    if max_weight:
        for key in ("hyper_env_mass", "hyper_module_mass"):
            mass = org.get(key)
            if torch.is_tensor(mass):
                mass = mass.to(device=pred.device, dtype=pred.dtype)
                loss = loss + pred.new_tensor(max_weight) * torch.relu(mass.amax(dim=-1) - max_fraction).square().mean()

    duplicate_weight = float(cfg.get("duplicate_weight", 0.0))
    if duplicate_weight:
        src = org.get("hyper_source_coords")
        dst = org.get("hyper_region_coords")
        if torch.is_tensor(src) and torch.is_tensor(dst) and src.shape[-2] > 1:
            vectors = torch.cat([src, dst], dim=-1).to(device=pred.device, dtype=pred.dtype)
            vectors = vectors - vectors.mean(dim=-2, keepdim=True)
            vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(EPS)
            sim = torch.einsum("bkh,blh->bkl", vectors, vectors)
            eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool).unsqueeze(0)
            threshold_sim = float(cfg.get("duplicate_similarity_threshold", 0.95))
            dup = torch.relu(sim.masked_fill(eye, -1.0) - threshold_sim).square()
            loss = loss + pred.new_tensor(duplicate_weight) * dup.mean()
    return loss
