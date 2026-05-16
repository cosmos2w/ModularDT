from __future__ import annotations

"""Guided posterior search over the target-agnostic latent design prior."""

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from channelthermal_datasets import CHANNEL_ORDER
except Exception:  # pragma: no cover
    CHANNEL_ORDER = ("u", "v", "p", "omega", "temperature")

try:
    from field_functional_objective import FieldFunctionalObjective
    from layout_search_baselines import (
        LayoutSearchConfig,
        decode_design_vec_to_layout,
        encode_layout_to_design_vec,
        geometry_penalty,
    )
    from model_design_prior import LatentModularDesignPrior
    from thermal_hypergraph_consistency import compare_hypergraph_plans
    from thermal_inverse_kpi import compute_steady_thermal_kpis
    from train_inverse import (
        build_hypergraph_plan_from_forward_prediction,
        decode_hypergraph_plan_vector,
        infer_hypergraph_plan_num_edges,
        load_forward_model,
        predict_candidate_with_forward,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "search_design_prior.py requires the existing ChannelThermal forward/inverse helper modules to be importable. "
        "Expected helpers include train_inverse.load_forward_model, predict_candidate_with_forward, "
        "build_hypergraph_plan_from_forward_prediction, and decode_hypergraph_plan_vector."
    ) from exc


@dataclass
class GuidedSearchConfig:
    method: str = "latent_cem"
    num_initial_samples: int = 512
    num_return: int = 16
    latent_cem_iterations: int = 8
    latent_cem_population: int = 128
    latent_cem_elite_frac: float = 0.15
    latent_cem_init_std: float = 1.0
    latent_cem_min_std: float = 0.05
    latent_cem_smoothing: float = 0.5
    posterior_beta: float = 1.0
    prior_energy_weight: float = 0.01
    geometry_penalty_weight: float = 1.0
    hypergraph_consistency_weight: float = 1.0
    diversity_weight: float = 0.0
    random_seed: int = 0


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sum_geometry_penalty(layout: Mapping[str, Any], config: LayoutSearchConfig) -> float:
    validity = layout.get("validity", {}) if isinstance(layout.get("validity"), Mapping) else {}
    penalties = validity.get("geometry_penalty") if isinstance(validity, Mapping) else None
    if not isinstance(penalties, Mapping):
        penalties = geometry_penalty(layout, config)
    return float(sum(float(v) for v in penalties.values()))


def _layout_vector(candidate: Mapping[str, Any], max_num_modules: int, config: LayoutSearchConfig) -> np.ndarray:
    if candidate.get("design_vec") is not None:
        arr = np.asarray(candidate["design_vec"], dtype=np.float32).reshape(-1)
        return arr
    layout = candidate.get("layout", {})
    return encode_layout_to_design_vec(layout, config) if isinstance(layout, Mapping) else np.zeros((max_num_modules * 3,), dtype=np.float32)


def diversity_rerank(candidates: Sequence[Dict[str, Any]], *, weight: float, config: LayoutSearchConfig) -> List[Dict[str, Any]]:
    rows = list(candidates)
    if float(weight) <= 0.0 or len(rows) <= 2:
        for rank, row in enumerate(rows, start=1):
            row["rank"] = int(rank)
        return rows
    vectors = [_layout_vector(row, config.max_num_modules, config) for row in rows]
    selected = [0]
    remaining = set(range(1, len(rows)))
    while remaining:
        best_idx = None
        best_adjusted = float("inf")
        for idx in sorted(remaining):
            min_dist = min(float(np.linalg.norm(vectors[idx] - vectors[j])) for j in selected)
            adjusted = float(rows[idx].get("total_score", float("inf"))) - float(weight) * min_dist
            if adjusted < best_adjusted:
                best_adjusted = adjusted
                best_idx = idx
        if best_idx is None:
            break
        rows[best_idx]["diversity_adjusted_score"] = float(best_adjusted)
        selected.append(best_idx)
        remaining.remove(best_idx)
    reranked = [rows[idx] for idx in selected] + [rows[idx] for idx in sorted(remaining)]
    for rank, row in enumerate(reranked, start=1):
        row["rank"] = int(rank)
    return reranked


def _decode_hypergraph(vector: Optional[Any], max_num_modules: int) -> Optional[Dict[str, Any]]:
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    num_edges = infer_hypergraph_plan_num_edges(int(arr.size), int(max_num_modules))
    if num_edges <= 0:
        return None
    return decode_hypergraph_plan_vector(arr, max_num_modules=int(max_num_modules), num_edges=int(num_edges))


def _fallback_hypergraph_compare(planned: Optional[Mapping[str, Any]], realized: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(planned, Mapping) or not isinstance(realized, Mapping):
        return {"total": 0.0, "diagnostics_available": False, "reason": "planned_or_realized_hypergraph_missing"}
    p_strength = np.asarray(planned.get("hyper_strength", []), dtype=np.float64).reshape(-1)
    r_strength = np.asarray(realized.get("hyper_strength", []), dtype=np.float64).reshape(-1)
    n = min(p_strength.size, r_strength.size)
    strength = float(np.mean(np.abs(p_strength[:n] - r_strength[:n]))) if n else 0.0
    active = abs(float(planned.get("active_edge_count", 0.0) or 0.0) - float(realized.get("active_edge_count", 0.0) or 0.0))
    return {"total": float(strength + 0.1 * active), "strength_mae": strength, "active_count_error": active, "diagnostics_available": True, "fallback": True}


def compare_hypergraphs(planned: Optional[Mapping[str, Any]], realized: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(planned, Mapping) or not isinstance(realized, Mapping):
        return {"total": 0.0, "diagnostics_available": False, "reason": "planned_or_realized_hypergraph_missing"}
    try:
        out = compare_hypergraph_plans(planned, realized)
        out["diagnostics_available"] = True
        return out
    except Exception:
        return _fallback_hypergraph_compare(planned, realized)


class ForwardHONFEvaluator:
    """Thin adapter around existing frozen forward model utilities.

    Contract:
        evaluate_design_vec(design_vec, context) -> candidate payload containing
        forward_prediction, kpis, layout, realized_hypergraph, and optional fields.
    """

    def __init__(
        self,
        forward_model: Any,
        forward_metadata: Mapping[str, Any],
        device: torch.device | str,
        layout_config: LayoutSearchConfig,
        *,
        query_batch_size: int = 32768,
        generate_heat_power: bool = False,
        heat_load_policy: str = "preserve_total_heat",
        fixed_heat_per_module: Optional[float] = None,
        target_heat_power_total: Optional[float] = None,
    ) -> None:
        self.forward_model = forward_model
        self.forward_metadata = dict(forward_metadata)
        self.device = torch.device(device)
        self.layout_config = layout_config
        self.query_batch_size = int(query_batch_size)
        self.generate_heat_power = bool(generate_heat_power)
        self.heat_load_policy = str(heat_load_policy)
        self.fixed_heat_per_module = fixed_heat_per_module
        self.target_heat_power_total = target_heat_power_total
        self.num_forward_calls = 0

    @classmethod
    def from_config(
        cls,
        forward_cfg: Mapping[str, Any],
        device: torch.device | str,
        layout_config: LayoutSearchConfig,
    ) -> "ForwardHONFEvaluator":
        model, metadata, _ = load_forward_model(forward_cfg, torch.device(device))
        return cls(
            model,
            metadata,
            device,
            layout_config,
            query_batch_size=int(forward_cfg.get("query_batch_size", forward_cfg.get("batch_size", 32768))),
            generate_heat_power=bool(layout_config.generate_heat_power),
        )

    def evaluate_design_vec(self, design_vec: Any, context: Mapping[str, Any]) -> Dict[str, Any]:
        layout = decode_design_vec_to_layout(design_vec, self.layout_config)
        return self.evaluate_layout(layout, context)

    def evaluate_layout(self, layout_or_design_vec: Any, context: Mapping[str, Any]) -> Dict[str, Any]:
        if isinstance(layout_or_design_vec, Mapping):
            layout = dict(layout_or_design_vec)
        else:
            layout = decode_design_vec_to_layout(layout_or_design_vec, self.layout_config)
        record = context.get("record") if isinstance(context, Mapping) else None
        if record is None:
            raise ValueError("ForwardHONFEvaluator requires context['record'] for frozen forward verification.")
        design_vec = encode_layout_to_design_vec(layout, self.layout_config)
        prediction = predict_candidate_with_forward(
            self.forward_model,
            self.forward_metadata,
            record,
            layout,
            self.device,
            max_num_modules=int(self.layout_config.max_num_modules),
            generate_heat_power=self.generate_heat_power,
            heat_load_policy=str(context.get("heat_load_policy", self.heat_load_policy)),
            fixed_heat_per_module=self.fixed_heat_per_module,
            target_heat_power_total=self.target_heat_power_total,
            query_batch_size=int(context.get("query_batch_size", self.query_batch_size)),
        )
        self.num_forward_calls += 1
        kpis = compute_steady_thermal_kpis(
            prediction["pred_field_grid"],
            x_grid=record.x_grid,
            y_grid=record.y_grid,
            channel_order=CHANNEL_ORDER,
            module_centers=prediction.get("centers_padded"),
            module_present=prediction.get("module_present"),
            heat_powers=prediction.get("heat_powers", getattr(record, "heat_powers", None)),
            module_internal_temperature=prediction.get("pred_internal_temperature"),
            module_internal_mask=getattr(record, "module_internal_mask", None),
            interface_target=prediction.get("pred_interface"),
            interface_condition=prediction.get("pred_port_condition"),
            domain={"domain_length_x": record.domain_length_x, "domain_length_y": record.domain_length_y, "module_radius": record.module_radius},
            material_params=getattr(record, "material_params", None),
        )
        centers = np.asarray(layout.get("centers", []), dtype=np.float32).reshape(-1, 2)
        kpis["num_modules"] = int(layout.get("count", centers.shape[0]))
        realized_plan = build_hypergraph_plan_from_forward_prediction(
            prediction,
            max_num_modules=int(self.layout_config.max_num_modules),
            domain_length_x=float(record.domain_length_x),
            domain_length_y=float(record.domain_length_y),
            num_edges=None,
        )
        realized = _decode_hypergraph(realized_plan.get("vector"), int(self.layout_config.max_num_modules))
        return {
            "forward_prediction": prediction,
            "kpis": kpis,
            "layout": layout,
            "design_vec": design_vec,
            "realized_hypergraph": realized,
            "realized_hypergraph_raw": realized_plan,
            "num_forward_calls": int(self.num_forward_calls),
        }


class LatentPosteriorDesignSearcher:
    def __init__(
        self,
        prior_model: LatentModularDesignPrior,
        forward_evaluator: Any,
        objective: FieldFunctionalObjective,
        layout_config: LayoutSearchConfig,
        search_config: GuidedSearchConfig,
    ) -> None:
        self.prior_model = prior_model
        self.forward_evaluator = forward_evaluator
        self.objective = objective
        self.layout_config = layout_config
        self.search_config = search_config
        self.device = next(prior_model.parameters()).device
        self.forward_calls = 0

    def _context_tensor(self, context: Optional[Mapping[str, Any]], n: int = 1) -> Optional[torch.Tensor]:
        dim = int(getattr(self.prior_model.cfg, "context_dim", 0))
        if dim <= 0:
            return None
        raw = None if context is None else context.get("context_vec")
        if raw is None:
            arr = np.zeros((n, dim), dtype=np.float32)
        else:
            arr = np.asarray(raw, dtype=np.float32).reshape(-1)
            padded = np.zeros((dim,), dtype=np.float32)
            padded[: min(dim, arr.size)] = arr[: min(dim, arr.size)]
            arr = np.repeat(padded[None, :], n, axis=0)
        return torch.from_numpy(arr).to(self.device)

    def _evaluate_decoded(
        self,
        *,
        z: np.ndarray,
        decoded: Mapping[str, torch.Tensor],
        context: Mapping[str, Any],
        source: str,
        method: str,
    ) -> Dict[str, Any]:
        design_vec = decoded["design_vec"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        planned_vec = decoded.get("hypergraph_vec")
        behavior = decoded.get("behavior_vec")
        planned_np = planned_vec.detach().cpu().numpy().reshape(-1).astype(np.float32) if planned_vec is not None else None
        planned = _decode_hypergraph(planned_np, int(self.layout_config.max_num_modules))
        forward_payload = self.forward_evaluator.evaluate_design_vec(design_vec, context)
        self.forward_calls += 1
        layout = forward_payload["layout"]
        realized = forward_payload.get("realized_hypergraph")
        comparison = compare_hypergraphs(planned, realized)
        objective_result = self.objective.evaluate(
            forward_prediction=forward_payload.get("forward_prediction"),
            kpis=forward_payload.get("kpis"),
            layout=layout,
            planned_hypergraph=planned,
            realized_hypergraph=realized,
            hypergraph_consistency=comparison,
        )
        geom = _sum_geometry_penalty(layout, self.layout_config)
        z_tensor = torch.from_numpy(np.asarray(z, dtype=np.float32)[None]).to(self.device)
        prior_energy = float(self.prior_model.latent_prior_energy(z_tensor).detach().cpu().numpy().reshape(-1)[0])
        hyper_score = float(comparison.get("total", 0.0) or 0.0) if bool(comparison.get("diagnostics_available", False)) else 0.0
        total = (
            float(objective_result.get("total_score", 0.0))
            + float(self.search_config.geometry_penalty_weight) * geom
            + float(self.search_config.hypergraph_consistency_weight) * hyper_score
            + float(self.search_config.prior_energy_weight) * prior_energy
        )
        return _jsonable(
            {
                "method": method,
                "design_vec": np.asarray(forward_payload.get("design_vec", design_vec), dtype=np.float32),
                "layout": layout,
                "centers": layout.get("centers"),
                "count": int(layout.get("count", 0)),
                "num_modules": int(layout.get("count", 0)),
                "latent_z": np.asarray(z, dtype=np.float32),
                "planned_hypergraph": planned,
                "planned_hypergraph_vec": planned_np,
                "realized_hypergraph": realized,
                "hypergraph_consistency": comparison,
                "hypergraph_consistency_score": hyper_score,
                "behavior_vec_hat": behavior.detach().cpu().numpy().reshape(-1).astype(np.float32) if behavior is not None else None,
                "kpis": forward_payload.get("kpis", {}),
                "forward_prediction": forward_payload.get("forward_prediction", {}),
                "objective_result": objective_result,
                "total_score": total,
                "hard_violation_score": float(objective_result.get("hard_violation_score", 0.0)),
                "satisfied": bool(objective_result.get("satisfied", False)),
                "num_satisfied": int(objective_result.get("num_satisfied", 0)),
                "num_terms": int(objective_result.get("num_terms", 0)),
                "prior_energy": prior_energy,
                "geometry_penalty": geom,
                "repair_distance": float(layout.get("repair_distance", 0.0)),
                "forward_calls": int(self.forward_calls),
                "source": source,
            }
        )

    def sample_prior_candidates(self, context: Mapping[str, Any], num_samples: int, *, temperature: float = 1.0) -> List[Dict[str, Any]]:
        self.prior_model.eval()
        candidates: List[Dict[str, Any]] = []
        with torch.no_grad():
            context_vec = self._context_tensor(context, int(num_samples))
            torch.manual_seed(int(self.search_config.random_seed))
            z = torch.randn(int(num_samples), int(self.prior_model.cfg.latent_dim), device=self.device) * float(temperature)
            decoded_all = self.prior_model.decode_latent(z, context_vec)
            for idx in range(int(num_samples)):
                decoded = {key: value[idx : idx + 1] for key, value in decoded_all.items()}
                candidates.append(
                    self._evaluate_decoded(
                        z=z[idx].detach().cpu().numpy(),
                        decoded=decoded,
                        context=context,
                        source=f"prior_{idx}",
                        method="atlas_prior",
                    )
                )
        candidates.sort(key=lambda row: float(row.get("total_score", float("inf"))))
        return diversity_rerank(candidates, weight=float(self.search_config.diversity_weight), config=self.layout_config)

    def latent_cem_search(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        self.prior_model.eval()
        rng = np.random.default_rng(int(self.search_config.random_seed))
        latent_dim = int(self.prior_model.cfg.latent_dim)
        mean = np.zeros((latent_dim,), dtype=np.float64)
        std = np.full((latent_dim,), max(float(self.search_config.latent_cem_init_std), float(self.search_config.latent_cem_min_std)), dtype=np.float64)
        all_candidates: List[Dict[str, Any]] = []
        history = []
        pop = max(int(self.search_config.latent_cem_population), 2)
        elite_n = max(1, int(math.ceil(pop * float(self.search_config.latent_cem_elite_frac))))
        for iteration in range(max(int(self.search_config.latent_cem_iterations), 1)):
            z_np = rng.normal(mean[None, :], std[None, :], size=(pop, latent_dim)).astype(np.float32)
            rows = []
            with torch.no_grad():
                z = torch.from_numpy(z_np).to(self.device)
                context_vec = self._context_tensor(context, pop)
                decoded_all = self.prior_model.decode_latent(z, context_vec)
                for idx in range(pop):
                    decoded = {key: value[idx : idx + 1] for key, value in decoded_all.items()}
                    rows.append(
                        self._evaluate_decoded(
                            z=z_np[idx],
                            decoded=decoded,
                            context=context,
                            source=f"latent_cem_iteration_{iteration}",
                            method="atlas_guided",
                        )
                    )
            rows.sort(key=lambda row: float(row.get("total_score", float("inf"))))
            elites = rows[:elite_n]
            elite_z = np.stack([np.asarray(row["latent_z"], dtype=np.float64).reshape(-1) for row in elites], axis=0)
            smoothing = min(max(float(self.search_config.latent_cem_smoothing), 0.0), 1.0)
            mean = smoothing * mean + (1.0 - smoothing) * np.mean(elite_z, axis=0)
            std = smoothing * std + (1.0 - smoothing) * np.maximum(np.std(elite_z, axis=0), float(self.search_config.latent_cem_min_std))
            all_candidates.extend(rows)
            history.append(
                {
                    "iteration": int(iteration),
                    "best_score": float(rows[0]["total_score"]),
                    "mean_elite_score": float(np.mean([float(row["total_score"]) for row in elites])),
                    "num_forward_calls": int(self.forward_calls),
                }
            )
        all_candidates.sort(key=lambda row: float(row.get("total_score", float("inf"))))
        reranked = diversity_rerank(all_candidates, weight=float(self.search_config.diversity_weight), config=self.layout_config)
        top = reranked[: max(int(self.search_config.num_return), 0)]
        for rank, row in enumerate(top, start=1):
            row["rank"] = int(rank)
        return {"method": "atlas_guided", "best_candidates": top, "history": history, "num_forward_calls": int(self.forward_calls)}

    def smc_search(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("SMC posterior search is reserved for a later prompt; latent_cem is implemented for Prompt 3.")
