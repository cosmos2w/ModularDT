from __future__ import annotations

"""Mechanism-prior search over hypergraph-conditioned layout realization."""

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

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
class MechanismGuidedSearchConfig:
    method: str = "mechanism_cem"
    num_initial_samples: int = 512
    num_return: int = 16

    mechanism_cem_iterations: int = 8
    mechanism_cem_population: int = 128
    mechanism_cem_elite_frac: float = 0.15
    mechanism_cem_init_std: float = 0.5
    mechanism_cem_min_std: float = 0.03
    mechanism_cem_smoothing: float = 0.5

    layouts_per_mechanism: int = 2
    layout_sample_steps: int = 16
    layout_temperature: float = 1.0

    mechanism_prior_weight: float = 0.05
    geometry_penalty_weight: float = 1.0
    hypergraph_realization_weight: float = 1.0
    diversity_weight: float = 0.0

    filter_mechanisms_by_count: bool = True
    mechanism_jitter_std: float = 0.05
    random_seed: int = 0


GuidedSearchConfig = MechanismGuidedSearchConfig


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


def objective_has_plan_realization_term(objective: Any) -> bool:
    """Return True when the objective already scores planned-vs-realized consistency."""

    spec = getattr(objective, "spec", {})
    if not isinstance(spec, Mapping):
        return False
    for term in spec.get("hypergraph_terms", []) or []:
        if not isinstance(term, Mapping):
            continue
        operator = str(term.get("operator", "")).lower().strip()
        name = str(term.get("name", "")).lower()
        if operator == "plan_realization_distance" or "planned_realized" in name:
            return True
    return False


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


def _count_range_from_context(config: LayoutSearchConfig, context: Optional[Mapping[str, Any]]) -> Tuple[int, int]:
    hard: Dict[str, Any] = {}
    if isinstance(context, Mapping):
        if isinstance(context.get("hard_constraints"), Mapping):
            hard.update(context["hard_constraints"])  # type: ignore[index]
        if isinstance(context.get("objective_spec"), Mapping) and isinstance(context["objective_spec"].get("hard_constraints"), Mapping):  # type: ignore[index]
            hard.update(context["objective_spec"]["hard_constraints"])  # type: ignore[index]
    lo = int(hard.get("num_modules_min", config.min_num_modules))
    hi = int(hard.get("num_modules_max", config.max_active_modules or config.max_num_modules))
    if isinstance(hard.get("num_modules"), Sequence) and len(hard["num_modules"]) >= 2:
        lo = int(hard["num_modules"][0])
        hi = int(hard["num_modules"][1])
    lo = max(0, min(lo, int(config.max_num_modules)))
    hi = max(lo, min(hi, int(config.max_num_modules)))
    return lo, hi


class HypergraphMechanismDesignSearcher:
    def __init__(
        self,
        mechanism_atlas: Any,
        layout_realizer: Any,
        forward_evaluator: Any,
        objective: FieldFunctionalObjective,
        layout_config: LayoutSearchConfig,
        search_config: MechanismGuidedSearchConfig,
    ) -> None:
        self.mechanism_atlas = mechanism_atlas
        self.layout_realizer = layout_realizer
        self.forward_evaluator = forward_evaluator
        self.objective = objective
        self.layout_config = layout_config
        self.search_config = search_config
        self.device = next(layout_realizer.parameters()).device
        self.forward_calls = 0
        self._objective_counts_plan_realization = objective_has_plan_realization_term(objective)

    def _context_tensor(self, context: Optional[Mapping[str, Any]], n: int = 1) -> Optional[torch.Tensor]:
        dim = int(getattr(self.layout_realizer.cfg, "context_dim", 0))
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

    def _evaluate_forward(self, design_vec: np.ndarray, context: Mapping[str, Any]) -> Dict[str, Any]:
        if hasattr(self.forward_evaluator, "evaluate_design_vec"):
            return self.forward_evaluator.evaluate_design_vec(design_vec, context)
        if callable(self.forward_evaluator):
            return self.forward_evaluator(design_vec, context)
        raise TypeError("forward_evaluator must be callable or expose evaluate_design_vec(design_vec, context).")

    def _sample_layouts(self, mechanism_features: np.ndarray, context: Mapping[str, Any], *, num_layouts: int) -> np.ndarray:
        mech = np.asarray(mechanism_features, dtype=np.float32).reshape(-1, int(self.layout_realizer.cfg.mechanism_dim))
        if num_layouts > 1:
            mech = np.repeat(mech, int(num_layouts), axis=0)
        with torch.no_grad():
            mech_t = torch.from_numpy(mech).to(self.device)
            context_t = self._context_tensor(context, mech_t.shape[0])
            out = self.layout_realizer.sample_layout(
                mech_t,
                context_t,
                num_samples=int(mech_t.shape[0]),
                steps=int(self.search_config.layout_sample_steps),
                temperature=float(self.search_config.layout_temperature),
            )
        return out["design_vec"].detach().cpu().numpy().astype(np.float32)

    def _eligible_cluster_ids(self, count_range: Optional[Tuple[float, float]]) -> Optional[np.ndarray]:
        centers = np.asarray(getattr(self.mechanism_atlas, "cluster_centers", []), dtype=np.float32)
        if count_range is None or centers.size == 0:
            return None
        decoded = self.mechanism_atlas.decode_feature_to_parts(centers)
        count_descriptor = decoded.get("count_descriptor")
        if count_descriptor is None:
            return None
        counts = np.asarray(count_descriptor, dtype=np.float32).reshape(-1)
        lo, hi = float(count_range[0]), float(count_range[1])
        eligible = np.where((counts >= lo) & (counts <= hi))[0].astype(np.int64)
        return eligible if eligible.size else None

    def _sample_atlas_features(
        self,
        num_samples: int,
        *,
        rng: np.random.Generator,
        count_range: Optional[Tuple[float, float]],
    ) -> np.ndarray:
        cluster_ids = None
        eligible = self._eligible_cluster_ids(count_range)
        if eligible is not None:
            cluster_counts = np.asarray(getattr(self.mechanism_atlas, "cluster_counts", []), dtype=np.float64)
            weights = cluster_counts[eligible] if cluster_counts.size >= int(np.max(eligible)) + 1 else np.ones((eligible.size,), dtype=np.float64)
            weights = weights / weights.sum() if weights.sum() > 0.0 else np.full((eligible.size,), 1.0 / max(eligible.size, 1))
            cluster_ids = rng.choice(eligible, size=int(num_samples), replace=True, p=weights)
        sampled = self.mechanism_atlas.sample_features(
            int(num_samples),
            rng=rng,
            count_range=count_range if cluster_ids is None else None,
            cluster_ids=cluster_ids,
            jitter_std=float(self.search_config.mechanism_jitter_std),
        )
        return np.asarray(sampled["features"], dtype=np.float32)

    def _evaluate_mechanism_layout(
        self,
        *,
        mechanism_feature: np.ndarray,
        design_vec: np.ndarray,
        context: Mapping[str, Any],
        source: str,
        method: str,
        layout_index: int = 0,
    ) -> Dict[str, Any]:
        mechanism = np.asarray(mechanism_feature, dtype=np.float32).reshape(1, -1)
        decoded_parts = self.mechanism_atlas.decode_feature_to_parts(mechanism)
        desired_vec = np.asarray(decoded_parts.get("hypergraph_vec", np.zeros((1, 0), dtype=np.float32))[0], dtype=np.float32)
        desired = _decode_hypergraph(desired_vec, int(self.layout_config.max_num_modules))
        desired_behavior = np.asarray(decoded_parts.get("behavior_vec", np.zeros((1, 0), dtype=np.float32))[0], dtype=np.float32)
        forward_payload = self._evaluate_forward(np.asarray(design_vec, dtype=np.float32).reshape(-1), context)
        self.forward_calls += 1
        layout = forward_payload.get("layout", decode_design_vec_to_layout(design_vec, self.layout_config))
        realized = forward_payload.get("realized_hypergraph")
        comparison = compare_hypergraphs(desired, realized)
        objective_result = self.objective.evaluate(
            forward_prediction=forward_payload.get("forward_prediction"),
            kpis=forward_payload.get("kpis"),
            layout=layout,
            planned_hypergraph=desired,
            realized_hypergraph=realized,
            hypergraph_consistency=comparison,
        )
        geom = _sum_geometry_penalty(layout, self.layout_config)
        prior_energy = float(np.asarray(self.mechanism_atlas.prior_energy(mechanism), dtype=np.float32).reshape(-1)[0])
        hyper_score = float(comparison.get("total", 0.0) or 0.0) if bool(comparison.get("diagnostics_available", False)) else 0.0
        hyper_extra_score = float(self.search_config.hypergraph_realization_weight) * hyper_score
        if self._objective_counts_plan_realization:
            hyper_extra_score = 0.0
        total = (
            float(objective_result.get("total_score", 0.0))
            + float(self.search_config.geometry_penalty_weight) * geom
            + hyper_extra_score
            + float(self.search_config.mechanism_prior_weight) * prior_energy
        )
        cluster_id = int(np.asarray(self.mechanism_atlas.nearest_cluster(mechanism)).reshape(-1)[0])
        return _jsonable(
            {
                "method": method,
                "mechanism_feature": mechanism.reshape(-1).astype(np.float32),
                "mechanism_cluster_id": cluster_id,
                "desired_hypergraph": desired,
                "desired_hypergraph_vec": desired_vec,
                "desired_behavior": desired_behavior,
                "planned_hypergraph": desired,
                "planned_hypergraph_vec": desired_vec,
                "design_vec": np.asarray(forward_payload.get("design_vec", design_vec), dtype=np.float32),
                "layout": layout,
                "centers": layout.get("centers"),
                "count": int(layout.get("count", 0)),
                "num_modules": int(layout.get("count", 0)),
                "realized_hypergraph": realized,
                "hypergraph_consistency": comparison,
                "hypergraph_realization_score": hyper_score,
                "hypergraph_consistency_score": hyper_score,
                "hypergraph_extra_score": hyper_extra_score,
                "kpis": forward_payload.get("kpis", {}),
                "forward_prediction": forward_payload.get("forward_prediction", {}),
                "objective_result": objective_result,
                "total_score": total,
                "objective_score": float(objective_result.get("total_score", 0.0)),
                "mechanism_prior_score": prior_energy,
                "hard_violation_score": float(objective_result.get("hard_violation_score", 0.0)),
                "satisfied": bool(objective_result.get("satisfied", False)),
                "num_satisfied": int(objective_result.get("num_satisfied", 0)),
                "num_terms": int(objective_result.get("num_terms", 0)),
                "prior_energy": prior_energy,
                "geometry_penalty": geom,
                "repair_distance": float(layout.get("repair_distance", 0.0)),
                "forward_calls": int(self.forward_calls),
                "layout_index": int(layout_index),
                "source": source,
            }
        )

    def _evaluate_mechanisms(
        self,
        mechanism_features: np.ndarray,
        context: Mapping[str, Any],
        *,
        method: str,
        source: str,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        features = np.asarray(mechanism_features, dtype=np.float32).reshape(-1, int(self.layout_realizer.cfg.mechanism_dim))
        layouts_per = max(int(self.search_config.layouts_per_mechanism), 1)
        design_vecs = self._sample_layouts(features, context, num_layouts=layouts_per)
        for idx in tqdm(range(features.shape[0]), desc=method, unit="mechanism", leave=False, dynamic_ncols=True):
            rows = []
            for layout_idx in range(layouts_per):
                design_idx = idx * layouts_per + layout_idx
                rows.append(
                    self._evaluate_mechanism_layout(
                        mechanism_feature=features[idx],
                        design_vec=design_vecs[design_idx],
                        context=context,
                        source=f"{source}_{idx}",
                        method=method,
                        layout_index=layout_idx,
                    )
                )
            rows.sort(key=lambda row: float(row.get("total_score", float("inf"))))
            candidates.extend(rows)
        return candidates

    def sample_mechanism_prior_candidates(
        self,
        context: Mapping[str, Any],
        num_samples: int,
        *,
        count_range: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(int(self.search_config.random_seed))
        if count_range is None and bool(self.search_config.filter_mechanisms_by_count):
            count_range = _count_range_from_context(self.layout_config, context)
        features = self._sample_atlas_features(
            int(num_samples),
            rng=rng,
            count_range=count_range,
        )
        candidates = self._evaluate_mechanisms(features, context, method="mechanism_prior", source="mechanism_prior")
        candidates.sort(key=lambda row: float(row.get("total_score", float("inf"))))
        return diversity_rerank(candidates, weight=float(self.search_config.diversity_weight), config=self.layout_config)

    def mechanism_cem_search(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        rng = np.random.default_rng(int(self.search_config.random_seed))
        mechanism_dim = int(self.layout_realizer.cfg.mechanism_dim)
        count_range = _count_range_from_context(self.layout_config, context) if bool(self.search_config.filter_mechanisms_by_count) else None
        seed_features = self._sample_atlas_features(
            int(max(self.search_config.num_initial_samples, self.search_config.mechanism_cem_population)),
            rng=rng,
            count_range=count_range,
        ).astype(np.float64).reshape(-1, mechanism_dim)
        mean = np.mean(seed_features, axis=0)
        std = np.maximum(np.std(seed_features, axis=0), float(self.search_config.mechanism_cem_init_std))
        all_candidates: List[Dict[str, Any]] = []
        history = []
        pop = max(int(self.search_config.mechanism_cem_population), 2)
        elite_n = max(1, int(math.ceil(pop * float(self.search_config.mechanism_cem_elite_frac))))
        total_iterations = max(int(self.search_config.mechanism_cem_iterations), 1)
        cumulative_best = float("inf")
        progress = tqdm(range(total_iterations), desc="mechanism_guided cem", unit="iter", dynamic_ncols=True)
        for iteration in progress:
            iter_start = time.time()
            if iteration == 0:
                mechanism_np = seed_features[:pop].astype(np.float32)
            else:
                mechanism_np = rng.normal(mean[None, :], std[None, :], size=(pop, mechanism_dim)).astype(np.float32)
            rows = self._evaluate_mechanisms(mechanism_np, context, method="mechanism_guided", source=f"mechanism_cem_iteration_{iteration}")
            rows.sort(key=lambda row: float(row.get("total_score", float("inf"))))
            best_by_mechanism: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                key = tuple(np.round(np.asarray(row["mechanism_feature"], dtype=np.float64), 8).tolist())
                if key not in best_by_mechanism or float(row["total_score"]) < float(best_by_mechanism[key]["total_score"]):
                    best_by_mechanism[key] = row
            mechanism_rows = sorted(best_by_mechanism.values(), key=lambda row: float(row.get("total_score", float("inf"))))
            elites = mechanism_rows[:elite_n]
            elite_features = np.stack([np.asarray(row["mechanism_feature"], dtype=np.float64).reshape(-1) for row in elites], axis=0)
            smoothing = min(max(float(self.search_config.mechanism_cem_smoothing), 0.0), 1.0)
            mean = smoothing * mean + (1.0 - smoothing) * np.mean(elite_features, axis=0)
            std = smoothing * std + (1.0 - smoothing) * np.maximum(np.std(elite_features, axis=0), float(self.search_config.mechanism_cem_min_std))
            all_candidates.extend(rows)
            round_best = float(rows[0]["total_score"]) if rows else float("inf")
            cumulative_best = min(cumulative_best, round_best)
            history.append(
                {
                    "iteration": int(iteration),
                    "best_score": float(cumulative_best),
                    "round_best_score": float(round_best),
                    "mean_elite_score": float(np.mean([float(row["total_score"]) for row in elites])),
                    "num_forward_calls": int(self.forward_calls),
                    "runtime_seconds": float(time.time() - iter_start),
                }
            )
            progress.set_postfix(best=f"{cumulative_best:.4g}", calls=self.forward_calls)
        all_candidates.sort(key=lambda row: float(row.get("total_score", float("inf"))))
        reranked = diversity_rerank(all_candidates, weight=float(self.search_config.diversity_weight), config=self.layout_config)
        top = reranked[: max(int(self.search_config.num_return), 0)]
        for rank, row in enumerate(top, start=1):
            row["rank"] = int(rank)
        return {"method": "mechanism_guided", "best_candidates": top, "history": history, "num_forward_calls": int(self.forward_calls)}

    def smc_search(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("SMC posterior search is reserved for a later prompt; mechanism_cem is implemented.")


LatentPosteriorDesignSearcher = HypergraphMechanismDesignSearcher
