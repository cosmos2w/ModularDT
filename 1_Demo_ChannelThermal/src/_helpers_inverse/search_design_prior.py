from __future__ import annotations

"""Mechanism-prior search over hypergraph-conditioned layout realization."""

from dataclasses import dataclass, replace
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
        RawLayoutCEMOptimizer,
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
    method: str = "retrieve_refine"
    mode: str = "retrieve_refine"
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
    ranking_score_key: str = "internal_total_score"

    use_representative_bank: bool = True
    screen_representatives: bool = True
    representative_screen_samples: int = 256
    screen_top_k_representatives: int = 32
    screen_top_k_clusters: int = 8
    include_witness_layouts: bool = True

    local_refine_enabled: bool = True
    local_refine_top_k: int = 8
    local_refine_iterations: int = 4
    local_refine_population: int = 48
    local_refine_init_std: float = 0.08
    local_refine_num_return: int = 8

    def __post_init__(self) -> None:
        key = str(self.ranking_score_key or "internal_total_score").strip()
        if key not in {"internal_total_score", "fair_objective_score"}:
            raise ValueError("ranking_score_key must be 'internal_total_score' or 'fair_objective_score'.")
        self.ranking_score_key = key
        mode = str(self.mode or self.method or "retrieve_refine").strip()
        if mode == "mechanism_cem":
            mode = "mechanism_feature_cem"
        if mode not in {"retrieve_refine", "mechanism_feature_cem"}:
            raise ValueError("mode must be 'retrieve_refine' or 'mechanism_feature_cem'.")
        self.mode = mode
        if str(self.method or "").strip() in {"", "mechanism_cem"}:
            self.method = mode


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


def diversity_rerank(candidates: Sequence[Dict[str, Any]], *, weight: float, config: LayoutSearchConfig, score_key: str = "total_score") -> List[Dict[str, Any]]:
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
            adjusted = float(rows[idx].get(score_key, row_score(rows[idx]))) - float(weight) * min_dist
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


def row_score(row: Mapping[str, Any], key: str = "total_score") -> float:
    try:
        return float(row.get(key, row.get("total_score", float("inf"))))
    except (TypeError, ValueError):
        return float("inf")


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
        representative_bank: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.mechanism_atlas = mechanism_atlas
        self.layout_realizer = layout_realizer
        self.forward_evaluator = forward_evaluator
        self.objective = objective
        self.layout_config = layout_config
        self.search_config = search_config
        self.representative_bank = self._normalize_representative_bank(representative_bank)
        self.device = next(layout_realizer.parameters()).device
        self.forward_calls = 0
        self._objective_counts_plan_realization = objective_has_plan_realization_term(objective)
        self.ranking_score_key = str(search_config.ranking_score_key)

    def _normalize_representative_bank(self, bank: Optional[Mapping[str, Any]]) -> Optional[Dict[str, np.ndarray]]:
        if not isinstance(bank, Mapping) or "features" not in bank or "design_vecs" not in bank:
            return None
        out: Dict[str, np.ndarray] = {}
        for key in ("features", "design_vecs", "hypergraph_vecs", "behavior_vecs", "true_counts", "cluster_ids", "sample_indices", "source_tags"):
            if key in bank:
                out[key] = np.asarray(bank[key])
        if out.get("features", np.zeros((0,))).size == 0 or out.get("design_vecs", np.zeros((0,))).size == 0:
            return None
        return out

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
        objective_score = float(objective_result.get("total_score", 0.0))
        ranking_score = objective_score if self.ranking_score_key == "fair_objective_score" else total
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
                "internal_total_score": total,
                "fair_objective_score": objective_score,
                "objective_score": objective_score,
                "ranking_score": ranking_score,
                "ranking_score_key": self.ranking_score_key,
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
            rows.sort(key=lambda row: row_score(row, "ranking_score"))
            candidates.extend(rows)
        return candidates

    def _annotate_layout_candidate_with_mechanism(
        self,
        row: Mapping[str, Any],
        mechanism_feature: np.ndarray,
        *,
        source: str,
        method: str = "mechanism_guided",
    ) -> Dict[str, Any]:
        mechanism = np.asarray(mechanism_feature, dtype=np.float32).reshape(1, -1)
        decoded_parts = self.mechanism_atlas.decode_feature_to_parts(mechanism)
        desired_vec = np.asarray(decoded_parts.get("hypergraph_vec", np.zeros((1, 0), dtype=np.float32))[0], dtype=np.float32)
        desired = _decode_hypergraph(desired_vec, int(self.layout_config.max_num_modules))
        desired_behavior = np.asarray(decoded_parts.get("behavior_vec", np.zeros((1, 0), dtype=np.float32))[0], dtype=np.float32)
        realized = row.get("realized_hypergraph")
        forward_payload = row.get("forward_prediction", {}) if isinstance(row.get("forward_prediction"), Mapping) else {}
        if realized is None and isinstance(forward_payload, Mapping):
            realized = forward_payload.get("realized_hypergraph")
        comparison = compare_hypergraphs(desired, realized)
        objective_score = float(row.get("fair_objective_score", row.get("objective_score", row.get("total_score", 0.0))) or 0.0)
        layout = row.get("layout", {})
        geom = _sum_geometry_penalty(layout, self.layout_config) if isinstance(layout, Mapping) else 0.0
        prior_energy = float(np.asarray(self.mechanism_atlas.prior_energy(mechanism), dtype=np.float32).reshape(-1)[0])
        hyper_score = float(comparison.get("total", 0.0) or 0.0) if bool(comparison.get("diagnostics_available", False)) else 0.0
        hyper_extra_score = float(self.search_config.hypergraph_realization_weight) * hyper_score
        if self._objective_counts_plan_realization:
            hyper_extra_score = 0.0
        total = objective_score + float(self.search_config.geometry_penalty_weight) * geom + hyper_extra_score + float(self.search_config.mechanism_prior_weight) * prior_energy
        ranking_score = objective_score if self.ranking_score_key == "fair_objective_score" else total
        cluster_id = int(np.asarray(self.mechanism_atlas.nearest_cluster(mechanism)).reshape(-1)[0])
        out = dict(row)
        out.update(
            {
                "method": method,
                "mechanism_feature": mechanism.reshape(-1).astype(np.float32),
                "mechanism_cluster_id": cluster_id,
                "desired_hypergraph": desired,
                "desired_hypergraph_vec": desired_vec,
                "desired_behavior": desired_behavior,
                "planned_hypergraph": desired,
                "planned_hypergraph_vec": desired_vec,
                "realized_hypergraph": realized,
                "kpis": row.get("kpis", forward_payload.get("kpis", {})),
                "hypergraph_consistency": comparison,
                "hypergraph_realization_score": hyper_score,
                "hypergraph_consistency_score": hyper_score,
                "hypergraph_extra_score": hyper_extra_score,
                "total_score": total,
                "internal_total_score": total,
                "fair_objective_score": objective_score,
                "objective_score": objective_score,
                "ranking_score": ranking_score,
                "ranking_score_key": self.ranking_score_key,
                "mechanism_prior_score": prior_energy,
                "prior_energy": prior_energy,
                "geometry_penalty": geom,
                "source": source,
            }
        )
        return _jsonable(out)

    def retrieve_representatives(self, context: Mapping[str, Any]) -> List[Dict[str, Any]]:
        bank = self.representative_bank
        self._last_retrieval_history: List[Dict[str, Any]] = []
        if bank is None or not bool(self.search_config.use_representative_bank):
            return []
        features = np.asarray(bank["features"], dtype=np.float32)
        designs = np.asarray(bank["design_vecs"], dtype=np.float32)
        counts = np.asarray(bank.get("true_counts", np.zeros((features.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
        cluster_ids = np.asarray(bank.get("cluster_ids", np.zeros((features.shape[0],), dtype=np.int64)), dtype=np.int64).reshape(-1)
        sample_indices = np.asarray(bank.get("sample_indices", np.arange(features.shape[0])), dtype=np.int64).reshape(-1)
        if features.size == 0 or designs.size == 0:
            return []
        idx = np.arange(features.shape[0])
        if bool(self.search_config.filter_mechanisms_by_count) and counts.size:
            lo, hi = _count_range_from_context(self.layout_config, context)
            eligible = idx[(counts[: idx.size] >= float(lo)) & (counts[: idx.size] <= float(hi))]
            if eligible.size:
                idx = eligible
        rng = np.random.default_rng(int(self.search_config.random_seed))
        if idx.size > int(self.search_config.representative_screen_samples):
            idx = rng.choice(idx, size=int(self.search_config.representative_screen_samples), replace=False)

        rows: List[Dict[str, Any]] = []
        if bool(self.search_config.screen_representatives) or bool(self.search_config.include_witness_layouts):
            for local_i, rep_idx in enumerate(tqdm(idx, desc="mechanism retrieval", unit="rep", leave=False, dynamic_ncols=True)):
                row = self._evaluate_mechanism_layout(
                    mechanism_feature=features[int(rep_idx)],
                    design_vec=designs[int(rep_idx)],
                    context=context,
                    source="representative_witness",
                    method="mechanism_guided",
                    layout_index=local_i,
                )
                row["representative_index"] = int(rep_idx)
                row["sample_index"] = int(sample_indices[int(rep_idx)]) if sample_indices.size > int(rep_idx) else int(rep_idx)
                row["source"] = "representative_witness"
                rows.append(row)
            rows.sort(key=lambda row: row_score(row, "fair_objective_score" if bool(self.search_config.screen_representatives) else "ranking_score"))
        else:
            order = sorted(idx.tolist(), key=lambda rep_idx: float(np.asarray(self.mechanism_atlas.prior_energy(features[int(rep_idx)][None, :])).reshape(-1)[0]))
            for rep_idx in order:
                rows.append(
                    {
                        "feature": features[int(rep_idx)].astype(np.float32),
                        "mechanism_feature": features[int(rep_idx)].astype(np.float32),
                        "design_vec": designs[int(rep_idx)].astype(np.float32),
                        "mechanism_cluster_id": int(cluster_ids[int(rep_idx)]) if cluster_ids.size > int(rep_idx) else int(self.mechanism_atlas.nearest_cluster(features[int(rep_idx)][None, :])[0]),
                        "representative_index": int(rep_idx),
                        "sample_index": int(sample_indices[int(rep_idx)]) if sample_indices.size > int(rep_idx) else int(rep_idx),
                        "source": "representative_witness",
                    }
                )
        if rows and int(self.search_config.screen_top_k_clusters) > 0:
            kept: List[Dict[str, Any]] = []
            seen_clusters: set[int] = set()
            allowed_clusters: set[int] = set()
            for row in rows:
                cid = int(row.get("mechanism_cluster_id", -1))
                if cid not in allowed_clusters and len(allowed_clusters) >= int(self.search_config.screen_top_k_clusters):
                    continue
                allowed_clusters.add(cid)
                kept.append(row)
                seen_clusters.add(cid)
                if len(kept) >= int(self.search_config.screen_top_k_representatives):
                    break
            rows = kept
        else:
            rows = rows[: int(self.search_config.screen_top_k_representatives)]
        best_objective = row_score(rows[0], "fair_objective_score") if rows and "fair_objective_score" in rows[0] else float("inf")
        self._last_retrieval_history = [
            {
                "iteration": 0,
                "method": "mechanism_retrieval",
                "round_best_score": float(best_objective),
                "best_score": float(best_objective),
                "best_objective_score": float(best_objective),
                "best_internal_total_score": row_score(rows[0], "internal_total_score") if rows and "internal_total_score" in rows[0] else float("inf"),
                "num_forward_calls": int(self.forward_calls),
                "ranking_score_key": "fair_objective_score",
            }
        ]
        return rows

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
        candidates.sort(key=lambda row: row_score(row, "ranking_score"))
        return diversity_rerank(candidates, weight=float(self.search_config.diversity_weight), config=self.layout_config, score_key="ranking_score")

    def retrieve_refine_search(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        all_candidates: List[Dict[str, Any]] = []
        history: List[Dict[str, Any]] = []
        cumulative_objective = float("inf")
        cumulative_internal = float("inf")

        representatives = self.retrieve_representatives(context)
        if getattr(self, "_last_retrieval_history", None):
            history.extend(self._last_retrieval_history)
            if representatives:
                cumulative_objective = min(cumulative_objective, row_score(representatives[0], "fair_objective_score"))
                cumulative_internal = min(cumulative_internal, row_score(representatives[0], "internal_total_score"))
                history[-1]["best_objective_score"] = float(cumulative_objective)
                history[-1]["best_internal_total_score"] = float(cumulative_internal)
        if bool(self.search_config.include_witness_layouts):
            all_candidates.extend([row for row in representatives if "total_score" in row])

        if representatives:
            selected_features = np.stack([np.asarray(row.get("mechanism_feature", row.get("feature")), dtype=np.float32).reshape(-1) for row in representatives], axis=0)
        else:
            rng = np.random.default_rng(int(self.search_config.random_seed))
            count_range = _count_range_from_context(self.layout_config, context) if bool(self.search_config.filter_mechanisms_by_count) else None
            selected_features = self._sample_atlas_features(
                max(int(self.search_config.screen_top_k_representatives), 1),
                rng=rng,
                count_range=count_range,
            )

        iter_start = time.time()
        generated = self._evaluate_mechanisms(selected_features, context, method="mechanism_guided", source="mechanism_realizer_sample")
        for row in generated:
            row["source"] = "mechanism_realizer_sample"
        all_candidates.extend(generated)
        if generated:
            generated.sort(key=lambda row: row_score(row, "ranking_score"))
            round_obj = min(row_score(row, "fair_objective_score") for row in generated)
            round_internal = min(row_score(row, "internal_total_score") for row in generated)
            cumulative_objective = min(cumulative_objective, round_obj)
            cumulative_internal = min(cumulative_internal, round_internal)
            history.append(
                {
                    "iteration": len(history),
                    "method": "mechanism_realizer_sample",
                    "round_best_score": float(round_internal if self.ranking_score_key == "internal_total_score" else round_obj),
                    "best_score": float(cumulative_internal if self.ranking_score_key == "internal_total_score" else cumulative_objective),
                    "best_objective_score": float(cumulative_objective),
                    "best_internal_total_score": float(cumulative_internal),
                    "num_forward_calls": int(self.forward_calls),
                    "ranking_score_key": self.ranking_score_key,
                    "runtime_seconds": float(time.time() - iter_start),
                }
            )

        if bool(self.search_config.local_refine_enabled) and all_candidates:
            seed_rows = sorted(all_candidates, key=lambda row: row_score(row, "ranking_score"))[: max(int(self.search_config.local_refine_top_k), 0)]
            for seed_idx, seed in enumerate(seed_rows):
                local_cfg = replace(
                    self.layout_config,
                    cem_iterations=max(int(self.search_config.local_refine_iterations), 1),
                    cem_population=max(int(self.search_config.local_refine_population), 2),
                    random_seed=int(self.search_config.random_seed) + 1009 + seed_idx,
                )
                optimizer = RawLayoutCEMOptimizer(local_cfg, self.forward_evaluator, self.objective)
                before_calls = int(self.forward_calls)
                local = optimizer.search(
                    context,
                    num_return=max(int(self.search_config.local_refine_num_return), 1),
                    initial_mean_vec=np.asarray(seed.get("design_vec"), dtype=np.float32).reshape(-1),
                    initial_std=float(self.search_config.local_refine_init_std),
                )
                local_calls = int(local.get("num_forward_calls", 0))
                self.forward_calls += local_calls
                feature = np.asarray(seed.get("mechanism_feature"), dtype=np.float32).reshape(-1)
                local_rows = []
                for row in local.get("best_candidates", []):
                    annotated = self._annotate_layout_candidate_with_mechanism(
                        row,
                        feature,
                        source="mechanism_local_refine",
                        method="mechanism_guided",
                    )
                    annotated["forward_calls"] = int(before_calls + int(row.get("forward_calls", 0)))
                    annotated["seed_source"] = seed.get("source", "")
                    annotated["seed_rank"] = int(seed.get("rank", seed_idx + 1) or seed_idx + 1)
                    local_rows.append(annotated)
                all_candidates.extend(local_rows)
                if local_rows:
                    round_obj = min(row_score(row, "fair_objective_score") for row in local_rows)
                    round_internal = min(row_score(row, "internal_total_score") for row in local_rows)
                    cumulative_objective = min(cumulative_objective, round_obj)
                    cumulative_internal = min(cumulative_internal, round_internal)
                for local_hist in local.get("history", []):
                    history.append(
                        {
                            "iteration": len(history),
                            "method": "mechanism_local_refine",
                            "seed_index": int(seed_idx),
                            "round_best_score": float(local_hist.get("round_best_score", local_hist.get("best_score", float("inf")))),
                            "best_score": float(cumulative_internal if self.ranking_score_key == "internal_total_score" else cumulative_objective),
                            "best_objective_score": float(cumulative_objective),
                            "best_internal_total_score": float(cumulative_internal),
                            "mean_elite_score": local_hist.get("mean_elite_score", ""),
                            "num_forward_calls": int(before_calls + int(local_hist.get("num_forward_calls", 0))),
                            "ranking_score_key": self.ranking_score_key,
                            "runtime_seconds": local_hist.get("runtime_seconds", ""),
                        }
                    )

        all_candidates.sort(key=lambda row: row_score(row, "ranking_score"))
        reranked = diversity_rerank(all_candidates, weight=float(self.search_config.diversity_weight), config=self.layout_config, score_key="ranking_score")
        top = reranked[: max(int(self.search_config.num_return), 0)]
        for rank, row in enumerate(top, start=1):
            row["rank"] = int(rank)
        return {"method": "mechanism_guided", "best_candidates": top, "history": history, "num_forward_calls": int(self.forward_calls)}

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
            rows.sort(key=lambda row: row_score(row, "ranking_score"))
            best_by_mechanism: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                key = tuple(np.round(np.asarray(row["mechanism_feature"], dtype=np.float64), 8).tolist())
                if key not in best_by_mechanism or row_score(row, "ranking_score") < row_score(best_by_mechanism[key], "ranking_score"):
                    best_by_mechanism[key] = row
            mechanism_rows = sorted(best_by_mechanism.values(), key=lambda row: row_score(row, "ranking_score"))
            elites = mechanism_rows[:elite_n]
            elite_features = np.stack([np.asarray(row["mechanism_feature"], dtype=np.float64).reshape(-1) for row in elites], axis=0)
            smoothing = min(max(float(self.search_config.mechanism_cem_smoothing), 0.0), 1.0)
            mean = smoothing * mean + (1.0 - smoothing) * np.mean(elite_features, axis=0)
            std = smoothing * std + (1.0 - smoothing) * np.maximum(np.std(elite_features, axis=0), float(self.search_config.mechanism_cem_min_std))
            all_candidates.extend(rows)
            round_best = row_score(rows[0], "ranking_score") if rows else float("inf")
            cumulative_best = min(cumulative_best, round_best)
            best_objective = min([row_score(row, "fair_objective_score") for row in all_candidates], default=float("inf"))
            best_internal = min([row_score(row, "internal_total_score") for row in all_candidates], default=float("inf"))
            history.append(
                {
                    "iteration": int(iteration),
                    "best_score": float(cumulative_best),
                    "round_best_score": float(round_best),
                    "best_objective_score": float(best_objective),
                    "best_internal_total_score": float(best_internal),
                    "mean_elite_score": float(np.mean([row_score(row, "ranking_score") for row in elites])),
                    "ranking_score_key": self.ranking_score_key,
                    "num_forward_calls": int(self.forward_calls),
                    "runtime_seconds": float(time.time() - iter_start),
                }
            )
            progress.set_postfix(best=f"{cumulative_best:.4g}", calls=self.forward_calls)
        all_candidates.sort(key=lambda row: row_score(row, "ranking_score"))
        reranked = diversity_rerank(all_candidates, weight=float(self.search_config.diversity_weight), config=self.layout_config, score_key="ranking_score")
        top = reranked[: max(int(self.search_config.num_return), 0)]
        for rank, row in enumerate(top, start=1):
            row["rank"] = int(rank)
        return {"method": "mechanism_guided", "best_candidates": top, "history": history, "num_forward_calls": int(self.forward_calls)}

    def mechanism_feature_cem(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        return self.mechanism_cem_search(context)

    def smc_search(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("SMC posterior search is reserved for a later prompt; mechanism_cem is implemented.")


LatentPosteriorDesignSearcher = HypergraphMechanismDesignSearcher
