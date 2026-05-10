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
from model_inverse import ThermalInverseDesignFlow, channel_clearance_diagnostics, repair_channel_design
from thermal_inverse_kpi import (
    DEFAULT_KPI_NAMES,
    build_target_spec_vector,
    calibrate_target_spec_to_kpi_quantiles,
    compute_steady_thermal_kpis,
    layout_spread_metrics,
    score_candidate_kpis,
)
from thermal_design_intent import (
    DEFAULT_FIELD_MAP_SHAPE,
    STRUCTURE_FEATURE_NAMES,
    STRUCTURE_INTENT_DIM,
    build_design_intent_arrays,
    build_layout_structure_maps,
    compute_design_intent_score,
    compute_layout_structure_features,
    is_design_intent_payload,
    normalize_intent_payload,
)
from train_inverse import (
    ThermalInverseDesignDataset,
    load_forward_model,
    predict_candidate_with_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a ChannelThermal inverse-design checkpoint.")
    parser.add_argument("--inverse-run", type=str, default="auto", help="Inverse run directory, checkpoint path, or auto.")
    parser.add_argument("--checkpoint-name", type=str, default="best_verified_model.pt", help="Inverse checkpoint filename or fallback selector.")
    parser.add_argument("--target", type=str, default=None, help="Target JSON path. If omitted, derive the target from --reference-split/--reference-case-index.")
    parser.add_argument("--dataset", type=str, default=None, help="Packed HDF5 override for fixed conditions/reference grid.")
    parser.add_argument("--reference-split", type=str, default="test", help="Dataset split used for fixed conditions.")
    parser.add_argument("--reference-case-index", type=int, default=0, help="Reference case index in split.")
    parser.add_argument("--n-samples", type=int, default=128, help="Number of inverse candidates to sample. Default: 128, or 8 with --quick/--smoke.")
    parser.add_argument("--n-steps", type=int, default=4, help="Rectified-flow ODE steps. Default: 16, or 4 with --quick/--smoke.")
    parser.add_argument(
        "--count-mode",
        type=str,
        default="uniform",
        choices=("uniform", "sample", "argmax"),
        help="How to choose generated module counts. 'uniform' samples within target count constraints.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    parser.add_argument("--output-dir", type=str, default=None, help="Evaluation output directory.")
    parser.add_argument("--query-batch-size", type=int, default=32768, help="Forward verifier grid query batch size.")
    parser.add_argument("--forward-run-dir", type=str, default=None, help="Override forward_model.run_dir.")
    parser.add_argument("--forward-checkpoint-name", type=str, default=None, help="Override forward_model.checkpoint_name.")
    parser.add_argument("--local-surrogate-checkpoint-path", type=str, default=None, help="Override local surrogate checkpoint path.")
    parser.add_argument("--diagnostic-teacher-mode", action="store_true", help="Reserved diagnostic flag; ranking still uses predicted/autonomous mode.")
    parser.add_argument("--quick", action="store_true", help="Use small smoke-test defaults when n-samples/n-steps are omitted.")
    parser.add_argument("--smoke", action="store_true", help="Alias for --quick.")
    parser.add_argument("--calibrate-target-to-data", action="store_true", help="Relax aggressive target bounds to training-distribution quantiles and save the resolved calibrated target.")
    parser.add_argument("--diversity-rerank-weight", type=float, default=0.15, help="MMR-style diversity reranking strength for displayed/saved top candidates.")
    parser.add_argument("--diversity-rerank-top-k", type=int, default=8, help="Number of top candidates to diversity-rerank for plots/NPZ.")
    parser.add_argument("--candidate-pool-multiplier", type=float, default=1.0, help="Sample a larger pool before selecting n-samples candidates.")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Conditional/unconditional velocity guidance scale for v2 design-intent sampling.")
    parser.add_argument("--reference-structure-strength", type=float, default=0.7, help="Strength for structure-family conditioning when deriving a target from a reference case.")
    parser.add_argument("--reference-heat-mode", type=str, default="exact", choices=("exact", "total", "uniform", "none"), help="Heat conditioning mode for reference-case evaluation.")
    parser.add_argument("--reference-anchor-mode", type=str, default="none", choices=("none", "soft", "exact"), help="Whether reference layouts are used as no anchors, soft map anchors, or exact center anchors.")
    parser.add_argument("--disable-reference-structure", action="store_true", help="Do not derive structure constraints/conditioning from the reference case.")
    parser.add_argument("--disable-reference-heat", action="store_true", help="Do not derive per-module heat conditions from the reference case.")
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
    incompatible = model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        print(f"[inverse] missing checkpoint keys ({len(missing)}): {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    if unexpected:
        print(f"[inverse] unexpected checkpoint keys ({len(unexpected)}): {unexpected[:12]}{' ...' if len(unexpected) > 12 else ''}")
    model.eval()
    return model, checkpoint


def load_target_payload(path: str | Path) -> Dict[str, Any]:
    target_path = resolve_demo_path(path)
    with target_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "kpis" not in payload and not is_design_intent_payload(payload):
        raise ValueError(f"Target JSON must contain a 'kpis' block or v2 design-intent blocks: {target_path}")
    payload["_path"] = str(target_path)
    return payload


def load_kpi_distribution_summary(checkpoint: Mapping[str, Any], inverse_path: Path) -> Dict[str, Any]:
    summary = checkpoint.get("kpi_distribution_summary")
    if isinstance(summary, Mapping) and summary:
        return dict(summary)
    path = inverse_path.parent / "kpi_distribution_summary.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _target_entry_values(entry: Any) -> List[Tuple[str, float]]:
    if not isinstance(entry, Mapping):
        try:
            return [("value", float(entry))]
        except (TypeError, ValueError):
            return []
    mode = str(entry.get("mode", "exact")).lower().strip()
    out: List[Tuple[str, float]] = []
    if mode in {"range", "between"}:
        for label, keys in (("low", ("low", "lower")), ("high", ("high", "upper"))):
            for key in keys:
                if entry.get(key) is not None:
                    out.append((label, float(entry[key])))
                    break
    elif mode in {"max", "upper", "at_most"}:
        value = entry.get("high", entry.get("upper"))
        if value is not None:
            out.append(("high", float(value)))
    elif mode in {"min", "lower", "at_least"}:
        value = entry.get("low", entry.get("lower"))
        if value is not None:
            out.append(("low", float(value)))
    else:
        value = entry.get("value", entry.get("target"))
        if value is not None:
            out.append(("value", float(value)))
    return [(label, value) for label, value in out if math.isfinite(value)]


def _percentile_estimate(value: float, stats: Mapping[str, Any]) -> Optional[float]:
    points = [
        ("min", 0.0),
        ("p01", 1.0),
        ("p05", 5.0),
        ("p10", 10.0),
        ("p25", 25.0),
        ("p50", 50.0),
        ("p75", 75.0),
        ("p90", 90.0),
        ("p95", 95.0),
        ("p99", 99.0),
        ("max", 100.0),
    ]
    pairs = [(float(stats[key]), pct) for key, pct in points if key in stats and math.isfinite(float(stats[key]))]
    if not pairs:
        return None
    pairs.sort(key=lambda pair: pair[0])
    if value <= pairs[0][0]:
        return pairs[0][1]
    if value >= pairs[-1][0]:
        return pairs[-1][1]
    for (v0, p0), (v1, p1) in zip(pairs[:-1], pairs[1:]):
        if v0 <= value <= v1:
            if abs(v1 - v0) < 1.0e-12:
                return 0.5 * (p0 + p1)
            return p0 + (value - v0) * (p1 - p0) / (v1 - v0)
    return None


def target_feasibility_report(payload: Mapping[str, Any], kpi_distribution_summary: Mapping[str, Any]) -> Dict[str, Any]:
    kpi_stats = kpi_distribution_summary.get("kpis", kpi_distribution_summary)
    entries = []
    warnings = []
    for name, raw_entry in dict(payload.get("kpis", {}) or {}).items():
        stats = kpi_stats.get(str(name), {}) if isinstance(kpi_stats, Mapping) else {}
        if not isinstance(stats, Mapping) or int(stats.get("count", 0) or 0) <= 0:
            continue
        mean = float(stats.get("mean", float("nan")))
        std = float(stats.get("std", float("nan")))
        p01 = float(stats.get("p01", float("nan")))
        p99 = float(stats.get("p99", float("nan")))
        vmin = float(stats.get("min", float("nan")))
        vmax = float(stats.get("max", float("nan")))
        for bound_name, value in _target_entry_values(raw_entry):
            z = (value - mean) / std if math.isfinite(mean) and math.isfinite(std) and abs(std) > 1.0e-12 else float("nan")
            pct = _percentile_estimate(value, stats)
            outside_min_max = (math.isfinite(vmin) and value < vmin) or (math.isfinite(vmax) and value > vmax)
            outside_p01_p99 = (math.isfinite(p01) and value < p01) or (math.isfinite(p99) and value > p99)
            entry = {
                "kpi": str(name),
                "bound": bound_name,
                "target_value": float(value),
                "train_min": vmin,
                "train_p01": p01,
                "train_p05": float(stats.get("p05", float("nan"))),
                "train_p50": float(stats.get("p50", float("nan"))),
                "train_p95": float(stats.get("p95", float("nan"))),
                "train_p99": p99,
                "train_max": vmax,
                "z_score": z,
                "percentile_estimate": pct,
                "outside_p01_p99": bool(outside_p01_p99),
                "outside_min_max": bool(outside_min_max),
            }
            entries.append(entry)
            if outside_min_max or outside_p01_p99:
                scope = "min-max" if outside_min_max else "p01-p99"
                msg = f"{name}.{bound_name}={value:.6g} is outside training {scope} (p01={p01:.6g}, p99={p99:.6g}, min={vmin:.6g}, max={vmax:.6g})."
                warnings.append(msg)
                print(f"[target-feasibility] warning: {msg}")
    return {"entries": entries, "warnings": warnings, "summary_available": bool(entries)}


def heat_condition_arrays_from_values(values: Any, *, max_num_modules: int, heat_power_scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    heat = np.asarray(values if values is not None else [], dtype=np.float32).reshape(-1)
    max_n = int(max_num_modules)
    vec_heat = np.zeros((max_n,), dtype=np.float32)
    mask = np.zeros((max_n,), dtype=np.float32)
    n = min(max_n, heat.size)
    if n > 0:
        vec_heat[:n] = heat[:n] / max(float(heat_power_scale), 1.0e-8)
        mask[:n] = 1.0
    active = heat[:n]
    total = float(np.sum(active)) if active.size else 0.0
    mean = float(np.mean(active)) if active.size else 0.0
    std = float(np.std(active)) if active.size else 0.0
    max_heat = float(np.max(active)) if active.size else 0.0
    high_frac = float(np.mean(active >= mean + std)) if active.size and std > 1.0e-8 else 0.0
    stats = np.asarray(
        [
            total / max(float(heat_power_scale) * max(float(max_n), 1.0), 1.0e-8),
            mean / max(float(heat_power_scale), 1.0e-8),
            std / max(float(heat_power_scale), 1.0e-8),
            max_heat / max(float(heat_power_scale), 1.0e-8),
            high_frac,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([vec_heat, mask, stats], axis=0).astype(np.float32), mask, stats


def apply_structure_and_heat_payload(
    spec: Dict[str, Any],
    payload: Mapping[str, Any],
    model: ThermalInverseDesignFlow,
) -> Dict[str, Any]:
    structure = payload.get("structure_constraints", {}) if isinstance(payload.get("structure_constraints"), Mapping) else {}
    reference_centers = payload.get("reference_centers")
    reference_heat = payload.get("reference_heat_powers")
    if bool(structure.get("enabled", False)) and reference_centers is not None:
        centers = np.asarray(reference_centers, dtype=np.float32).reshape(-1, 2)
        present = np.ones((centers.shape[0],), dtype=np.float32)
        struct_vec, struct_meta = compute_layout_structure_features(
            centers,
            present,
            reference_heat,
            domain_length_x=float(model.cfg.domain_length_x),
            domain_length_y=float(model.cfg.domain_length_y),
            module_radius=float(model.cfg.module_radius),
            max_num_modules=model.max_num_modules,
        )
        ref_map = None
        if str(structure.get("anchor_mode", "none")).lower().strip() in {"soft", "exact"}:
            ref_map = build_layout_structure_maps(
                centers,
                present,
                reference_heat,
                domain_length_x=float(model.cfg.domain_length_x),
                domain_length_y=float(model.cfg.domain_length_y),
                module_radius=float(model.cfg.module_radius),
                max_num_modules=model.max_num_modules,
            )[0][0]
        struct_maps, _ = build_layout_structure_maps(
            centers,
            present,
            reference_heat,
            domain_length_x=float(model.cfg.domain_length_x),
            domain_length_y=float(model.cfg.domain_length_y),
            module_radius=float(model.cfg.module_radius),
            max_num_modules=model.max_num_modules,
            reference_layout_soft_map=ref_map,
        )
        spec["structure_intent_vector"] = struct_vec
        spec["structure_intent_maps"] = struct_maps
        spec["structure_strength"] = float(structure.get("strength", 1.0))
        spec["structure_constraints"] = dict(structure)
        spec["structure_feature_metadata"] = struct_meta
    else:
        spec.setdefault("structure_intent_vector", np.zeros((STRUCTURE_INTENT_DIM,), dtype=np.float32))
        spec.setdefault("structure_intent_maps", np.zeros((5, DEFAULT_FIELD_MAP_SHAPE[1], DEFAULT_FIELD_MAP_SHAPE[0]), dtype=np.float32))
        spec.setdefault("structure_strength", 0.0)
        spec["structure_constraints"] = dict(structure)

    heat_loads = payload.get("heat_loads", {}) if isinstance(payload.get("heat_loads"), Mapping) else {}
    values = heat_loads.get("values")
    if values is not None and str(heat_loads.get("mode", "per_module")).lower().strip() != "none":
        vec, mask, stats = heat_condition_arrays_from_values(values, max_num_modules=model.max_num_modules, heat_power_scale=float(model.cfg.heat_power_scale))
        spec["heat_condition_vector"] = vec
        spec["heat_condition_mask"] = mask
        spec["heat_condition_stats"] = stats
        spec["heat_loads"] = dict(heat_loads)
    else:
        spec.setdefault("heat_condition_vector", np.zeros((model.max_num_modules * 2 + 7,), dtype=np.float32))
        spec.setdefault("heat_condition_mask", np.zeros((model.max_num_modules,), dtype=np.float32))
        spec.setdefault("heat_condition_stats", np.zeros((7,), dtype=np.float32))
        spec["heat_loads"] = dict(heat_loads)
    return spec


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
    intent_arrays = build_design_intent_arrays(
        payload,
        max_num_modules=model.max_num_modules,
        domain_length_x=float(model.cfg.domain_length_x),
        domain_length_y=float(model.cfg.domain_length_y),
        heat_power_scale=float(model.cfg.heat_power_scale),
    )
    normalized_intent = intent_arrays["intent"]
    kpis = dict(payload.get("kpis", {}))
    preference_warnings: List[str] = []
    if bool(prefs.get("avoid_wall_hotspots", False)) and "wall_hot_area_fraction" not in kpis:
        kpis["wall_hot_area_fraction"] = {"mode": "max", "high": 0.08, "weight": 0.5}
    scenario = normalized_intent.get("scenario", {}) if isinstance(normalized_intent.get("scenario"), Mapping) else {}
    geometry = normalized_intent.get("geometry_constraints", {}) if isinstance(normalized_intent.get("geometry_constraints"), Mapping) else {}
    constraints = {
        "num_modules_min": scenario.get("num_modules_min", payload.get("num_modules_min", payload.get("num_cylinders_min"))),
        "num_modules_max": scenario.get("num_modules_max", payload.get("num_modules_max", payload.get("num_cylinders_max"))),
        "min_center_distance": geometry.get("min_center_distance", payload.get("min_center_distance", prefs.get("min_center_distance"))),
        "wall_clearance": geometry.get("wall_clearance", payload.get("wall_clearance", prefs.get("wall_clearance"))),
        "inlet_clearance": geometry.get("inlet_clearance", payload.get("inlet_clearance", prefs.get("inlet_clearance"))),
        "outlet_clearance": geometry.get("outlet_clearance", payload.get("outlet_clearance", prefs.get("outlet_clearance"))),
        "heat_power_total": scenario.get("heat_power_total", payload.get("heat_power_total")),
    }
    vector = build_target_spec_vector(
        kpi_targets=kpis,
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
    thermal_limits_payload = payload.get("temperature_limits", target_cfg.get("temperature_limits"))
    if thermal_limits_payload is None and isinstance(normalized_intent.get("thermal_limits"), Mapping):
        thermal_limits_payload = {
            "wall_hot_delta_T": normalized_intent["thermal_limits"].get("wall_hot_delta_T"),
            "outlet_hot_delta_T": normalized_intent["thermal_limits"].get("outlet_hot_delta_T"),
        }
    spec = {
        "name": payload.get("name", "inverse_target"),
        "vector": vector,
        "kpi_names": list(kpi_names),
        "kpi_targets": kpis,
        "kpi_stats": kpi_stats,
        "constraints": constraints,
        "preferences": dict(prefs),
        "preference_warnings": preference_warnings,
        "temperature_limits": thermal_limits_payload,
        "target_payload": dict(payload),
        "design_intent": normalized_intent,
        "is_design_intent": bool(is_design_intent_payload(payload)),
        "design_intent_vector": intent_arrays["design_intent_vector"],
        "objective_weight_vector": intent_arrays["objective_weight_vector"],
        "field_intent_maps": intent_arrays["field_intent_maps"],
        "x_bounds": intent_arrays["x_bounds"],
        "y_bounds": intent_arrays["y_bounds"],
    }
    return apply_structure_and_heat_payload(spec, payload, model)


def target_payload_from_reference_record(
    record: Any,
    model: ThermalInverseDesignFlow,
    *,
    structure_strength: float = 0.7,
    heat_mode: str = "exact",
    anchor_mode: str = "none",
    enable_structure: bool = True,
    enable_heat: bool = True,
) -> Dict[str, Any]:
    centers = np.asarray(record.module_centers[record.module_present > 0.5], dtype=np.float32).reshape(-1, 2)
    heat = np.asarray(record.heat_powers[record.module_present > 0.5], dtype=np.float32).reshape(-1)
    struct_vec, _ = compute_layout_structure_features(
        centers,
        np.ones((centers.shape[0],), dtype=np.float32),
        heat,
        domain_length_x=float(record.domain_length_x),
        domain_length_y=float(record.domain_length_y),
        module_radius=float(record.module_radius),
        max_num_modules=model.max_num_modules,
    )
    heat_mode_norm = "none" if not enable_heat else str(heat_mode).lower().strip()
    if heat_mode_norm == "exact":
        heat_loads = {"mode": "per_module", "values": heat.tolist(), "sort_mode": "heat_desc_then_xy"}
    elif heat_mode_norm == "uniform":
        heat_loads = {"mode": "uniform", "total": float(np.sum(heat)), "values": np.full((centers.shape[0],), float(np.mean(heat)) if heat.size else 0.0, dtype=np.float32).tolist()}
    elif heat_mode_norm == "total":
        heat_loads = {"mode": "total_only", "total": float(np.sum(heat)), "values": None}
    else:
        heat_loads = {"mode": "none", "values": None}
    return {
        "name": f"reference_{record.split}_{record.case_id}",
        "scenario": {
            "num_modules_min": int(record.true_count),
            "num_modules_max": int(record.true_count),
            "heat_load_policy": getattr(model.cfg, "heat_load_policy", "preserve_total_heat"),
        },
        "geometry_constraints": {
            "min_center_distance": float(model.cfg.min_center_distance),
            "wall_clearance": float(model.cfg.wall_clearance),
            "inlet_clearance": float(model.cfg.inlet_clearance),
            "outlet_clearance": float(model.cfg.outlet_clearance),
            "x_span": [0.0, float(record.domain_length_x)],
            "y_span": [0.0, float(record.domain_length_y)],
            "keepout_boxes": [],
            "protected_boxes": [],
        },
        "thermal_limits": {
            "solid_temperature_max": float(record.kpi_dict.get("max_solid_temperature", 0.0)),
            "module_temperature_spread_max": float(record.kpi_dict.get("module_peak_temperature_spread", 0.0)),
            "pressure_drop_max": float(record.kpi_dict.get("pressure_drop", 0.0)),
            "wall_hot_delta_T": None,
            "outlet_hot_delta_T": None,
        },
        "objective_weights": {
            "safety": 1.0,
            "uniformity": 1.0,
            "pressure": 0.6,
            "outlet_mixing": 0.6,
            "wall_protection": 0.4,
            "plume_avoidance": 0.6,
            "coverage": 0.2,
        },
        "field_preferences": {
            "avoid_downstream_hot_plumes": True,
            "protect_wall_band": True,
            "protect_outlet_uniformity": True,
        },
        "structure_constraints": {
            "enabled": bool(enable_structure),
            "strength": float(structure_strength) if enable_structure else 0.0,
            "x_coverage_min": float(struct_vec[5] * record.domain_length_x),
            "y_coverage_min": float(struct_vec[6] * record.domain_length_y),
            "mean_pair_distance_min": float(struct_vec[11] * max(record.domain_length_x, record.domain_length_y)),
            "centroid": [float(np.mean(centers[:, 0])) if centers.size else 0.0, float(np.mean(centers[:, 1])) if centers.size else 0.0],
            "centroid_tolerance": [2.0, 1.0],
            "x_bin_occupancy": None,
            "y_bin_occupancy": None,
            "pair_distance_hist": None,
            "avoid_vertical_stack": False,
            "match_reference_layout_features": True,
            "anchor_mode": str(anchor_mode),
        },
        "heat_loads": heat_loads,
        "reference_centers": centers.tolist(),
        "reference_heat_powers": heat.tolist(),
        "kpis": {
            name: {"mode": "exact", "value": float(record.kpi_dict[name]), "weight": 1.0}
            for name in DEFAULT_KPI_NAMES
            if name in record.kpi_dict and name not in set(record.kpi_dict.get("unavailable_kpis", []))
        },
        "_source": "reference_case",
        "_reference_case_id": record.case_id,
        "_reference_split": record.split,
    }


def target_spec_from_reference_item(
    record: Any,
    item: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    model: ThermalInverseDesignFlow,
    *,
    structure_strength: float = 0.7,
    heat_mode: str = "exact",
    anchor_mode: str = "none",
    enable_structure: bool = True,
    enable_heat: bool = True,
) -> Dict[str, Any]:
    payload = target_payload_from_reference_record(
        record,
        model,
        structure_strength=structure_strength,
        heat_mode=heat_mode,
        anchor_mode=anchor_mode,
        enable_structure=enable_structure,
        enable_heat=enable_heat,
    )
    spec = target_spec_from_payload(payload, checkpoint, model)
    spec["name"] = payload["name"]
    spec["vector"] = np.asarray(item["target_spec_vector"], dtype=np.float32)
    spec["kpi_targets"] = dict(item.get("target_kpi_targets", spec.get("kpi_targets", {})))
    spec["design_intent_vector"] = np.asarray(item.get("design_intent_vector", spec["design_intent_vector"]), dtype=np.float32)
    spec["objective_weight_vector"] = np.asarray(item.get("objective_weight_vector", spec["objective_weight_vector"]), dtype=np.float32)
    spec["field_intent_maps"] = np.asarray(item.get("field_intent_maps", spec["field_intent_maps"]), dtype=np.float32)
    if enable_structure:
        spec["structure_strength"] = float(structure_strength)
    if enable_heat and str(heat_mode).lower().strip() == "exact":
        spec["heat_condition_vector"] = np.asarray(spec.get("heat_condition_vector", item.get("heat_condition_vector")), dtype=np.float32)
        spec["heat_condition_mask"] = np.asarray(spec.get("heat_condition_mask", item.get("heat_condition_mask")), dtype=np.float32)
        spec["heat_condition_stats"] = np.asarray(spec.get("heat_condition_stats", item.get("heat_condition_stats")), dtype=np.float32)
    spec["reference_case_id"] = record.case_id
    spec["reference_ground_truth"] = {
        "centers": np.asarray(record.module_centers[record.module_present > 0.5], dtype=np.float32),
        "count": int(record.true_count),
        "verified_kpis": dict(record.kpi_dict),
    }
    return spec


def _span_bounds(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    lo, hi = sorted((float(arr[0]), float(arr[1])))
    return lo, hi


def preference_bounds(target_spec: Mapping[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], List[str]]:
    prefs = target_spec.get("preferences", {})
    warnings: List[str] = []
    if not isinstance(prefs, Mapping):
        return None, None, warnings
    x_bounds = _span_bounds(prefs.get("x_span"))
    y_bounds = _span_bounds(prefs.get("y_span"))
    supported = {
        "x_span",
        "y_span",
        "avoid_wall_hotspots",
        "min_center_distance",
        "wall_clearance",
        "inlet_clearance",
        "outlet_clearance",
        "min_x_coverage",
        "min_y_coverage",
        "min_bbox_area",
        "min_mean_pair_distance",
        "prefer_count",
        "description",
    }
    for key in prefs:
        if key not in supported:
            warning = f"Unsupported preference {key!r} was parsed but not applied."
            print(f"[warning] {warning}")
            warnings.append(warning)
    return x_bounds, y_bounds, warnings


def apply_preferences_to_candidate(
    candidate: Mapping[str, Any],
    record: Any,
    target_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    x_bounds, y_bounds, warnings = preference_bounds(target_spec)
    if x_bounds is None and y_bounds is None:
        out = dict(candidate)
        out["preference_warnings"] = warnings
        return out
    centers = np.asarray(candidate.get("centers", []), dtype=np.float32).reshape(-1, 2).copy()
    if centers.size:
        if x_bounds is not None:
            centers[:, 0] = np.clip(centers[:, 0], x_bounds[0], x_bounds[1])
        if y_bounds is not None:
            centers[:, 1] = np.clip(centers[:, 1], y_bounds[0], y_bounds[1])
    constraints = target_spec.get("constraints", {}) if isinstance(target_spec.get("constraints"), Mapping) else {}
    repaired, validity = repair_channel_design(
        centers,
        count=int(candidate.get("count", centers.shape[0])),
        domain_length_x=float(record.domain_length_x),
        domain_length_y=float(record.domain_length_y),
        module_radius=float(record.module_radius),
        min_center_distance=float(constraints.get("min_center_distance") or 1.1),
        max_num_modules=int(len(candidate.get("mask", [])) or centers.shape[0] or 1),
        min_count=int(constraints.get("num_modules_min") or 0),
        wall_clearance=float(constraints.get("wall_clearance") or 0.0),
        inlet_clearance=float(constraints.get("inlet_clearance") or 0.0),
        outlet_clearance=float(constraints.get("outlet_clearance") or 0.0),
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    )
    out = dict(candidate)
    out["centers"] = repaired
    out["count"] = int(repaired.shape[0])
    out["validity"] = validity
    out["preference_warnings"] = warnings
    return out


def _candidate_kpi_payload(
    record: Any,
    prediction: Mapping[str, Any],
    candidate: Mapping[str, Any],
    model: ThermalInverseDesignFlow,
    target_spec: Optional[Mapping[str, Any]] = None,
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
        interface_condition=prediction.get("pred_port_condition"),
        domain={"domain_length_x": record.domain_length_x, "domain_length_y": record.domain_length_y, "module_radius": record.module_radius},
        material_params=record.material_params,
        temperature_limits=target_spec.get("temperature_limits") if isinstance(target_spec, Mapping) and isinstance(target_spec.get("temperature_limits"), Mapping) else None,
    )
    centers = np.asarray(candidate.get("centers", []), dtype=np.float32).reshape(-1, 2)
    kpis.update(channel_clearance_diagnostics(centers, domain_length_x=record.domain_length_x, domain_length_y=record.domain_length_y, module_radius=record.module_radius))
    kpis.update(layout_spread_metrics(centers, num_modules=int(candidate.get("count", centers.shape[0]))))
    kpis["num_modules"] = int(candidate.get("count", centers.shape[0]))
    heat_used = np.asarray(prediction.get("heat_powers", record.heat_powers), dtype=np.float32).reshape(-1)
    kpis["heat_power_total"] = float(np.sum(heat_used[: kpis["num_modules"]])) if heat_used.size else 0.0
    kpis["valid"] = bool(candidate.get("validity", {}).get("valid", False))
    return kpis


def compute_structure_match_score(candidate: Mapping[str, Any], target_spec: Mapping[str, Any], record: Any) -> Dict[str, Any]:
    centers = np.asarray(candidate.get("centers", []), dtype=np.float32).reshape(-1, 2)
    heat = np.asarray(candidate.get("heat_powers", []), dtype=np.float32).reshape(-1) if candidate.get("heat_powers") is not None else None
    present = np.ones((centers.shape[0],), dtype=np.float32)
    vec, meta = compute_layout_structure_features(
        centers,
        present,
        heat,
        domain_length_x=float(record.domain_length_x),
        domain_length_y=float(record.domain_length_y),
        module_radius=float(record.module_radius),
        max_num_modules=int(getattr(record, "max_num_modules", centers.shape[0] or 12)),
    )
    target_vec = np.asarray(target_spec.get("structure_intent_vector", []), dtype=np.float32).reshape(-1)
    constraints = target_spec.get("structure_constraints", {}) if isinstance(target_spec.get("structure_constraints"), Mapping) else {}
    descriptor_error = float(np.mean(np.abs(vec[: min(vec.size, target_vec.size)] - target_vec[: min(vec.size, target_vec.size)]))) if target_vec.size else 0.0
    centroid_error = 0.0
    if constraints.get("centroid") is not None and centers.size:
        centroid = np.mean(centers, axis=0)
        target_centroid = np.asarray(constraints.get("centroid"), dtype=np.float32).reshape(-1)[:2]
        tol = np.asarray(constraints.get("centroid_tolerance", [record.domain_length_x, record.domain_length_y]), dtype=np.float32).reshape(-1)[:2]
        centroid_error = float(np.mean(np.abs(centroid - target_centroid) / np.maximum(tol, 1.0e-8)))
    coverage_error = 0.0
    if constraints.get("x_coverage_min") is not None:
        coverage_error += max(0.0, float(constraints["x_coverage_min"]) - float(vec[5] * record.domain_length_x)) / max(float(constraints["x_coverage_min"]), 1.0)
    if constraints.get("y_coverage_min") is not None:
        coverage_error += max(0.0, float(constraints["y_coverage_min"]) - float(vec[6] * record.domain_length_y)) / max(float(constraints["y_coverage_min"]), 1.0)
    if constraints.get("mean_pair_distance_min") is not None:
        coverage_error += max(0.0, float(constraints["mean_pair_distance_min"]) - float(vec[11] * max(record.domain_length_x, record.domain_length_y))) / max(float(constraints["mean_pair_distance_min"]), 1.0)
    hist_error = 0.0
    if target_vec.size >= STRUCTURE_INTENT_DIM:
        hist_error = float(np.mean(np.abs(vec[23:39] - target_vec[23:39])))
    heat_weighted_centroid_error = float(np.mean(np.abs(vec[3:5] - target_vec[3:5]))) if target_vec.size >= 5 else 0.0
    high_power_placement_error = 0.0
    ref_centers = np.asarray(target_spec.get("reference_ground_truth", {}).get("centers", target_spec.get("target_payload", {}).get("reference_centers", [])), dtype=np.float32).reshape(-1, 2)
    ref_heat = np.asarray(target_spec.get("target_payload", {}).get("reference_heat_powers", []), dtype=np.float32).reshape(-1)
    if heat is not None and heat.size and centers.size and ref_centers.size and ref_heat.size:
        n = min(centers.shape[0], heat.size)
        m = min(ref_centers.shape[0], ref_heat.size)
        cand_cut = float(np.percentile(heat[:n], 75.0))
        ref_cut = float(np.percentile(ref_heat[:m], 75.0))
        cand_high = centers[:n][heat[:n] >= cand_cut]
        ref_high = ref_centers[:m][ref_heat[:m] >= ref_cut]
        if cand_high.size and ref_high.size:
            cand_hc = np.asarray([np.mean(cand_high[:, 0]) / record.domain_length_x, np.mean(cand_high[:, 1]) / record.domain_length_y])
            ref_hc = np.asarray([np.mean(ref_high[:, 0]) / record.domain_length_x, np.mean(ref_high[:, 1]) / record.domain_length_y])
            high_power_placement_error = float(np.mean(np.abs(cand_hc - ref_hc)))
    vertical_stack_penalty = float(vec[-1]) if bool(constraints.get("avoid_vertical_stack", False)) else 0.0
    total = descriptor_error + coverage_error + centroid_error + hist_error + heat_weighted_centroid_error + high_power_placement_error + vertical_stack_penalty
    return {
        "structure_match_score": float(total),
        "descriptor_error": descriptor_error,
        "coverage_error": coverage_error,
        "centroid_error": centroid_error,
        "histogram_l1_error": hist_error,
        "heat_weighted_centroid_error": heat_weighted_centroid_error,
        "high_power_placement_error": high_power_placement_error,
        "vertical_stack_penalty": vertical_stack_penalty,
        "features": vec,
        "feature_metadata": meta,
    }


def write_candidates_csv(candidates: Sequence[Mapping[str, Any]], path: Path) -> None:
    keys = [
        "rank",
        "raw_score_rank",
        "diversity_rank",
        "sample_index",
        "count",
        "valid",
        "total_score",
        "kpi_score",
        "constraint_penalty",
        "spread_preference_penalty",
        "min_center_distance",
        "wall_clearance",
        "inlet_clearance",
        "outlet_clearance",
        "x_coverage",
        "y_coverage",
        "bbox_area",
        "mean_pair_distance",
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


def _draw_layout_with_style(ax: Any, record: Any, centers: np.ndarray, *, color: str, label: str, linestyle: str = "-", linewidth: float = 1.4) -> None:
    first = True
    for cx, cy in np.asarray(centers, dtype=np.float32).reshape(-1, 2):
        ax.add_patch(
            plt.Circle(
                (float(cx), float(cy)),
                float(record.module_radius),
                fill=False,
                lw=linewidth,
                color=color,
                linestyle=linestyle,
                label=label if first else None,
            )
        )
        first = False


def plot_reference_layout_comparison(candidate: Mapping[str, Any], record: Any, out_path: Path) -> None:
    gt_centers = np.asarray(record.module_centers[record.module_present > 0.5], dtype=np.float32).reshape(-1, 2)
    pred_centers = np.asarray(candidate.get("centers", []), dtype=np.float32).reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(8.2, 3.2), constrained_layout=True)
    ax.set_xlim(0.0, float(record.domain_length_x))
    ax.set_ylim(0.0, float(record.domain_length_y))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _draw_layout_with_style(ax, record, gt_centers, color="#222222", label="reference layout", linestyle="--", linewidth=1.7)
    _draw_layout_with_style(ax, record, pred_centers, color="#1f77b4", label="generated best", linestyle="-", linewidth=1.5)
    ax.set_title(f"Generated vs reference layout, case {record.case_id}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.18)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


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
        ax.text(0.01, 0.98, "circle = generated module footprint", transform=ax.transAxes, va="top", ha="left", fontsize=7, bbox={"facecolor": "white", "alpha": 0.65, "pad": 2})
    for ax in axes_arr[n:]:
        ax.axis("off")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_temperature_field(candidate: Mapping[str, Any], record: Any, out_path: Path) -> None:
    pred = candidate["prediction"]["pred_field_grid"]
    names = list(CHANNEL_ORDER)
    wanted = [name for name in ("u", "v", "p", "temperature") if name in names]
    if not wanted:
        wanted = [names[min(pred.shape[-1] - 1, 0)] if names else "field"]
    cols = min(2, len(wanted))
    rows = int(math.ceil(len(wanted) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.0 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    centers = np.asarray(candidate["centers"], dtype=np.float32).reshape(-1, 2)
    for ax, name in zip(axes_arr, wanted):
        idx = names.index(name) if name in names else pred.shape[-1] - 1
        cmap = "inferno" if name == "temperature" else "viridis"
        im = ax.imshow(pred[..., idx], origin="lower", extent=_extent(record), cmap=cmap, aspect="equal")
        _draw_layout(ax, record, centers, title=f"{name} field")
        ax.text(0.01, 0.98, "blue circles: generated modules", transform=ax.transAxes, va="top", ha="left", fontsize=8, color="white", bbox={"facecolor": "black", "alpha": 0.35, "pad": 2})
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    for ax in axes_arr[len(wanted):]:
        ax.axis("off")
    fig.suptitle(f"Best generated global fields, score={candidate['total_score']:.4f}", fontsize=11)
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
    centers = np.asarray(candidate["centers"], dtype=np.float32).reshape(-1, 2)
    count = min(int(candidate.get("count", centers.shape[0])), internal.shape[0], 8)
    if count <= 0:
        return
    cols = min(4, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 2.5 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for m, ax in enumerate(axes_arr[:count]):
        local = _local_disk_image(internal[m, :, 0] if internal.ndim == 4 else internal[m], record.module_internal_mask)
        im = ax.imshow(local, origin="lower", cmap="inferno")
        ax.set_title(f"M{m} internal T")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    for ax in axes_arr[count:]:
        ax.axis("off")
    fig.suptitle("Generated module-internal temperature disks", fontsize=11)
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
        axes[row, 0].plot(theta, interface[row, :, 0], label="predicted surface T")
        axes[row, 0].set_title(f"M{row} surface temperature")
        axes[row, 1].plot(theta, interface[row, :, 1], label="predicted normal heat flux")
        axes[row, 1].set_title(f"M{row} interface heat flux")
        for col in range(2):
            axes[row, col].set_xlabel("theta")
            axes[row, col].legend(fontsize=8)
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
    ax.legend(["target bound/value", "verified"], fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def normalized_target_violation_rows(candidate: Mapping[str, Any], target_spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    targets = dict(target_spec.get("kpi_targets", {}) or {})
    if not targets and isinstance(target_spec.get("design_intent"), Mapping):
        thermal = target_spec["design_intent"].get("thermal_limits", {})
        if isinstance(thermal, Mapping):
            mapped = {
                "max_solid_temperature": ("solid_temperature_max", "max"),
                "module_peak_temperature_spread": ("module_temperature_spread_max", "max"),
                "pressure_drop": ("pressure_drop_max", "max"),
            }
            for kpi_name, (limit_name, mode) in mapped.items():
                if thermal.get(limit_name) is not None:
                    targets[kpi_name] = {"mode": mode, "high": thermal.get(limit_name)}
    names = [name for name in target_spec.get("kpi_names", []) if name in targets]
    stats = target_spec.get("kpi_stats", {})
    rows: List[Dict[str, Any]] = []
    for name in names:
        raw_entry = targets[name]
        entry = raw_entry if isinstance(raw_entry, Mapping) else {"mode": "exact", "value": raw_entry}
        mode = str(entry.get("mode", "exact")).lower().strip()
        verified = float(candidate["verified_kpis"].get(name, float("nan")))
        stat_entry = stats.get(name, {}) if isinstance(stats, Mapping) else {}
        stat_std = float(stat_entry.get("std", 1.0)) if isinstance(stat_entry, Mapping) else 1.0
        bound_value = float("nan")
        bound_label = "value"
        violation = 0.0
        if not math.isfinite(verified):
            violation = float("nan")
        elif mode in {"range", "between"}:
            lo = entry.get("low", entry.get("lower"))
            hi = entry.get("high", entry.get("upper"))
            lo_f = float(lo) if lo is not None else float("-inf")
            hi_f = float(hi) if hi is not None else float("inf")
            if verified < lo_f:
                bound_value, bound_label = lo_f, "low"
                violation = lo_f - verified
            elif verified > hi_f:
                bound_value, bound_label = hi_f, "high"
                violation = verified - hi_f
            else:
                bound_value, bound_label, violation = 0.5 * (lo_f + hi_f), "range", 0.0
        elif mode in {"max", "upper", "at_most"}:
            bound_value = float(entry.get("high", entry.get("upper", float("nan"))))
            bound_label = "high"
            violation = max(0.0, verified - bound_value)
        elif mode in {"min", "lower", "at_least"}:
            bound_value = float(entry.get("low", entry.get("lower", float("nan"))))
            bound_label = "low"
            violation = max(0.0, bound_value - verified)
        else:
            bound_value = float(entry.get("value", entry.get("target", float("nan"))))
            bound_label = "value"
            violation = abs(verified - bound_value)
        scale = max(abs(bound_value) if math.isfinite(bound_value) else 0.0, abs(stat_std) if math.isfinite(stat_std) else 0.0, 1.0)
        rows.append(
            {
                "kpi": str(name),
                "mode": mode,
                "verified": verified,
                "target_bound_label": bound_label,
                "target_bound": bound_value,
                "normalized_violation": float(max(violation, 0.0) / scale) if math.isfinite(violation) else float("nan"),
            }
        )
    return rows


def write_normalized_violation_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    keys = ["kpi", "mode", "verified", "target_bound_label", "target_bound", "normalized_violation"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def plot_target_vs_verified_normalized(candidate: Mapping[str, Any], target_spec: Mapping[str, Any], out_path: Path) -> List[Dict[str, Any]]:
    rows = normalized_target_violation_rows(candidate, target_spec)
    if not rows:
        return rows
    names = [str(row["kpi"]) for row in rows]
    values = [float(row["normalized_violation"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.45 * len(names)), 4.2), constrained_layout=True)
    x = np.arange(len(names))
    ax.bar(x, values, color="#d55e00", alpha=0.82)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("normalized violation")
    ax.set_title("Target KPI violations, normalized")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return rows


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


def write_all_kpis_verified(candidate: Mapping[str, Any], kpi_names: Sequence[str], path: Path) -> None:
    kpis = candidate.get("verified_kpis", {})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kpi", "verified_value"])
        writer.writeheader()
        for name in kpi_names:
            writer.writerow({"kpi": name, "verified_value": kpis.get(name, "")})


def write_design_intent_breakdown(candidate: Mapping[str, Any], out_csv: Path, out_png: Path) -> None:
    detail = candidate.get("design_intent_score_detail", {})
    rows = []
    for group, payload in (("objective", detail.get("components", {})), ("field", detail.get("field_penalties", {}))):
        if isinstance(payload, Mapping):
            for name, value in payload.items():
                rows.append({"group": group, "component": name, "value": float(value)})
    rows.extend(
        [
            {"group": "total", "component": "hard_feasibility_penalty", "value": float(detail.get("hard_feasibility_penalty", 0.0))},
            {"group": "total", "component": "objective_score", "value": float(detail.get("objective_score", 0.0))},
            {"group": "total", "component": "field_penalty", "value": float(detail.get("field_penalty", 0.0))},
        ]
    )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "component", "value"])
        writer.writeheader()
        writer.writerows(rows)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    labels = [row["component"] for row in rows]
    values = [float(row["value"]) for row in rows]
    ax.bar(np.arange(len(rows)), values, color="#4c78a8")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("penalty / score")
    ax.set_title("Design intent score breakdown")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_candidate_pareto_scatter(candidates: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    if not candidates:
        return
    pressure = [float(c.get("verified_kpis", {}).get("pressure_drop", np.nan)) for c in candidates]
    tmax = [float(c.get("verified_kpis", {}).get("max_solid_temperature", np.nan)) for c in candidates]
    color = [float(c.get("verified_kpis", {}).get("module_peak_temperature_spread", np.nan)) for c in candidates]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    sc = ax.scatter(pressure, tmax, c=color, cmap="magma", s=35)
    ax.set_xlabel("pressure_drop")
    ax.set_ylabel("max_solid_temperature")
    ax.set_title("Candidate Pareto view")
    fig.colorbar(sc, ax=ax, label="module temperature spread")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_field_risk_overlay(candidate: Mapping[str, Any], target_spec: Mapping[str, Any], record: Any, out_path: Path) -> None:
    pred = candidate["prediction"]["pred_field_grid"]
    names = list(CHANNEL_ORDER)
    t_idx = names.index("temperature") if "temperature" in names else pred.shape[-1] - 1
    maps = np.asarray(target_spec.get("field_intent_maps", []), dtype=np.float32)
    risk = None
    if maps.ndim == 3 and maps.shape[0] >= 4:
        risk = np.maximum.reduce([maps[0], maps[1], maps[2], maps[3]])
    fig, ax = plt.subplots(figsize=(9.5, 3.4), constrained_layout=True)
    im = ax.imshow(pred[..., t_idx], origin="lower", extent=_extent(record), cmap="inferno", aspect="equal")
    if risk is not None and np.any(risk > 0):
        ax.imshow(risk, origin="lower", extent=_extent(record), cmap="Blues", alpha=0.35, aspect="equal")
    _draw_layout(ax, record, candidate["centers"], title="Best layout thermal field + intent risk")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_reference_field_comparison(candidate: Mapping[str, Any], record: Any, out_path: Path) -> None:
    pred = np.asarray(candidate["prediction"]["pred_field_grid"], dtype=np.float32)
    ref = np.asarray(record.steady_field, dtype=np.float32)
    names = list(CHANNEL_ORDER)
    t_idx = names.index("temperature") if "temperature" in names else pred.shape[-1] - 1
    p_idx = names.index("p") if "p" in names else min(2, pred.shape[-1] - 1)
    panels = [
        ("reference temperature", ref[..., t_idx], "inferno"),
        ("generated temperature", pred[..., t_idx], "inferno"),
        ("temperature error", pred[..., t_idx] - ref[..., t_idx], "coolwarm"),
        ("pressure error", pred[..., p_idx] - ref[..., p_idx], "coolwarm"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.0), constrained_layout=True)
    for ax, (title, values, cmap) in zip(np.asarray(axes).reshape(-1), panels):
        im = ax.imshow(values, origin="lower", extent=_extent(record), cmap=cmap, aspect="equal")
        _draw_layout(ax, record, candidate["centers"], title=title)
        if title.startswith("reference"):
            _draw_layout_with_style(ax, record, record.module_centers[record.module_present > 0.5], color="#ffffff", label="reference modules", linestyle="--")
            ax.legend(loc="upper right", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"Generated-vs-reference fields for case {record.case_id}", fontsize=11)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_reference_kpi_comparison(candidate: Mapping[str, Any], record: Any, kpi_names: Sequence[str], out_path: Path) -> None:
    selected = [
        "max_solid_temperature",
        "module_peak_temperature_spread",
        "pressure_drop",
        "outlet_temperature_nonuniformity",
        "wall_hot_area_fraction",
        "thermal_plume_length",
        "downstream_reheat_index",
    ]
    names = [name for name in selected if name in kpi_names and name in candidate.get("verified_kpis", {}) and name in record.kpi_dict]
    if not names:
        names = [name for name in kpi_names if name in candidate.get("verified_kpis", {}) and name in record.kpi_dict][:10]
    if not names:
        return
    ref_vals = [float(record.kpi_dict.get(name, np.nan)) for name in names]
    gen_vals = [float(candidate["verified_kpis"].get(name, np.nan)) for name in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(names)), 4.2), constrained_layout=True)
    ax.bar(x - 0.18, ref_vals, width=0.36, label="reference case", color="#666666")
    ax.bar(x + 0.18, gen_vals, width=0.36, label="generated best", color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("KPI value")
    ax.set_title("Reference vs generated verified KPIs")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_reference_comparison_csv(candidate: Mapping[str, Any], record: Any, kpi_names: Sequence[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kpi", "reference_value", "generated_value", "absolute_error"])
        writer.writeheader()
        for name in kpi_names:
            if name not in record.kpi_dict or name not in candidate.get("verified_kpis", {}):
                continue
            ref = float(record.kpi_dict.get(name, np.nan))
            gen = float(candidate["verified_kpis"].get(name, np.nan))
            writer.writerow({"kpi": name, "reference_value": ref, "generated_value": gen, "absolute_error": abs(gen - ref) if math.isfinite(gen) and math.isfinite(ref) else ""})


def write_structure_outputs(candidates: Sequence[Mapping[str, Any]], target_spec: Mapping[str, Any], record: Any, out_dirs: Mapping[str, Path]) -> None:
    rows = []
    feature_rows = []
    target_vec = np.asarray(target_spec.get("structure_intent_vector", []), dtype=np.float32).reshape(-1)
    for row in candidates:
        detail = row.get("structure_score_detail", {})
        rows.append(
            {
                "rank": row.get("rank", ""),
                "sample_index": row.get("sample_index", ""),
                "structure_match_score": detail.get("structure_match_score", ""),
                "descriptor_error": detail.get("descriptor_error", ""),
                "coverage_error": detail.get("coverage_error", ""),
                "centroid_error": detail.get("centroid_error", ""),
                "histogram_l1_error": detail.get("histogram_l1_error", ""),
                "heat_weighted_centroid_error": detail.get("heat_weighted_centroid_error", ""),
                "high_power_placement_error": detail.get("high_power_placement_error", ""),
            }
        )
        features = np.asarray(detail.get("features", []), dtype=np.float32).reshape(-1)
        feature_row = {"rank": row.get("rank", ""), "sample_index": row.get("sample_index", "")}
        for idx, name in enumerate(STRUCTURE_FEATURE_NAMES):
            feature_row[name] = float(features[idx]) if idx < features.size else ""
            feature_row[f"target_{name}"] = float(target_vec[idx]) if idx < target_vec.size else ""
        feature_rows.append(feature_row)
    if rows:
        with (out_dirs["data"] / "layout_structure_comparison.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (out_dirs["data"] / "candidate_structure_features.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
            writer.writeheader()
            writer.writerows(feature_rows)
    heat_rows = []
    for row in candidates:
        heat = np.asarray(row.get("heat_powers", row.get("prediction", {}).get("heat_powers", [])), dtype=np.float32).reshape(-1)
        for idx, value in enumerate(heat[: int(row.get("count", heat.size))]):
            heat_rows.append({"rank": row.get("rank", ""), "sample_index": row.get("sample_index", ""), "slot_id": idx, "heat_power": float(value), "heat_source": row.get("prediction", {}).get("heat_source", "")})
    with (out_dirs["data"] / "heat_assignment_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "sample_index", "slot_id", "heat_power", "heat_source"])
        writer.writeheader()
        writer.writerows(heat_rows)
    best = candidates[0] if candidates else None
    write_json(out_dirs["data"] / "layout_structure_summary.json", json_safe({"target_feature_names": list(STRUCTURE_FEATURE_NAMES), "target_features": target_vec, "best": best.get("structure_score_detail", {}) if best else {}}))


def plot_structure_diagnostics(best: Mapping[str, Any], target_spec: Mapping[str, Any], record: Any, out_dirs: Mapping[str, Path]) -> None:
    if not best:
        return
    target_vec = np.asarray(target_spec.get("structure_intent_vector", []), dtype=np.float32).reshape(-1)
    cand_vec = np.asarray(best.get("structure_score_detail", {}).get("features", []), dtype=np.float32).reshape(-1)
    if target_vec.size and cand_vec.size:
        names = list(STRUCTURE_FEATURE_NAMES[:23]) + ["occupancy_entropy", "heat_density_entropy", "anisotropy_score"]
        idxs = list(range(23)) + [39, 40, 41]
        idxs = [idx for idx in idxs if idx < target_vec.size and idx < cand_vec.size]
        fig, ax = plt.subplots(figsize=(max(8.0, 0.28 * len(idxs)), 4.0), constrained_layout=True)
        x = np.arange(len(idxs))
        ax.bar(x - 0.18, target_vec[idxs], width=0.36, label="target/reference", color="#666666")
        ax.bar(x + 0.18, cand_vec[idxs], width=0.36, label="generated", color="#4c78a8")
        ax.set_xticks(x)
        ax.set_xticklabels([names[i] if i < len(names) else STRUCTURE_FEATURE_NAMES[idxs[i]] for i in range(len(idxs))], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("normalized descriptor")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        fig.savefig(out_dirs["diagnostics"] / "generated_vs_reference_layout_descriptor_bars.png", dpi=170)
        fig.savefig(out_dirs["diagnostics"] / "layout_structure_comparison.png", dpi=170)
        plt.close(fig)
    ref_centers = np.asarray(target_spec.get("reference_ground_truth", {}).get("centers", []), dtype=np.float32).reshape(-1, 2)
    gen_centers = np.asarray(best.get("centers", []), dtype=np.float32).reshape(-1, 2)
    if ref_centers.size and gen_centers.size:
        ref_maps = build_layout_structure_maps(ref_centers, np.ones((ref_centers.shape[0],), dtype=np.float32), None, domain_length_x=record.domain_length_x, domain_length_y=record.domain_length_y, module_radius=record.module_radius)[0]
        gen_maps = build_layout_structure_maps(gen_centers, np.ones((gen_centers.shape[0],), dtype=np.float32), best.get("heat_powers"), domain_length_x=record.domain_length_x, domain_length_y=record.domain_length_y, module_radius=record.module_radius)[0]
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), constrained_layout=True)
        axes[0].imshow(ref_maps[0], origin="lower", extent=_extent(record), cmap="Blues", aspect="equal")
        _draw_layout(axes[0], record, ref_centers, title="reference occupancy")
        axes[1].imshow(gen_maps[0], origin="lower", extent=_extent(record), cmap="Blues", aspect="equal")
        _draw_layout(axes[1], record, gen_centers, title="generated occupancy")
        fig.savefig(out_dirs["diagnostics"] / "reference_vs_generated_occupancy_maps.png", dpi=170)
        plt.close(fig)
    heat = np.asarray(best.get("heat_powers", best.get("prediction", {}).get("heat_powers", [])), dtype=np.float32).reshape(-1)
    if gen_centers.size and heat.size:
        fig, ax = plt.subplots(figsize=(8.2, 3.2), constrained_layout=True)
        ax.set_xlim(0.0, float(record.domain_length_x))
        ax.set_ylim(0.0, float(record.domain_length_y))
        ax.set_aspect("equal", adjustable="box")
        sizes = 80.0 + 180.0 * (heat[: gen_centers.shape[0]] / max(float(np.max(heat)), 1.0e-8))
        sc = ax.scatter(gen_centers[:, 0], gen_centers[:, 1], c=heat[: gen_centers.shape[0]], s=sizes, cmap="magma", edgecolor="black")
        fig.colorbar(sc, ax=ax, label="heat power")
        ax.set_title("Generated heat-power layout overlay")
        ax.grid(True, alpha=0.18)
        fig.savefig(out_dirs["diagnostics"] / "heat_power_layout_overlay.png", dpi=170)
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


def _layout_vector(candidate: Mapping[str, Any], max_num_modules: int, record: Any) -> np.ndarray:
    centers = np.asarray(candidate.get("centers", []), dtype=np.float32).reshape(-1, 2)
    padded = np.zeros((max_num_modules, 2), dtype=np.float32)
    n = min(centers.shape[0], max_num_modules)
    if n > 0:
        padded[:n, 0] = centers[:n, 0] / max(float(record.domain_length_x), 1.0e-8)
        padded[:n, 1] = centers[:n, 1] / max(float(record.domain_length_y), 1.0e-8)
    count = np.asarray([float(candidate.get("count", n)) / max(float(max_num_modules), 1.0)], dtype=np.float32)
    return np.concatenate([padded.reshape(-1), count], axis=0)


def diversity_rerank_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    weight: float,
    top_k: int,
    max_num_modules: int,
    record: Any,
) -> List[Dict[str, Any]]:
    raw_sorted = list(candidates)
    if not raw_sorted:
        return []
    if weight <= 0.0 or top_k <= 1:
        for rank, row in enumerate(raw_sorted):
            row["diversity_rank"] = int(rank)
        return raw_sorted
    pool = raw_sorted[: max(int(top_k), 1)]
    remainder = raw_sorted[len(pool) :]
    vectors = [_layout_vector(row, max_num_modules, record) for row in pool]
    selected = [0]
    remaining = set(range(1, len(pool)))
    while remaining:
        best_idx = None
        best_adjusted = float("inf")
        for idx in sorted(remaining):
            min_dist = min(float(np.linalg.norm(vectors[idx] - vectors[j])) for j in selected)
            adjusted = float(pool[idx]["total_score"]) - float(weight) * min_dist
            if adjusted < best_adjusted:
                best_adjusted = adjusted
                best_idx = idx
        assert best_idx is not None
        pool[best_idx]["diversity_adjusted_score"] = float(best_adjusted)
        selected.append(best_idx)
        remaining.remove(best_idx)
    reranked = [pool[idx] for idx in selected] + remainder
    for rank, row in enumerate(reranked):
        row["diversity_rank"] = int(rank)
    reranked[0]["diversity_adjusted_score"] = float(reranked[0]["total_score"])
    return reranked


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


def make_output_dirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": out_dir,
        "data": out_dir / "data",
        "plots": out_dir / "plots",
        "layouts": out_dir / "plots" / "layouts",
        "fields": out_dir / "plots" / "fields",
        "kpis": out_dir / "plots" / "kpis",
        "diagnostics": out_dir / "plots" / "diagnostics",
        "reference": out_dir / "reference",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> int:
    args = parse_args()
    quick = bool(args.quick or args.smoke)
    n_samples = int(args.n_samples if args.n_samples is not None else (8 if quick else 128))
    n_steps = int(args.n_steps if args.n_steps is not None else (4 if quick else 16))
    device = select_device(args.device)
    inverse_path = resolve_inverse_checkpoint(args.inverse_run, args.checkpoint_name)
    inverse_model, checkpoint = load_inverse_checkpoint(inverse_path, device)
    kpi_distribution_summary = load_kpi_distribution_summary(checkpoint, inverse_path)
    train_cfg = checkpoint.get("train_config", {})
    dataset_cfg = train_cfg.get("dataset", {}) if isinstance(train_cfg, Mapping) else {}
    target_cfg = train_cfg.get("target_kpis", {}) if isinstance(train_cfg, Mapping) else {}
    conditioning_cfg = train_cfg.get("conditioning", {}) if isinstance(train_cfg, Mapping) and isinstance(train_cfg.get("conditioning", {}), Mapping) else {}
    intent_aug_cfg = train_cfg.get("intent_augmentation", {}) if isinstance(train_cfg, Mapping) and isinstance(train_cfg.get("intent_augmentation", {}), Mapping) else {}
    structure_conditioning_cfg = train_cfg.get("structure_conditioning", {}) if isinstance(train_cfg, Mapping) and isinstance(train_cfg.get("structure_conditioning", {}), Mapping) else {}
    heat_conditioning_cfg = train_cfg.get("heat_conditioning", {}) if isinstance(train_cfg, Mapping) and isinstance(train_cfg.get("heat_conditioning", {}), Mapping) else {}
    packed_path = args.dataset or dataset_cfg.get("packed_h5_path", "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5")
    if args.target:
        target_payload = load_target_payload(args.target)
        original_target_payload = dict(target_payload)
        if args.calibrate_target_to_data:
            calibration_cfg = target_payload.get("calibration", {}) if isinstance(target_payload.get("calibration"), Mapping) else {}
            target_payload = calibrate_target_spec_to_kpi_quantiles(
                target_payload,
                kpi_distribution_summary,
                max_replacement_quantile=str(calibration_cfg.get("max_replacement_quantile", "p10")),
                min_replacement_quantile=str(calibration_cfg.get("min_replacement_quantile", "p90")),
            )
            target_payload["_path"] = original_target_payload.get("_path")
        target_spec = target_spec_from_payload(target_payload, checkpoint, inverse_model)
        dataset_temperature_limits = target_spec.get("temperature_limits") if isinstance(target_spec.get("temperature_limits"), Mapping) else None
    else:
        target_payload = {}
        target_spec = {
            "kpi_names": list(checkpoint.get("kpi_names", DEFAULT_KPI_NAMES)),
            "temperature_limits": target_cfg.get("temperature_limits"),
        }
        dataset_temperature_limits = target_cfg.get("temperature_limits") if isinstance(target_cfg.get("temperature_limits"), Mapping) else None
    dataset = ThermalInverseDesignDataset(
        packed_path,
        split=args.reference_split,
        kpi_names=target_spec["kpi_names"],
        kpi_stats=checkpoint.get("kpi_stats"),
        normalize_targets=False,
        target_augmentation={},
        temperature_limits=dataset_temperature_limits,
        max_num_modules=inverse_model.max_num_modules,
        generate_heat_power=bool(inverse_model.cfg.generate_heat_power),
        heat_power_scale=float(inverse_model.cfg.heat_power_scale),
        max_cases=max(args.reference_case_index + 1, 1),
        use_all_if_split_missing=True,
        seed=int(args.seed),
        behavior_latent_dim=int(inverse_model.cfg.behavior_latent_dim),
        organization_latent_dim=int(inverse_model.cfg.organization_latent_dim),
        conditioning_mode=str(conditioning_cfg.get("mode", getattr(inverse_model.cfg, "conditioning_mode", "legacy_kpi"))),
        intent_augmentation={**intent_aug_cfg, "field_preference_dropout": 1.0},
        structure_conditioning=structure_conditioning_cfg,
        heat_conditioning=heat_conditioning_cfg,
    )
    dataset.set_kpi_distribution_summary(kpi_distribution_summary)
    record = dataset.records[min(max(int(args.reference_case_index), 0), len(dataset.records) - 1)]
    if not args.target:
        item = dataset[min(max(int(args.reference_case_index), 0), len(dataset) - 1)]
        target_spec = target_spec_from_reference_item(
            record,
            item,
            checkpoint,
            inverse_model,
            structure_strength=float(args.reference_structure_strength),
            heat_mode=str(args.reference_heat_mode),
            anchor_mode=str(args.reference_anchor_mode),
            enable_structure=not bool(args.disable_reference_structure),
            enable_heat=not bool(args.disable_reference_heat),
        )
        target_payload = dict(target_spec.get("target_payload", {}))
        print(f"[target] derived target from {record.split} case {record.case_id} (index {args.reference_case_index}).")
    forward_cfg = apply_forward_overrides(train_cfg.get("forward_model", {}) if isinstance(train_cfg, Mapping) else {}, args)
    forward_model, forward_metadata, _ = load_forward_model(forward_cfg, device)
    inverse_cfg = train_cfg.get("inverse_model", {}) if isinstance(train_cfg, Mapping) else {}
    heat_load_policy = str(inverse_cfg.get("heat_load_policy", getattr(inverse_model.cfg, "heat_load_policy", "preserve_total_heat"))).lower().strip()
    fixed_heat_per_module = inverse_cfg.get("fixed_heat_per_module")
    if target_spec.get("constraints", {}).get("heat_power_total") is not None:
        heat_load_policy = "target_heat_power_total" if heat_load_policy == "preserve_total_heat" else heat_load_policy
    x_bounds, y_bounds, preference_warnings = preference_bounds(target_spec)
    x_bounds = target_spec.get("x_bounds", x_bounds)
    y_bounds = target_spec.get("y_bounds", y_bounds)
    base_out = Path(args.output_dir) if args.output_dir else inverse_path.parent / "evaluation" / f"inverse_eval_{current_timestamp()}"
    out_dir = resolve_demo_path(base_out)
    out_dirs = make_output_dirs(out_dir)
    feasibility_report = target_feasibility_report(target_payload, kpi_distribution_summary) if kpi_distribution_summary else {"entries": [], "warnings": ["kpi_distribution_summary.json not found; target feasibility was not checked."], "summary_available": False}
    if not kpi_distribution_summary:
        print("[target-feasibility] warning: kpi_distribution_summary.json not found; target feasibility was not checked.")
    write_json(out_dirs["data"] / "target_feasibility_report.json", json_safe(feasibility_report))
    if args.calibrate_target_to_data:
        write_json(out_dirs["data"] / "calibrated_target_spec_resolved.json", json_safe(target_payload))

    target_vec = np.asarray(target_spec["vector"], dtype=np.float32)
    candidate_pool_size = max(int(math.ceil(n_samples * max(float(args.candidate_pool_multiplier), 1.0))), n_samples)
    sampled = inverse_model.sample_designs(
        target_vec,
        n_samples=int(candidate_pool_size),
        n_steps=int(n_steps),
        seed=int(args.seed),
        count_mode=str(args.count_mode),
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        design_intent_vector=target_spec.get("design_intent_vector"),
        objective_weight_vector=target_spec.get("objective_weight_vector"),
        field_intent_maps=target_spec.get("field_intent_maps"),
        structure_intent_vector=target_spec.get("structure_intent_vector"),
        structure_intent_maps=target_spec.get("structure_intent_maps"),
        heat_condition_vector=target_spec.get("heat_condition_vector"),
        heat_condition_mask=target_spec.get("heat_condition_mask"),
        guidance_scale=float(args.guidance_scale),
        device=device,
    )
    candidates: List[Dict[str, Any]] = []
    for sample_idx, raw_cand in enumerate(tqdm(sampled, desc="verify", unit="candidate", dynamic_ncols=True)):
        cand = apply_preferences_to_candidate(raw_cand, record, target_spec)
        prediction = predict_candidate_with_forward(
            forward_model,
            forward_metadata,
            record,
            cand,
            device,
            max_num_modules=inverse_model.max_num_modules,
            generate_heat_power=bool(inverse_model.cfg.generate_heat_power),
            heat_load_policy=heat_load_policy,
            fixed_heat_per_module=float(fixed_heat_per_module) if fixed_heat_per_module is not None else target_payload.get("fixed_heat_per_module"),
            target_heat_power_total=target_spec.get("constraints", {}).get("heat_power_total") if isinstance(target_spec.get("constraints"), Mapping) else None,
            query_batch_size=int(args.query_batch_size),
        )
        verified_kpis = _candidate_kpi_payload(record, prediction, cand, inverse_model, target_spec)
        score = score_candidate_kpis(verified_kpis, target_spec)
        intent_score = compute_design_intent_score(verified_kpis, {"centers": cand.get("centers", [])}, target_spec)
        structure_score = compute_structure_match_score({**cand, "heat_powers": prediction.get("heat_powers")}, target_spec, record)
        use_intent_score = bool(target_spec.get("is_design_intent", False) or getattr(inverse_model.cfg, "conditioning_mode", "legacy_kpi") == "design_intent")
        structure_weight = float(target_spec.get("structure_strength", target_spec.get("structure_constraints", {}).get("strength", 0.0) if isinstance(target_spec.get("structure_constraints"), Mapping) else 0.0))
        primary_base = float(intent_score["design_intent_score"] if use_intent_score else score["total_score"])
        primary_score = primary_base + structure_weight * float(structure_score["structure_match_score"])
        row = {
            "sample_index": int(sample_idx),
            "count": int(cand.get("count", 0)),
            "centers": np.asarray(cand["centers"], dtype=np.float32),
            "heat_powers": np.asarray(prediction.get("heat_powers", cand.get("heat_powers", [])), dtype=np.float32),
            "slot_ids": np.asarray(cand.get("slot_ids", []), dtype=np.int64),
            "valid": bool(cand.get("validity", {}).get("valid", False)),
            "validity": cand.get("validity", {}),
            "verified_kpis": verified_kpis,
            "score_detail": score,
            "design_intent_score_detail": intent_score,
            "design_intent_score": float(intent_score["design_intent_score"]),
            "structure_match_score": float(structure_score["structure_match_score"]),
            "structure_score_detail": structure_score,
            "total_score": primary_score,
            "primary_score_without_structure": primary_base,
            "legacy_total_score": float(score["total_score"]),
            "kpi_score": float(score.get("kpi_score", score["total_score"])),
            "constraint_penalty": float(score.get("constraint_penalty", 0.0)),
            "spread_preference_penalty": float(score.get("spread_preference_penalty", 0.0)),
            "feasibility_penalty": float(score.get("feasibility_penalty", score.get("constraint_penalty", 0.0))),
            "kpi_violation": float(score.get("kpi_violation", score.get("kpi_score", 0.0))),
            "preference_reward": float(score.get("preference_reward", 0.0)),
            "prediction": prediction,
        }
        for key in ("min_center_distance", "wall_clearance", "inlet_clearance", "outlet_clearance", "x_coverage", "y_coverage", "bbox_area", "mean_pair_distance"):
            row[key] = float(verified_kpis.get(key, float("nan")))
        candidates.append(row)
    candidates.sort(key=lambda row: (0 if row["valid"] else 1, float(row["total_score"]), -int(row["count"])))
    for rank, row in enumerate(candidates):
        row["raw_score_rank"] = int(rank)
        row["rank"] = int(rank)
    ranked_candidates = diversity_rerank_candidates(
        candidates,
        weight=float(args.diversity_rerank_weight),
        top_k=min(int(args.diversity_rerank_top_k), len(candidates)),
        max_num_modules=inverse_model.max_num_modules,
        record=record,
    )
    for row in ranked_candidates:
        row["rank"] = int(row.get("diversity_rank", row.get("raw_score_rank", 0)))
    candidates_for_outputs = ranked_candidates[: min(n_samples, len(ranked_candidates))]

    serializable = []
    for row in candidates:
        lite = {key: value for key, value in row.items() if key != "prediction"}
        serializable.append(json_safe(lite))
    write_json(out_dirs["data"] / "candidates.json", {"target": json_safe(target_spec), "candidates": serializable, "displayed_candidate_indices": [int(row["sample_index"]) for row in candidates_for_outputs]})
    write_candidates_csv(candidates_for_outputs, out_dirs["data"] / "candidates.csv")
    write_kpi_scores_csv(candidates_for_outputs, target_spec["kpi_names"], out_dirs["data"] / "kpi_scores.csv")
    write_structure_outputs(candidates_for_outputs, target_spec, record, out_dirs)
    write_json(out_dirs["data"] / "target_spec_resolved.json", json_safe(target_spec))
    save_top_npz(candidates_for_outputs, out_dirs["data"] / "top_candidates.npz", max_num_modules=inverse_model.max_num_modules)

    best = candidates[0] if candidates else None
    summary = {
        "inverse_checkpoint": str(inverse_path),
        "target_path": target_payload.get("_path"),
        "reference_case_id": record.case_id,
        "n_samples": int(n_samples),
        "candidate_pool_size": int(candidate_pool_size),
        "n_steps": int(n_steps),
        "guidance_scale": float(args.guidance_scale),
        "count_mode": str(args.count_mode),
        "best_score": best["total_score"] if best else None,
        "best_valid": bool(best["valid"]) if best else None,
        "best_raw_score_rank": int(best.get("raw_score_rank", 0)) if best else None,
        "best_diversity_rank": int(best.get("diversity_rank", 0)) if best else None,
        "validity_rate": float(np.mean([float(c["valid"]) for c in candidates])) if candidates else 0.0,
        "forward_checkpoint": forward_metadata.get("checkpoint_path"),
        "local_surrogate_checkpoint": forward_metadata.get("local_surrogate_checkpoint_path"),
        "verification_mode": "predicted",
        "local_surrogate_used": bool(forward_metadata.get("local_surrogate_used", False)),
        "predicted_port_condition_kpis_available": bool(best and "mean_interface_T_env" in best.get("verified_kpis", {}).get("available_kpis", [])),
        "heat_load_policy": heat_load_policy,
        "heat_source": best.get("prediction", {}).get("heat_source") if best else None,
        "structure_weight": float(target_spec.get("structure_strength", 0.0)),
        "diversity_rerank_weight": float(args.diversity_rerank_weight),
        "diversity_rerank_top_k": int(args.diversity_rerank_top_k),
        "best_preference_penalties": best.get("score_detail", {}).get("per_preference_penalties", {}) if best else {},
        "best_design_intent_score_breakdown": best.get("design_intent_score_detail", {}) if best else {},
        "target_feasibility_warnings": feasibility_report.get("warnings", []),
        "target_calibration": target_payload.get("_calibration", {"enabled": False}),
        "preference_warnings": sorted(set(preference_warnings + sum((list(c.get("preference_warnings", [])) for c in candidates), []))),
    }
    write_json(out_dirs["data"] / "verification_summary.json", json_safe(summary))

    if best is not None:
        plot_target_vs_verified(best, target_spec, out_dirs["kpis"] / "target_vs_verified_kpis.png")
        normalized_rows = plot_target_vs_verified_normalized(best, target_spec, out_dirs["kpis"] / "target_vs_verified_kpis_normalized.png")
        write_normalized_violation_table(normalized_rows, out_dirs["data"] / "target_vs_verified_kpis_normalized.csv")
        write_json(out_dirs["data"] / "target_vs_verified_kpis_normalized.json", json_safe({"rows": normalized_rows}))
        plot_candidate_layouts(candidates_for_outputs, record, out_dirs["layouts"] / "candidate_layouts_ranked.png")
        write_design_intent_breakdown(best, out_dirs["data"] / "design_intent_score_breakdown.csv", out_dirs["diagnostics"] / "design_intent_score_breakdown.png")
        write_all_kpis_verified(best, target_spec["kpi_names"], out_dirs["data"] / "all_kpis_verified.csv")
        write_json(out_dirs["data"] / "field_penalty_summary.json", json_safe(best.get("design_intent_score_detail", {}).get("field_penalties", {})))
        plot_field_risk_overlay(best, target_spec, record, out_dirs["fields"] / "best_layout_field_risk_overlay.png")
        plot_candidate_pareto_scatter(candidates, out_dirs["diagnostics"] / "candidate_pareto_scatter.png")
        plot_temperature_field(best, record, out_dirs["fields"] / "best_layout_global_fields.png")
        plot_composite_internal(best, record, out_dirs["fields"] / "best_layout_module_internal_disks.png")
        plot_internal_bars(best, out_dirs["kpis"] / "best_layout_module_temperature_bars.png")
        plot_interface_curves(best, out_dirs["fields"] / "best_layout_interface_curves.png")
        try_plot_organization(best, record, out_dirs["diagnostics"] / "best_layout_organization_overview.png")
        plot_diversity(candidates, out_dirs["diagnostics"] / "candidate_diversity.png")
        plot_structure_diagnostics(best, target_spec, record, out_dirs)
        if target_spec.get("reference_ground_truth"):
            plot_reference_layout_comparison(best, record, out_dirs["reference"] / "generated_vs_reference_layout.png")
            plot_reference_field_comparison(best, record, out_dirs["reference"] / "generated_vs_reference_fields.png")
            plot_reference_kpi_comparison(best, record, target_spec["kpi_names"], out_dirs["reference"] / "generated_vs_reference_kpis.png")
            write_reference_comparison_csv(best, record, target_spec["kpi_names"], out_dirs["reference"] / "generated_vs_reference_kpis.csv")
    print(f"[done] inverse evaluation saved to {out_dir}")
    if best is not None:
        print(f"[best] score={best['total_score']:.6f}, valid={best['valid']}, count={best['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
