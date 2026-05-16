from __future__ import annotations

"""Evaluate target-agnostic design-prior search against field-functional specs."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channelthermal-design-prior-eval")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

import _bootstrap_imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from channelthermal_model_utils import current_timestamp, ensure_dir, resolve_demo_path, select_device, write_json
from design_candidate_io import plot_layout_candidate, plot_score_vs_calls, to_jsonable, write_candidates_csv, write_summary_json
from field_functional_objective import FieldFunctionalObjective, extract_field_array
from layout_search_baselines import LayoutSearchConfig, RandomValidLayoutSampler, RawLayoutCEMOptimizer, sample_random_valid_layout
from model_design_prior import LatentModularDesignPrior
from search_design_prior import ForwardHONFEvaluator, GuidedSearchConfig, LatentPosteriorDesignSearcher
from train_inverse import ThermalInverseDesignDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ChannelThermal design-prior inverse search.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--calibration-samples", type=int, default=256)
    return parser.parse_args()


def _read_json(path: str | Path) -> Dict[str, Any]:
    with resolve_demo_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _path_status(path: str | Path | None) -> Dict[str, Any]:
    if path is None or str(path) == "":
        return {"path": "", "exists": False}
    resolved = resolve_demo_path(path)
    return {"path": str(resolved), "exists": bool(resolved.exists())}


def _forward_cfg(raw_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = {}
    config_path = raw_cfg.get("config_path")
    if config_path:
        path = resolve_demo_path(config_path)
        if path.exists():
            loaded = _read_json(path)
            if isinstance(loaded.get("forward_model"), Mapping):
                cfg.update(dict(loaded["forward_model"]))
    if raw_cfg.get("checkpoint_path"):
        checkpoint = resolve_demo_path(raw_cfg["checkpoint_path"])
        cfg["run_dir"] = str(checkpoint.parent)
        cfg["checkpoint_name"] = checkpoint.name
    cfg.update({k: v for k, v in raw_cfg.items() if k not in {"config_path", "checkpoint_path", "device", "batch_size"}})
    if raw_cfg.get("batch_size") and "query_batch_size" not in cfg:
        cfg["query_batch_size"] = int(raw_cfg["batch_size"])
    cfg.setdefault("enabled", True)
    return cfg


def _load_prior(path: str | Path, device: torch.device) -> LatentModularDesignPrior:
    checkpoint = torch.load(resolve_demo_path(path), map_location=device, weights_only=False)
    model = LatentModularDesignPrior(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _normalize_run_id(value: Any, fallback: str = "0001") -> str:
    raw = str(value if value is not None and str(value).strip() else fallback).strip()
    if not raw.isdigit():
        raise ValueError(f"Run_ID must be a numeric serial such as '0001'; got {raw!r}.")
    return raw.zfill(4)


def _latest_design_prior_run(saved_root: Path, run_id: str) -> Path:
    normalized = _normalize_run_id(run_id)
    pattern = re.compile(rf"^Run_{re.escape(normalized)}_\d{{8}}_\d{{6}}$")
    runs = sorted([path for path in saved_root.glob(f"Run_{normalized}_*") if path.is_dir() and pattern.match(path.name)])
    if not runs:
        raise FileNotFoundError(f"No design-prior runs found under {saved_root} with Run_ID={normalized!r}.")
    return runs[-1]


def _resolve_design_prior_checkpoint(prior_cfg: Mapping[str, Any], *, required: bool = False) -> Optional[Path]:
    checkpoint_path = str(prior_cfg.get("checkpoint_path", "auto") or "auto")
    if checkpoint_path.lower() not in {"", "auto", "latest"}:
        path = resolve_demo_path(checkpoint_path)
        if path.exists() or required:
            return path
        return None
    saved_root = resolve_demo_path(prior_cfg.get("saved_root", "Saved_Model_Prior"))
    run_id = _normalize_run_id(prior_cfg.get("Run_ID", prior_cfg.get("run_id", "0001")))
    checkpoint_name = str(prior_cfg.get("checkpoint_name", "best_model.pt"))
    try:
        run_dir = _latest_design_prior_run(saved_root, run_id)
    except FileNotFoundError:
        if required:
            raise
        return None
    path = run_dir / checkpoint_name
    if not path.exists() and checkpoint_name != "latest_model.pt":
        latest = run_dir / "latest_model.pt"
        if latest.exists():
            return latest
    if path.exists() or required:
        return path
    return None


def _context_dataset_path(cfg: Mapping[str, Any], prior_checkpoint: Optional[Mapping[str, Any]] = None) -> str:
    context_cfg = cfg.get("context", {}) if isinstance(cfg.get("context"), Mapping) else {}
    if context_cfg.get("dataset_path"):
        return str(context_cfg["dataset_path"])
    if isinstance(prior_checkpoint, Mapping):
        train_cfg = prior_checkpoint.get("train_config", {})
        data_cfg = train_cfg.get("data", {}) if isinstance(train_cfg, Mapping) and isinstance(train_cfg.get("data"), Mapping) else {}
        if data_cfg.get("source_dataset_path"):
            return str(data_cfg["source_dataset_path"])
    return "./Data_Saved/Processed_ChannelThermal_Dataset/packed_dataset.h5"


def _load_context(cfg: Mapping[str, Any], max_num_modules: int, generate_heat_power: bool) -> Dict[str, Any]:
    context_cfg = cfg.get("context", {}) if isinstance(cfg.get("context"), Mapping) else {}
    path = _context_dataset_path(cfg)
    split = str(context_cfg.get("split", "test"))
    case_index = int(context_cfg.get("case_index", 0))
    dataset = ThermalInverseDesignDataset(
        path,
        split=split,
        max_num_modules=max_num_modules,
        normalize_targets=False,
        generate_heat_power=generate_heat_power,
        max_cases=max(case_index + 1, int(context_cfg.get("max_cases", 0) or 0)),
        use_all_if_split_missing=True,
    )
    record = dataset.records[min(max(case_index, 0), len(dataset.records) - 1)]
    return {
        "record": record,
        "context_vec": np.asarray([float(record.re), float(record.u_in)], dtype=np.float32),
        "dataset_path": str(resolve_demo_path(path)),
        "split": split,
        "case_index": int(case_index),
        "case_id": str(record.case_id),
    }


def _layout_config(cfg: Mapping[str, Any], override: Optional[Mapping[str, Any]] = None) -> LayoutSearchConfig:
    layout = dict(cfg.get("layout", {}) if isinstance(cfg.get("layout"), Mapping) else {})
    if override:
        layout.update({k: v for k, v in override.items() if k in LayoutSearchConfig.__dataclass_fields__})
    return LayoutSearchConfig(**{k: v for k, v in layout.items() if k in LayoutSearchConfig.__dataclass_fields__})


def _guided_config(payload: Mapping[str, Any]) -> GuidedSearchConfig:
    return GuidedSearchConfig(**{k if k != "seed" else "random_seed": v for k, v in dict(payload).items() if k in GuidedSearchConfig.__dataclass_fields__ or k == "seed"})


def _success(candidate: Mapping[str, Any], tolerance: float) -> bool:
    result = candidate.get("objective_result", {})
    hard = float(candidate.get("hard_violation_score", result.get("hard_violation_score", 0.0)) or 0.0) if isinstance(result, Mapping) else 0.0
    return bool(candidate.get("satisfied", result.get("satisfied", False) if isinstance(result, Mapping) else False)) and hard <= float(tolerance)


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _topk_diversity(candidates: Sequence[Mapping[str, Any]], top_k: int = 8) -> float:
    vecs = [np.asarray(row.get("design_vec", []), dtype=np.float64).reshape(-1) for row in list(candidates)[:top_k]]
    vecs = [v for v in vecs if v.size]
    if len(vecs) < 2:
        return 0.0
    vals = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            n = min(vecs[i].size, vecs[j].size)
            vals.append(float(np.linalg.norm(vecs[i][:n] - vecs[j][:n])))
    return float(np.mean(vals)) if vals else 0.0


def _method_summary(result: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], *, enabled: bool, skipped: bool = False, skip_reason: str = "", tolerance: float = 1.0e-6, runtime: float = 0.0) -> Dict[str, Any]:
    scores = np.asarray([float(row.get("total_score", np.nan)) for row in candidates], dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    successes = [_success(row, tolerance) for row in candidates]
    first_success = None
    for row, ok in zip(candidates, successes):
        if ok:
            first_success = int(row.get("forward_calls", result.get("num_forward_calls", 0)))
            break
    return {
        "enabled": bool(enabled),
        "skipped": bool(skipped),
        "skip_reason": str(skip_reason),
        "num_candidates": int(len(candidates)),
        "num_forward_calls": int(result.get("num_forward_calls", 0)),
        "best_score": float(np.min(finite)) if finite.size else None,
        "median_topk_score": float(np.median(finite[: min(8, finite.size)])) if finite.size else None,
        "success_rate": float(np.mean(successes)) if candidates else 0.0,
        "time_to_first_success_calls": first_success,
        "valid_fraction_before_repair": None,
        "mean_repair_distance": float(np.mean([float(row.get("repair_distance", 0.0) or 0.0) for row in candidates])) if candidates else 0.0,
        "mean_hypergraph_consistency_score": float(np.mean([float(row.get("hypergraph_consistency_score", 0.0) or 0.0) for row in candidates])) if candidates else 0.0,
        "best_hypergraph_consistency_score": float(np.min([float(row.get("hypergraph_consistency_score", 0.0) or 0.0) for row in candidates])) if candidates else 0.0,
        "topk_diversity": _topk_diversity(candidates),
        "runtime_seconds": float(runtime),
    }


def _write_score_vs_calls_csv(histories: Mapping[str, Sequence[Mapping[str, Any]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "iteration", "num_forward_calls", "best_score", "mean_elite_score"])
        writer.writeheader()
        for method, rows in histories.items():
            for row in rows:
                writer.writerow(
                    {
                        "method": method,
                        "iteration": row.get("iteration", ""),
                        "num_forward_calls": row.get("num_forward_calls", ""),
                        "best_score": row.get("best_score", ""),
                        "mean_elite_score": row.get("mean_elite_score", ""),
                    }
                )


def _candidate_field(candidate: Mapping[str, Any]) -> Any:
    prediction = candidate.get("forward_prediction", {})
    if not isinstance(prediction, Mapping):
        return None
    field = extract_field_array(prediction, "temperature")
    return field


def _strip_dense(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(candidate)
    prediction = out.get("forward_prediction")
    if isinstance(prediction, Mapping):
        slim = dict(prediction)
        for key in ("pred_field_grid", "pred_internal_temperature", "pred_interface", "pred_port_condition"):
            if key in slim:
                arr = np.asarray(slim[key])
                slim[key] = {"omitted": True, "shape": list(arr.shape)}
        out["forward_prediction"] = slim
    return out


def _write_top_artifacts(candidates: Sequence[Mapping[str, Any]], out_dir: Path, *, top_k: int, save_dense: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in list(candidates)[:top_k]:
        method = str(row.get("method", "method"))
        rank = int(row.get("rank", 0) or 0)
        prefix = f"{method}_{rank:03d}"
        field = _candidate_field(row)
        plot_layout_candidate(row.get("layout", row), out_dir / f"{prefix}_layout.png", title=f"{method} #{rank}", field=field)
        write_summary_json(row.get("objective_result", {}), out_dir / f"{prefix}_score_terms.json")
        write_summary_json(row.get("kpis", {}), out_dir / f"{prefix}_kpis.json")
        write_summary_json(
            {"planned": row.get("planned_hypergraph"), "realized": row.get("realized_hypergraph"), "comparison": row.get("hypergraph_consistency")},
            out_dir / f"{prefix}_hypergraph.json",
        )
        if field is not None:
            fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
            im = ax.imshow(np.asarray(field), origin="lower", aspect="auto")
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"{method} #{rank} temperature preview")
            fig.savefig(str(out_dir / f"{prefix}_field_preview.png"), dpi=160)
            plt.close(fig)
        if save_dense:
            write_summary_json(to_jsonable(row), out_dir / f"{prefix}_candidate_full.json")


def _plot_method_comparison(summary: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    enabled = [(name, payload) for name, payload in summary.items() if payload.get("enabled") and not payload.get("skipped")]
    if not enabled:
        return
    names = [name for name, _ in enabled]
    best = [_float_or_nan(payload.get("best_score")) for _, payload in enabled]
    success = [_float_or_nan(payload.get("success_rate", 0.0) or 0.0) for _, payload in enabled]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(max(8.0, 1.1 * len(names)), 3.8), constrained_layout=True)
    axes[0].bar(x, best)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right")
    axes[0].set_ylabel("Best total score")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(x, success)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.savefig(str(path), dpi=170)
    plt.close(fig)


def _rank(method: str, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = sorted([dict(row) for row in candidates], key=lambda row: float(row.get("total_score", float("inf"))))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = int(idx)
        row["method"] = method
    return rows


def _dry_run(cfg: Mapping[str, Any]) -> int:
    objective_path = cfg.get("objective", {}).get("spec_path") if isinstance(cfg.get("objective"), Mapping) else None
    objective = FieldFunctionalObjective.from_json(resolve_demo_path(objective_path)) if objective_path and resolve_demo_path(objective_path).exists() else None
    methods = cfg.get("methods", {}) if isinstance(cfg.get("methods"), Mapping) else {}
    layout_cfg = _layout_config(cfg)
    guided_cfg = _guided_config(cfg.get("atlas_guided", {}) if isinstance(cfg.get("atlas_guided"), Mapping) else {})
    prior_cfg = cfg.get("design_prior", {}) if isinstance(cfg.get("design_prior"), Mapping) else {}
    resolved_prior = _resolve_design_prior_checkpoint(prior_cfg, required=False)
    print("[dry-run] objective:", objective.name if objective else f"missing ({objective_path})")
    print("[dry-run] layout:", layout_cfg)
    print("[dry-run] guided:", guided_cfg)
    print("[dry-run] forward checkpoint:", _path_status(cfg.get("forward_model", {}).get("checkpoint_path") if isinstance(cfg.get("forward_model"), Mapping) else None))
    print("[dry-run] design prior checkpoint:", _path_status(resolved_prior) if resolved_prior is not None else {"path": "auto", "exists": False, "saved_root": str(resolve_demo_path(prior_cfg.get("saved_root", "Saved_Model_Prior"))), "Run_ID": _normalize_run_id(prior_cfg.get("Run_ID", prior_cfg.get("run_id", "0001")))})
    print("[dry-run] methods:", {name: bool(enabled) for name, enabled in methods.items()})
    return 0


CALIBRATION_KPIS = (
    "max_solid_temperature",
    "pressure_drop",
    "outlet_temperature_nonuniformity",
    "wall_hot_area_fraction",
    "thermal_plume_length",
    "downstream_reheat_index",
)


def _finite_series(values: Sequence[Any]) -> np.ndarray:
    parsed = []
    for value in values:
        if value is None or str(value) == "":
            continue
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    arr = np.asarray(parsed, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _quantile_summary(values: Sequence[Any]) -> Dict[str, Any]:
    arr = _finite_series(values)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p25": float(np.percentile(arr, 25.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p75": float(np.percentile(arr, 75.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def _write_calibration_histograms(series: Mapping[str, Sequence[Any]], path: Path) -> None:
    finite_items = [(name, _finite_series(values)) for name, values in series.items()]
    finite_items = [(name, arr) for name, arr in finite_items if arr.size]
    if not finite_items:
        return
    n = len(finite_items)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (name, arr) in zip(axes_arr, finite_items):
        ax.hist(arr, bins=min(24, max(4, int(math.sqrt(arr.size)) + 1)), alpha=0.85)
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
    for ax in axes_arr[len(finite_items) :]:
        ax.axis("off")
    fig.savefig(str(path), dpi=170)
    plt.close(fig)


def _run_calibration(
    *,
    cfg: Mapping[str, Any],
    objective: FieldFunctionalObjective,
    layout_config: LayoutSearchConfig,
    context: Mapping[str, Any],
    evaluator: ForwardHONFEvaluator,
    num_samples: int,
    run_dir: Path,
) -> Dict[str, Any]:
    search_context = dict(context)
    search_context["objective_spec"] = objective.spec
    search_context["hard_constraints"] = objective.spec.get("hard_constraints", {})
    seed = int((cfg.get("random_valid", {}) if isinstance(cfg.get("random_valid"), Mapping) else {}).get("seed", 0))
    rng = np.random.default_rng(seed)
    term_rows: List[Dict[str, Any]] = []
    series: Dict[str, List[Any]] = {"total_score": [], "hard_violation_score": []}
    for key in CALIBRATION_KPIS:
        series[key] = []
    term_series: Dict[str, List[Any]] = {}
    for idx in tqdm(range(max(int(num_samples), 0)), desc="calibration", unit="layout", dynamic_ncols=True):
        layout = sample_random_valid_layout(layout_config, rng, search_context)
        forward_payload = evaluator.evaluate_layout(layout, search_context)
        objective_result = objective.evaluate(
            forward_prediction=forward_payload.get("forward_prediction"),
            kpis=forward_payload.get("kpis"),
            layout=forward_payload.get("layout", layout),
            realized_hypergraph=forward_payload.get("realized_hypergraph"),
        )
        total = objective_result.get("total_score")
        hard = objective_result.get("hard_violation_score")
        series["total_score"].append(total)
        series["hard_violation_score"].append(hard)
        kpis = forward_payload.get("kpis", {}) if isinstance(forward_payload.get("kpis"), Mapping) else {}
        for key in CALIBRATION_KPIS:
            series[key].append(kpis.get(key))
        for term in objective_result.get("term_results", []) or []:
            if not isinstance(term, Mapping):
                continue
            name = str(term.get("name", "term"))
            raw = term.get("raw_value")
            term_series.setdefault(f"term_raw:{name}", []).append(raw)
            details = term.get("details", {}) if isinstance(term.get("details"), Mapping) else {}
            term_rows.append(
                {
                    "sample_index": idx,
                    "term_name": name,
                    "raw_value": raw,
                    "value": term.get("value"),
                    "satisfied": term.get("satisfied"),
                    "mode": term.get("mode"),
                    "scale": details.get("scale", ""),
                    "unscaled_penalty": details.get("unscaled_penalty", ""),
                    "scaled_penalty": details.get("scaled_penalty", ""),
                }
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "objective": objective.name,
        "num_samples": int(num_samples),
        "num_forward_calls": int(getattr(evaluator, "num_forward_calls", 0)),
        "quantiles": {name: _quantile_summary(values) for name, values in {**series, **term_series}.items()},
        "run_dir": str(run_dir),
    }
    write_summary_json(summary, run_dir / "calibration_summary.json")
    with (run_dir / "calibration_terms.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "term_name",
                "raw_value",
                "value",
                "satisfied",
                "mode",
                "scale",
                "unscaled_penalty",
                "scaled_penalty",
            ],
        )
        writer.writeheader()
        writer.writerows(term_rows)
    _write_calibration_histograms({**series, **term_series}, run_dir / "calibration_histograms.png")
    best = _quantile_summary(series["total_score"])
    print(
        "[calibrate] wrote calibration to"
        f" {run_dir} | samples={num_samples}"
        f" total_score_p50={best.get('p50')}"
        f" total_score_p95={best.get('p95')}"
    )
    return summary


def main() -> int:
    args = parse_args()
    cfg = _read_json(args.config)
    if args.dry_run:
        return _dry_run(cfg)
    objective_cfg = cfg.get("objective", {}) if isinstance(cfg.get("objective"), Mapping) else {}
    objective = FieldFunctionalObjective.from_json(resolve_demo_path(objective_cfg.get("spec_path")))
    tolerance = float(objective_cfg.get("success_hard_violation_tolerance", 1.0e-6))
    layout_config = _layout_config(cfg)
    context = _load_context(cfg, int(layout_config.max_num_modules), bool(layout_config.generate_heat_power))
    search_context = dict(context)
    search_context["objective_spec"] = objective.spec
    search_context["hard_constraints"] = objective.spec.get("hard_constraints", {})
    forward_cfg = _forward_cfg(cfg.get("forward_model", {}) if isinstance(cfg.get("forward_model"), Mapping) else {})
    if args.calibrate_only:
        checkpoint_path = (cfg.get("forward_model", {}) if isinstance(cfg.get("forward_model"), Mapping) else {}).get("checkpoint_path")
        if checkpoint_path and not resolve_demo_path(checkpoint_path).exists():
            raise FileNotFoundError(f"Forward checkpoint is required for --calibrate-only and was not found: {resolve_demo_path(checkpoint_path)}")
    forward_device = select_device(None if str(cfg.get("forward_model", {}).get("device", "auto")).lower() == "auto" else str(cfg.get("forward_model", {}).get("device")))
    evaluator = ForwardHONFEvaluator.from_config(forward_cfg, forward_device, layout_config)
    output_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), Mapping) else {}
    run_dir_cfg = str(output_cfg.get("run_dir", "") or "").strip()
    if run_dir_cfg:
        run_dir = ensure_dir(resolve_demo_path(run_dir_cfg))
    else:
        run_dir = ensure_dir(resolve_demo_path(f"Saved_Model_Prior/DesignPrior_Eval/eval_{current_timestamp()}"))
    if args.calibrate_only:
        _run_calibration(
            cfg=cfg,
            objective=objective,
            layout_config=layout_config,
            context=search_context,
            evaluator=evaluator,
            num_samples=int(args.calibration_samples),
            run_dir=run_dir,
        )
        return 0
    methods_cfg = cfg.get("methods", {}) if isinstance(cfg.get("methods"), Mapping) else {}
    prior_model = None
    design_prior_cfg = cfg.get("design_prior", {}) if isinstance(cfg.get("design_prior"), Mapping) else {}
    need_prior = bool(methods_cfg.get("atlas_prior", True) or methods_cfg.get("atlas_guided", True))
    prior_checkpoint = _resolve_design_prior_checkpoint(design_prior_cfg, required=False) if need_prior else None
    if need_prior and prior_checkpoint is not None and prior_checkpoint.exists():
        prior_device = select_device(None if str(design_prior_cfg.get("device", "auto")).lower() == "auto" else str(design_prior_cfg.get("device")))
        prior_model = _load_prior(prior_checkpoint, prior_device)
    histories_dir = ensure_dir(run_dir / "histories")
    top_dir = ensure_dir(run_dir / "top_candidates")
    results: Dict[str, Any] = {}
    summaries: Dict[str, Any] = {}
    all_candidates: List[Dict[str, Any]] = []
    histories: Dict[str, Sequence[Mapping[str, Any]]] = {}

    def record_result(method: str, result: Mapping[str, Any], runtime: float, enabled: bool = True, skipped: bool = False, reason: str = "") -> None:
        candidates = _rank(method, list(result.get("best_candidates", [])))
        results[method] = dict(result)
        summaries[method] = _method_summary(result, candidates, enabled=enabled, skipped=skipped, skip_reason=reason, tolerance=tolerance, runtime=runtime)
        all_candidates.extend(candidates)
        if result.get("history"):
            histories[method] = list(result["history"])
            write_summary_json({"history": list(result["history"])}, histories_dir / f"{method}_history.json")
        if skipped:
            print(f"[method:{method}] skipped in {runtime:.2f}s: {reason}")
        else:
            best = summaries[method].get("best_score")
            calls = summaries[method].get("num_forward_calls")
            best_text = "nan" if best is None else f"{float(best):.6g}"
            print(f"[method:{method}] done in {runtime:.2f}s | forward_calls={calls} | best_score={best_text}")

    if bool(methods_cfg.get("random_valid", True)):
        print("[method:random_valid] starting")
        start = time.time()
        rv_cfg = cfg.get("random_valid", {}) if isinstance(cfg.get("random_valid"), Mapping) else {}
        lcfg = _layout_config(cfg, {"random_seed": int(rv_cfg.get("seed", 0))})
        sampler = RandomValidLayoutSampler(lcfg, evaluator, objective, num_samples=int(rv_cfg.get("num_samples", 512)))
        result = sampler.search(search_context, num_return=int(rv_cfg.get("num_return", 16)), num_samples=int(rv_cfg.get("num_samples", 512)))
        record_result("random_valid", result, time.time() - start)
    if bool(methods_cfg.get("raw_layout_cem", True)):
        print("[method:raw_layout_cem] starting")
        start = time.time()
        raw_cfg = cfg.get("raw_layout_cem", {}) if isinstance(cfg.get("raw_layout_cem"), Mapping) else {}
        lcfg = _layout_config(
            cfg,
            {
                "cem_iterations": int(raw_cfg.get("cem_iterations", 8)),
                "cem_population": int(raw_cfg.get("cem_population", 128)),
                "cem_elite_frac": float(raw_cfg.get("cem_elite_frac", 0.15)),
                "cem_init_std": float(raw_cfg.get("cem_init_std", 0.35)),
                "cem_min_std": float(raw_cfg.get("cem_min_std", 0.03)),
                "cem_smoothing": float(raw_cfg.get("cem_smoothing", 0.5)),
                "random_seed": int(raw_cfg.get("seed", 0)),
            },
        )
        result = RawLayoutCEMOptimizer(lcfg, evaluator, objective).search(search_context, num_return=int(raw_cfg.get("num_return", 16)))
        record_result("raw_layout_cem", result, time.time() - start)
    if bool(methods_cfg.get("current_inverse", False)):
        record_result("current_inverse", {"method": "current_inverse", "best_candidates": [], "history": [], "num_forward_calls": 0}, 0.0, skipped=True, reason="Current KPI-conditioned inverse baseline integration is deferred; A/B/D/E remain runnable.")
    if bool(methods_cfg.get("atlas_prior", True)):
        if prior_model is None:
            reason = "design_prior.checkpoint_path missing or not found."
            record_result("atlas_prior", {"method": "atlas_prior", "best_candidates": [], "history": [], "num_forward_calls": 0}, 0.0, skipped=True, reason=reason)
        else:
            print("[method:atlas_prior] starting")
            start = time.time()
            prior_cfg = cfg.get("atlas_prior", {}) if isinstance(cfg.get("atlas_prior"), Mapping) else {}
            search_cfg = GuidedSearchConfig(num_return=int(prior_cfg.get("num_return", 16)), random_seed=int(prior_cfg.get("seed", 0)))
            searcher = LatentPosteriorDesignSearcher(prior_model, evaluator, objective, layout_config, search_cfg)
            candidates = searcher.sample_prior_candidates(
                search_context,
                int(prior_cfg.get("num_samples", 512)),
                temperature=float(prior_cfg.get("temperature", 1.0)),
            )[: int(prior_cfg.get("num_return", 16))]
            result = {"method": "atlas_prior", "best_candidates": candidates, "history": [{"iteration": 0, "best_score": float(candidates[0]["total_score"]) if candidates else float("inf"), "num_forward_calls": int(searcher.forward_calls)}], "num_forward_calls": int(searcher.forward_calls)}
            record_result("atlas_prior", result, time.time() - start)
    if bool(methods_cfg.get("atlas_guided", True)):
        if prior_model is None:
            reason = "design_prior.checkpoint_path missing or not found."
            record_result("atlas_guided", {"method": "atlas_guided", "best_candidates": [], "history": [], "num_forward_calls": 0}, 0.0, skipped=True, reason=reason)
        else:
            print("[method:atlas_guided] starting")
            start = time.time()
            guided_cfg = _guided_config(cfg.get("atlas_guided", {}) if isinstance(cfg.get("atlas_guided"), Mapping) else {})
            searcher = LatentPosteriorDesignSearcher(prior_model, evaluator, objective, layout_config, guided_cfg)
            result = searcher.latent_cem_search(search_context)
            record_result("atlas_guided", result, time.time() - start)

    save_dense = bool(output_cfg.get("save_dense_forward_outputs", False))
    top_k = int(output_cfg.get("save_top_k_per_method", 5))
    top_candidates = []
    for method in results:
        candidates = _rank(method, list(results[method].get("best_candidates", [])))
        top_candidates.extend(candidates[:top_k])
        _write_top_artifacts(candidates, top_dir, top_k=top_k, save_dense=save_dense)
    csv_candidates = [_strip_dense(row) for row in all_candidates] if not save_dense else all_candidates
    csv_top = [_strip_dense(row) for row in top_candidates] if not save_dense else top_candidates
    write_candidates_csv(csv_candidates, run_dir / "candidates_all.csv")
    write_candidates_csv(csv_top, run_dir / "candidates_top.csv")
    _write_score_vs_calls_csv(histories, run_dir / "score_vs_forward_calls.csv")
    if histories:
        plot_score_vs_calls(histories, run_dir / "score_vs_calls.png")
    _plot_method_comparison(summaries, run_dir / "method_comparison.png")
    write_summary_json(
        {
            "objective": objective.name,
            "context": {key: value for key, value in context.items() if key != "record"},
            "methods": summaries,
            "run_dir": str(run_dir),
        },
        run_dir / "summary.json",
    )
    print(f"[evaluate_design_prior] wrote outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
