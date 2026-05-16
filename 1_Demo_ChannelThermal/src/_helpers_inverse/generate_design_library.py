from __future__ import annotations

"""Generate a behavior-aware design library for the latent design prior.

Data enhancement here is not about inventing arbitrary valid structures. It is
about populating a behavior-diverse atlas of layout -> full-state behavior ->
realized hypergraph organization using the frozen forward HONF verifier.
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from channelthermal_model_utils import resolve_demo_path, select_device
    from design_prior_dataset import FALLBACK_KPI_NAMES
    from layout_search_baselines import LayoutSearchConfig, sample_random_valid_layout, encode_layout_to_design_vec
    from thermal_inverse_kpi import compute_steady_thermal_kpis, layout_spread_metrics
    from train_inverse import (
        ThermalInverseDesignDataset,
        build_hypergraph_plan_from_forward_prediction,
        load_forward_model,
        predict_candidate_with_forward,
    )
except Exception:  # pragma: no cover
    from .channelthermal_model_utils import resolve_demo_path, select_device
    from .design_prior_dataset import FALLBACK_KPI_NAMES
    from .layout_search_baselines import LayoutSearchConfig, sample_random_valid_layout, encode_layout_to_design_vec
    from .thermal_inverse_kpi import compute_steady_thermal_kpis, layout_spread_metrics
    from .train_inverse import (
        ThermalInverseDesignDataset,
        build_hypergraph_plan_from_forward_prediction,
        load_forward_model,
        predict_candidate_with_forward,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a ChannelThermal design-prior library.")
    parser.add_argument("--source", choices=("existing", "random", "mixed"), default="existing")
    parser.add_argument("--forward-config", type=str, default=None)
    parser.add_argument("--forward-checkpoint", type=str, default=None)
    parser.add_argument("--input-data", type=str, default=None)
    parser.add_argument("--output", type=str, default="Data_Saved/DesignPrior_Library/design_library.h5")
    parser.add_argument("--num-random-layouts", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-existing", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--query-batch-size", type=int, default=32768)
    parser.add_argument("--save-fields", action="store_true")
    parser.add_argument("--existing-sample-weight", type=float, default=1.0)
    parser.add_argument("--random-sample-weight", type=float, default=0.3)
    parser.add_argument("--max-num-modules", type=int, default=12)
    return parser.parse_args()


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with resolve_demo_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _forward_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_json(args.forward_config)
    if "forward_model" in cfg and isinstance(cfg["forward_model"], Mapping):
        cfg = dict(cfg["forward_model"])
    if args.forward_checkpoint:
        path = resolve_demo_path(args.forward_checkpoint)
        cfg["run_dir"] = str(path.parent)
        cfg["checkpoint_name"] = path.name
    cfg.setdefault("enabled", True)
    return cfg


def _behavior_from_kpis(kpis: Mapping[str, Any], layout_desc: np.ndarray, hyper_summary: Mapping[str, Any], field_desc: np.ndarray) -> np.ndarray:
    kpi_vals = [float(kpis.get(name, 0.0) or 0.0) for name in FALLBACK_KPI_NAMES]
    hyper_vals = [
        float(hyper_summary.get("active_edge_count", 0.0) or 0.0),
        float(hyper_summary.get("strength_mean", hyper_summary.get("mean_strength", 0.0)) or 0.0),
        float(hyper_summary.get("strength_max", hyper_summary.get("max_strength", 0.0)) or 0.0),
    ]
    raw = np.concatenate([np.asarray(kpi_vals, dtype=np.float32), np.asarray(hyper_vals, dtype=np.float32), layout_desc.astype(np.float32), field_desc.astype(np.float32)], axis=0)
    out = np.zeros((32,), dtype=np.float32)
    out[: min(out.size, raw.size)] = raw[: out.size]
    return out


def _field_descriptors(prediction: Mapping[str, Any]) -> np.ndarray:
    field = prediction.get("pred_field_grid")
    if field is None:
        return np.zeros((7,), dtype=np.float32)
    arr = np.asarray(field, dtype=np.float32)
    temp = arr[..., -1] if arr.ndim == 3 else arr
    finite = temp[np.isfinite(temp)]
    if finite.size == 0:
        return np.zeros((7,), dtype=np.float32)
    outlet = temp[:, int(0.92 * temp.shape[1]) :] if temp.ndim == 2 and temp.shape[1] > 1 else temp
    threshold = float(np.percentile(finite, 90.0))
    return np.asarray(
        [
            float(np.max(finite)),
            float(np.mean(finite)),
            float(np.std(finite)),
            float(np.percentile(finite, 95.0)),
            float(np.std(outlet[np.isfinite(outlet)])) if np.isfinite(outlet).any() else 0.0,
            float(np.mean(finite > threshold)),
            float(np.mean(np.maximum(finite - threshold, 0.0))),
        ],
        dtype=np.float32,
    )


def _layout_descriptors(layout: Mapping[str, Any]) -> np.ndarray:
    centers = np.asarray(layout.get("centers", []), dtype=np.float32).reshape(-1, 2)
    spread = layout_spread_metrics(centers, num_modules=int(layout.get("count", centers.shape[0])))
    return np.asarray(
        [
            float(layout.get("count", centers.shape[0])),
            float(spread.get("x_coverage", 0.0)),
            float(spread.get("y_coverage", 0.0)),
            float(spread.get("bbox_area", 0.0)),
            float(spread.get("mean_pair_distance", 0.0)),
        ],
        dtype=np.float32,
    )


def _kpis_for_prediction(record: Any, prediction: Mapping[str, Any], layout: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        kpis = compute_steady_thermal_kpis(
            prediction["pred_field_grid"],
            x_grid=record.x_grid,
            y_grid=record.y_grid,
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
    except Exception:
        kpis = {}
    centers = np.asarray(layout.get("centers", []), dtype=np.float32).reshape(-1, 2)
    kpis.update(layout_spread_metrics(centers, num_modules=int(layout.get("count", centers.shape[0]))))
    kpis["num_modules"] = int(layout.get("count", centers.shape[0]))
    return kpis


def _record_layout(record: Any) -> Dict[str, Any]:
    present = np.asarray(record.module_present).reshape(-1) > 0.5
    centers = np.asarray(record.module_centers, dtype=np.float32).reshape(-1, 2)[present]
    heat = np.asarray(record.heat_powers, dtype=np.float32).reshape(-1)[present] if getattr(record, "heat_powers", None) is not None else None
    layout = {
        "centers": centers,
        "count": int(centers.shape[0]),
        "module_radius": float(record.module_radius),
        "domain": {"domain_length_x": float(record.domain_length_x), "domain_length_y": float(record.domain_length_y), "module_radius": float(record.module_radius)},
    }
    if heat is not None:
        layout["heat_powers"] = heat
    return layout


def _context(record: Any) -> np.ndarray:
    return np.asarray([float(getattr(record, "re", 0.0)), float(getattr(record, "u_in", 0.0))], dtype=np.float32)


def _append_sample(rows: List[Dict[str, Any]], *, layout: Mapping[str, Any], record: Any, prediction: Mapping[str, Any], max_num_modules: int, source: str, weight: float, forward_checkpoint: str, save_fields: bool) -> None:
    cfg = LayoutSearchConfig(
        max_num_modules=max_num_modules,
        domain_length_x=float(record.domain_length_x),
        domain_length_y=float(record.domain_length_y),
        module_radius=float(record.module_radius),
    )
    design_vec = encode_layout_to_design_vec(layout, cfg)
    plan = build_hypergraph_plan_from_forward_prediction(
        prediction,
        max_num_modules=max_num_modules,
        domain_length_x=float(record.domain_length_x),
        domain_length_y=float(record.domain_length_y),
        num_edges=None,
    )
    hyper_vec = np.asarray(plan.get("vector", []), dtype=np.float32).reshape(-1)
    hyper_mask = np.asarray(plan.get("mask", np.ones_like(hyper_vec)), dtype=np.float32).reshape(-1)
    summary = plan.get("summary", {}) if isinstance(plan.get("summary"), Mapping) else {}
    layout_desc = _layout_descriptors(layout)
    field_desc = _field_descriptors(prediction)
    kpis = _kpis_for_prediction(record, prediction, layout)
    behavior = _behavior_from_kpis(kpis, layout_desc, summary, field_desc)
    row = {
        "design_vec": design_vec.astype(np.float32),
        "hypergraph_vec": hyper_vec.astype(np.float32),
        "hypergraph_mask": hyper_mask.astype(np.float32),
        "behavior_vec": behavior.astype(np.float32),
        "context_vec": _context(record),
        "kpi_descriptor_vec": np.asarray([float(kpis.get(name, 0.0) or 0.0) for name in FALLBACK_KPI_NAMES], dtype=np.float32),
        "layout_descriptor_vec": layout_desc.astype(np.float32),
        "sample_weight": float(weight),
        "source_tag": source,
        "case_id": str(getattr(record, "case_id", len(rows))),
        "forward_model_checkpoint_id": str(forward_checkpoint),
    }
    if save_fields and prediction.get("pred_field_grid") is not None:
        row["pred_field_grid"] = np.asarray(prediction["pred_field_grid"], dtype=np.float32)
    rows.append(row)


def _pad_stack(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    arrays = [np.asarray(row.get(key, []), dtype=np.float32).reshape(-1) for row in rows]
    width = max((arr.size for arr in arrays), default=0)
    out = np.zeros((len(arrays), width), dtype=np.float32)
    for i, arr in enumerate(arrays):
        out[i, : arr.size] = arr
    return out


def _write_h5(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for key in ("design_vec", "hypergraph_vec", "hypergraph_mask", "behavior_vec", "context_vec", "kpi_descriptor_vec", "layout_descriptor_vec"):
            h5.create_dataset(key, data=_pad_stack(rows, key), compression="gzip")
        h5.create_dataset("sample_weight", data=np.asarray([float(row.get("sample_weight", 1.0)) for row in rows], dtype=np.float32))
        for key in ("source_tag", "case_id", "forward_model_checkpoint_id"):
            values = np.asarray([str(row.get(key, "")) for row in rows], dtype=h5py.string_dtype("utf-8"))
            h5.create_dataset(key, data=values)
        h5.attrs["num_samples"] = int(len(rows))
        h5.attrs["library_version"] = 1


def main() -> int:
    args = parse_args()
    if not args.input_data:
        raise ValueError("--input-data is required so forward verification has physical context records.")
    device = select_device(args.device if args.device and args.device != "auto" else None)
    forward_model, forward_metadata, forward_path = load_forward_model(_forward_cfg(args), device)
    dataset = ThermalInverseDesignDataset(
        args.input_data,
        split=args.split,
        max_num_modules=int(args.max_num_modules),
        normalize_targets=False,
        max_cases=int(args.max_existing) if int(args.max_existing) > 0 else 0,
        use_all_if_split_missing=True,
    )
    rows: List[Dict[str, Any]] = []
    include_existing = args.source in {"existing", "mixed"}
    include_random = args.source in {"random", "mixed"} or int(args.num_random_layouts) > 0
    if include_existing:
        for record in tqdm(dataset.records, desc="existing", unit="layout"):
            layout = _record_layout(record)
            prediction = predict_candidate_with_forward(
                forward_model,
                forward_metadata,
                record,
                layout,
                device,
                max_num_modules=int(args.max_num_modules),
                query_batch_size=int(args.query_batch_size),
            )
            _append_sample(rows, layout=layout, record=record, prediction=prediction, max_num_modules=int(args.max_num_modules), source="existing", weight=float(args.existing_sample_weight), forward_checkpoint=str(forward_path), save_fields=bool(args.save_fields))
    if include_random:
        rng = np.random.default_rng(int(args.seed))
        n_random = int(args.num_random_layouts)
        if args.source == "random" and n_random <= 0:
            n_random = len(dataset.records)
        for idx in tqdm(range(max(n_random, 0)), desc="random", unit="layout"):
            record = dataset.records[idx % len(dataset.records)]
            cfg = LayoutSearchConfig(
                max_num_modules=int(args.max_num_modules),
                domain_length_x=float(record.domain_length_x),
                domain_length_y=float(record.domain_length_y),
                module_radius=float(record.module_radius),
                random_seed=int(args.seed) + idx,
            )
            layout = sample_random_valid_layout(cfg, rng, {"hard_constraints": {"num_modules": [1, min(int(args.max_num_modules), 8)]}})
            prediction = predict_candidate_with_forward(
                forward_model,
                forward_metadata,
                record,
                layout,
                device,
                max_num_modules=int(args.max_num_modules),
                query_batch_size=int(args.query_batch_size),
            )
            _append_sample(rows, layout=layout, record=record, prediction=prediction, max_num_modules=int(args.max_num_modules), source="random_forward_synthetic", weight=float(args.random_sample_weight), forward_checkpoint=str(forward_path), save_fields=bool(args.save_fields))
    if not rows:
        raise RuntimeError("No design-library rows were generated.")
    _write_h5(rows, resolve_demo_path(args.output))
    print(f"[library] wrote {len(rows)} samples to {resolve_demo_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
