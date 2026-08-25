"""Physical metrics, organizer exports, and canonical result paths."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np

from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.evaluation_tools.plots import (
    error_metrics,
    masked_error_metrics,
    module_and_fluid_masks,
)
from honf_runtime.artifact_layout import EvaluationArtifactLayout, default_evaluation_root
from honf_runtime.compat import current_timestamp, resolve_demo_path


def denormalize_predictions(predictions: Dict[str, Any], dataset: GlobalChannelThermalDataset, normalize_targets: bool) -> Dict[str, Any]:
    """Convert normalized predictions."""

    if not normalize_targets:
        return predictions
    out = dict(predictions)
    out["pred_field_grid"] = dataset.normalizer.denormalize_fields(out["pred_field_grid"])
    if np.asarray(out["pred_internal_temperature"]).size:
        out["pred_internal_temperature"] = dataset.normalizer.denormalize_internal_temperature(out["pred_internal_temperature"])
    if np.asarray(out["pred_interface"]).size:
        out["pred_interface"] = dataset.normalizer.denormalize_interface_targets(out["pred_interface"])
    return out


def safe_path_name(value: object) -> str:
    """Perform the safe path name operation used by this module."""

    raw = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw) or "case"


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for artifact provenance."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_output_dir(base_dir_arg: str | None, checkpoint_path: Path, case_id: object) -> Path:
    """Return one timestamped job under the canonical single-case evaluation root."""

    base_dir = (
        Path(base_dir_arg)
        if base_dir_arg
        else default_evaluation_root(checkpoint_path, evaluation_kind="single_case")
    )
    return resolve_demo_path(base_dir) / f"{safe_path_name(case_id)}_{current_timestamp()}"


def extract_organization_arrays(sample: Dict[str, Any], aux: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract organization arrays."""

    centers = np.asarray(sample["structure"]["module_centers"], dtype=np.float32)
    present = np.asarray(sample["structure"]["module_present"] > 0.5)
    env_coords = np.asarray(aux.get("env_coords", np.zeros((0, 2))), dtype=np.float32)
    A_eh = np.asarray(aux.get("A_eh", np.zeros((env_coords.shape[0], 1))), dtype=np.float32)
    A_mh = np.asarray(aux.get("A_mh", np.zeros((centers.shape[0], A_eh.shape[-1]))), dtype=np.float32)
    strength = np.asarray(aux.get("hyper_strength", np.ones((A_eh.shape[-1],), dtype=np.float32)), dtype=np.float32)
    active_mask = np.asarray(
        aux.get(
            "effective_edge_mask",
            aux.get("edge_active_mask", np.ones((A_eh.shape[-1],), dtype=np.float32)),
        ),
        dtype=np.float32,
    )
    return {
        "centers": centers,
        "present": present,
        "heat": np.asarray(sample["structure"].get("heat_powers", np.zeros((centers.shape[0],))), dtype=np.float32),
        "env_coords": env_coords,
        "A_eh": A_eh,
        "A_mh": A_mh,
        "strength": strength,
        "active_hyperedge_mask": active_mask,
        "module_mass": np.asarray(aux.get("hyper_module_mass", np.zeros_like(strength)), dtype=np.float32),
        "env_mass": np.asarray(aux.get("hyper_env_mass", np.zeros_like(strength)), dtype=np.float32),
        "src": np.asarray(aux.get("hyper_source_coords", np.zeros((strength.shape[0], 2))), dtype=np.float32),
        "dst": np.asarray(aux.get("hyper_thermal_region_coords", aux.get("hyper_region_coords", np.zeros((strength.shape[0], 2)))), dtype=np.float32),
    }


def _entropy(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Perform the entropy operation used by this module."""

    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, 1.0e-12, None)
    return -np.sum(arr * np.log(arr), axis=axis)


def hypergraph_diagnostics(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Perform the hypergraph diagnostics operation used by this module."""

    aux = predictions["organizer_aux"]
    base = predictions.get("base_organizer_aux", {})
    A_mh = np.asarray(aux.get("A_mh", np.zeros((0, 0))), dtype=np.float64)
    A_eh = np.asarray(aux.get("A_eh", np.zeros((0, 0))), dtype=np.float64)
    strength = np.asarray(aux.get("hyper_strength", np.zeros((0,))), dtype=np.float64)
    module_mass = np.asarray(aux.get("hyper_module_mass", np.zeros_like(strength)), dtype=np.float64)
    env_mass = np.asarray(aux.get("hyper_env_mass", np.zeros_like(strength)), dtype=np.float64)
    num_h = max(int(strength.shape[0]), 1)
    static = {
        "active_edge_count": float(np.sum(strength > 0.05)),
        "A_mh_entropy": float(np.mean(_entropy(A_mh, axis=-1))) if A_mh.size else 0.0,
        "A_eh_entropy": float(np.mean(_entropy(A_eh, axis=-1))) if A_eh.size else 0.0,
        "module_mass_entropy_norm": float(_entropy(module_mass, axis=-1) / np.log(max(num_h, 2))) if module_mass.size else 0.0,
        "env_mass_entropy_norm": float(_entropy(env_mass, axis=-1) / np.log(max(num_h, 2))) if env_mass.size else 0.0,
        "module_mass_max": float(np.max(module_mass)) if module_mass.size else 0.0,
        "env_mass_max": float(np.max(env_mass)) if env_mass.size else 0.0,
        "hyper_strength_mean": float(np.mean(strength)) if strength.size else 0.0,
        "hyper_strength_max": float(np.max(strength)) if strength.size else 0.0,
    }
    for key in (
        "pre_fallback_zero_support_module_rows",
        "post_fallback_zero_support_module_rows",
        "pre_fallback_zero_support_environment_rows",
        "post_fallback_zero_support_environment_rows",
    ):
        if key in aux:
            static[key] = float(np.mean(np.asarray(aux[key], dtype=np.float64)))
    routing_maps = predictions.get("routing_maps", {})
    routing_aux = predictions.get("routing_aux", {})
    routing = {
        key: float(routing_aux[key])
        for key in (
            "routed_module_retained_mass_mean",
            "routed_module_retained_mass_p05",
            "routed_module_retained_mass_min",
            "routed_query_edge_pair_count",
        )
        if key in routing_aux
    }
    if routing_maps:
        alpha = np.asarray(routing_maps.get("query_hyper_attention", np.zeros((0, 0, 0))), dtype=np.float64)
        pair = np.asarray(routing_maps.get("pairwise_edge_contribution", np.zeros_like(alpha)), dtype=np.float64)
        c_h = np.asarray(routing_maps.get("c_H_norm", np.zeros((0, 0))), dtype=np.float64)
        c_pair = np.asarray(routing_maps.get("c_pair_norm", np.zeros((0, 0))), dtype=np.float64)
        if alpha.size:
            entropy = _entropy(alpha, axis=-1)
            routing.update(
                {
                    "query_attention_entropy": float(np.mean(entropy)),
                    "query_attention_effective_edges": float(np.mean(np.exp(entropy))),
                    "query_attention_max": float(np.mean(np.max(alpha, axis=-1))),
                    "pairwise_edge_contribution_mean": float(np.mean(pair)) if pair.size else 0.0,
                    "c_H_norm_mean": float(np.mean(c_h)) if c_h.size else 0.0,
                    "c_pair_norm_mean": float(np.mean(c_pair)) if c_pair.size else 0.0,
                }
            )
    changes = {}
    for key, out_key in (
        ("A_mh", "A_mh_change_norm"),
        ("A_eh", "A_eh_change_norm"),
        ("hyper_source_coords", "source_coordinate_shift"),
        ("hyper_region_coords", "region_coordinate_shift"),
        ("hyper_module_mass", "module_mass_shift"),
        ("hyper_env_mass", "env_mass_shift"),
        ("hyper_strength", "strength_shift"),
    ):
        if key in aux and key in base:
            final_arr = np.asarray(aux[key], dtype=np.float64)
            base_arr = np.asarray(base[key], dtype=np.float64)
            if final_arr.shape == base_arr.shape:
                changes[out_key] = float(np.linalg.norm(final_arr - base_arr))
    return {
        "static_organization": static,
        "routing": routing,
        "base_vs_final": changes,
        "note": "Organization/routing/plan diagnostics are computed from predicted mode when both teacher and predicted modes are evaluated.",
    }


def summarize(
    raw_sample: Dict[str, Any],
    predictions: Dict[str, Any],
    checkpoint_path: Path,
    layout: EvaluationArtifactLayout,
    channel_order: list[str],
) -> Dict[str, Any]:
    """Perform the summarize operation used by this module."""

    pred = predictions["pred_field_grid"]
    gt = raw_sample["steady_field"][..., : pred.shape[-1]]
    _, fluid_mask = module_and_fluid_masks(raw_sample, pred)
    suffix = str(predictions.get("suffix", "predicted"))
    layout.ensure("arrays", "metrics")
    npz_path = layout.arrays / f"evaluation_outputs_{suffix}.npz"
    np.savez_compressed(
        npz_path,
        pred_field_grid=pred.astype(np.float32),
        gt_field_grid=gt.astype(np.float32),
        pred_internal_temperature=predictions["pred_internal_temperature"].astype(np.float32),
        pred_interface=predictions["pred_interface"].astype(np.float32),
        pred_port_condition=predictions["pred_port_condition"].astype(np.float32),
    )
    channel_metrics = {
        str(name): masked_error_metrics(pred[..., idx], gt[..., idx], fluid_mask)
        for idx, name in enumerate(channel_order[: pred.shape[-1]])
    }
    field_metrics = error_metrics(pred, gt)
    field_metrics_fluid = masked_error_metrics(pred, gt, fluid_mask)
    temperature_metrics_fluid = masked_error_metrics(pred[..., 4], gt[..., 4], fluid_mask) if pred.shape[-1] >= 5 else None
    metrics_csv_path = layout.metrics / f"metrics_{suffix}.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "suffix",
                "field_mse",
                "field_rmse",
                "field_mae",
                "field_relative_l2",
                "fluid_mse",
                "fluid_rmse",
                "fluid_mae",
                "fluid_relative_l2",
                "temperature_fluid_mse",
                "temperature_fluid_rmse",
                "temperature_fluid_mae",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": str(raw_sample["case_id"]),
                "suffix": suffix,
                "field_mse": field_metrics.get("mse"),
                "field_rmse": field_metrics.get("rmse"),
                "field_mae": field_metrics.get("mae"),
                "field_relative_l2": field_metrics.get("relative_l2"),
                "fluid_mse": field_metrics_fluid.get("mse"),
                "fluid_rmse": field_metrics_fluid.get("rmse"),
                "fluid_mae": field_metrics_fluid.get("mae"),
                "fluid_relative_l2": field_metrics_fluid.get("relative_l2"),
                "temperature_fluid_mse": None if temperature_metrics_fluid is None else temperature_metrics_fluid.get("mse"),
                "temperature_fluid_rmse": None if temperature_metrics_fluid is None else temperature_metrics_fluid.get("rmse"),
                "temperature_fluid_mae": None if temperature_metrics_fluid is None else temperature_metrics_fluid.get("mae"),
            }
        )
    return {
        "checkpoint": str(checkpoint_path),
        "case_id": str(raw_sample["case_id"]),
        "phase": "prompt3_physical_coupling",
        "field_metrics": field_metrics,
        "field_metrics_fluid": field_metrics_fluid,
        "temperature_metrics_fluid": temperature_metrics_fluid,
        "field_channel_metrics_fluid": channel_metrics,
        "internal_interface_note": "Skipped only when internal/interface tensors are empty.",
        "interface_flux_mode": str(predictions.get("interface_flux_mode", "unknown")),
        "outputs": {
            "global_field_quicklook": str(layout.fields / f"global_field_quicklook_{suffix}.png"),
            "npz": str(npz_path),
            "metrics_csv": str(metrics_csv_path),
        },
    }
