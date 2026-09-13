"""Checkpoint-owned sampled WindFarm validation/test evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_forward_core.config import UnifiedForwardConfig
from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_runtime.compat import load_trusted_checkpoint, select_device
from honf_runtime.paths import resolve_path
from torch.utils.data import DataLoader

from ..data import COMPACT_GEOMETRY_KEYS, WindFarmNativeDataset, WindFarmNativeView, case_batch, collate_windfarm
from ..geometry import ENV_TOKEN_SHAPE
from ..model import WindFarmForwardModel
from ..normalization import VelocityNormalizer, VerticalProfileBaseline
from ..splits import GroupSplit, make_group_split
from ..study_spatial import downstream_envelope
from .train_forward import _as_device_batch, _split_from_checkpoint


def _compact_metadata(path: str | Path) -> dict[str, np.ndarray]:
    compact_path = resolve_path(str(path))
    if not compact_path.is_file():
        raise FileNotFoundError(f"WindFarm compact metadata resource not found: {compact_path}")
    with np.load(compact_path, allow_pickle=False) as archive:
        missing = [name for name in COMPACT_GEOMETRY_KEYS if name not in archive.files]
        if missing:
            raise ValueError(f"WindFarm compact metadata is missing keys {missing}: {compact_path}")
        return {name: np.asarray(archive[name]).copy() for name in COMPACT_GEOMETRY_KEYS}


def _load_split(view: WindFarmNativeView, derived_view: Path) -> GroupSplit:
    path = derived_view / "split_indices.npz"
    if not path.is_file():
        return make_group_split(np.asarray(view.volume.array("layout_index")), seed=42)
    with np.load(path, allow_pickle=False) as archive:
        split = GroupSplit(
            train=np.asarray(archive["train"], dtype=np.int64),
            validation=np.asarray(archive["validation"], dtype=np.int64),
            test=np.asarray(archive["test"], dtype=np.int64),
            metadata={},
        )
    canonical = make_group_split(np.asarray(view.volume.array("layout_index")), seed=42)
    if not (
        np.array_equal(split.train, canonical.train)
        and np.array_equal(split.validation, canonical.validation)
        and np.array_equal(split.test, canonical.test)
    ):
        raise ValueError("Stored WindFarm split indices do not match seed-42 layout grouping.")
    return canonical


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    materialization_batch: Any,
) -> tuple[WindFarmForwardModel, dict[str, Any]]:
    """Load a WindFarm checkpoint with trusted loading and strict state shape."""

    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = load_trusted_checkpoint(path, map_location="cpu")
    validate_checkpoint_identity(
        checkpoint,
        case_id="WindFarm",
        model_family="honf_forward",
        workflow="forward",
    )
    model_payload = checkpoint.get("model_config")
    if not isinstance(model_payload, Mapping):
        raise TypeError("WindFarm checkpoint lacks its resolved model_config.")
    model_config = UnifiedForwardConfig.from_dict(dict(model_payload))
    normalization_payload = checkpoint.get("normalization")
    if not isinstance(normalization_payload, Mapping):
        raise TypeError("WindFarm checkpoint lacks training-owned velocity normalization.")
    normalizer = VelocityNormalizer.from_dict(dict(normalization_payload))
    model = WindFarmForwardModel(model_config, velocity_transform=normalizer)
    target_device = torch.device(device)
    model = model.to(target_device)
    batch = materialization_batch
    if not hasattr(batch, "to"):
        batch = _as_device_batch(batch, target_device)
    else:
        batch = batch.to(target_device)
    model.materialize(batch)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def predict_standardized(
    model: WindFarmForwardModel,
    prepared: Any,
    query_coords: torch.Tensor,
    query_features: torch.Tensor,
    *,
    receiver_chunk_size: int = 128,
) -> torch.Tensor:
    """Public evaluation helper for standardized velocity predictions."""

    with torch.no_grad():
        return model.predict_standardized(
            prepared,
            query_coords,
            query_features,
            receiver_chunk_size=int(receiver_chunk_size),
        )


def predict_physical(
    model: WindFarmForwardModel,
    prepared: Any,
    query_coords: torch.Tensor,
    query_features: torch.Tensor,
    *,
    receiver_chunk_size: int = 128,
) -> torch.Tensor:
    """Public evaluation helper for m/s velocity predictions."""

    with torch.no_grad():
        return model.predict_physical(
            prepared,
            query_coords,
            query_features,
            receiver_chunk_size=int(receiver_chunk_size),
        )


def _case_metrics(
    *,
    raw_batch: Mapping[str, Any],
    prediction_physical: np.ndarray,
    prediction_standardized: np.ndarray,
    normalizer: VelocityNormalizer,
    baseline: VerticalProfileBaseline | None,
    volume_queries: int,
    module_present: np.ndarray,
) -> dict[str, Any]:
    target_standardized = np.asarray(raw_batch["target_field"], dtype=np.float32)
    target_physical = normalizer.denormalize(target_standardized)
    pred_phys = np.asarray(prediction_physical, dtype=np.float32)
    pred_std = np.asarray(prediction_standardized, dtype=np.float32)
    query_coords = np.asarray(raw_batch["query_xy"], dtype=np.float32)
    if target_standardized.shape != pred_std.shape or target_standardized.ndim != 2:
        raise ValueError("WindFarm endpoint metrics expect [Q,3] case arrays.")
    volume_queries = int(volume_queries)
    if not 0 < volume_queries < target_standardized.shape[0]:
        raise ValueError("volume_queries must split the sampled endpoint points.")
    metadata = dict(raw_batch["metadata"])
    centers = np.asarray(raw_batch.get("module_centers", np.empty((0, 3))), dtype=np.float32)
    active_centers = centers[np.asarray(module_present, dtype=np.float32) > 0.5]
    support_volume_D3 = float(metadata.get("support_volume_D3", np.nan))
    if not np.isfinite(support_volume_D3):
        support_volume_D3 = float(metadata["support_volume_m3"]) / 80.0**3
    support_extent_value = metadata.get("support_extent_D")
    if support_extent_value is None:
        global_context = np.asarray(raw_batch.get("global_context", []), dtype=np.float64).reshape(-1)
        support_extent_value = (
            global_context[8:11] * np.asarray((50.0, 38.0, 6.25), dtype=np.float64)
            if global_context.size >= 11 else [np.nan, np.nan, np.nan]
        )
    support_extent = np.asarray(support_extent_value, dtype=np.float64)
    finite_extent = np.all(np.isfinite(support_extent))
    aspect = float(np.max(support_extent[:2]) / np.min(support_extent[:2])) if finite_extent and np.all(support_extent[:2] > 0) else None
    if active_centers.shape[0] >= 2:
        deltas = active_centers[:, None, :2] - active_centers[None, :, :2]
        distances = np.sqrt(np.sum(deltas**2, axis=-1))
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        spacing = float(np.min(nearest))
        mean_nearest = float(np.mean(nearest))
        nearest_dispersion = float(np.std(nearest))
    else:
        spacing = mean_nearest = nearest_dispersion = None
    result: dict[str, Any] = {
        "source_index": int(metadata["source_index"]),
        "case": str(metadata["case"]),
        "layout_index": int(metadata["layout_index"]),
        "wind_direction_deg": float(metadata["wind_direction_deg"]),
        "n_turbines": int(metadata.get("n_turbines", int(np.asarray(module_present).sum()))),
        "shape_nxyz": list(metadata["shape_nxyz"]),
        "support_volume_m3": float(metadata["support_volume_m3"]),
        "support_volume_D3": support_volume_D3,
        "support_aspect_xy": aspect,
        "minimum_turbine_spacing_D": spacing,
        # These are evaluation-only geometry descriptors.  They are computed
        # from the row-specific active turbine centers and never enter model
        # inputs or target transforms.
        "min_sep_D": spacing,
        "mean_nn_D": mean_nearest,
        "nn_dispersion": nearest_dispersion,
    }
    slices: dict[str, np.ndarray] = {
        "volume": np.arange(0, volume_queries, dtype=np.int64),
        "hub_band": np.arange(volume_queries, target_standardized.shape[0], dtype=np.int64),
    }
    # The downstream envelope is reported conditionally on volume samples;
    # including the separately sampled hub band would make this stratum's
    # mixture weight depend on the endpoint sampler rather than geometry.
    wake = downstream_envelope(query_coords[:volume_queries], active_centers)
    slices["downstream_envelope"] = np.flatnonzero(wake)
    result["downstream_envelope_source"] = "volume_queries_only"
    raw_measure = raw_batch.get("query_measure_m3")
    if raw_measure is None:
        raise ValueError("endpoint query_measure_m3 native quadrature weights are required for volume integrals")
    point_weights = np.asarray(raw_measure, dtype=np.float64).reshape(-1)
    if (
        point_weights.shape != (target_standardized.shape[0],)
        or not np.all(np.isfinite(point_weights))
        or np.any(point_weights <= 0.0)
    ):
        raise ValueError("endpoint query_measure_m3 must align with finite positive query coordinates")
    for name, selected in slices.items():
        minimum_region_count = 32 if name == "downstream_envelope" else 1
        if selected.size < minimum_region_count:
            result[f"{name}_available"] = False
            result[f"{name}_count"] = int(selected.size)
            result[f"{name}_unavailable_reason"] = (
                "no points" if selected.size == 0 else f"fewer than {minimum_region_count} volume samples"
            )
            continue
        result[f"{name}_available"] = True
        result[f"{name}_count"] = int(selected.size)
        result[f"{name}_weight_sum"] = float(np.sum(point_weights[selected]))
        target = target_physical[selected]
        pred = pred_phys[selected]
        error = pred - target
        std_error = pred_std[selected] - target_standardized[selected]
        result[f"{name}_standardized_mse"] = float(np.mean(std_error**2))
        result[f"{name}_rmse_mps"] = float(np.sqrt(np.mean(error**2)))
        result[f"{name}_mae_mps"] = float(np.mean(np.abs(error)))
        result[f"{name}_channel_rmse_mps"] = np.sqrt(np.mean(error**2, axis=0)).astype(float).tolist()
        result[f"{name}_channel_mae_mps"] = np.mean(np.abs(error), axis=0).astype(float).tolist()
        result[f"{name}_channel_mae_over_u_ref"] = (
            np.mean(np.abs(error), axis=0) / float(normalizer.u_ref_mps)
        ).astype(float).tolist()
        result[f"{name}_channel_rmse_over_u_ref"] = (
            np.sqrt(np.mean(error**2, axis=0)) / float(normalizer.u_ref_mps)
        ).astype(float).tolist()
        result[f"{name}_target_energy_mps2"] = float(np.sum(target**2))
        target_energy = float(np.sum(target**2))
        result[f"{name}_vector_relative_l2"] = (
            float(np.sqrt(np.sum(error**2) / target_energy)) if target_energy > 0.0 else None
        )
        weighted_error = point_weights[selected, None] * error
        weighted_target = point_weights[selected, None] * target
        result[f"{name}_error_squared_integral"] = np.sum(weighted_error * error, axis=0).astype(float).tolist()
        result[f"{name}_absolute_error_integral"] = np.sum(np.abs(weighted_error), axis=0).astype(float).tolist()
        result[f"{name}_target_energy_integral"] = np.sum(weighted_target * target, axis=0).astype(float).tolist()
        if baseline is not None:
            coords = query_coords[selected]
            baseline_values = baseline.predict(coords[:, 2])
            baseline_error = baseline_values - target
            result[f"{name}_baseline_rmse_mps"] = float(np.sqrt(np.mean(baseline_error**2)))
            baseline_std = normalizer.normalize(baseline_values)
            result[f"{name}_baseline_standardized_mse"] = float(
                np.mean((baseline_std - target_standardized[selected]) ** 2)
            )
            target_residual = target[:, 0] - baseline_values[:, 0]
            prediction_residual = pred[:, 0] - baseline_values[:, 0]
            residual_error = prediction_residual - target_residual
            residual_energy = float(np.sum(target_residual**2))
            result[f"{name}_baseline_ux_residual_rmse_mps"] = float(np.sqrt(np.mean(residual_error**2)))
            result[f"{name}_baseline_ux_residual_energy_mps2"] = residual_energy
            result[f"{name}_baseline_ux_residual_relative_l2"] = (
                float(np.sqrt(np.sum(residual_error**2) / residual_energy))
                if residual_energy > 0.0 else None
            )
    return result


def evaluate_rows(
    model: WindFarmForwardModel,
    view: WindFarmNativeView,
    rows: Iterable[int],
    *,
    normalizer: VelocityNormalizer,
    q_volume: int,
    q_band: int,
    seed: int = 42,
    baseline: VerticalProfileBaseline | None = None,
    batch_size: int = 1,
    receiver_chunk_size: int = 128,
    strata_reference_rows: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Evaluate selected rows with fixed native samples and equal-case metrics."""

    total = int(q_volume) + int(q_band)
    fraction = int(q_volume) / max(total, 1)
    dataset = WindFarmNativeDataset(
        view,
        tuple(int(row) for row in rows),
        normalizer=normalizer,
        queries_per_case=total,
        volume_fraction=fraction,
        seed=int(seed),
        fixed_sampling=True,
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=collate_windfarm)
    case_rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = _as_device_batch(raw_batch, next(model.parameters()).device)
        with torch.no_grad():
            prepared = model.prepare_case(batch)
            prediction_std = model.predict_standardized(
                prepared,
                batch.query_xy,
                batch.query_features,
                receiver_chunk_size=int(receiver_chunk_size),
            )
            prediction_phys = model.velocity_transform
            if prediction_phys is None:
                raise ValueError("WindFarm evaluation model lacks its target transform.")
            prediction_phys_array = prediction_phys.denormalize(prediction_std.detach().cpu().numpy())
            prediction_std_array = prediction_std.detach().cpu().numpy()
        for index, metadata in enumerate(raw_batch["metadata"]):
            single_raw = {
                "target_field": np.asarray(raw_batch["target_field"])[index],
                "query_xy": np.asarray(raw_batch["query_xy"])[index],
                "module_centers": np.asarray(raw_batch["module_centers"])[index],
                "global_context": np.asarray(raw_batch["global_context"])[index],
                "query_loss_weight": (
                    None
                    if raw_batch.get("query_loss_weight") is None
                    else np.asarray(raw_batch["query_loss_weight"])[index]
                ),
                "query_measure_m3": (
                    None
                    if raw_batch.get("query_measure_m3") is None
                    else np.asarray(raw_batch["query_measure_m3"])[index]
                ),
                "metadata": metadata,
            }
            case_rows.append(
                _case_metrics(
                    raw_batch=single_raw,
                    prediction_physical=prediction_phys_array[index],
                    prediction_standardized=prediction_std_array[index],
                    normalizer=normalizer,
                    baseline=baseline,
                    volume_queries=int(q_volume),
                    module_present=np.asarray(raw_batch["module_present"])[index],
                )
            )
    if not case_rows:
        raise ValueError("No WindFarm rows were evaluated.")
    summary: dict[str, Any] = {
        "rows": len(case_rows),
        "cases": case_rows,
        "equal_case": {},
    }
    metric_keys = (
        "volume_standardized_mse", "hub_band_standardized_mse", "volume_rmse_mps", "hub_band_rmse_mps",
        "volume_mae_mps", "hub_band_mae_mps", "volume_vector_relative_l2", "hub_band_vector_relative_l2",
        "volume_baseline_rmse_mps", "hub_band_baseline_rmse_mps",
        "volume_baseline_standardized_mse", "hub_band_baseline_standardized_mse",
        "downstream_envelope_rmse_mps", "downstream_envelope_standardized_mse",
    )
    for key in metric_keys:
        values = np.asarray(
            [row[key] for row in case_rows if row.get(key) is not None and np.isfinite(row[key])],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        summary["equal_case"][f"mean_{key}"] = float(values.mean())
        summary["equal_case"][f"median_{key}"] = float(np.median(values))
        summary["equal_case"][f"p95_{key}"] = float(np.percentile(values, 95.0))
        summary["equal_case"][f"worst_{key}"] = float(values.max())
    pooled: dict[str, Any] = {}
    for name in ("volume", "hub_band", "downstream_envelope"):
        available = [row for row in case_rows if row.get(f"{name}_available")]
        if not available:
            pooled[name] = {"available": False, "count": 0}
            continue
        total_weight = float(sum(float(row[f"{name}_weight_sum"]) for row in available))
        squared = np.sum(
            np.asarray([row[f"{name}_error_squared_integral"] for row in available], dtype=np.float64),
            axis=0,
        )
        absolute = np.sum(
            np.asarray([row[f"{name}_absolute_error_integral"] for row in available], dtype=np.float64),
            axis=0,
        )
        energy = np.sum(
            np.asarray([row[f"{name}_target_energy_integral"] for row in available], dtype=np.float64),
            axis=0,
        )
        pooled[name] = {
            "available": True,
            "count": int(sum(int(row[f"{name}_count"]) for row in available)),
            "weight_sum": total_weight,
            "channel_rmse_mps": np.sqrt(squared / total_weight).tolist(),
            "channel_mae_mps": (absolute / total_weight).tolist(),
            "channel_rmse_over_u_ref": (np.sqrt(squared / total_weight) / float(normalizer.u_ref_mps)).tolist(),
            "channel_mae_over_u_ref": (absolute / total_weight / float(normalizer.u_ref_mps)).tolist(),
            "vector_relative_l2": float(np.sqrt(squared.sum() / energy.sum())) if float(energy.sum()) > 0.0 else None,
            "error_squared_integral": squared.tolist(),
            "target_energy_integral": energy.tolist(),
        }
    summary["pooled"] = pooled
    layouts: dict[int, list[float]] = {}
    for row in case_rows:
        layouts.setdefault(int(row["layout_index"]), []).append(float(row["volume_rmse_mps"]))
    summary["layout_level"] = {
        "layouts": len(layouts),
        "mean_direction_rmse_mps": float(np.mean([np.mean(values) for values in layouts.values()])),
        "per_layout": {
            str(layout): {
                "directions": len(values),
                "mean_volume_rmse_mps": float(np.mean(values)),
                "max_volume_rmse_mps": float(np.max(values)),
            }
            for layout, values in layouts.items()
        },
    }
    reference_rows = tuple(int(row) for row in (rows if strata_reference_rows is None else strata_reference_rows))
    reference_volumes = np.asarray(
        [float(view.run(row).support.volume_D3) for row in reference_rows], dtype=np.float64
    )
    if reference_volumes.size == 0 or not np.all(np.isfinite(reference_volumes)):
        raise ValueError("strata_reference_rows must provide finite native support volumes")
    volumes = np.asarray([float(row["support_volume_D3"]) for row in case_rows], dtype=np.float64)
    quartile = np.asarray(
        np.searchsorted(np.quantile(reference_volumes, [0.25, 0.5, 0.75]), volumes, side="right"), dtype=np.int64
    )
    reference_geometry: list[dict[str, float | int | None]] = []
    for reference_row in reference_rows:
        reference_case = view.run(reference_row)
        reference_centers = np.asarray(reference_case.module_centers, dtype=np.float64)
        if reference_centers.shape[0] >= 2:
            reference_delta = reference_centers[:, None, :2] - reference_centers[None, :, :2]
            reference_distances = np.sqrt(np.sum(reference_delta**2, axis=-1))
            np.fill_diagonal(reference_distances, np.inf)
            reference_nearest = np.min(reference_distances, axis=1)
            reference_min_sep = float(np.min(reference_nearest))
            reference_mean_nn = float(np.mean(reference_nearest))
            reference_nn_dispersion = float(np.std(reference_nearest))
        else:
            reference_min_sep = reference_mean_nn = reference_nn_dispersion = None
        extent = np.asarray(reference_case.support.extent_D, dtype=np.float64)
        reference_geometry.append(
            {
                "n_turbines": int(reference_case.n_turbines),
                "support_aspect_xy": float(np.max(extent[:2]) / np.min(extent[:2])),
                "min_sep_D": reference_min_sep,
                "mean_nn_D": reference_mean_nn,
                "nn_dispersion": reference_nn_dispersion,
            }
        )

    def _quartile_edges(name: str) -> list[float] | None:
        values = np.asarray(
            [float(item[name]) for item in reference_geometry if item[name] is not None], dtype=np.float64
        )
        if values.size == 0 or not np.all(np.isfinite(values)):
            return None
        return np.quantile(values, [0.25, 0.5, 0.75]).astype(float).tolist()

    aspect_edges = _quartile_edges("support_aspect_xy")
    descriptor_edges = {
        name: _quartile_edges(name) for name in ("min_sep_D", "mean_nn_D", "nn_dispersion")
    }
    strata: dict[str, list[int]] = {}

    def _m_bin_label(count: int) -> str:
        if 6 <= count <= 30:
            lower = 6 + 5 * ((count - 6) // 5)
            return f"{lower}-{lower + 4}"
        return str(count)

    for index, row in enumerate(case_rows):
        keys = {
            f"M={int(row['n_turbines'])}": index,
            f"M_bin={_m_bin_label(int(row['n_turbines']))}": index,
            f"direction={float(row['wind_direction_deg']):g}": index,
            f"volume_quartile={int(quartile[index]) + 1}": index,
        }
        if row.get("support_aspect_xy") is not None and aspect_edges is not None:
            aspect_bin = int(np.searchsorted(aspect_edges, float(row["support_aspect_xy"]), side="right")) + 1
            keys[f"aspect_xy_bin={aspect_bin}"] = index
        for descriptor_name, edges in descriptor_edges.items():
            value = row.get(descriptor_name)
            if value is not None and edges is not None:
                descriptor_bin = int(np.searchsorted(edges, float(value), side="right")) + 1
                keys[f"{descriptor_name}_bin={descriptor_bin}"] = index
        for key in keys:
            strata.setdefault(key, []).append(index)
    summary["strata"] = {}
    summary["strata_reference"] = {
        "scope": "training rows" if strata_reference_rows is not None else "evaluated rows",
        "rows": int(reference_volumes.size),
        "support_volume_D3_quartiles": np.quantile(reference_volumes, [0.25, 0.5, 0.75]).tolist(),
        "m_bins": ["6-10", "11-15", "16-20", "21-25", "26-30"],
        "support_aspect_xy_quartiles": aspect_edges,
        "descriptor_quartiles": descriptor_edges,
        "descriptor_definitions": {
            "min_sep_D": "minimum pairwise turbine-center spacing in rotor diameters",
            "mean_nn_D": "mean per-turbine nearest-neighbor spacing in rotor diameters",
            "nn_dispersion": "standard deviation of per-turbine nearest-neighbor spacing in rotor diameters",
        },
    }
    for key, indexes in sorted(strata.items()):
        values = np.asarray([case_rows[index]["volume_rmse_mps"] for index in indexes], dtype=np.float64)
        summary["strata"][key] = {
            "cases": len(indexes),
            "mean_volume_rmse_mps": float(values.mean()),
            "median_volume_rmse_mps": float(np.median(values)),
            "max_volume_rmse_mps": float(values.max()),
        }
    return summary


def _write_results(output_dir: Path, result: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(result.get("cases", []))
    if rows:
        fields = list(rows[0])
        with (output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def _parse_extra(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--split", default=None)
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--volume-queries", type=int, default=None)
    parser.add_argument("--band-queries", type=int, default=None)
    parser.add_argument("--case-batch-size", type=int, default=1)
    parser.add_argument("--receiver-chunk-size", type=int, default=None)
    parser.add_argument("--study-mode", choices=("native", "diagnostics", "timing"), default=None)
    parser.add_argument("--Run_ID", action="append", default=[])
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_known_args(list(argv))[0]


def evaluate_cli(
    *,
    config: Mapping[str, Any],
    checkpoint: str | None,
    volume_path: Path,
    derived_view: Path,
    device: str | None,
    output_dir: str | None,
    compact_path: Path | None = None,
    argv: Iterable[str] = (),
    workflow: str = "forward",
) -> int:
    """Entry point called by :class:`WindFarmPlugin` after root resolution."""

    args = _parse_extra(argv)
    if checkpoint is None:
        raise ValueError("WindFarm evaluation requires a selected checkpoint path.")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_config = payload.get("train_config")
    checkpoint_dataset = checkpoint_config.get("dataset", {}) if isinstance(checkpoint_config, Mapping) else {}
    checkpoint_evaluation = checkpoint_config.get("evaluation", {}) if isinstance(checkpoint_config, Mapping) else {}
    dataset_cfg = dict(checkpoint_dataset) if isinstance(checkpoint_dataset, Mapping) else dict(config.get("dataset", {}))
    evaluation_cfg = (
        dict(checkpoint_evaluation)
        if isinstance(checkpoint_evaluation, Mapping)
        else dict(config.get("evaluation", {}))
    )
    owned_volume = dataset_cfg.get("volume_path")
    owned_compact = dataset_cfg.get("compact_path")
    owned_derived = dataset_cfg.get("derived_view")
    if owned_volume is not None:
        volume_path = resolve_path(str(owned_volume))
    if owned_compact is not None:
        compact_path = resolve_path(str(owned_compact))
    if owned_derived is not None:
        derived_view = resolve_path(str(owned_derived))
    if args.study_mode is not None:
        from .study_evidence import run_study

        return int(
            run_study(
                checkpoint=checkpoint_path,
                volume_path=volume_path,
                derived_view=derived_view,
                device=select_device(device),
                output_dir=output_dir or checkpoint_path.parent / "evaluations" / args.study_mode,
                mode=args.study_mode,
                compact_path=compact_path,
            )
        )
    compact_metadata = None if compact_path is None else _compact_metadata(compact_path)
    view = WindFarmNativeView(
        volume_path,
        compact_metadata=compact_metadata,
        token_shape=tuple(dataset_cfg.get("env_token_shape", ENV_TOKEN_SHAPE)),
    )
    split = (
        _split_from_checkpoint(view, payload)
        if isinstance(payload.get("split_indices"), Mapping)
        else _load_split(view, resolve_path(str(derived_view)))
    )
    selected_split = str(args.split or evaluation_cfg.get("split", "validation"))
    if selected_split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown WindFarm evaluation split {selected_split!r}.")
    rows = getattr(split, selected_split)
    prefix = "validation" if selected_split == "validation" else selected_split
    q_volume = int(args.volume_queries or evaluation_cfg.get(f"{prefix}_volume_queries", dataset_cfg.get("q_volume", 768)))
    q_band = int(args.band_queries or evaluation_cfg.get(f"{prefix}_band_queries", dataset_cfg.get("q_band", 256)))
    normalization_payload = payload.get("normalization")
    if not isinstance(normalization_payload, Mapping):
        raise TypeError("Selected WindFarm checkpoint lacks normalization metadata.")
    normalizer = VelocityNormalizer.from_dict(dict(normalization_payload))
    materialization_case = view.run(int(split.train[0]))
    materialization_batch = case_batch(materialization_case, materialization_case.module_centers)
    target_device = select_device(device)
    model, payload = load_checkpoint(
        checkpoint_path,
        device=target_device,
        materialization_batch=materialization_batch,
    )
    profile_payload = payload.get("vertical_profile_baseline")
    baseline = None if not isinstance(profile_payload, Mapping) else VerticalProfileBaseline.from_dict(dict(profile_payload))
    if workflow == "compare":
        run_ids = args.Run_ID
        if not run_ids:
            raise ValueError("WindFarm compare evaluation requires repeated --Run_ID values.")
        raise ValueError("WindFarm compare is intentionally handled by the caller with explicit checkpoint paths.")
    result = evaluate_rows(
        model,
        view,
        rows,
        normalizer=normalizer,
        q_volume=q_volume,
        q_band=q_band,
        seed=int(dataset_cfg.get("sample_seed", 42)),
        baseline=baseline,
        batch_size=max(int(args.case_batch_size), 1),
        strata_reference_rows=split.train,
        receiver_chunk_size=int(
            args.receiver_chunk_size
            or payload.get("model_config", {}).get("interface_model", {}).get("receiver_chunk_size", 128)
        ),
    )
    result.update({"checkpoint": str(checkpoint_path), "split": selected_split, "workflow": workflow})
    target = Path(output_dir).expanduser().resolve() if output_dir else checkpoint_path.parent / "evaluations" / selected_split
    _write_results(target, result)
    print(f"[windfarm-eval] split={selected_split} rows={result['rows']} output={target}")
    return 0


__all__ = [
    "evaluate_cli",
    "evaluate_rows",
    "load_checkpoint",
    "predict_physical",
    "predict_standardized",
]
