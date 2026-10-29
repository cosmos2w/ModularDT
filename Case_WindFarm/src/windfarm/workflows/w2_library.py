"""Validation-only W2 selection among input-matched stored WindFarm layouts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_runtime.compat import load_trusted_checkpoint, select_device
from honf_runtime.paths import resolve_path

from ..data import COMPACT_GEOMETRY_KEYS, WindFarmNativeView, case_batch
from ..decision_library import (
    build_input_only_w2_cohorts,
    fit_grouped_ridge_head,
    regression_metrics,
    score_finite_library,
    spearman_rank_correlation,
    static_geometry_features,
)
from ..geometry import (
    ANCHOR_ROLE_ENVIRONMENT,
    ANCHOR_ROLE_ROTOR_EDGE,
    ANCHOR_ROLE_ROTOR_HUB,
    ANCHOR_ROLE_ROTOR_INTERIOR,
    ENV_TOKEN_SHAPE,
    windfarm_receiver_anchors,
)
from ..model import WindFarmForwardModel, build_windfarm_forward_config
from ..normalization import VelocityNormalizer
from ..splits import GroupSplit, make_group_split

_STATIC_CV_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def _geometry_metadata(compact_path: Path) -> dict[str, np.ndarray]:
    if not compact_path.is_file():
        raise FileNotFoundError(f"WindFarm compact metadata not found: {compact_path}")
    with np.load(compact_path, allow_pickle=False) as archive:
        missing = [name for name in COMPACT_GEOMETRY_KEYS if name not in archive.files]
        if missing:
            raise ValueError(f"WindFarm compact metadata is missing geometry keys {missing}")
        return {name: np.asarray(archive[name]).copy() for name in COMPACT_GEOMETRY_KEYS}


def _load_split(view: WindFarmNativeView, derived_view: Path) -> GroupSplit:
    canonical = make_group_split(np.asarray(view.volume.array("layout_index")), seed=42)
    path = derived_view / "split_indices.npz"
    if not path.is_file():
        return canonical
    with np.load(path, allow_pickle=False) as archive:
        saved = tuple(np.asarray(archive[name], dtype=np.int64) for name in ("train", "validation", "test"))
    expected = (canonical.train, canonical.validation, canonical.test)
    if not all(np.array_equal(left, right) for left, right in zip(saved, expected)):
        raise ValueError("Stored WindFarm split does not match the canonical seed-42 layout-group split")
    return canonical


def _freeze_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != serialized:
            raise FileExistsError(f"Refusing to overwrite a different frozen W2 cohort manifest: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, path)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _selected_checkpoint_provenance(checkpoint_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError("W2 requires an exact checkpoint inside its run-owned checkpoints directory")
    run_dir = checkpoint_path.parent.parent
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"W2 exact checkpoint lacks its run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_value = manifest.get("checkpoints", {}).get("best_field")
    if not selected_value:
        raise ValueError("W2 run manifest does not name a best_field checkpoint")
    selected_path = Path(str(selected_value)).expanduser()
    if not selected_path.is_absolute():
        selected_path = run_dir / selected_path
    if not selected_path.is_file():
        fallback = run_dir / "checkpoints" / Path(str(selected_value)).name
        if fallback.is_file():
            selected_path = fallback
    if selected_path.resolve() != checkpoint_path.resolve():
        raise ValueError(
            "W2 checkpoint is not the exact manifest-selected field checkpoint: "
            f"requested={checkpoint_path}, manifest={selected_path}"
        )
    expected_epoch = manifest.get("best_metrics", {}).get("best_epoch")
    epoch = payload.get("epoch", payload.get("current_epoch"))
    best_epoch = payload.get("best_epoch")
    if expected_epoch is None or epoch is None or best_epoch is None:
        raise ValueError("W2 exact checkpoint or run manifest lacks selected epoch metadata")
    if int(epoch) != int(expected_epoch) or int(best_epoch) != int(expected_epoch):
        raise ValueError(
            "W2 selected checkpoint epoch conflicts with its run manifest: "
            f"manifest={expected_epoch}, epoch={epoch}, best_epoch={best_epoch}"
        )
    expected_metric = manifest.get("best_metrics", {}).get("best_val_volume_mse")
    actual_metric = payload.get("best_metric")
    if expected_metric is not None and actual_metric is not None:
        tolerance = max(1.0e-10, 1.0e-7 * abs(float(expected_metric)))
        if abs(float(expected_metric) - float(actual_metric)) > tolerance:
            raise ValueError("W2 selected checkpoint best metric conflicts with its run manifest")
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "run_id": manifest.get("run_id"),
        "run_uuid": manifest.get("run_uuid"),
        "selected_epoch": int(expected_epoch),
        "best_metric": None if actual_metric is None else float(actual_metric),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_selected_best_field": str(selected_path.resolve()),
    }


def _receiver_observable_features(
    prediction_mps: np.ndarray,
    roles: np.ndarray,
    n_turbines: int,
) -> tuple[np.ndarray, list[str]]:
    components = ("Ux", "Uy", "Uz")
    role_map = {
        ANCHOR_ROLE_ROTOR_HUB: ("hub", 1),
        ANCHOR_ROLE_ROTOR_INTERIOR: ("interior", 8),
        ANCHOR_ROLE_ROTOR_EDGE: ("edge", 8),
    }
    if prediction_mps.shape != (int(roles.size), 3):
        raise ValueError("predicted receiver velocities must align with rotor anchor roles")
    if roles.size != int(n_turbines) * 17:
        raise ValueError("WindFarm rotor receiver universe must contain 17 points per active turbine")
    per_turbine: list[np.ndarray] = []
    names: list[str] = []
    for role_code, (role_name, points_per_turbine) in role_map.items():
        selected = prediction_mps[np.asarray(roles) == int(role_code)]
        expected = int(n_turbines) * points_per_turbine
        if selected.shape != (expected, 3):
            raise ValueError(f"{role_name} receiver predictions have shape {selected.shape}, expected {(expected, 3)}")
        by_turbine = selected.reshape(int(n_turbines), points_per_turbine, 3)
        per_turbine.append(by_turbine.mean(axis=1))
        for component in components:
            names.append(f"receiver_{role_name}_{component}")
    turbine_values = np.concatenate(per_turbine, axis=1)
    summaries = []
    summary_names: list[str] = []
    for summary_name, summary in (
        ("mean", np.mean(turbine_values, axis=0)),
        ("std", np.std(turbine_values, axis=0)),
        ("q10", np.quantile(turbine_values, 0.10, axis=0)),
        ("q50", np.quantile(turbine_values, 0.50, axis=0)),
        ("q90", np.quantile(turbine_values, 0.90, axis=0)),
    ):
        summaries.extend(summary.tolist())
        summary_names.extend(f"{name}_{summary_name}_mps" for name in names)
    features = np.asarray(summaries, dtype=np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("predicted receiver observables contain non-finite values")
    return features, summary_names


def _physical_device_identity(device: torch.device) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device.type != "cuda" or visible != "2":
        raise RuntimeError(
            "WindFarm W2 field prediction and scalar fitting must run on physical GPU 2; "
            "launch with CUDA_VISIBLE_DEVICES=2 and --device cuda:0"
        )
    if not torch.cuda.is_available() or device.index not in (None, 0):
        raise RuntimeError("CUDA_VISIBLE_DEVICES=2 must expose physical GPU 2 as logical cuda:0")
    completed = subprocess.run(
        ["nvidia-smi", "-i", "2", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "visible_devices": visible,
        "logical_device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "physical_uuid": completed.stdout.strip(),
    }


def _load_wake_loss_member(compact_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode the one compact outcome member, after the cohort manifest exists."""

    individual = compact_path.parent / "wake_loss_pct.npy"
    if individual.is_file():
        values = np.load(individual, mmap_mode="r", allow_pickle=False)
        return values, {
            "member_path": str(individual.resolve()),
            "storage_format": "individual_npy_mmap",
            "archive_member_rows_materialized": None,
        }
    with np.load(compact_path, allow_pickle=False) as archive:
        if "wake_loss_pct" not in archive.files:
            raise KeyError(f"Compact WindFarm source lacks wake_loss_pct: {compact_path}")
        values = np.asarray(archive["wake_loss_pct"], dtype=np.float32).copy()
    return values, {
        "member_path": f"{compact_path.resolve()}::wake_loss_pct",
        "storage_format": "compressed_npz_member",
        "archive_member_rows_materialized": int(values.shape[0]),
        "note": "The ZIP member is decoded as one array; split labels are indexed separately after decode.",
    }


def run_w2_library_selection(
    *,
    checkpoint: str | Path,
    volume_path: Path,
    compact_path: Path,
    derived_view: Path,
    device: str | None,
    output_dir: str | Path,
) -> int:
    """Run train-only empirical scalar heads and score frozen validation cohorts."""

    total_started = time.perf_counter()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    validate_checkpoint_identity(payload, case_id="WindFarm", model_family="honf_forward", workflow="forward")
    checkpoint_provenance = _selected_checkpoint_provenance(checkpoint_path, payload)

    checkpoint_config = payload.get("train_config")
    dataset_cfg = checkpoint_config.get("dataset", {}) if isinstance(checkpoint_config, Mapping) else {}
    if isinstance(dataset_cfg, Mapping):
        volume_ref = dataset_cfg.get("volume_path")
        compact_ref = dataset_cfg.get("compact_path")
        derived_ref = dataset_cfg.get("derived_view")
        if volume_ref is not None:
            volume_path = resolve_path(str(volume_ref))
        if compact_ref is not None:
            compact_path = resolve_path(str(compact_ref))
        if derived_ref is not None:
            derived_view = resolve_path(str(derived_ref))

    model_config_payload = payload.get("model_config")
    if not isinstance(model_config_payload, Mapping):
        raise TypeError("Selected WindFarm field checkpoint lacks model_config")
    architecture = str(model_config_payload.get("forward_architecture", ""))
    if architecture not in {"dense_pairwise_field", "three_term_full_access_honf", "adaptive_interaction_cover_honf"}:
        raise ValueError(f"W2 receiver scalar head does not support architecture {architecture!r}")
    token_shape = tuple(int(value) for value in dataset_cfg.get("env_token_shape", ENV_TOKEN_SHAPE))

    # Only the declared geometry columns are copied. The scalar target member
    # is intentionally absent until the input-only manifest has been written.
    geometry_metadata = _geometry_metadata(compact_path)
    view = WindFarmNativeView(
        volume_path,
        compact_metadata=geometry_metadata,
        token_shape=token_shape,
        include_receiver_anchors=(architecture == "adaptive_interaction_cover_honf"),
    )
    split = _load_split(view, derived_view)
    cohort_manifest = build_input_only_w2_cohorts(view.metadata, split)
    validation_cohorts = [
        item for item in cohort_manifest["cohorts"] if item.get("split") == "validation"
    ]
    if not validation_cohorts:
        raise ValueError("No eligible input-only validation cohorts are available for W2")
    validation_candidate_rows = sorted(
        {int(row) for item in validation_cohorts for row in item["row_indices"]}
    )
    cohort_manifest_path = output / "cohort_manifest.json"
    cohort_sha256 = _freeze_manifest(cohort_manifest_path, cohort_manifest)

    # Only now may the empirical scalar outcome member be opened.
    target_array, target_storage = _load_wake_loss_member(compact_path)
    if target_array.shape != (view.n_cases,):
        raise ValueError(f"wake_loss_pct has shape {target_array.shape}, expected {(view.n_cases,)}")
    train_target = np.asarray(target_array[split.train], dtype=np.float32).copy()
    if not np.all(np.isfinite(train_target)):
        raise ValueError("Train wake_loss_pct values are non-finite")

    target_device = select_device(device)
    device_identity = _physical_device_identity(target_device)
    torch.cuda.reset_peak_memory_stats(target_device)
    normalization_payload = payload.get("normalization")
    if not isinstance(normalization_payload, Mapping):
        raise TypeError("Selected WindFarm field checkpoint lacks velocity normalization")
    normalizer = VelocityNormalizer.from_dict(dict(normalization_payload))
    model = WindFarmForwardModel(build_windfarm_forward_config(dict(model_config_payload)), velocity_transform=normalizer)
    model = model.to(target_device)
    first_case = view.run(int(split.train[0]))
    materialization = case_batch(first_case, first_case.module_centers).to(target_device)
    model.materialize(materialization)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    interface_payload = model_config_payload.get("interface_model")
    receiver_chunk_size = int(dict(interface_payload or {}).get("receiver_chunk_size", 512))
    if receiver_chunk_size <= 0:
        raise ValueError("checkpoint receiver_chunk_size must be positive")

    selected_rows = sorted(set(map(int, split.train.tolist())) | set(validation_candidate_rows))
    static_features, static_names = static_geometry_features(
        np.asarray(view.metadata["turbine_xy_D"])[selected_rows],
        np.asarray(view.metadata["n_turbines"])[selected_rows],
        np.asarray(view.metadata["wd_deg"])[selected_rows],
    )
    static_by_row = {row: static_features[position] for position, row in enumerate(selected_rows)}
    receiver_by_row: dict[int, np.ndarray] = {}
    receiver_feature_names: list[str] | None = None
    torch.cuda.synchronize(target_device)
    prediction_started = time.perf_counter()
    for row in selected_rows:
        case = view.run(row)
        if case.receiver_anchor_coords is None:
            active_centers = case.module_centers[case.module_present > 0.5]
            anchor_coords, anchor_weights, anchor_roles = windfarm_receiver_anchors(
                case.environment,
                active_centers,
                rotor_radius_D=float(case.module_features[0, 0]),
            )
        else:
            anchor_coords = case.receiver_anchor_coords
            anchor_weights = case.receiver_anchor_weights
            anchor_roles = case.receiver_anchor_roles
        del anchor_weights
        rotor_mask = np.asarray(anchor_roles) != ANCHOR_ROLE_ENVIRONMENT
        rotor_coords = np.asarray(anchor_coords, dtype=np.float32)[rotor_mask]
        rotor_roles = np.asarray(anchor_roles, dtype=np.int64)[rotor_mask]
        if np.any(rotor_roles == ANCHOR_ROLE_ENVIRONMENT):
            raise RuntimeError("environment tokens must not enter W2 rotor receiver observables")
        batch = case_batch(
            case,
            rotor_coords,
            include_receiver_anchors=(architecture == "adaptive_interaction_cover_honf"),
        ).to(target_device)
        with torch.inference_mode():
            prepared = model.prepare_case(batch)
            velocity_mps = model.predict_physical(
                prepared,
                batch.query_xy,
                query_features=batch.query_features,
                receiver_chunk_size=receiver_chunk_size,
            )[0].float().cpu().numpy()
        features, names = _receiver_observable_features(velocity_mps, rotor_roles, case.n_turbines)
        if receiver_feature_names is None:
            receiver_feature_names = names
        elif receiver_feature_names != names:
            raise RuntimeError("WindFarm receiver feature order changed across rows")
        receiver_by_row[row] = features
    torch.cuda.synchronize(target_device)
    prediction_wall_seconds = float(time.perf_counter() - prediction_started)

    train_rows = np.asarray(split.train, dtype=np.int64)
    validation_rows = np.asarray(validation_candidate_rows, dtype=np.int64)
    static_train = np.stack([static_by_row[int(row)] for row in train_rows])
    static_validation = np.stack([static_by_row[int(row)] for row in validation_rows])
    receiver_train = np.stack([receiver_by_row[int(row)] for row in train_rows])
    receiver_validation = np.stack([receiver_by_row[int(row)] for row in validation_rows])
    train_groups = np.asarray(view.metadata["layout_index"])[train_rows]

    # Geometry-only ridge is the requested scalar/static control. The second
    # ridge adds observed model velocity at geometry-defined rotor locations.
    torch.cuda.synchronize(target_device)
    ridge_started = time.perf_counter()
    static_prediction, static_fit = fit_grouped_ridge_head(
        static_train,
        train_target,
        train_groups,
        static_validation,
        device=target_device,
        alphas=_STATIC_CV_ALPHAS,
        folds=5,
        seed=42,
    )
    receiver_train_features = np.concatenate((static_train, receiver_train), axis=1)
    receiver_validation_features = np.concatenate((static_validation, receiver_validation), axis=1)
    receiver_prediction, receiver_fit = fit_grouped_ridge_head(
        receiver_train_features,
        train_target,
        train_groups,
        receiver_validation_features,
        device=target_device,
        alphas=_STATIC_CV_ALPHAS,
        folds=5,
        seed=42,
    )
    torch.cuda.synchronize(target_device)
    ridge_fit_wall_seconds = float(time.perf_counter() - ridge_started)

    # Validation outcomes are indexed only after cohort membership is frozen
    # and each head's regularization has been selected by train-layout CV.
    validation_target = np.asarray(target_array[validation_rows], dtype=np.float32).copy()
    if not np.all(np.isfinite(validation_target)):
        raise ValueError("Eligible validation wake_loss_pct values are non-finite")

    target_length = int(view.n_cases)
    static_by_source = np.full(target_length, np.nan, dtype=np.float32)
    receiver_by_source = np.full(target_length, np.nan, dtype=np.float32)
    observed_validation = np.full(target_length, np.nan, dtype=np.float32)
    static_by_source[validation_rows] = static_prediction
    receiver_by_source[validation_rows] = receiver_prediction
    observed_validation[validation_rows] = validation_target
    static_metrics = score_finite_library(
        cohort_manifest,
        static_by_source,
        observed_validation,
        split_name="validation",
    )
    receiver_metrics = score_finite_library(
        cohort_manifest,
        receiver_by_source,
        observed_validation,
        split_name="validation",
    )

    def _prediction_metrics(prediction: np.ndarray) -> dict[str, Any]:
        return {
            **regression_metrics(validation_target, prediction),
            "spearman_rank_correlation": spearman_rank_correlation(validation_target, prediction),
            "scored_rows": validation_candidate_rows,
        }

    peak_memory = {
        "allocated_bytes": int(torch.cuda.max_memory_allocated(target_device)),
        "reserved_bytes": int(torch.cuda.max_memory_reserved(target_device)),
    }
    torch.cuda.synchronize(target_device)
    total_wall_seconds = float(time.perf_counter() - total_started)
    result = {
        "schema_version": 1,
        "study": "WindFarm W2 empirical finite-library selection",
        "status": "validation_only",
        "cohort_manifest": str(cohort_manifest_path.resolve()),
        "cohort_manifest_sha256": cohort_sha256,
        "cohort_counts": cohort_manifest["counts"],
        "eligible_validation_cohorts": len(validation_cohorts),
        "eligible_validation_candidate_rows": len(validation_candidate_rows),
        "selected_checkpoint": checkpoint_provenance,
        "field_architecture": architecture,
        "receiver_observable_contract": {
            "source": "checkpoint-predicted physical velocity at input-only rotor-neighborhood queries",
            "coordinates": "hub center plus 8 half-radius and 8 edge samples in the documented Y-Z rotor plane",
            "geometry_frame": "native downstream/crosswind frame; no second wind-direction rotation",
            "query_points_per_turbine": 17,
            "environment_tokens_used_as_receiver_queries": False,
            "target_or_case_outcome_used_to_choose_queries": False,
            "velocity_cubed_or_power_proxy_used": False,
            "exact_query_locations_have_native_target_support": False,
            "native_velocity_support": {
                "hub_band": "sampled native mesh centers in the global |z-hub_height| <= 0.5D slab; not turbine-local rotor points",
                "downstream_envelope": "sampled native volume cells in the geometry-defined 0 < dx <= 10D and yz radius <= 1.5D turbine envelope",
                "exact_rotor_plane_anchor_metrics": "not computed; anchors generally do not coincide with stored cell centers",
            },
            "selection_utility_interpretation": (
                "empirical validation-cohort association using predicted receiver features; "
                "not physical validation of the continuous rotor-anchor velocities"
            ),
        },
        "target_contract": {
            "source_column": "wake_loss_pct",
            "units": "percent",
            "documented_meaning": "Scalar wake-loss/regression target",
            "source_formula_verified": False,
            "interpretation": "empirical scalar target only; no power or AEP claim",
            "selection_direction": "minimize observed wake_loss_pct within each exact turbine-count/direction cohort",
        },
        "feasibility_assessment": {
            "available": False,
            "reason": "The inspected WindFarm data contract specifies no additional site, clearance, operating, or turbine-control constraint.",
            "cohort_match_is_not_a_claim_of_physical_feasibility": True,
        },
        "target_exposure": {
            "train_rows_used_for_ridge_fit": int(train_rows.size),
            "train_layout_groups_used_for_ridge_fit": int(np.unique(train_groups).size),
            "eligible_validation_candidate_rows_used_for_outcome_audit": int(validation_rows.size),
            "validation_cohorts_selected": len(validation_cohorts),
            "reserved_test_label_rows_indexed_or_used": 0,
            "reserved_test_cohort_outcomes_scored": False,
            "validation_outcomes_indexed_after_train_group_cv": True,
            "target_storage": target_storage,
            "selection_and_audit_note": (
                "All eligible stored validation outcomes are read after prediction-only cohort selection is frozen; "
                "no new CFD solves are launched. The reserved test is not indexed for outcomes."
            ),
        },
        "static_geometry_features": static_names,
        "predicted_receiver_features": receiver_feature_names,
        "scalar_head_fit": {
            "geometry_only_static_control": static_fit,
            "predicted_receiver_velocity_plus_geometry": receiver_fit,
        },
        "validation_prediction_metrics": {
            "geometry_only_static_control": _prediction_metrics(static_prediction),
            "predicted_receiver_velocity_plus_geometry": _prediction_metrics(receiver_prediction),
        },
        "finite_library_selection": {
            "geometry_only_static_control": static_metrics,
            "predicted_receiver_velocity_plus_geometry": receiver_metrics,
        },
        "execution": {
            **device_identity,
            "environment_token_shape": list(token_shape),
            "environment_token_count": int(np.prod(token_shape)),
            "receiver_queries_per_turbine": 17,
            "receiver_chunk_size": receiver_chunk_size,
            "forward_prediction_rows": len(selected_rows),
            "forward_prediction_train_rows": int(train_rows.size),
            "forward_prediction_validation_candidate_rows": int(validation_rows.size),
            "total_rotor_receiver_queries": int(
                17 * np.asarray(view.metadata["n_turbines"], dtype=np.int64)[selected_rows].sum()
            ),
            "environment_tokens_materialized": int(len(selected_rows) * np.prod(token_shape)),
            "forward_prediction_wall_seconds_synchronized": prediction_wall_seconds,
            "scalar_head_fit_wall_seconds_synchronized": ridge_fit_wall_seconds,
            "total_wall_seconds_synchronized": total_wall_seconds,
            "peak_cuda_memory": peak_memory,
        },
        "split": {
            "group_key": "layout_index",
            "seed": 42,
            "train_rows": int(split.train.size),
            "validation_rows": int(split.validation.size),
            "test_rows": int(split.test.size),
            "test_labels_used": False,
        },
        "no_new_cfd_solves": True,
    }
    result_path = output / "w2_selection_metrics.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite W2 result: {result_path}")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[windfarm-w2] validation cohorts="
        f"{len(validation_cohorts)} candidates={len(validation_candidate_rows)} output={result_path}"
    )
    return 0


__all__ = ["run_w2_library_selection"]
