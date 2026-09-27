"""Argumentized ThermalChannel response-control fitting runner.

The runner supports intact incumbent checkpoints and the historical explicit
``three_term_full_access_honf`` conversion path. It samples typed train
stencils and supports a one-update correctness preflight or paired staged
fits. All one-time state and reports go to an ignored output directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import signal
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from dataclasses import fields as dataclass_fields
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.evaluation.loading import load_model
from channelthermal.interaction_evidence.reference_adapter import (
    AnalyticWakeReferenceAdapter,
    load_stored_reference_case,
)
from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.interaction_evidence.response_dataset import ResponseStencil
from channelthermal.interaction_evidence.storage import load_solve_record
from channelthermal.interaction_evidence.types import EvidenceSplit, MeasuredQuantity
from channelthermal.model import ChannelThermalHONFModel
from honf_forward_core.config import UnifiedForwardConfig
from honf_forward_core.interface_fields.checkpoint_warm_start import (
    warm_start_three_term_full_access,
)
from honf_forward_core.interface_fields.core import InterfaceFieldCore

from .algebra import MixedResponseSpec, predict_stencil
from .contracts import (
    DesignInput,
    RoleQuery,
    context_inputs,
    role_queries_from_stencil,
)
from .derivative_check import check_pressure_peak_ad_fd
from .evaluation import evaluate_absolute_record, evaluate_stencil
from .historical import HistoricalValueSource, select_broad_evaluation_cases
from .losses import (
    FixedHeatNullControl,
    ThermalLossScales,
    compute_stencil_loss_terms,
    fixed_heat_control_loss_terms,
    fixed_heat_material_peak_coverage,
    historical_absolute_value_loss,
)
from .native import DifferentiableThermalOperator
from .paired import run_paired_staged_fits, write_paired_training_curves
from .resume_provenance import (
    _same,
    _validate_fit_arm_gate_accounting,
    training_config_mapping,
    validate_checkpoint_only_review_provenance,
    validate_paired_resume_provenance,
)
from .sampling import ReceiverSamplingConfig, SamplingSummary, sample_training_panel
from .thermal import pressure_section_masks
from .training import (
    StagedFitResult,
    StagedTrainingConfig,
    TrainingStage,
    calibrate_operator_weights,
    historical_replay_coverage,
    load_staged_training_config,
    restore_checkpoint_payload,
    run_staged_fit,
)


def _enable_deterministic_algorithms(device: torch.device) -> str | None:
    """Enable strict deterministic Torch kernels for native response work."""

    cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if device.type == "cuda" and cublas_workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "Deterministic CUDA response work requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "or :16:8 set before Python starts."
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return cublas_workspace


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.tmp-", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(_json_safe(payload), stream, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_default(value: Any) -> Any:
    return _json_safe(value)


def _json_safe(value: Any) -> Any:
    """Recursively convert scientific result values to JSON-native objects."""

    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if hasattr(value, "__dataclass_fields__"):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in dataclass_fields(value)
        }
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} to the run manifest.")


def _dataclass_record(value: Any) -> dict[str, Any]:
    """Extract dataclass fields without deepcopying immutable mapping values."""

    return {field.name: getattr(value, field.name) for field in dataclass_fields(value)}


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.tmp-", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_safe_response_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load response checkpoints through PyTorch's restricted unpickler.

    The legacy NumPy RNG tuple uses ndarray reconstruction and a uint32 dtype;
    those exact NumPy globals are the only additions to the weights-only
    allowlist. Arbitrary globals in a CLI-selected resume path remain rejected.
    """

    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:  # NumPy 1.x compatibility.
        numpy_core = np.core
    safe_numpy_globals = [
        numpy_core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(safe_numpy_globals):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Response checkpoint {path} must contain a mapping.")
    return payload


def _load_r0_projection_evidence(path: Path, checkpoint_path: Path) -> dict[str, Any]:
    """Validate measured train-only M10 projection evidence.

    Accept either the direct passed R0 report or the read-only recovery
    manifest whose 80-update fit and gradient diagnostic completed but whose
    outer serializer failed at the known lazy-buffer lookup. The recovered
    path is intentionally narrow: any other failed R0 wrapper is rejected.
    """

    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    source_digest = payload.get("checkpoint_sha256", payload.get("source_checkpoint_sha256"))
    if source_digest != _checkpoint_digest(checkpoint_path):
        raise ValueError("The supplied R0 projection evidence belongs to a different source checkpoint.")
    if payload.get("status") == "passed" and payload.get("mode") == "train_only_scope_diagnostic":
        justification = payload.get("projection_justification")
        if not isinstance(justification, Mapping):
            raise ValueError("The passed R0 report has no measured projection-justification record.")
        family_id = justification.get("family_id")
        justified = bool(justification.get("justified_by_measured_m10_combined_value_conflict"))
        detail = dict(justification)
        evidence_status = "passed_train_only_scope_diagnostic"
    elif (
        payload.get("mode") == "r0_missing_native_nonlinear_interface_scope"
        and payload.get("status") == "failed"
        and payload.get("error_type") == "KeyError"
        and payload.get("error") == "'core.position_fourier.frequencies'"
        and int(payload.get("optimizer_updates_completed", -1)) == 80
        and int(payload.get("optimizer_updates_attempted", -1)) == 80
        and int(payload.get("prior_optimizer_attempts_preserved", -1)) == 100
        and int(payload.get("total_r0_optimizer_attempts_including_prior", -1)) == 180
        and int(payload.get("reference_solver_calls", -1)) == 0
        and payload.get("m10_projection_justified") is True
    ):
        gradient_path_value = payload.get("m10_gradient_diagnostic")
        if not gradient_path_value:
            raise ValueError("Recovered R0 manifest is missing its combined M10 gradient diagnostic path.")
        gradient_path = Path(str(gradient_path_value)).expanduser().resolve()
        gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
        blocks = gradient.get("combined_response_gradient_vs_combined_value_gradient")
        expected_blocks = {"field_head", "port_head", "port_refinement_head"}
        if not isinstance(blocks, Mapping) or set(blocks) != expected_blocks:
            raise ValueError("Recovered M10 gradient diagnostic does not cover every trainable parameter block.")
        computed_blocks: dict[str, dict[str, Any]] = {}
        for name, item in blocks.items():
            if not isinstance(item, Mapping):
                raise TypeError(f"Recovered M10 gradient block {name!r} is malformed.")
            dot = float(item["dot_product"])
            value_norm = float(item["combined_value_gradient_norm"])
            if not np.isfinite(dot) or not np.isfinite(value_norm):
                raise ValueError(f"Recovered M10 gradient block {name!r} is non-finite.")
            adverse = dot < 0.0 and value_norm > 0.0
            if bool(item.get("adverse")) != adverse:
                raise ValueError(f"Recovered M10 gradient block {name!r} has an inconsistent conflict flag.")
            computed_blocks[str(name)] = {**dict(item), "adverse": adverse}
        family_id = gradient.get("family_id")
        justified = any(item["adverse"] for item in computed_blocks.values())
        if family_id != "stored_family:0350" or not justified:
            raise ValueError("Recovered M10 diagnostic does not establish the declared train-family conflict.")
        if not payload.get("m10_projection_justified") == justified:
            raise ValueError("Recovered R0 manifest and combined M10 gradient conflict flags disagree.")
        detail = {
            "family_id": family_id,
            "justified_by_measured_m10_combined_value_conflict": justified,
            "combined_gradient_gate_by_block": computed_blocks,
            "individual_response_term_diagnostics": gradient.get("individual_response_term_diagnostics"),
            "calibrated_weights": gradient.get("calibrated_weights"),
            "m10_gradient_diagnostic_path": str(gradient_path),
            "m10_gradient_diagnostic_sha256": _checkpoint_digest(gradient_path),
            "r0_outer_status": "failed_after_80_updates_on_lazy_buffer_KeyError",
            "r0_error": payload.get("error"),
            "charged_updates_including_prior": 180,
            "reference_solver_calls": 0,
        }
        evidence_status = "recovered_train_only_fit_and_m10_gradient; outer serialization failed"
    else:
        raise ValueError(
            "The supplied R0 evidence is neither a passed direct diagnostic nor the verified read-only "
            "u80 lazy-buffer recovery manifest."
        )
    if family_id != "stored_family:0350":
        raise ValueError("Projection evidence does not cover the declared M10 conflict family.")
    return {
        **detail,
        "family_id": family_id,
        "justified_by_measured_m10_combined_value_conflict": justified,
        "evidence_status": evidence_status,
        "r0_diagnostic_path": str(resolved),
        "r0_diagnostic_sha256": _checkpoint_digest(resolved),
        "projection_enabled_for_fit": justified,
    }


def _resolve_dataset_path(checkpoint: Mapping[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    train_config = checkpoint.get("train_config", {})
    dataset_config = train_config.get("dataset", {}) if isinstance(train_config, Mapping) else {}
    configured = dataset_config.get("packed_h5_path")
    if not configured:
        raise ValueError("Checkpoint has no packed_h5_path; pass --dataset explicitly.")
    return Path(str(configured)).expanduser().resolve()


def _validate_rehydrated_raw_baseline(
    atlas_baseline: Any,
    raw_baseline: Any,
    *,
    coordinate_atol: float = 1.0e-7,
) -> dict[str, Any]:
    """Bind an existing raw solver baseline to its atlas target without solving."""

    if atlas_baseline.output is None or raw_baseline.output is None:
        raise ValueError("Fixed-heat review baseline binding requires solved atlas and raw records.")
    if raw_baseline.status.value != "converged" or atlas_baseline.status.value != "converged":
        raise ValueError("Fixed-heat review baseline binding requires converged records.")
    if raw_baseline.source is not atlas_baseline.source:
        raise ValueError("Rehydrated raw baseline changes the atlas evidence source.")
    if raw_baseline.design != atlas_baseline.design:
        raise ValueError("Rehydrated raw baseline design differs from the atlas baseline.")
    if dict(raw_baseline.context.values) != dict(atlas_baseline.context.values):
        raise ValueError("Rehydrated raw baseline context differs from the atlas baseline.")
    atlas_output = atlas_baseline.output
    raw_output = raw_baseline.output
    if set(raw_output.roles) != set(atlas_output.roles):
        raise ValueError("Rehydrated raw baseline role set differs from the atlas baseline.")
    role_checks: dict[str, dict[str, Any]] = {}
    for role_name, atlas_role in atlas_output.roles.items():
        raw_role = raw_output.roles[role_name]
        identity_equal = (
            raw_role.role == atlas_role.role
            and raw_role.coordinate_kind == atlas_role.coordinate_kind
            and raw_role.channel_names == atlas_role.channel_names
            and raw_role.channel_units == atlas_role.channel_units
            and raw_role.query_ids == atlas_role.query_ids
            and raw_role.receiver_module_ids == atlas_role.receiver_module_ids
            and raw_role.values.shape == atlas_role.values.shape
            and raw_role.query_features.shape == atlas_role.query_features.shape
        )
        if not identity_equal:
            raise ValueError(f"Rehydrated raw baseline {role_name!r} schema differs from its atlas record.")
        coordinate_delta = float(np.max(np.abs(raw_role.query_features - atlas_role.query_features)))
        if coordinate_delta > coordinate_atol:
            raise ValueError(
                f"Rehydrated raw baseline {role_name!r} coordinates exceed the declared atlas-rounding "
                f"tolerance {coordinate_atol}: max_abs={coordinate_delta}."
            )
        if not np.array_equal(raw_role.values, atlas_role.values):
            raise ValueError(f"Rehydrated raw baseline {role_name!r} target values differ from the atlas record.")
        if not np.array_equal(raw_role.valid_mask, atlas_role.valid_mask):
            raise ValueError(f"Rehydrated raw baseline {role_name!r} validity mask differs from the atlas record.")
        role_checks[role_name] = {
            "identity_schema_equal": True,
            "coordinates_max_abs_difference": coordinate_delta,
            "coordinate_absolute_tolerance": coordinate_atol,
            "values_exact": True,
            "valid_mask_exact": True,
            "quadrature_weights_compared": False,
            "quadrature_note": "Atlas and raw solver retain separate documented quadrature schemas.",
        }
    atlas_pressure = atlas_output.quantities["pressure_drop"]
    raw_pressure = raw_output.quantities["pressure_drop"]
    if (
        not atlas_pressure.resolved
        or not raw_pressure.resolved
        or atlas_pressure.units != raw_pressure.units
        or float(atlas_pressure.value) != float(raw_pressure.value)
    ):
        raise ValueError("Rehydrated raw baseline pressure differs from the atlas baseline target.")
    if raw_output.module_peak_temperature != atlas_output.module_peak_temperature:
        raise ValueError("Rehydrated raw baseline per-module peak labels differ from the atlas baseline.")
    return {
        "status": "passed",
        "design_exact": True,
        "context_exact": True,
        "pressure_exact": True,
        "module_peaks_exact": True,
        "role_checks": role_checks,
        "raw_baseline_case_dir": raw_output.case_dir,
        "atlas_baseline_case_dir": atlas_output.case_dir,
    }


def _load_fixed_heat_null_controls(
    recipe_payload: Mapping[str, Any],
    raw_stencils: Sequence[ResponseStencil],
    sampled_panel: Sequence[Any],
    *,
    stencil_paths: Sequence[Path],
) -> tuple[tuple[FixedHeatNullControl, ...], dict[str, Any], tuple[tuple[str, ResponseStencil], ...]]:
    """Load the already-solved train-only fixed-geometry heating controls."""

    relative_panel = recipe_payload.get("fixed_heat_control_panel")
    family_id = str(recipe_payload.get("fixed_heat_control_family_id", ""))
    if not relative_panel or not family_id:
        raise ValueError("The named nonlinear recipe must identify its fixed-heat train panel and family.")
    project_root = Path(__file__).resolve().parents[4]
    panel_dir = (project_root / str(relative_panel)).resolve()
    frozen_path = panel_dir / "frozen_inputs.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_new_reference_calls" or frozen.get("reference_split") != "train":
        raise ValueError("Fixed-heat inputs lack their frozen train-only provenance.")
    family_atlas_stencil = next(
        (stencil for stencil in raw_stencils if stencil.physical_family_id == family_id),
        None,
    )
    if family_atlas_stencil is None or family_atlas_stencil.baseline.output is None:
        raise ValueError(f"The named fixed-heat family {family_id!r} is absent from the train panel.")
    atlas_path = next(
        (path for path in stencil_paths if path.name == "train_0001_responses.npz"),
        None,
    )
    if atlas_path is None or _checkpoint_digest(atlas_path) != frozen.get("source_atlas_sha256"):
        raise ValueError("Fixed-heat controls do not match the verified train-0001 source atlas.")
    raw_baseline_path = Path(str(family_atlas_stencil.baseline.output.case_dir)).expanduser().resolve()
    raw_baseline_config = raw_baseline_path / "case_config.json"
    if not raw_baseline_config.is_file():
        raise FileNotFoundError(f"Stored raw atlas baseline is missing: {raw_baseline_config}")
    raw_baseline_adapter = AnalyticWakeReferenceAdapter(
        demo_root=Path(__file__).resolve().parents[5] / "1_Demo_ChannelThermal",
        output_root=panel_dir,
        case_template=json.loads(raw_baseline_config.read_text(encoding="utf-8")),
    )
    raw_baseline = raw_baseline_adapter.load_record(
        raw_baseline_path,
        family_atlas_stencil.baseline.design,
        family_atlas_stencil.baseline.context,
        record_id=family_atlas_stencil.baseline.record_id,
        elapsed_seconds=family_atlas_stencil.baseline.elapsed_seconds,
    )
    raw_baseline_binding = _validate_rehydrated_raw_baseline(
        family_atlas_stencil.baseline,
        raw_baseline,
    )
    raw_baseline_binding["raw_case_config_sha256"] = _checkpoint_digest(raw_baseline_config)
    frame_index_path = raw_baseline_path / "frame_index.csv"
    raw_baseline_binding["raw_frame_index_sha256"] = _checkpoint_digest(frame_index_path)
    raw_baseline_binding["raw_grid_sha256"] = _checkpoint_digest(raw_baseline_path / "grid.npz")
    with frame_index_path.open("r", encoding="utf-8", newline="") as stream:
        frame_rows = list(csv.DictReader(stream))
    if not frame_rows:
        raise ValueError("Stored raw atlas baseline has no indexed solver frames.")
    raw_frame_path = (raw_baseline_path / "scene" / str(frame_rows[-1]["file"])).resolve()
    if raw_frame_path.parent != (raw_baseline_path / "scene").resolve() or not raw_frame_path.is_file():
        raise ValueError("Stored raw atlas baseline final frame path is invalid or missing.")
    raw_baseline_binding["raw_final_frame"] = str(raw_frame_path)
    raw_baseline_binding["raw_final_frame_sha256"] = _checkpoint_digest(raw_frame_path)
    raw_baseline_binding["read_only_rehydration"] = True
    raw_baseline_binding["reference_solver_calls"] = 0
    sampled_family = next(
        (item.stencil for item in sampled_panel if item.stencil.physical_family_id == family_id),
        None,
    )
    if sampled_family is None:
        raise ValueError("Fixed-heat controls cannot align to the sampled family-0001 response stencil.")
    candidate_rows = list(frozen.get("candidates", ()))
    if len(candidate_rows) != 4 or len({row.get("record_id") for row in candidate_rows}) != 4:
        raise ValueError("The fixed-heat train panel must contain its four unique frozen controls.")
    controls: list[FixedHeatNullControl] = []
    review_stencils: list[tuple[str, ResponseStencil]] = []
    record_hashes: list[dict[str, str]] = []
    summaries: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        control_id = str(candidate["record_id"])
        record_dir = panel_dir / "records" / control_id
        outcome_path = panel_dir / f"{control_id}.outcome.json"
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        if (
            outcome.get("solve_status") != "converged"
            or outcome.get("serialization_status") != "stored"
            or outcome.get("solver_invoked") is not True
            or outcome.get("raw_solver_completed") is not True
            or outcome.get("pressure_delta_from_existing_baseline") != 0.0
        ):
            raise ValueError(f"Fixed-heat train control {control_id!r} is not a verified stored null case.")
        record = load_solve_record(record_dir)
        control = FixedHeatNullControl(
            control_id=control_id,
            baseline=family_atlas_stencil.baseline,
            control=record,
        ).sampled_for_panel(sampled_family.baseline)
        controls.append(control)
        review_stencil = ResponseStencil(raw_baseline, {control_id: record})
        for role_name in raw_baseline.output.roles:  # type: ignore[union-attr]
            # This call keeps the physical comparison strict: IDs, coordinates,
            # masks, and the raw solver's quadrature weights must all align.
            review_stencil.finite_change(control_id, role_name)
        review_stencils.append((control_id, review_stencil))
        record_hashes.append({
            "record_id": control_id,
            "record_json_sha256": _checkpoint_digest(record_dir / "record.json"),
            "physical_output_sha256": _checkpoint_digest(record_dir / "physical_output.npz"),
            "outcome_sha256": _checkpoint_digest(outcome_path),
        })
        summaries.append({
            "record_id": control_id,
            "family_id": control.family_id,
            "role_query_counts": {
                role: len(indices) for role, indices in control.sampled_role_indices.items()
            },
            "solid_temperature_full_receiver_universe": (
                len(control.sampled_role_indices["solid_temperature"])
                == len(record.output.roles["solid_temperature"].query_ids)
            ),
            "solid_temperature_receiver_ids_exact": tuple(
                record.output.roles["solid_temperature"].query_ids[index]
                for index in control.sampled_role_indices["solid_temperature"]
            ) == tuple(record.output.roles["solid_temperature"].query_ids),
            "solid_temperature_peak_scoring": "all stored receiver IDs using per-record validity masks; quadrature weights do not filter peaks",
            "solid_temperature_peak_coverage": {
                "baseline": fixed_heat_material_peak_coverage(family_atlas_stencil.baseline),
                "control": fixed_heat_material_peak_coverage(record),
            },
            "uses_control_quadrature_with_sample_factors": True,
            "review_uses_original_raw_baseline_and_control": True,
            "review_strict_response_alignment_passed": True,
            "pressure_drop_reducer": "maintained unweighted 8-percent inlet/outlet means over protected sampled rows",
            "stored_pressure_delta": float(outcome["pressure_delta_from_existing_baseline"]),
        })
    provenance = {
        "panel_path": str(panel_dir),
        "panel_policy": str(recipe_payload["fixed_heat_control_policy"]),
        "frozen_inputs_sha256": _checkpoint_digest(frozen_path),
        "source_atlas": str(atlas_path.resolve()),
        "source_atlas_sha256": _checkpoint_digest(atlas_path),
        "family_id": family_id,
        "review_raw_baseline_binding": raw_baseline_binding,
        "control_count": len(controls),
        "control_records": record_hashes,
        "sampling_adapter": summaries,
        "reference_solver_calls_during_fit": 0,
    }
    return tuple(controls), provenance, tuple(review_stencils)


def _make_refit_model(
    source_model: ChannelThermalHONFModel,
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[ChannelThermalHONFModel, dict[str, Any]]:
    model = copy.deepcopy(source_model)
    model_config = copy.deepcopy(source_model.config)
    core_payload = model_config.core_honf.to_dict()
    core_payload["forward_architecture"] = "three_term_full_access_honf"
    interface = dict(core_payload.get("interface_model") or {})
    # The clean full-access architecture fixes these normalizers. This is a
    # documented refit of the inherited organizer controls, not an identity
    # conversion of Run1508 outputs.
    interface["source_normalizer"] = "entmax15"
    interface["query_normalizer"] = "entmax15"
    interface["environment_refinement_normalizer"] = "entmax15"
    for name in ("module_temperature", "environment_temperature", "query_temperature"):
        interface[name] = 1.0
    core_payload["interface_model"] = interface
    core_config = UnifiedForwardConfig.from_dict(core_payload)
    model_config.core_honf = core_config
    model.config = model_config
    model.core = InterfaceFieldCore(core_config).to(device)
    normalization_config = checkpoint.get(
        "global_normalization_config",
        checkpoint.get("train_config", {}).get("dataset", {}),
    )
    model.set_global_target_normalization(
        checkpoint.get("global_normalization_stats", {}),
        normalize_targets=bool(normalization_config.get("normalize_targets", False)),
    )
    return model.to(device), {
        "forward_architecture": core_config.forward_architecture,
        "source_normalizer": interface["source_normalizer"],
        "query_normalizer": interface["query_normalizer"],
        "environment_refinement_normalizer": interface["environment_refinement_normalizer"],
        "source_prediction_mode": source_model.config.channelthermal.internal_prediction_mode,
        "local_module_params_from_used_ports": bool(
            source_model.config.channelthermal.local_module_params_from_used_ports
        ),
        "prediction_identity_claim": False,
    }


def _materialize_and_warm_start(
    source_model: ChannelThermalHONFModel,
    target_model: ChannelThermalHONFModel,
    operator: DifferentiableThermalOperator,
    stencil: ResponseStencil,
    *,
    device: torch.device,
) -> dict[str, Any]:
    queries = role_queries_from_stencil(stencil, device=device)
    design = DesignInput.from_state(stencil.baseline.design, device=device)
    with torch.no_grad():
        operator(design, context_inputs(stencil.baseline.context), queries)
    source_state = source_model.core.state_dict()
    target_state = target_model.core.state_dict()
    initialized, inventory = warm_start_three_term_full_access(
        source_state,
        target_state,
        source_architecture=str(source_model.config.core_honf.forward_architecture),
    )
    target_model.core.load_state_dict(initialized, strict=True)
    return {
        "copied_tensor_count": len(inventory["copied"]),
        "initialized_tensor_count": len(inventory["initialized"]),
        "discarded_source_tensor_count": len(inventory["discarded_source"]),
        "source_architecture": inventory["source_architecture"],
        "prediction_identity_claim": inventory["prediction_identity_claim"],
        "copied_tensors": list(inventory["copied"]),
        "initialized_tensors": list(inventory["initialized"]),
        "discarded_source_tensors": list(inventory["discarded_source"]),
    }


def _native_checkpoint_initialization(
    source_model: ChannelThermalHONFModel,
) -> tuple[ChannelThermalHONFModel, dict[str, Any]]:
    """Keep the selected, intact architecture and every checkpoint tensor."""

    architecture = str(source_model.config.core_honf.forward_architecture)
    if architecture not in {
        "dense_pairwise_field",
        "sparse_incidence_group_control_honf",
    }:
        raise ValueError(
            "native_checkpoint initialization requires an incumbent native architecture; "
            f"got {architecture!r}."
        )
    return copy.deepcopy(source_model), {
        "initialization_mode": "native_checkpoint",
        "forward_architecture": architecture,
        "native_checkpoint_tensor_count": len(source_model.state_dict()),
        "prediction_identity_claim": True,
        "warm_start_or_conversion_invoked": False,
    }


def _final_linear(module: nn.Module, *, label: str) -> tuple[nn.Linear, str]:
    linears = [
        (name, child)
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear)
    ]
    if not linears:
        raise TypeError(f"Native trainable head {label!r} has no materialized Linear output layer.")
    relative_name, layer = linears[-1]
    return layer, relative_name


def _configure_native_output_head_scope(
    model: ChannelThermalHONFModel,
) -> dict[str, Any]:
    """Freeze the intact model except its existing field and port output layers."""

    if not model.local_coupling.has_local_surrogate:
        raise RuntimeError("Native output-head fitting requires the checkpoint's attached local surrogate.")
    heads = {
        "field_head_output": model.core.common.field_head,
        "port_head_output": model.local_coupling.port_head,
        "port_refinement_output": model.local_coupling.port_refinement_head,
    }
    selected: dict[str, nn.Parameter] = {}
    selected_modules: dict[str, str] = {}
    for label, head in heads.items():
        layer, relative_name = _final_linear(head, label=label)
        module_name = {
            "field_head_output": "core.common.field_head",
            "port_head_output": "local_coupling.port_head",
            "port_refinement_output": "local_coupling.port_refinement_head",
        }[label]
        selected_modules[label] = f"{module_name}.{relative_name}".rstrip(".")
        for parameter_name, parameter in layer.named_parameters(recurse=False):
            selected[f"{selected_modules[label]}.{parameter_name}"] = parameter
    if not selected:
        raise RuntimeError("Native output-head scope selected no parameters.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in selected.values():
        parameter.requires_grad_(True)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    expected_names = set(selected)
    if set(trainable_names) != expected_names:
        raise RuntimeError(
            "Native output-head scope mismatch: "
            f"expected {sorted(expected_names)}, got {sorted(trainable_names)}."
        )
    return {
        "name": "native_output_heads",
        "trainable_module_names": selected_modules,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(parameter.numel() for parameter in selected.values()),
        "frozen_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
    }


def _configure_native_nonlinear_interface_scope(
    model: ChannelThermalHONFModel,
) -> dict[str, Any]:
    """Train all existing layers in the native field and port interface heads."""

    if not model.local_coupling.has_local_surrogate:
        raise RuntimeError("Native nonlinear-interface fitting requires the attached local surrogate.")
    blocks = {
        "field_head": ("core.common.field_head", model.core.common.field_head),
        "port_head": ("local_coupling.port_head", model.local_coupling.port_head),
        "port_refinement_head": (
            "local_coupling.port_refinement_head",
            model.local_coupling.port_refinement_head,
        ),
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: dict[str, nn.Parameter] = {}
    parameter_counts: dict[str, int] = {}
    selected_modules: dict[str, str] = {}
    for label, (module_path, module) in blocks.items():
        selected_modules[label] = module_path
        block_parameters = dict(module.named_parameters(recurse=True))
        if not block_parameters:
            raise TypeError(f"Native interface block {module_path!r} has no materialized parameters.")
        for relative_name, parameter in block_parameters.items():
            full_name = f"{module_path}.{relative_name}" if relative_name else module_path
            selected[full_name] = parameter
            parameter.requires_grad_(True)
        parameter_counts[label] = sum(parameter.numel() for parameter in block_parameters.values())
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if set(trainable_names) != set(selected):
        raise RuntimeError(
            "Native nonlinear-interface scope mismatch: "
            f"expected {sorted(selected)}, got {sorted(trainable_names)}."
        )
    # Full evaluation mode fixes BatchNorm buffers and disables dropout for
    # both absolute states in every finite stencil response.
    model.eval()
    stochastic_modules = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.p > 0.0
    ]
    training_modules = [name for name, module in model.named_modules() if module.training]
    if training_modules:
        raise RuntimeError(f"Deterministic native response mode left modules in training mode: {training_modules}.")
    return {
        "name": "native_nonlinear_interface",
        "trainable_module_names": selected_modules,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count_by_block": parameter_counts,
        "trainable_parameter_count": sum(parameter.numel() for parameter in selected.values()),
        "frozen_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
        "frozen_buffer_names": [name for name, _ in model.named_buffers()],
        "model_mode": "eval_for_all_updates_and_absolute_stencil_states",
        "dropout_modules_disabled_by_eval": stochastic_modules,
        "non_eval_modules": training_modules,
    }


def _configure_native_expanded_response_interface_scope(
    model: ChannelThermalHONFModel,
) -> dict[str, Any]:
    """Train the native message/update/read path that feeds Thermal P0/P1.

    This is a separately named expansion of the older nonlinear-head scope.
    Encoders, common context builders, local surrogate and response assembly
    remain frozen so the fit isolates the native interaction path plus the
    established field and port heads.
    """

    if not model.local_coupling.has_local_surrogate:
        raise RuntimeError("Expanded native response fitting requires the attached local surrogate.")
    if not isinstance(getattr(model.core, "backend", None), nn.Module):
        raise RuntimeError("Expanded native response fitting requires a materialized native backend.")  # noqa: TRY004
    blocks: dict[str, tuple[str, ...]] = {
        "field_head": ("core.common.field_head",),
        "port_head": ("local_coupling.port_head",),
        "port_refinement_head": ("local_coupling.port_refinement_head",),
        "native_module_environment_updates": (
            "core.backend.mm_message",
            "core.backend.me_message",
            "core.backend.em_message",
            "core.backend.module_update",
            "core.backend.env_update",
        ),
        "native_fine_receiver_reads": (
            "core.backend.query_module_message",
            "core.backend.query_module_output",
            "core.backend.env_query",
            "core.backend.env_attention",
            "core.backend.env_geometry_bias",
        ),
    }
    named_modules = dict(model.named_modules())
    missing_modules = sorted({name for names in blocks.values() for name in names} - set(named_modules))
    if missing_modules:
        raise RuntimeError(f"Expanded native response scope is missing required modules: {missing_modules}.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: dict[str, nn.Parameter] = {}
    parameter_counts: dict[str, int] = {}
    selected_modules: dict[str, list[str]] = {}
    selected_parameter_ids: set[int] = set()
    for label, module_names in blocks.items():
        selected_modules[label] = list(module_names)
        block_parameters: dict[str, nn.Parameter] = {}
        for module_name in module_names:
            module = named_modules[module_name]
            if not isinstance(module, nn.Module):
                raise TypeError(f"Expanded native block {module_name!r} is not a module.")
            parameters = dict(module.named_parameters(recurse=True))
            if not parameters:
                raise TypeError(f"Expanded native block {module_name!r} has no materialized parameters.")
            for relative_name, parameter in parameters.items():
                if isinstance(parameter, UninitializedParameter):
                    raise RuntimeError(f"Expanded native block {module_name!r} is still lazy.")  # noqa: TRY004
                full_name = f"{module_name}.{relative_name}" if relative_name else module_name
                if id(parameter) in selected_parameter_ids:
                    raise RuntimeError(f"Expanded native parameter is selected by multiple blocks: {full_name}.")
                selected_parameter_ids.add(id(parameter))
                block_parameters[full_name] = parameter
                selected[full_name] = parameter
                parameter.requires_grad_(True)
        parameter_counts[label] = sum(parameter.numel() for parameter in block_parameters.values())
    if not selected:
        raise RuntimeError("Expanded native response scope selected no parameters.")
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if set(trainable_names) != set(selected):
        raise RuntimeError(
            "Expanded native response scope mismatch: "
            f"expected {sorted(selected)}, got {sorted(trainable_names)}."
        )
    model.eval()
    stochastic_modules = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.p > 0.0
    ]
    training_modules = [name for name, module in model.named_modules() if module.training]
    if training_modules:
        raise RuntimeError(f"Deterministic expanded response mode left modules in training mode: {training_modules}.")
    return {
        "name": "native_expanded_response_interface",
        "trainable_module_names_by_block": selected_modules,
        "trainable_module_names": [name for names in selected_modules.values() for name in names],
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count_by_block": parameter_counts,
        "trainable_parameter_count": sum(parameter.numel() for parameter in selected.values()),
        "frozen_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
        "frozen_buffer_names": [name for name, _ in model.named_buffers()],
        "frozen_modules": [
            "global_encoder",
            "module_feature_encoder",
            "module_position_encoder",
            "env_encoder",
            "core.common.context_builders_except_field_head",
            "local_coupling.local_latent_fusion",
            "local_coupling.local_response_summary_proj",
            "local_coupling.flux_correction_head",
            "local_coupling.local_surrogate",
            "all_other_parameters_and_buffers",
        ],
        "model_mode": "eval_for_all_updates_and_absolute_stencil_states",
        "dropout_modules_disabled_by_eval": stochastic_modules,
        "non_eval_modules": training_modules,
    }


def _optimizer_hyperparameter_inventory(
    optimizer: torch.optim.Optimizer,
    *,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    actual_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    actual_decays = [float(group.get("weight_decay", 0.0)) for group in optimizer.param_groups]
    if not actual_lrs or any(not np.isclose(value, learning_rate, rtol=0.0, atol=1.0e-15) for value in actual_lrs):
        raise RuntimeError(f"Optimizer learning rate differs from resolved {learning_rate}: {actual_lrs}.")
    if any(not np.isclose(value, weight_decay, rtol=0.0, atol=1.0e-15) for value in actual_decays):
        raise RuntimeError(f"Optimizer weight decay differs from resolved {weight_decay}: {actual_decays}.")
    return {
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "optimizer_group_learning_rates": actual_lrs,
        "optimizer_group_weight_decays": actual_decays,
    }


def _parameter_inventory(model: torch.nn.Module) -> dict[str, Any]:
    uninitialized = [
        name for name, parameter in model.named_parameters()
        if isinstance(parameter, UninitializedParameter)
    ]
    if uninitialized:
        raise RuntimeError(f"Model has lazy parameters after role materialization: {uninitialized}.")
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    local = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("local_coupling.local_surrogate.")
    ]
    local_trainable = [name for name, parameter in local if parameter.requires_grad]
    return {
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameter_shapes": {name: list(parameter.shape) for name, parameter in trainable},
        "core_trainable_parameter_count": sum(
            parameter.numel() for name, parameter in trainable if name.startswith("core.")
        ),
        "core_trainable_parameter_names": [
            name for name, _ in trainable if name.startswith("core.")
        ],
        "local_surrogate_attached": bool(model.local_coupling.has_local_surrogate),
        "local_surrogate_parameter_count": sum(parameter.numel() for _, parameter in local),
        "local_surrogate_trainable_parameter_names": local_trainable,
        "local_surrogate_frozen": bool(local) and not local_trainable,
        "local_surrogate_frozen_by_loader": bool(model.local_coupling.local_surrogate_frozen),
        "local_surrogate_checkpoint_provenance": model.local_coupling.local_surrogate_checkpoint_path,
        "local_surrogate_configured_checkpoint": model.config.channelthermal.local_surrogate_checkpoint_path,
        "optimizer_includes_all_trainable_parameters": True,
        "lazy_parameters_remaining": uninitialized,
    }


def _audit_named_buffers(
    model: torch.nn.Module,
    snapshot: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Compare registered buffers without assuming state_dict persistence.

    Lazy positional encoders can register a nonpersistent frequency buffer on
    their first real input. The audit compares common names, reports additions
    separately, and never indexes a state_dict with a name that may not be
    present there.
    """

    current = dict(model.named_buffers())
    previous_names = set(snapshot)
    current_names = set(current)
    missing = sorted(previous_names - current_names)
    added = sorted(current_names - previous_names)
    changed = sorted(
        name
        for name in previous_names & current_names
        if not torch.equal(current[name].detach().cpu(), snapshot[name].detach().cpu())
    )
    persistent_names = set(model.state_dict())
    nonpersistent = sorted(name for name in current_names if name not in persistent_names)
    persistent_additions = sorted(set(added) - set(nonpersistent))
    return {
        "passed": not missing and not changed and not persistent_additions,
        "buffer_count": len(current),
        "missing_names": missing,
        "added_names": added,
        "persistent_added_names": persistent_additions,
        "changed_names": changed,
        "nonpersistent_names": nonpersistent,
        "added_nonpersistent_names": sorted(set(added) & set(nonpersistent)),
    }


def _restore_review_arm_state(
    model: torch.nn.Module,
    arm: str,
    checkpoint_payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    """Load the named arm state before reusing a shared review operator."""

    payload = checkpoint_payloads.get(arm)
    if payload is None or not isinstance(payload.get("model"), Mapping):
        raise ValueError(f"The matched review checkpoint for {arm!r} is missing its model state.")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()


def _verify_optimizer_inventory(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, int | bool]:
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if actual != expected:
        raise RuntimeError(
            f"Optimizer parameter mismatch: {len(expected - actual)} trainable parameters omitted, "
            f"{len(actual - expected)} nontrainable/duplicate parameters included."
        )
    return {
        "trainable_parameter_tensors": len(expected),
        "optimizer_parameter_tensors": len(actual),
        "all_trainable_parameters_in_optimizer": True,
    }


def _make_input_template(dataset: GlobalChannelThermalDataset) -> dict[str, Any]:
    if len(dataset) == 0:
        raise ValueError("Checkpoint packed dataset has no train cases for module-slot capacity.")
    sample = dataset[0]
    structure = sample.get("structure")
    if not isinstance(structure, Mapping):
        raise TypeError("Packed dataset sample has no structure mapping.")
    # Only these input-schema arrays reach the native callback. Targets and
    # masks from the sample are deliberately dropped at this boundary.
    return {
        "structure": {
            "module_centers": np.array(structure["module_centers"], copy=True),
            "material_params": np.array(structure["material_params"], copy=True),
        }
    }


def _rms_from_arrays(values: Sequence[np.ndarray], masks: Sequence[np.ndarray], channel: int) -> float:
    squares: list[np.ndarray] = []
    for array, mask in zip(values, masks):
        target = np.asarray(array, dtype=np.float64)
        valid = np.asarray(mask, dtype=bool)
        if valid.ndim == 1:
            valid = np.broadcast_to(valid[:, None], target.shape)
        selected = target[valid[:, channel], channel]
        if selected.size:
            squares.append(np.square(selected))
    if not squares:
        return 1.0
    return max(float(np.sqrt(np.mean(np.concatenate(squares)))), 1.0e-6)


def _historical_calibration_case_map(
    family_ids: Sequence[str], historical_value_source: HistoricalValueSource
) -> dict[str, str]:
    """Assign one distinct, deterministically ordered historical train case per family slot."""

    ordered_families = tuple(sorted(set(family_ids)))
    if len(historical_value_source.case_ids) < len(ordered_families):
        raise ValueError("Balanced calibration needs at least one historical train case per family.")
    selected = tuple(historical_value_source.case_ids[:len(ordered_families)])
    if len(set(selected)) != len(selected):
        raise ValueError("Balanced calibration historical case IDs must be distinct.")
    return dict(zip(ordered_families, selected, strict=True))


def derive_training_scales(
    stencils: Sequence[ResponseStencil],
    *,
    smooth_peak_beta: float = 1.0,
    historical_value_source: HistoricalValueSource | None = None,
) -> ThermalLossScales:
    """Freeze train-only scales with equal weight per physical family.

    When supplied, one historical absolute-value record contributes to each
    family's absolute role scale. Finite and pressure-response scales remain
    derived from matched stencil changes because historical cases have no
    paired perturbation labels.
    """

    if not stencils or any(stencil.split is not EvidenceSplit.TRAIN for stencil in stencils):
        raise ValueError("Loss-scale derivation requires only nonempty train stencils.")
    family_groups: dict[str, list[ResponseStencil]] = {}
    for stencil in stencils:
        family_groups.setdefault(stencil.physical_family_id, []).append(stencil)
    family_ids = tuple(sorted(family_groups))
    roles = tuple(stencils[0].baseline.output.roles)  # type: ignore[union-attr]
    historical_by_family: dict[str, Any] = {}
    if historical_value_source is not None:
        historical_case_map = _historical_calibration_case_map(family_ids, historical_value_source)
        historical_by_family = {
            family_id: historical_value_source.load(case_id)
            for family_id, case_id in historical_case_map.items()
        }
        for family_id, record in historical_by_family.items():
            if record.output is None or record.design.split is not EvidenceSplit.TRAIN:
                raise ValueError(
                    f"Historical scale example for {family_id!r} must be a solved train record."
                )
            if set(roles) - set(record.output.roles):
                raise ValueError(
                    f"Historical scale example for {family_id!r} is missing a required role."
                )
    value_scales: dict[str, tuple[float, ...]] = {}
    finite_scales: dict[str, tuple[float, ...]] = {}
    mixed_scales: dict[str, tuple[float, ...]] = {}
    for role_name in roles:
        family_role_examples = {
            family: [
                record.output.roles[role_name]
                for stencil in group_stencils
                for record in stencil.records
                if record.output is not None
            ] + (
                [historical_by_family[family].output.roles[role_name]]
                if family in historical_by_family
                else []
            )
            for family, group_stencils in family_groups.items()
        }
        # One RMS per physical family, then an equal-family RMS. This keeps
        # larger module counts and query panels from setting the global scale.
        role_examples = next(iter(family_role_examples.values()))
        channels = len(role_examples[0].channel_names)
        value_scales[role_name] = tuple(
            max(float(np.sqrt(np.mean([
                _rms_from_arrays(
                    [role.values for role in family_role_examples[family]],
                    [role.valid_mask for role in family_role_examples[family]],
                    channel,
                ) ** 2
                for family in sorted(family_role_examples)
            ]))), 1.0e-6)
            for channel in range(channels)
        )
        finite_scales[role_name] = tuple(
            max(float(np.sqrt(np.mean([
                _rms_from_arrays(
                    [block.delta for stencil in family_groups[family] for block in (
                        stencil.finite_change(label, role_name) for label in stencil.variants
                    )],
                    [block.valid_mask for stencil in family_groups[family] for block in (
                        stencil.finite_change(label, role_name) for label in stencil.variants
                    )],
                    channel,
                ) ** 2
                for family in sorted(family_groups)
            ]))), 1.0e-6)
            for channel in range(channels)
        )
        # There is no mixed noise floor in the current atlas. These scales
        # define units only; the loss builder leaves those labels unknown.
        mixed_scales[role_name] = finite_scales[role_name]

    baseline_pressure = {
        family: [
            float(stencil.baseline.output.quantities["pressure_drop"].value)  # type: ignore[union-attr]
            for stencil in group
        ]
        for family, group in family_groups.items()
    }
    pressure_deltas = {
        family: [
            float(stencil.variants[label].output.quantities["pressure_drop"].value)  # type: ignore[union-attr]
            - float(stencil.baseline.output.quantities["pressure_drop"].value)  # type: ignore[union-attr]
            for stencil in group
            for label in stencil.variants
        ]
        for family, group in family_groups.items()
    }
    pressure_value = max(float(np.sqrt(np.mean([
        np.mean(np.square(values)) for values in baseline_pressure.values()
    ]))), 1.0e-6)
    pressure_response = max(float(np.sqrt(np.mean([
        np.mean(np.square(values)) for values in pressure_deltas.values()
    ]))), pressure_value * 1.0e-4, 1.0e-6)
    peak_values = {
        family: [
            float(value)
            for stencil in group
            for record in stencil.records
            for value in (record.output.module_peak_temperature.values() if record.output else ())
        ]
        for family, group in family_groups.items()
    }
    solid_scale = max(float(np.sqrt(np.mean([
        np.mean(np.square(values)) for values in peak_values.values()
    ]))), 1.0e-6)
    family_pressure_means = [float(np.mean(values)) for values in baseline_pressure.values()]
    scales = ThermalLossScales(
        value=value_scales,
        finite=finite_scales,
        mixed=mixed_scales,
        pressure_value=pressure_value,
        pressure_response=pressure_response,
        pressure_limit=float(np.median(family_pressure_means) * 1.05),
        pressure_boundary=max(pressure_value * 0.05, 1.0e-6),
        solid_temperature=solid_scale,
        smooth_peak_beta=float(smooth_peak_beta),
        near_limit_band=max(pressure_value * 0.05, 1.0e-6),
    )
    return scales.with_train_reference_limits(stencils)


def _load_frozen_loss_scales(
    path: Path,
    training_stencils: Sequence[ResponseStencil],
    *,
    smooth_peak_beta: float,
    expected_calibration_scope: str | None = None,
    expected_family_ids: Sequence[str] | None = None,
    expected_historical_case_map: Mapping[str, str] | None = None,
) -> tuple[ThermalLossScales, dict[str, Any]]:
    """Load and validate a train-only calibration frozen by an earlier recipe."""

    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    frozen = payload.get("frozen_scales", payload)
    if not isinstance(frozen, Mapping):
        raise TypeError("Frozen scale JSON must contain an object named frozen_scales.")
    if expected_calibration_scope is not None:
        if payload.get("calibration_scope") != expected_calibration_scope:
            raise ValueError(
                "Frozen R1 scales must carry the eight-family plus historical calibration scope."
            )
        if payload.get("equal_family_weighting") is not True:
            raise ValueError("Frozen R1 scales must declare equal-family weighting.")
        if list(payload.get("train_family_ids", ())) != sorted(set(expected_family_ids or ())):
            raise ValueError("Frozen R1 scale family IDs do not match the current eight-family panel.")
        expected_cases = dict(expected_historical_case_map or {})
        if dict(payload.get("historical_train_examples_by_family", {})) != expected_cases:
            raise ValueError("Frozen R1 historical calibration cases do not match the deterministic train cohort.")
    required = {
        "value",
        "finite",
        "mixed",
        "pressure_value",
        "pressure_response",
        "pressure_boundary",
        "solid_temperature",
        "smooth_peak_beta",
        "near_limit_band",
        "pressure_limit_by_family",
    }
    missing = required - set(frozen)
    if missing:
        raise ValueError(f"Frozen loss scales are missing fields: {sorted(missing)}")
    if not training_stencils or any(stencil.split is not EvidenceSplit.TRAIN for stencil in training_stencils):
        raise ValueError("Frozen scale reconstruction requires only train stencils.")
    train_baseline_pressures: dict[str, float] = {}
    for stencil in training_stencils:
        output = stencil.baseline.output
        if output is None:
            raise ValueError("Frozen pressure limits require train baseline outputs.")
        pressure = output.quantities["pressure_drop"]
        if not pressure.resolved:
            raise ValueError("Frozen pressure limits require resolved train baseline pressures.")
        train_baseline_pressures[stencil.physical_family_id] = float(pressure.value)
    derived_scalar_limit = float(np.median(list(train_baseline_pressures.values())) * 1.05)
    if "pressure_limit" in frozen and not np.isclose(
        float(frozen["pressure_limit"]), derived_scalar_limit, rtol=1.0e-10, atol=1.0e-12
    ):
        raise ValueError("Frozen scalar pressure limit does not match the train baseline median.")
    scales = ThermalLossScales(
        value=frozen["value"],
        finite=frozen["finite"],
        mixed=frozen["mixed"],
        pressure_value=float(frozen["pressure_value"]),
        pressure_response=float(frozen["pressure_response"]),
        pressure_limit=derived_scalar_limit,
        pressure_boundary=float(frozen["pressure_boundary"]),
        solid_temperature=float(frozen["solid_temperature"]),
        smooth_peak_beta=float(frozen["smooth_peak_beta"]),
        near_limit_band=float(frozen["near_limit_band"]),
        near_limit_multiplier=float(frozen.get("near_limit_multiplier", 2.0)),
        pressure_limit_by_family=frozen["pressure_limit_by_family"],
    )
    family_ids = {stencil.physical_family_id for stencil in training_stencils}
    if set(scales.pressure_limit_by_family or {}) != family_ids:
        raise ValueError("Frozen per-family pressure limits must exactly match the training panel.")
    expected_family_limits = {
        family_id: pressure * 1.05 for family_id, pressure in train_baseline_pressures.items()
    }
    for family_id, expected_limit in expected_family_limits.items():
        if not np.isclose(
            scales.pressure_limit_by_family[family_id], expected_limit, rtol=1.0e-10, atol=1.0e-12
        ):
            raise ValueError(
                f"Frozen pressure limit for {family_id!r} does not match 1.05x its train baseline."
            )
    if not np.isclose(scales.smooth_peak_beta, smooth_peak_beta, rtol=0.0, atol=1.0e-12):
        raise ValueError("Frozen smooth_peak_beta differs from the requested training functional.")
    return scales, {
        "path": str(source),
        "sha256": _checkpoint_digest(source),
        "scope": "frozen_train_only_recipe_calibration",
        "family_ids": sorted(family_ids),
    }


def _load_frozen_response_weights(
    path: Path, *, required_terms: Sequence[str] | None = None
) -> tuple[dict[str, float], dict[str, Any]]:
    """Load a previously calibrated paired response multiplier set."""

    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_weights = payload.get("calibrated_response_weights")
    if not isinstance(raw_weights, Mapping):
        raise TypeError("Frozen weight JSON must contain calibrated_response_weights.")
    weights = {str(name): float(value) for name, value in raw_weights.items()}
    required = (
        {"value", *required_terms}
        if required_terms is not None
        else {"value", "finite", "decision", "constraint"}
    )
    if not required.issubset(weights):
        raise ValueError(f"Frozen response weights are missing: {sorted(required - set(weights))}")
    if any(not np.isfinite(value) or value <= 0.0 for value in weights.values()):
        raise ValueError("Frozen response weights must be positive and finite.")
    if not np.isclose(weights["value"], 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("The frozen response calibration must keep the value multiplier at one.")
    return weights, {
        "path": str(source),
        "sha256": _checkpoint_digest(source),
        "scope": "frozen_train_only_recipe_calibration",
    }


def _validate_review_gate(
    review_cap: int,
    config: StagedTrainingConfig,
    training_stencil_count: int,
) -> int:
    """Return the actual update cap and reject review gates beyond it."""

    if training_stencil_count <= 0:
        raise ValueError("At least one training stencil is required to compute the update cap.")
    if review_cap not in config.review_updates:
        raise ValueError(f"Review gate {review_cap} is not configured: {config.review_updates}.")
    effective_cap = min(
        config.max_optimizer_updates,
        config.total_optimizer_update_ceiling,
        config.max_epochs * training_stencil_count,
    )
    if review_cap > effective_cap:
        raise ValueError(
            f"Requested review gate {review_cap} is unreachable: {training_stencil_count} train "
            f"stencils x max_epochs={config.max_epochs} limit each arm to {effective_cap} updates."
        )
    return effective_cap


def _review_stage_cap_record(
    *,
    review_cap: int,
    resume_payloads: Mapping[str, Mapping[str, Any]] | None,
    review_continuations: Sequence[int],
) -> dict[str, Any]:
    """Serialize the cumulative update range and stop behavior for one fit segment."""

    start_updates = (
        {int(payload.get("actual_optimizer_updates", -1)) for payload in resume_payloads.values()}
        if resume_payloads is not None
        else {0}
    )
    if len(start_updates) != 1 or min(start_updates) < 0:
        raise ValueError("A paired review stage requires both arms at one valid resume update.")
    start_update = start_updates.pop()
    if review_cap <= start_update:
        raise ValueError("The paired review stage cap must be ahead of its resume update.")
    continuations = sorted({int(update) for update in review_continuations})
    if any(update >= review_cap for update in continuations):
        raise ValueError("A review continuation cannot reach or pass the selected stage cap.")
    return {
        "start_update": start_update,
        "stop_update": int(review_cap),
        "new_update_budget_per_arm": int(review_cap - start_update),
        "review_continuations": continuations,
        "stop_at_cap": True,
    }


def _mixed_specs(stencil: ResponseStencil) -> tuple[MixedResponseSpec, ...]:
    variants = set(stencil.variants)
    corners = (("mm", "i_minus", "j_minus"), ("mp", "i_minus", "j_plus"),
               ("pm", "i_plus", "j_minus"), ("pp", "i_plus", "j_plus"))
    roles = tuple(stencil.baseline.output.roles)  # type: ignore[union-attr]
    return tuple(
        MixedResponseSpec(role, joint, first, second, f"{role}_{joint}")
        for joint, first, second in corners
        if {joint, first, second}.issubset(variants)
        for role in roles
    )


def _config_for_one_update(max_wall_seconds: float) -> StagedTrainingConfig:
    return StagedTrainingConfig(
        arm="B_response",
        max_optimizer_updates=1,
        max_epochs=1,
        total_optimizer_update_ceiling=20000,
        checkpoint_every_updates=1,
        review_updates=(),
        random_seed=2317,
        stages=(TrainingStage("preflight_value", 0, 1, ("value",)),),
        max_wall_seconds=max_wall_seconds,
    )


def _roundtrip_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: StagedTrainingConfig,
    payload: Mapping[str, Any],
    label: str,
) -> Path:
    path = output_dir / f"response_control_{label}.pt"
    _atomic_torch_save(path, payload)
    restored_payload = _load_safe_response_checkpoint(path)
    restore_checkpoint_payload(model, optimizer, restored_payload, config=config)
    return path


def _all_close_predictions(left: Any, right: Any, *, atol: float = 1.0e-6) -> bool:
    if set(left.role_values) != set(right.role_values):
        return False
    values_match = all(
        torch.allclose(left.role_values[name], right.role_values[name], atol=atol, rtol=1.0e-6)
        for name in left.role_values
    )
    if not values_match:
        return False
    if left.receiver_world_xy is None or right.receiver_world_xy is None:
        return left.receiver_world_xy is right.receiver_world_xy
    return all(
        torch.allclose(left.receiver_world_xy[name], right.receiver_world_xy[name], atol=atol, rtol=1.0e-6)
        for name in left.receiver_world_xy
    )


def _prediction_difference_summary(
    left: Any,
    right: Any,
    role_queries: Mapping[str, RoleQuery],
) -> dict[str, Any]:
    def summarize(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
        difference = (a - b).detach().abs()
        scale = torch.maximum(a.detach().abs(), b.detach().abs()).clamp_min(1.0e-12)
        return {
            "max_absolute": float(difference.max().cpu()),
            "max_relative": float((difference / scale).max().cpu()),
            "allclose_at_2e-5_atol_1e-6_rtol": bool(
                torch.allclose(a, b, atol=2.0e-5, rtol=1.0e-6)
            ),
            "allclose_at_3e-6_atol_3e-5_rtol": bool(
                torch.allclose(a, b, atol=3.0e-6, rtol=3.0e-5)
            ),
        }

    result: dict[str, Any] = {
        "role_values": {
            name: summarize(left.role_values[name], right.role_values[name])
            for name in left.role_values
        },
        "role_channels": {},
    }
    for name, query in role_queries.items():
        a = left.role_values[name]
        b = right.role_values[name]
        channel_rows = []
        for index, (channel_name, unit) in enumerate(zip(query.channel_names, query.channel_units)):
            channel_a = a[:, index]
            channel_b = b[:, index]
            item = summarize(channel_a, channel_b)
            item.update(
                {
                    "channel": channel_name,
                    "unit": unit,
                    "reference_max_absolute": float(channel_a.detach().abs().max().cpu()),
                    "candidate_max_absolute": float(channel_b.detach().abs().max().cpu()),
                }
            )
            channel_scale = max(
                item["reference_max_absolute"], item["candidate_max_absolute"]
            )
            tolerance = 3.0e-6 + 3.0e-6 * channel_scale
            item.update(
                {
                    "permutation_absolute_tolerance": 3.0e-6,
                    "permutation_relative_channel_scale_tolerance": 3.0e-6,
                    "permutation_allowed_max_absolute": tolerance,
                    "permutation_scale_gate_passed": item["max_absolute"] <= tolerance,
                }
            )
            channel_rows.append(item)
        result["role_channels"][name] = channel_rows
    result["permutation_world_coordinates_passed"] = True
    if left.receiver_world_xy is not None and right.receiver_world_xy is not None:
        result["receiver_world_xy"] = {
            name: summarize(left.receiver_world_xy[name], right.receiver_world_xy[name])
            for name in left.receiver_world_xy
        }
        result["permutation_world_coordinates_passed"] = all(
            torch.allclose(
                left.receiver_world_xy[name],
                right.receiver_world_xy[name],
                atol=1.0e-6,
                rtol=1.0e-6,
            )
            for name in left.receiver_world_xy
        )
    result["permutation_equivalence_criterion"] = (
        "per channel max_abs_delta <= 3e-6 + 3e-6*max_abs_channel; "
        "receiver_world_xy atol=1e-6, rtol=1e-6"
    )
    result["permutation_gate_passed"] = bool(
        result["permutation_world_coordinates_passed"]
        and all(
            item["permutation_scale_gate_passed"]
            for rows in result["role_channels"].values()
            for item in rows
        )
    )
    return result


def _poison_targets(stencil: ResponseStencil) -> ResponseStencil:
    record = stencil.baseline
    if record.output is None:
        raise ValueError("Target-poison regression requires a solved baseline.")
    poisoned_roles = {
        name: replace(
            role,
            values=np.full_like(role.values, 9.87654321e5),
            valid_mask=np.zeros_like(role.valid_mask, dtype=bool),
            noise_floor=None,
        )
        for name, role in record.output.roles.items()
    }
    quantities = dict(record.output.quantities)
    pressure = quantities["pressure_drop"]
    quantities["pressure_drop"] = MeasuredQuantity(
        -9.87654321e8, pressure.units, resolved=True, metadata={"poisoned": True}
    )
    poisoned_output = replace(
        record.output,
        roles=poisoned_roles,
        quantities=quantities,
        module_peak_temperature={
            module_id: -9.87654321e7 for module_id in record.output.active_module_ids
        },
    )
    poisoned_record = replace(record, output=poisoned_output)
    return ResponseStencil(poisoned_record, dict(stencil.variants))


def _permuted_role_queries(
    queries: Mapping[str, RoleQuery], permutation: torch.Tensor
) -> dict[str, RoleQuery]:
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    result = {}
    for name, query in queries.items():
        slots = query.receiver_slots
        if slots is not None:
            slot_tensor = torch.as_tensor(slots, dtype=torch.long, device=permutation.device)
            slots = tuple(int(value) for value in inverse[slot_tensor].detach().cpu().tolist())
        result[name] = RoleQuery(
            role=query.role,
            query_features=query.query_features,
            channel_names=query.channel_names,
            channel_units=query.channel_units,
            receiver_slots=slots,
            coordinate_kind=query.coordinate_kind,
        )
    return result


def _pressure_counts(stencil: ResponseStencil, device: torch.device) -> dict[str, int]:
    queries = role_queries_from_stencil(stencil, device=device)
    design = DesignInput.from_state(stencil.baseline.design, device=device)
    inlet, outlet, _ = pressure_section_masks(
        queries["fluid_fields"], design, context_inputs(stencil.baseline.context)
    )
    return {
        "inlet_queries_retained": int(inlet.sum().detach().cpu()),
        "outlet_queries_retained": int(outlet.sum().detach().cpu()),
    }


def _sampling_summary_mapping(summary: SamplingSummary) -> dict[str, Any]:
    """Convert immutable sampling metadata to plain JSON-ready containers."""

    return {
        "physical_family_id": summary.physical_family_id,
        "split": summary.split,
        "original_counts": dict(summary.original_counts),
        "sampled_counts": dict(summary.sampled_counts),
        "protected_counts": dict(summary.protected_counts),
        "solid_peak_query_coverage": dict(summary.solid_peak_query_coverage),
        "inverse_probability_weighting": summary.inverse_probability_weighting,
    }


def _select_geometry_x_probe(stencil: ResponseStencil) -> tuple[int, float]:
    """Choose an active module and a centered x step inside the domain."""

    modules = stencil.baseline.design.modules
    domain_x = float(stencil.baseline.context.values["domain_length_x"])
    module_radius = float(stencil.baseline.context.values["module_radius"])
    candidates = [
        (index, float(module.position_xy[0]))
        for index, module in enumerate(modules)
        if module.active
    ]
    if not candidates:
        raise ValueError("Geometry AD/FD requires at least one active module.")
    clearances = [
        min(position_x - module_radius, domain_x - module_radius - position_x)
        for _, position_x in candidates
    ]
    best_index = int(np.argmax(clearances))
    clearance = float(clearances[best_index])
    if clearance <= 0.0:
        raise ValueError("No active module center has positive x clearance for the geometry AD/FD check.")
    return candidates[best_index][0], min(1.0e-3, clearance * 0.25)


def run_one_update_preflight(
    *,
    checkpoint_path: Path,
    stencil_path: Path,
    dataset_path: Path | None,
    output_dir: Path,
    device: torch.device,
    sampling: ReceiverSamplingConfig,
    max_wall_seconds: float,
    initialization_mode: str = "three_term_conversion",
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    smooth_peak_beta: float = 1.0,
    query_batch_size: int = 2048,
) -> dict[str, Any]:
    """Run exactly one successful optimizer update from one train family."""

    started = time.monotonic()
    source_model, checkpoint = load_model(checkpoint_path, device)
    source_model.eval()
    stencil, _ = load_response_atlas_stencil(stencil_path)
    if stencil.split is not EvidenceSplit.TRAIN:
        raise ValueError("The one-update preflight requires one EvidenceSplit.TRAIN stencil.")
    sampled = sample_training_panel((stencil,), config=sampling)[0]
    sampled_stencil = sampled.stencil
    scales = derive_training_scales((sampled_stencil,), smooth_peak_beta=smooth_peak_beta)
    dataset_root = _resolve_dataset_path(checkpoint, str(dataset_path) if dataset_path else None)
    train_config = checkpoint.get("train_config", {})
    dataset_config = train_config.get("dataset", {})
    raw_dataset = GlobalChannelThermalDataset(
        dataset_root,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    template = _make_input_template(raw_dataset)
    native_scope = None
    if initialization_mode == "native_checkpoint":
        target_model, refit_config = _native_checkpoint_initialization(source_model)
        native_scope = _configure_native_output_head_scope(target_model)
    elif initialization_mode == "three_term_conversion":
        target_model, refit_config = _make_refit_model(source_model, checkpoint, device=device)
    else:
        raise ValueError(f"Unsupported initialization_mode {initialization_mode!r}.")
    operator = DifferentiableThermalOperator(
        target_model,
        template,
        dataset_config=dataset_config,
        normalization_stats=checkpoint.get("global_normalization_stats", {}),
        query_batch_size=query_batch_size,
    )
    if initialization_mode == "native_checkpoint":
        # One target-free call verifies the complete native checkpoint and
        # materializes any lazy output shape without replacing or adapting it.
        native_queries = role_queries_from_stencil(sampled_stencil, device=device)
        native_design = DesignInput.from_state(sampled_stencil.baseline.design, device=device)
        with torch.no_grad():
            operator(native_design, context_inputs(sampled_stencil.baseline.context), native_queries)
        transfer = dict(refit_config)
        transfer["native_output_scope"] = native_scope
    else:
        transfer = _materialize_and_warm_start(
            source_model, target_model, operator, sampled_stencil, device=device
        )
    target_model.eval()

    regression_started = time.perf_counter()
    derivative_started = time.perf_counter()
    derivative = check_pressure_peak_ad_fd(
        operator,
        sampled_stencil,
        variable="heating",
        module_slot=0,
        step=max(abs(float(sampled_stencil.baseline.design.modules[0].heating)) * 1.0e-2, 1.0e-5),
        relative_tolerance=0.35,
        absolute_tolerance=1.0e-4,
        device=device,
    )
    if not derivative["passed"]:
        raise RuntimeError(f"Whole-wrapper pressure/peak AD-FD check failed: {derivative}")
    derivative_seconds = time.perf_counter() - derivative_started

    geometry_slot, geometry_step = _select_geometry_x_probe(sampled_stencil)
    geometry_started = time.perf_counter()
    geometry_derivative = check_pressure_peak_ad_fd(
        operator,
        sampled_stencil,
        variable="position_x",
        module_slot=geometry_slot,
        step=geometry_step,
        relative_tolerance=0.35,
        absolute_tolerance=1.0e-4,
        device=device,
    )
    if not geometry_derivative["passed"]:
        raise RuntimeError(
            f"Whole-wrapper position pressure/peak AD-FD check failed: {geometry_derivative}"
        )
    geometry_derivative_seconds = time.perf_counter() - geometry_started

    # Target leakage regression: poisoned atlas labels and poisoned irrelevant
    # packed outputs do not enter the target-free native callback.
    original_queries = role_queries_from_stencil(sampled_stencil, device=device)
    design = DesignInput.from_state(sampled_stencil.baseline.design, device=device)
    context = context_inputs(sampled_stencil.baseline.context)
    with torch.no_grad():
        first = operator(design, context, original_queries)
        poison = {
            "structure": {
                "module_centers": np.full_like(template["structure"]["module_centers"], 777.0),
                "material_params": np.full_like(template["structure"]["material_params"], -123.0),
            },
            "field_targets": np.full((7, 3), -9.0, dtype=np.float32),
            "interface_condition": np.full((5, 8), -8.0e4, dtype=np.float32),
            "teacher_port_tokens": np.full((5, 5), 3.0e6, dtype=np.float32),
            "interface_target": np.full((5, 2), 1.0e4, dtype=np.float32),
            "module_internal_temperature_points": np.full((9, 1), 9.0e5, dtype=np.float32),
            "module_internal_mask": np.zeros((9,), dtype=np.uint8),
            "interface_condition_valid_mask": np.zeros((5,), dtype=np.uint8),
        }
        poisoned_operator = DifferentiableThermalOperator(
            target_model,
            poison,
            dataset_config=dataset_config,
            normalization_stats=checkpoint.get("global_normalization_stats", {}),
            query_batch_size=query_batch_size,
        )
        poisoned = poisoned_operator(design, context, original_queries)
    if not _all_close_predictions(first, poisoned):
        raise AssertionError("Native operator changed when target/irrelevant sample arrays were poisoned.")
    poisoned_stencil = _poison_targets(sampled_stencil)
    poisoned_queries = role_queries_from_stencil(poisoned_stencil, device=device)
    with torch.no_grad():
        poisoned_labels = operator(
            design, context_inputs(poisoned_stencil.baseline.context), poisoned_queries
        )
    if not _all_close_predictions(first, poisoned_labels):
        raise AssertionError(
            "Native prediction consumed atlas role values, masks, scalar targets, or peak labels."
        )

    # Re changes both an explicit input and its family-specific material
    # vector; a static packed template must not overwrite either value.
    changed_context = dict(context)
    changed_context["re"] = float(changed_context["re"]) * 1.07
    changed_context["nu"] = float(changed_context["nu"]) / 1.07
    original_inputs = operator._context_structure(context)
    changed_inputs = operator._context_structure(changed_context)
    if torch.equal(original_inputs["re"], changed_inputs["re"]) or torch.equal(
        original_inputs["material_params"], changed_inputs["material_params"]
    ):
        raise AssertionError("Per-stencil Reynolds/material context did not reach the native model inputs.")
    with torch.no_grad():
        changed_re_prediction = operator(design, changed_context, original_queries)
    if _all_close_predictions(first, changed_re_prediction, atol=1.0e-8):
        raise AssertionError("Changing per-stencil Re/nu did not change the native absolute prediction.")

    module_count = int(design.module_positions.shape[0])
    permutation = torch.arange(module_count - 1, -1, -1, device=device)
    permuted_design = DesignInput(
        design.module_positions.index_select(0, permutation),
        design.module_heating.index_select(0, permutation),
        design.module_present.index_select(0, permutation),
    )
    permuted_queries = _permuted_role_queries(original_queries, permutation)
    with torch.no_grad():
        permuted_prediction = operator(permuted_design, context, permuted_queries)
    permutation_diagnostic = _prediction_difference_summary(
        first, permuted_prediction, original_queries
    )
    if not permutation_diagnostic["permutation_gate_passed"]:
        raise AssertionError(
            "Native outputs changed under module-row permutation with receiver slots remapped: "
            + json.dumps(permutation_diagnostic, sort_keys=True)
        )

    spectator_changed = False
    if module_count >= 3:
        spectator_positions = design.module_positions.detach().clone()
        spectator_positions[-1, 0] += 0.03
        spectator_design = DesignInput(
            spectator_positions, design.module_heating, design.module_present
        )
        with torch.no_grad():
            spectator_prediction = operator(spectator_design, context, original_queries)
        spectator_changed = any(
            not torch.allclose(
                first.role_values[name],
                spectator_prediction.role_values[name],
                atol=1.0e-8,
                rtol=1.0e-7,
            )
            for name in first.role_values
        )
        if not spectator_changed:
            raise AssertionError("Moving an unchanged spectator did not change the recomputed absolute prediction.")
    regression_seconds = time.perf_counter() - regression_started
    parameter_inventory = _parameter_inventory(target_model)

    target_model.train()
    model_config = checkpoint.get("train_config", {}).get("training", {})
    resolved_lr = float(
        learning_rate
        if learning_rate is not None
        else 1.0e-5
        if initialization_mode == "native_checkpoint"
        else model_config.get("learning_rate", 3.0e-4)
    )
    resolved_decay = float(
        weight_decay
        if weight_decay is not None
        else model_config.get("weight_decay", 1.0e-5)
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in target_model.parameters() if parameter.requires_grad),
        lr=resolved_lr,
        weight_decay=resolved_decay,
    )
    optimizer_inventory = _verify_optimizer_inventory(target_model, optimizer)
    optimizer_inventory.update(
        _optimizer_hyperparameter_inventory(
            optimizer, learning_rate=resolved_lr, weight_decay=resolved_decay
        )
    )
    config = _config_for_one_update(max_wall_seconds)
    checkpoint_paths: list[str] = []

    def save_and_roundtrip(payload: Mapping[str, Any], label: str) -> None:
        checkpoint_file = _roundtrip_checkpoint(output_dir, target_model, optimizer, config, payload, label)
        checkpoint_paths.append(str(checkpoint_file))

    preflight_peak_allocated = 0
    preflight_peak_reserved = 0
    free_bytes_before_fit = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        preflight_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        preflight_peak_reserved = int(torch.cuda.max_memory_reserved(device))
        free_bytes_before_fit, _ = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
    fit_started = time.perf_counter()
    fit = run_staged_fit(
        operator,
        target_model,
        optimizer,
        (sampled_stencil,),
        scales=scales,
        loss_weights={"value": 1.0},
        config=config,
        device=device,
        on_checkpoint=save_and_roundtrip,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - fit_started
    if fit.actual_optimizer_updates != 1 or fit.attempted_optimizer_steps != 1:
        raise RuntimeError(
            f"Expected exactly one successful update, got {fit.actual_optimizer_updates} completed "
            f"and {fit.attempted_optimizer_steps} attempted."
        )
    if len(checkpoint_paths) < 2:
        raise RuntimeError("Expected both zero-update and post-update checkpoint round trips.")
    return {
        "status": "passed",
        "mode": "one_update_preflight",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _checkpoint_digest(checkpoint_path),
        "selected_epoch": checkpoint.get("epoch", checkpoint.get("current_epoch")),
        "initialization_mode": initialization_mode,
        "atlas_npz": str(stencil_path.resolve()),
        "atlas_family_id": stencil.physical_family_id,
        "atlas_split": stencil.split.value,
        "atlas_context": dict(sampled_stencil.baseline.context.values),
        "atlas_record_count": len(sampled_stencil.records),
        "refit_config": refit_config,
        "warm_start": transfer,
        "parameter_inventory": parameter_inventory,
        "native_trainable_scope": native_scope,
        "optimizer_inventory": optimizer_inventory,
        "sampling": _sampling_summary_mapping(sampled.summary),
        "role_query_counts": {
            name: query.query_features.shape[0]
            for name, query in original_queries.items()
        },
        "pressure_sections": _pressure_counts(sampled_stencil, device),
        "whole_wrapper_ad_fd": derivative,
        "whole_wrapper_ad_fd_wall_seconds": derivative_seconds,
        "whole_wrapper_geometry_ad_fd": geometry_derivative,
        "whole_wrapper_geometry_ad_fd_wall_seconds": geometry_derivative_seconds,
        "wrapper_regression_wall_seconds": regression_seconds,
        "target_poison_invariance": True,
        "atlas_target_poison_invariance": True,
        "per_stencil_reynolds_material_route": True,
        "reynolds_change_prediction_changed": True,
        "module_row_permutation_invariance": True,
        "module_row_permutation_diagnostic": permutation_diagnostic,
        "spectator_intervention_changed_prediction": spectator_changed,
        "optimizer_updates": fit.actual_optimizer_updates,
        "attempted_optimizer_steps": fit.attempted_optimizer_steps,
        "losses": [_dataclass_record(step) for step in fit.history],
        "checkpoint_roundtrip_paths": checkpoint_paths,
        "fit_wall_seconds": fit_seconds,
        "total_wall_seconds": float(time.monotonic() - started),
        "cuda": _cuda_evidence(
            device,
            free_bytes_before_fit=free_bytes_before_fit,
            preflight_peak_allocated_bytes=preflight_peak_allocated,
            preflight_peak_reserved_bytes=preflight_peak_reserved,
        ),
    }


_R0_RESPONSE_TERMS = ("finite", "finite_peak", "pressure_value", "pressure_response")
_R0_TERMS = ("value", *_R0_RESPONSE_TERMS)


def _response_parameter_blocks(model: torch.nn.Module) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    groups: dict[str, list[torch.nn.Parameter]] = {
        "field_head": [],
        "port_head": [],
        "port_refinement_head": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("core.common.field_head."):
            groups["field_head"].append(parameter)
        elif name.startswith("local_coupling.port_head."):
            groups["port_head"].append(parameter)
        elif name.startswith("local_coupling.port_refinement_head."):
            groups["port_refinement_head"].append(parameter)
        else:
            raise ValueError(f"Trainable parameter {name!r} is outside the response blocks.")
    if any(not values for values in groups.values()):
        raise ValueError(f"Native response scope is missing a trainable block: {groups.keys()}.")
    return {name: tuple(values) for name, values in groups.items()}


def _combine_r0_response_gradient(
    value_gradient: np.ndarray,
    response_gradients: Mapping[str, np.ndarray],
    response_weights: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Combine calibrated response gradients before deciding on projection."""

    value = np.asarray(value_gradient, dtype=np.float64).reshape(-1)
    if not response_gradients or set(response_gradients) != set(response_weights):
        raise ValueError("Projection evidence needs aligned response gradients and calibrated weights.")
    combined = np.zeros_like(value)
    for term, gradient in response_gradients.items():
        vector = np.asarray(gradient, dtype=np.float64).reshape(-1)
        if vector.shape != value.shape:
            raise ValueError(f"Response gradient {term!r} does not match the value-gradient block.")
        weight = float(response_weights[term])
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Response gradient weight {term!r} must be positive and finite.")
        combined += weight * vector
    dot = float(np.dot(value, combined))
    value_norm = float(np.linalg.norm(value))
    response_norm = float(np.linalg.norm(combined))
    denominator = value_norm * response_norm
    return combined, {
        "combined_value_gradient_norm": value_norm,
        "calibrated_combined_response_gradient_norm": response_norm,
        "dot_product": dot,
        "cosine": float(dot / denominator) if denominator > 0.0 else None,
        "adverse": bool(dot < 0.0),
        "response_term_weights": {name: float(value) for name, value in response_weights.items()},
    }


def _r0_gradient_loss_terms(
    stencil_losses: Mapping[str, torch.Tensor],
    historical_loss: torch.Tensor,
    calibrated_weights: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    """Build gradient objectives with keys matching calibrated loss names."""

    missing_weights = set(_R0_TERMS) - set(calibrated_weights)
    if missing_weights:
        raise ValueError(
            "R0 cannot construct calibrated gradient objectives; "
            f"missing weights for {sorted(missing_weights)}."
        )
    missing_losses = set(_R0_TERMS) - set(stencil_losses)
    if missing_losses:
        raise ValueError(
            "R0 cannot construct gradient objectives; "
            f"missing stencil losses for {sorted(missing_losses)}."
        )
    return {
        "stencil_value": stencil_losses["value"],
        "historical_value": historical_loss,
        "combined_value": float(calibrated_weights["value"])
        * (stencil_losses["value"] + historical_loss),
        **{term: stencil_losses[term] for term in _R0_RESPONSE_TERMS},
    }


def _r0_gradient_diagnostic(
    operator: DifferentiableThermalOperator,
    model: ChannelThermalHONFModel,
    stencils: Sequence[ResponseStencil],
    historical_source: HistoricalValueSource,
    scales: ThermalLossScales,
    calibrated_weights: Mapping[str, float],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Measure train-only task gradients by family and native head block."""

    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    blocks = _response_parameter_blocks(model)
    historical_ids = historical_source.case_ids[:len(stencils)]
    family_rows: list[dict[str, Any]] = []
    for index, stencil in enumerate(stencils):
        predictions = predict_stencil(operator, stencil, device=device)
        losses = compute_stencil_loss_terms(
            predictions,
            stencil,
            scales=scales,
            enabled_terms=_R0_TERMS,
            include_feasibility_bce=False,
        )
        historical_record = historical_source.load(historical_ids[index])
        historical_loss = historical_absolute_value_loss(
            operator, historical_record, scales=scales, device=device
        )
        gradient_losses = _r0_gradient_loss_terms(
            losses.terms,
            historical_loss,
            calibrated_weights,
        )
        block_vectors: dict[str, dict[str, np.ndarray]] = {term: {} for term in gradient_losses}
        for term_index, (term_name, loss) in enumerate(gradient_losses.items()):
            if not loss.requires_grad:
                gradients = tuple(None for _ in parameters)
            else:
                gradients = torch.autograd.grad(
                    loss,
                    parameters,
                    retain_graph=term_index + 1 < len(gradient_losses),
                    allow_unused=True,
                )
            gradient_by_id = {
                id(parameter): gradient
                for parameter, gradient in zip(parameters, gradients, strict=True)
            }
            for block_name, block_parameters in blocks.items():
                pieces = []
                for parameter in block_parameters:
                    gradient = gradient_by_id[id(parameter)]
                    pieces.append(
                        np.zeros(parameter.numel(), dtype=np.float64)
                        if gradient is None
                        else gradient.detach().reshape(-1).double().cpu().numpy()
                    )
                flat = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)
                block_vectors[term_name][block_name] = flat
            del gradients
        projection_gate: dict[str, dict[str, Any]] = {}
        for block_name in blocks:
            combined, evidence = _combine_r0_response_gradient(
                block_vectors["combined_value"][block_name],
                {
                    term: block_vectors[term][block_name]
                    for term in _R0_RESPONSE_TERMS
                },
                {term: float(calibrated_weights[term]) for term in _R0_RESPONSE_TERMS},
            )
            block_vectors.setdefault("combined_response", {})[block_name] = combined
            projection_gate[block_name] = evidence
        norms = {
            term: {block: float(np.linalg.norm(vector)) for block, vector in per_block.items()}
            for term, per_block in block_vectors.items()
        }
        cosines: dict[str, dict[str, dict[str, float | None]]] = {}
        for block_name in blocks:
            term_names = tuple(gradient_losses)
            cosines[block_name] = {}
            for left_index, left in enumerate(term_names):
                cosines[block_name][left] = {}
                for right in term_names[left_index + 1:]:
                    left_vector = block_vectors[left][block_name]
                    right_vector = block_vectors[right][block_name]
                    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
                    cosine = (
                        float(np.dot(left_vector, right_vector) / denominator)
                        if denominator > 0.0 else None
                    )
                    cosines[block_name][left][right] = cosine
        family_rows.append({
            "family_id": stencil.physical_family_id,
            "module_count": len(stencil.baseline.design.active_modules),
            "historical_case_id": historical_ids[index],
            "loss_values": {
                name: float(loss.detach().cpu()) for name, loss in gradient_losses.items()
            },
            "gradient_term_display_names": {"finite": "finite_field"},
            "gradient_norms_by_term_and_block": norms,
            "pairwise_gradient_cosines_by_block": cosines,
            "calibrated_combined_projection_gate_by_block": projection_gate,
        })
    return {
        "scope_parameter_counts": {
            name: sum(parameter.numel() for parameter in values)
            for name, values in blocks.items()
        },
        "families": family_rows,
        "family_equal_weighting": True,
        "historical_train_case_ids": list(historical_ids),
    }


def _evaluate_r0_train_objective(
    operator: DifferentiableThermalOperator,
    stencils: Sequence[ResponseStencil],
    historical_source: HistoricalValueSource,
    scales: ThermalLossScales,
    weights: Mapping[str, float],
    *,
    device: torch.device,
) -> dict[str, Any]:
    family_rows: list[dict[str, Any]] = []
    for index, stencil in enumerate(stencils):
        with torch.no_grad():
            predictions = predict_stencil(operator, stencil, device=device)
            losses = compute_stencil_loss_terms(
                predictions,
                stencil,
                scales=scales,
                enabled_terms=_R0_TERMS,
                include_feasibility_bce=False,
            )
            historical_loss = historical_absolute_value_loss(
                operator,
                historical_source.load(historical_source.case_ids[index]),
                scales=scales,
                device=device,
            )
        terms = {name: float(value.detach().cpu()) for name, value in losses.terms.items()}
        terms["historical_value"] = float(historical_loss.detach().cpu())
        value_objective = float(weights.get("value", 1.0)) * (
            terms.get("value", 0.0) + terms["historical_value"]
        )
        response_objective = sum(
            float(weights.get(name, 0.0)) * terms.get(name, 0.0)
            for name in _R0_RESPONSE_TERMS
        )
        family_rows.append({
            "family_id": stencil.physical_family_id,
            "module_count": len(stencil.baseline.design.active_modules),
            "term_losses": terms,
            "weighted_value_objective": value_objective,
            "weighted_response_objective": response_objective,
            "weighted_total_objective": value_objective + response_objective,
        })
    return {
        "families": family_rows,
        "equal_family_mean": {
            key: float(np.mean([row[key] for row in family_rows]))
            for key in (
                "weighted_value_objective",
                "weighted_response_objective",
                "weighted_total_objective",
            )
        },
        "equal_family_mean_term_losses": {
            term: float(np.mean([
                row["term_losses"].get(term, 0.0) for row in family_rows
            ]))
            for term in (*_R0_TERMS, "historical_value")
        },
    }


def _r0_fit_config(
    *, rate_probe: bool, max_wall_seconds: float, max_optimizer_updates: int = 100
) -> StagedTrainingConfig:
    if rate_probe:
        return StagedTrainingConfig(
            arm="R_response",
            max_optimizer_updates=10,
            max_epochs=3,
            total_optimizer_update_ceiling=400,
            checkpoint_every_updates=5,
            max_wall_seconds=max_wall_seconds,
            review_updates=(10,),
            random_seed=2317,
            deterministic_eval_mode=True,
            deterministic_algorithms=True,
            include_feasibility_bce=False,
            stages=(TrainingStage("rate_probe_response", 0, 10, _R0_TERMS),),
        )
    response_terms = ("value", *_R0_RESPONSE_TERMS)
    if max_optimizer_updates < 50:
        raise ValueError("An R0 scope comparison requires at least 50 updates so response terms are active.")
    return StagedTrainingConfig(
        arm="R_response",
        max_optimizer_updates=max_optimizer_updates,
        max_epochs=(max_optimizer_updates + 3) // 4,
        total_optimizer_update_ceiling=400,
        checkpoint_every_updates=50,
        max_wall_seconds=max_wall_seconds,
        review_updates=(max_optimizer_updates,),
        random_seed=2317,
        deterministic_eval_mode=True,
        deterministic_algorithms=True,
        include_feasibility_bce=False,
        response_ramp_start_update=10,
        response_ramp_end_update=50,
        response_ramp_terms=_R0_RESPONSE_TERMS,
        stages=(
            TrainingStage("value_warmup", 0, 10, ("value",)),
            TrainingStage("balanced_response_ramp", 10, 50, response_terms),
            TrainingStage("response_probe", 50, max_optimizer_updates, response_terms),
        ),
    )


def run_train_only_scope_diagnostic(
    *,
    checkpoint_path: Path,
    stencil_paths: Sequence[Path],
    dataset_path: Path | None,
    output_dir: Path,
    device: torch.device,
    sampling: ReceiverSamplingConfig,
    query_batch_size: int,
    max_wall_seconds: float = 1800.0,
    weight_decay: float = 1.0e-5,
) -> dict[str, Any]:
    """Run the bounded R0 train-only gradient, scope, and rate diagnosis."""

    started = time.monotonic()
    cublas_workspace = _enable_deterministic_algorithms(device)
    if max_wall_seconds <= 0.0:
        raise ValueError("R0 max_wall_seconds must be positive.")
    deadline = started + max_wall_seconds
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "r0_train_only_scope_diagnostic.json"
    progress: dict[str, Any] = {
        "status": "running",
        "mode": "train_only_scope_diagnostic",
        "reference_solver_calls": 0,
        "max_wall_seconds": float(max_wall_seconds),
        "deterministic_algorithms_enabled": True,
        "cublas_workspace_config": cublas_workspace,
        "planned_optimizer_updates_max": 180,
        "optimizer_updates_completed_total": 0,
        "attempted_optimizer_updates_total": 0,
        "current_phase": "input_validation_and_load",
        "phase_update_counts": {},
        "optimizer_trial_status": [],
        "started_unix_seconds": time.time(),
    }
    _atomic_json(result_path, progress)

    def save_progress() -> None:
        progress["total_wall_seconds"] = float(time.monotonic() - started)
        _atomic_json(result_path, progress)

    def ensure_time_remaining(phase: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            progress["current_phase"] = phase
            progress["status"] = "timed_out"
            progress["error"] = f"R0 total wall cap of {max_wall_seconds:.1f} seconds expired."
            progress["finished_unix_seconds"] = time.time()
            save_progress()
            raise TimeoutError(progress["error"])
        return remaining

    def begin_trial(name: str, update_cap: int, learning_rate: float) -> None:
        progress["current_phase"] = name
        progress["optimizer_trial_status"].append({
            "name": name,
            "learning_rate": float(learning_rate),
            "planned_update_cap": int(update_cap),
            "status": "running",
            "updates_completed": 0,
            "attempted_optimizer_steps": 0,
        })
        save_progress()

    def record_trial_checkpoint(name: str, payload: Mapping[str, Any], label: str) -> None:
        trial = next(row for row in reversed(progress["optimizer_trial_status"]) if row["name"] == name)
        trial["last_checkpoint_label"] = label
        trial["updates_completed"] = max(
            int(trial["updates_completed"]), int(payload["actual_optimizer_updates"])
        )
        trial["attempted_optimizer_steps"] = max(
            int(trial["attempted_optimizer_steps"]), int(payload["attempted_optimizer_steps"])
        )
        counts = progress["phase_update_counts"]
        counts[name] = {
            "updates_completed": trial["updates_completed"],
            "attempted_optimizer_steps": trial["attempted_optimizer_steps"],
        }
        progress["optimizer_updates_completed_total"] = sum(
            item["updates_completed"] for item in counts.values()
        )
        progress["attempted_optimizer_updates_total"] = sum(
            item["attempted_optimizer_steps"] for item in counts.values()
        )
        save_progress()

    def record_trial_attempt(name: str, completed: int, attempted: int) -> None:
        trial = next(row for row in reversed(progress["optimizer_trial_status"]) if row["name"] == name)
        trial["updates_completed"] = max(int(trial["updates_completed"]), int(completed))
        trial["attempted_optimizer_steps"] = max(
            int(trial["attempted_optimizer_steps"]), int(attempted)
        )
        counts = progress["phase_update_counts"]
        counts[name] = {
            "updates_completed": trial["updates_completed"],
            "attempted_optimizer_steps": trial["attempted_optimizer_steps"],
        }
        progress["optimizer_updates_completed_total"] = sum(
            item["updates_completed"] for item in counts.values()
        )
        progress["attempted_optimizer_updates_total"] = sum(
            item["attempted_optimizer_steps"] for item in counts.values()
        )
        save_progress()

    def finish_trial(name: str, fit: StagedFitResult, expected_updates: int) -> None:
        trial = next(row for row in reversed(progress["optimizer_trial_status"]) if row["name"] == name)
        complete = int(fit.actual_optimizer_updates) == int(expected_updates)
        trial.update({
            "status": "passed" if complete else "timed_out_or_partial",
            "updates_completed": int(fit.actual_optimizer_updates),
            "attempted_optimizer_steps": int(fit.attempted_optimizer_steps),
            "wall_seconds": float(fit.wall_seconds),
        })
        record_trial_checkpoint(name, {
            "actual_optimizer_updates": fit.actual_optimizer_updates,
            "attempted_optimizer_steps": fit.attempted_optimizer_steps,
        }, "fit_complete")
        if not complete:
            progress["status"] = "timed_out_or_partial"
            progress["error"] = (
                f"{name} completed {fit.actual_optimizer_updates}/{expected_updates} updates; "
                "partial checkpoints and attempted-step counts were preserved."
            )
            progress["finished_unix_seconds"] = time.time()
            save_progress()
            raise TimeoutError(progress["error"])

    if len(stencil_paths) != 4:
        raise ValueError("R0 requires exactly four preselected train families spanning M=3,5,7,10.")
    loaded = [load_response_atlas_stencil(path) for path in stencil_paths]
    raw_stencils = [stencil for stencil, _metadata in loaded]
    if any(stencil.split is not EvidenceSplit.TRAIN for stencil in raw_stencils):
        raise ValueError("R0 accepts train-split atlas stencils only.")
    module_counts = {len(stencil.baseline.design.active_modules) for stencil in raw_stencils}
    if module_counts != {3, 5, 7, 10} or len({stencil.physical_family_id for stencil in raw_stencils}) != 4:
        raise ValueError("R0 requires four distinct train families with module counts 3, 5, 7, and 10.")
    m10_family_ids = {
        stencil.physical_family_id
        for stencil in raw_stencils
        if len(stencil.baseline.design.active_modules) == 10
    }
    if m10_family_ids != {"stored_family:0350"}:
        raise ValueError(
            "R0 requires the established M10 conflict family stored_family:0350; "
            f"got {sorted(m10_family_ids)}."
        )
    source_model, checkpoint = load_model(checkpoint_path, device)
    if int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))) != 4738:
        raise ValueError("R0 is fixed to the intact Run1804 e4738 checkpoint.")
    if str(source_model.config.core_honf.forward_architecture) != "dense_pairwise_field":
        raise ValueError("R0 requires the intact Run1804 dense_pairwise_field architecture.")
    source_model.eval()
    dataset_root = _resolve_dataset_path(checkpoint, str(dataset_path) if dataset_path else None)
    train_config = checkpoint.get("train_config", {})
    dataset_config = train_config.get("dataset", {})
    raw_dataset = GlobalChannelThermalDataset(
        dataset_root,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    template = _make_input_template(raw_dataset)
    source_operator = DifferentiableThermalOperator(
        source_model,
        template,
        dataset_config=dataset_config,
        normalization_stats=checkpoint.get("global_normalization_stats", {}),
        query_batch_size=query_batch_size,
    )
    historical_sampling = replace(sampling, max_fluid_queries=max(3072, sampling.max_fluid_queries))
    historical_source = HistoricalValueSource.from_dataset(raw_dataset, sampling=historical_sampling)
    sampled_panel = sample_training_panel(raw_stencils, config=sampling)
    training_stencils = tuple(item.stencil for item in sampled_panel)
    scales = derive_training_scales(training_stencils, smooth_peak_beta=1.0)
    progress.update({
        "checkpoint": str(checkpoint_path.expanduser().resolve()),
        "training_atlas_paths": [str(path.expanduser().resolve()) for path in stencil_paths],
        "family_ids": [stencil.physical_family_id for stencil in training_stencils],
        "module_counts": [len(stencil.baseline.design.active_modules) for stencil in training_stencils],
        "current_phase": "update_zero_scope_parity_and_gradient_diagnostics",
        "reference_solver_calls": 0,
    })
    save_progress()

    def build_scope(scope_name: str) -> tuple[ChannelThermalHONFModel, DifferentiableThermalOperator, dict[str, Any]]:
        target_model = copy.deepcopy(source_model)
        operator = DifferentiableThermalOperator(
            target_model,
            template,
            dataset_config=dataset_config,
            normalization_stats=checkpoint.get("global_normalization_stats", {}),
            query_batch_size=query_batch_size,
        )
        first = training_stencils[0]
        with torch.no_grad():
            operator(
                DesignInput.from_state(first.baseline.design, device=device),
                context_inputs(first.baseline.context),
                role_queries_from_stencil(first, device=device),
            )
        scope = (
            _configure_native_output_head_scope(target_model)
            if scope_name == "native_output_heads"
            else _configure_native_nonlinear_interface_scope(target_model)
        )
        target_model.eval()
        scope["parameter_inventory"] = _parameter_inventory(target_model)
        output_differences: dict[str, float] = {}
        with torch.no_grad():
            for stencil in training_stencils:
                queries = role_queries_from_stencil(stencil, device=device)
                for label, record in (("baseline", stencil.baseline), *stencil.variants.items()):
                    design = DesignInput.from_state(record.design, device=device)
                    context = context_inputs(record.context)
                    reference_prediction = source_operator(design, context, queries)
                    target_prediction = operator(design, context, queries)
                    if not isinstance(reference_prediction, type(target_prediction)):
                        raise TypeError("Native update-zero parity returned a different prediction type.")
                    for role_name in target_prediction.role_values:
                        error = float((
                            reference_prediction.role_values[role_name]
                            - target_prediction.role_values[role_name]
                        ).abs().max().cpu())
                        output_differences[f"{stencil.physical_family_id}/{label}/{role_name}"] = error
        if any(value != 0.0 for value in output_differences.values()):
            raise AssertionError(f"R0 scope initialization changed native outputs: {output_differences}")
        scope["update_zero_max_abs_output_difference_by_family_state_role"] = output_differences
        return target_model, operator, scope

    scope_models: dict[str, tuple[ChannelThermalHONFModel, DifferentiableThermalOperator, dict[str, Any]]] = {}
    for scope_name in ("native_output_heads", "native_nonlinear_interface"):
        scope_models[scope_name] = build_scope(scope_name)
        ensure_time_remaining(f"update_zero_scope_parity_{scope_name}")
    gradient_results: dict[str, Any] = {}
    calibration_weights: dict[str, Mapping[str, float]] = {}
    for scope_name, (model, operator, _scope) in scope_models.items():
        calibration_weights[scope_name] = dict(calibrate_operator_weights(
            operator,
            training_stencils,
            scales=scales,
            historical_value_source=historical_source,
            parameters=model.parameters(),
            device=device,
            enabled_terms=_R0_TERMS,
            include_feasibility_bce=False,
        ))
        missing_response_weights = set(_R0_RESPONSE_TERMS) - set(calibration_weights[scope_name])
        if missing_response_weights:
            raise ValueError(
                f"R0 {scope_name} calibration has no nonzero gradient weight for "
                f"{sorted(missing_response_weights)}."
            )
        gradient_results[scope_name] = _r0_gradient_diagnostic(
            operator,
            model,
            training_stencils,
            historical_source,
            scales,
            calibration_weights[scope_name],
            device=device,
        )
        ensure_time_remaining(f"gradient_calibration_{scope_name}")

    rate_rows: list[dict[str, Any]] = []
    selected_learning_rate: float
    nonlinear_template, _nonlinear_operator, _nonlinear_scope = scope_models["native_nonlinear_interface"]
    initial_state = {name: value.detach().clone() for name, value in nonlinear_template.state_dict().items()}
    source_checkpoint_sha256 = _checkpoint_digest(checkpoint_path)
    recipe_path = Path(__file__).resolve().parents[3] / "configs" / "response_control_native_nonlinear_interface.json"

    def save_r0_checkpoint(
        root: Path, label: str, payload: Mapping[str, Any], *, scope_name: str
    ) -> Path:
        update = int(payload["actual_optimizer_updates"])
        safe_label = "".join(character if character.isalnum() else "_" for character in label)
        persisted_payload = dict(payload)
        persisted_payload["response_control_run_provenance"] = {
            "source_checkpoint": str(checkpoint_path.expanduser().resolve()),
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "training_recipe": str(recipe_path.resolve()),
            "training_recipe_sha256": _checkpoint_digest(recipe_path),
            "plan_schema": 1,
            "active_scope_name": scope_name,
            "active_scope_inventory": scope_models[scope_name][2],
            "training_atlas_paths": [str(path.expanduser().resolve()) for path in stencil_paths],
            "training_family_ids": [stencil.physical_family_id for stencil in training_stencils],
            "reference_solver_calls": 0,
        }
        path = root / f"checkpoint_{safe_label}_u{update:05d}.pt"
        _atomic_torch_save(path, persisted_payload)
        return path

    rate_probe_config: StagedTrainingConfig
    for learning_rate in (1.0e-5, 3.0e-5):
        remaining = ensure_time_remaining(f"rate_probe_lr_{learning_rate:.0e}")
        trial_name = f"rate_probe_lr_{learning_rate:.0e}"
        begin_trial(trial_name, 10, learning_rate)
        rate_probe_config = _r0_fit_config(rate_probe=True, max_wall_seconds=remaining)
        probe_model = copy.deepcopy(nonlinear_template)
        probe_operator = DifferentiableThermalOperator(
            probe_model,
            template,
            dataset_config=dataset_config,
            normalization_stats=checkpoint.get("global_normalization_stats", {}),
            query_batch_size=query_batch_size,
        )
        probe_model.load_state_dict(initial_state, strict=True)
        probe_model.eval()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in probe_model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        probe_dir = output_dir / f"rate_probe_lr_{learning_rate:.0e}"

        def save_rate_probe(
            payload: Mapping[str, Any], label: str, *, root: Path = probe_dir, name: str = trial_name
        ) -> None:
            save_r0_checkpoint(root, label, payload, scope_name="native_nonlinear_interface")
            record_trial_checkpoint(name, payload, label)

        initial_metrics = _evaluate_r0_train_objective(
            probe_operator,
            training_stencils,
            historical_source,
            scales,
            calibration_weights["native_nonlinear_interface"],
            device=device,
        )
        fit_started = time.perf_counter()
        fit = run_staged_fit(
            probe_operator,
            probe_model,
            optimizer,
            training_stencils,
            scales=scales,
            loss_weights=calibration_weights["native_nonlinear_interface"],
            historical_value_source=historical_source,
            config=rate_probe_config,
            device=device,
            on_checkpoint=save_rate_probe,
            on_optimizer_attempt=lambda completed, attempted, name=trial_name: record_trial_attempt(
                name, completed, attempted
            ),
        )
        finish_trial(trial_name, fit, 10)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fit_seconds = time.perf_counter() - fit_started
        final_metrics = _evaluate_r0_train_objective(
            probe_operator,
            training_stencils,
            historical_source,
            scales,
            calibration_weights["native_nonlinear_interface"],
            device=device,
        )
        rate_rows.append({
            "learning_rate": learning_rate,
            "optimizer_updates": fit.actual_optimizer_updates,
            "attempted_optimizer_steps": fit.attempted_optimizer_steps,
            "fit_wall_seconds": fit_seconds,
            "initial": initial_metrics,
            "final": final_metrics,
            "checkpoint_paths": sorted(str(path) for path in probe_dir.glob("checkpoint_*.pt")),
        })
        ensure_time_remaining("scope_comparison_fits")
    candidates_under_value_guard = [
        row for row in rate_rows
        if row["final"]["equal_family_mean"]["weighted_value_objective"]
        <= 1.02 * row["initial"]["equal_family_mean"]["weighted_value_objective"]
    ]
    selection_pool = candidates_under_value_guard or rate_rows
    selected_rate_row = min(
        selection_pool,
        key=lambda row: row["final"]["equal_family_mean"]["weighted_response_objective"],
    )
    selected_learning_rate = float(selected_rate_row["learning_rate"])
    m10_termwise_conflicts: list[dict[str, Any]] = []
    m10_combined_projection_gates: list[dict[str, Any]] = []
    for scope_name, diagnostic in gradient_results.items():
        for family in diagnostic["families"]:
            if family["family_id"] != "stored_family:0350":
                continue
            for block_name, terms in family["pairwise_gradient_cosines_by_block"].items():
                response_cosines = terms.get("combined_value", {})
                for diagnostic_term, response_term in (
                    ("finite", "finite"),
                    ("finite_peak", "finite_peak"),
                    ("pressure_value", "pressure_value"),
                    ("pressure_response", "pressure_response"),
                ):
                    cosine = response_cosines.get(diagnostic_term)
                    if cosine is not None:
                        m10_termwise_conflicts.append({
                            "scope": scope_name,
                            "family_id": family["family_id"],
                            "block": block_name,
                            "response_term": response_term,
                            "combined_value_cosine": float(cosine),
                            "adverse": bool(cosine < 0.0),
                        })
                combined_gate = family["calibrated_combined_projection_gate_by_block"][block_name]
                m10_combined_projection_gates.append({
                    "scope": scope_name,
                    "family_id": family["family_id"],
                    "block": block_name,
                    **combined_gate,
                })
    projection_justified = any(
        row["adverse"] and row["scope"] == "native_nonlinear_interface"
        for row in m10_combined_projection_gates
    )

    scope_fit_rows: dict[str, Any] = {}
    for scope_name, (scope_model, _old_operator, scope) in scope_models.items():
        remaining = ensure_time_remaining(f"scope_fit_{scope_name}")
        trial_name = f"scope_fit_{scope_name}"
        scope_update_cap = 80
        begin_trial(trial_name, scope_update_cap, selected_learning_rate)
        model = copy.deepcopy(scope_model)
        operator = DifferentiableThermalOperator(
            model,
            template,
            dataset_config=dataset_config,
            normalization_stats=checkpoint.get("global_normalization_stats", {}),
            query_batch_size=query_batch_size,
        )
        model.eval()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=selected_learning_rate,
            weight_decay=weight_decay,
        )
        scope_config = _r0_fit_config(
            rate_probe=False,
            max_wall_seconds=remaining,
            max_optimizer_updates=scope_update_cap,
        )
        initial_metrics = _evaluate_r0_train_objective(
            operator,
            training_stencils,
            historical_source,
            scales,
            calibration_weights[scope_name],
            device=device,
        )
        buffer_snapshot = {
            name: value.detach().clone()
            for name, value in model.named_buffers()
        }
        checkpoint_root = output_dir / scope_name

        def save_scope_fit(
            payload: Mapping[str, Any], label: str, *, root: Path = checkpoint_root,
            name: str = trial_name, active_scope: str = scope_name,
        ) -> None:
            save_r0_checkpoint(root, label, payload, scope_name=active_scope)
            record_trial_checkpoint(name, payload, label)

        fit_started = time.perf_counter()
        fit = run_staged_fit(
            operator,
            model,
            optimizer,
            training_stencils,
            scales=scales,
            loss_weights=calibration_weights[scope_name],
            historical_value_source=historical_source,
            config=scope_config,
            device=device,
            on_checkpoint=save_scope_fit,
            on_optimizer_attempt=lambda completed, attempted, name=trial_name: record_trial_attempt(
                name, completed, attempted
            ),
        )
        finish_trial(trial_name, fit, scope_update_cap)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fit_seconds = time.perf_counter() - fit_started
        final_metrics = _evaluate_r0_train_objective(
            operator,
            training_stencils,
            historical_source,
            scales,
            calibration_weights[scope_name],
            device=device,
        )
        changed_buffers = [
            name for name, value in model.named_buffers()
            if name not in buffer_snapshot or not torch.equal(value, buffer_snapshot[name])
        ]
        if changed_buffers:
            raise AssertionError(f"R0 deterministic fit changed frozen buffers: {changed_buffers}.")
        scope_fit_rows[scope_name] = {
            "parameter_scope": scope,
            "learning_rate": selected_learning_rate,
            "weight_decay": weight_decay,
            "optimizer_updates": fit.actual_optimizer_updates,
            "attempted_optimizer_steps": fit.attempted_optimizer_steps,
            "fit_wall_seconds": fit_seconds,
            "initial": initial_metrics,
            "final": final_metrics,
            "response_objective_reduction_fraction": (
                1.0 - final_metrics["equal_family_mean"]["weighted_response_objective"]
                / max(initial_metrics["equal_family_mean"]["weighted_response_objective"], 1.0e-12)
            ),
            "buffer_values_unchanged": True,
            "checkpoint_paths": sorted(str(path) for path in checkpoint_root.glob("checkpoint_*.pt")),
            "update_history": [_dataclass_record(step) for step in fit.history],
        }
    result = {
        "status": "passed",
        "mode": "train_only_scope_diagnostic",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _checkpoint_digest(checkpoint_path),
        "selected_epoch": checkpoint.get("epoch", checkpoint.get("current_epoch")),
        "reference_solver_calls": 0,
        "training_atlas_paths": [str(path.resolve()) for path in stencil_paths],
        "training_atlas_hashes": [
            {"path": str(path.resolve()), "sha256": _checkpoint_digest(path),
             "metadata_sha256": _checkpoint_digest(path.with_suffix(".json"))}
            for path in stencil_paths
        ],
        "family_ids": [stencil.physical_family_id for stencil in training_stencils],
        "module_counts": [len(stencil.baseline.design.active_modules) for stencil in training_stencils],
        "sampling": [_sampling_summary_mapping(item.summary) for item in sampled_panel],
        "historical_value_replay": {
            "dataset": str(historical_source.dataset_path),
            "dataset_size_bytes": historical_source.dataset_path.stat().st_size,
            "dataset_mtime_ns": historical_source.dataset_path.stat().st_mtime_ns,
            "train_case_count": len(historical_source.case_ids),
            "balanced_calibration_case_ids": list(historical_source.case_ids[:len(training_stencils)]),
            "one_case_per_update": True,
        },
        "loss_scales": _dataclass_record(scales),
        "loss_scale_calibration": {
            "scope": "four_family_train_diagnostic_only",
            "family_ids": [stencil.physical_family_id for stencil in training_stencils],
            "reusable_for_formal_fit": False,
        },
        "gradient_diagnostics": gradient_results,
        "gradient_calibration_weights": calibration_weights,
        "gradient_calibration_scope": {
            "scope": "four_family_train_diagnostic_only",
            "family_ids": [stencil.physical_family_id for stencil in training_stencils],
            "reusable_for_formal_fit": False,
        },
        "learning_rate_candidates": rate_rows,
        "learning_rate_selection": {
            "selected_learning_rate": selected_learning_rate,
            "selection_rule": "lowest equal-family calibrated response objective among candidates with no more than 2 percent value-objective regression; if none meet the value guard, choose the lowest response objective",
            "value_guard_passed": bool(candidates_under_value_guard),
        },
        "scope_fit_comparison": scope_fit_rows,
        "actual_optimizer_updates_total": sum(
            int(row["optimizer_updates"])
            for row in rate_rows
        ) + sum(int(row["optimizer_updates"]) for row in scope_fit_rows.values()),
        "attempted_optimizer_updates_total": sum(
            int(row["attempted_optimizer_steps"]) for row in rate_rows
        ) + sum(int(row["attempted_optimizer_steps"]) for row in scope_fit_rows.values()),
        "optimizer_trial_status": list(progress["optimizer_trial_status"]),
        "gradient_diagnostic_optimizer_updates": 0,
        "matched_arm_names_for_subsequent_formal_fit": ["R_value", "R_response"],
        "projection_justification": {
            "justified_by_measured_m10_combined_value_conflict": projection_justified,
            "family_id": "stored_family:0350",
            "scope": "native_nonlinear_interface",
            "combined_calibrated_response_vs_value_by_block": m10_combined_projection_gates,
            "individual_response_term_vs_value_by_block": m10_termwise_conflicts,
            "policy": (
                "Enable blockwise response-gradient projection only if the calibrated weighted "
                "combined response gradient has a negative dot with the combined value gradient "
                "in a nonlinear-interface block on train-only M10. Individual response-term "
                "conflicts are recorded but do not independently activate projection."
            ),
        },
        "scope_fit_update_caps": {
            "native_output_heads": 80,
            "native_nonlinear_interface": 80,
            "native_nonlinear_interface_rate_candidates_combined": 20,
        },
        "total_wall_seconds": float(time.monotonic() - started),
        "deterministic_eval_mode": True,
        "deterministic_algorithms_enabled": True,
        "cublas_workspace_config": cublas_workspace,
        "projection_in_r0_scope_fits": False,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result["cuda"] = {
            "logical_device": str(device),
            "physical_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    ensure_time_remaining("final_manifest_write")
    result["finished_unix_seconds"] = time.time()
    progress.update(result)
    progress["status"] = "passed"
    progress["current_phase"] = "complete"
    save_progress()
    return result


def _evaluate_expanded_train_objective(
    operator: DifferentiableThermalOperator,
    stencils: Sequence[ResponseStencil],
    historical_source: HistoricalValueSource,
    historical_case_by_family: Mapping[str, str],
    scales: ThermalLossScales,
    weights: Mapping[str, float],
    fixed_heat_controls: Sequence[FixedHeatNullControl],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Measure the expanded recipe's train-only value and response objectives."""

    response_terms = (
        "finite", "finite_peak", "pressure_value", "pressure_response",
        "fixed_heat_null", "fixed_heat_thermal",
    )
    controls_by_family: dict[str, list[FixedHeatNullControl]] = {}
    for control in fixed_heat_controls:
        controls_by_family.setdefault(control.family_id, []).append(control)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for stencil in stencils:
            family_id = stencil.physical_family_id
            if family_id not in historical_case_by_family:
                raise ValueError(f"No balanced historical train case is assigned to {family_id!r}.")
            predictions = predict_stencil(operator, stencil, device=device)
            terms = compute_stencil_loss_terms(
                predictions,
                stencil,
                scales=scales,
                enabled_terms=("value", "finite", "finite_peak", "pressure_value", "pressure_response"),
                include_feasibility_bce=False,
            )
            historical_loss = historical_absolute_value_loss(
                operator,
                historical_source.load(historical_case_by_family[family_id]),
                scales=scales,
                device=device,
            )
            control_terms_by_name: dict[str, list[float]] = {}
            for control in controls_by_family.get(family_id, ()):
                control_terms, _diagnostics = fixed_heat_control_loss_terms(
                    operator, control, scales=scales, device=device
                )
                for name in ("fixed_heat_null", "fixed_heat_thermal"):
                    if name in control_terms:
                        control_terms_by_name.setdefault(name, []).append(
                            float(control_terms[name].detach().cpu())
                        )
            term_values = {
                name: float(value.detach().cpu())
                for name, value in terms.terms.items()
            }
            term_values["historical_value"] = float(historical_loss.detach().cpu())
            for name, values in control_terms_by_name.items():
                term_values[name] = float(np.mean(values))
            continuous_response_terms = (
                "finite", "finite_peak", "pressure_value", "pressure_response"
            )
            missing = set(continuous_response_terms) - set(term_values)
            if missing:
                raise ValueError(
                    f"Train family {family_id!r} has no measured probe objective for {sorted(missing)}."
                )
            for name in ("fixed_heat_null", "fixed_heat_thermal"):
                term_values.setdefault(name, 0.0)
            value_objective = float(weights["value"]) * (
                term_values.get("value", 0.0) + term_values["historical_value"]
            )
            response_objective = sum(
                float(weights[name]) * term_values[name] for name in response_terms
            )
            if not np.isfinite(value_objective) or not np.isfinite(response_objective):
                raise FloatingPointError(f"Expanded train objective is non-finite for {family_id!r}.")
            rows.append({
                "family_id": family_id,
                "module_count": len(stencil.baseline.design.active_modules),
                "historical_case_id": historical_case_by_family[family_id],
                "term_losses": term_values,
                "weighted_value_objective": value_objective,
                "weighted_response_objective": response_objective,
                "weighted_total_objective": value_objective + response_objective,
            })
    return {
        "families": rows,
        "equal_family_mean": {
            name: float(np.mean([row[name] for row in rows]))
            for name in (
                "weighted_value_objective",
                "weighted_response_objective",
                "weighted_total_objective",
            )
        },
        "equal_family_mean_term_losses": {
            name: float(np.mean([row["term_losses"].get(name, 0.0) for row in rows]))
            for name in (*response_terms, "value", "historical_value")
        },
    }


def run_expanded_response_fit_capability_probe(
    *,
    checkpoint_path: Path,
    stencil_paths: Sequence[Path],
    dataset_path: Path | None,
    recipe_config_path: Path,
    output_dir: Path,
    device: torch.device,
    sampling: ReceiverSamplingConfig,
    query_batch_size: int,
    max_wall_seconds: float,
    smooth_peak_beta: float,
) -> dict[str, Any]:
    """Run an 80-update CUDA-only train cohort probe for the expanded scope."""

    started = time.monotonic()
    deadline = started + max_wall_seconds
    if device.type != "cuda" or os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError(
            "The expanded fit-capability probe requires CUDA_VISIBLE_DEVICES=2 and logical --device cuda:0."
        )
    if device.index not in {None, 0} or not torch.cuda.is_available():
        raise RuntimeError("The expanded fit-capability probe requires the allocated logical CUDA device 0.")
    cublas_workspace = _enable_deterministic_algorithms(device)
    if max_wall_seconds <= 0.0:
        raise ValueError("Expanded fit-capability probe max_wall_seconds must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "expanded_response_fit_capability_probe.json"
    recipe_path = recipe_config_path.expanduser().resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("name") != "native_expanded_response_interface":
        raise ValueError("Expanded probe requires the native_expanded_response_interface recipe.")
    if int(recipe.get("probe_optimizer_updates", -1)) != 80:
        raise ValueError("Expanded fit-capability probe is bounded to exactly 80 optimizer updates.")
    if int(recipe.get("shared_remedy_optimizer_call_ceiling", -1)) != 600:
        raise ValueError("Expanded probe must use the shared 600-attempt short-remedy ledger.")
    if len(stencil_paths) != 8:
        raise ValueError("Expanded fit-capability probe requires exactly eight train response stencils.")
    source_digest = _checkpoint_digest(checkpoint_path)
    expected_digest = str(recipe.get("expected_source_checkpoint_sha256", ""))
    if source_digest != expected_digest:
        raise ValueError(
            "Expanded probe source checkpoint does not match the frozen Run1804 e4738 identity: "
            f"expected {expected_digest}, got {source_digest}."
        )
    comparator = recipe.get("read_only_comparator", {})
    comparator_path = (
        Path(__file__).resolve().parents[4] / str(comparator.get("checkpoint_relative_path", ""))
    ).resolve()
    if not comparator_path.is_file() or _checkpoint_digest(comparator_path) != comparator.get("checkpoint_sha256"):
        raise ValueError("Read-only Run1502 e4794 comparator is missing or no longer matches its pinned identity.")
    report: dict[str, Any] = {
        "status": "running",
        "mode": "expanded_response_fit_capability_probe",
        "checkpoint": str(checkpoint_path.expanduser().resolve()),
        "checkpoint_sha256": source_digest,
        "read_only_comparator": {
            "checkpoint": str(comparator_path),
            "checkpoint_sha256": _checkpoint_digest(comparator_path),
            "loaded_or_modified": False,
        },
        "recipe": str(recipe_path),
        "recipe_sha256": _checkpoint_digest(recipe_path),
        "train_atlas_paths": [str(path.expanduser().resolve()) for path in stencil_paths],
        "train_atlas_hashes": [],
        "planned_optimizer_updates": 80,
        "optimizer_updates_attempted": 0,
        "optimizer_updates_completed": 0,
        "optimizer_calls_charged_to_shared_remedy_ledger": 0,
        "shared_remedy_optimizer_call_ceiling": 600,
        "reference_solver_calls": 0,
        "development_stencil_count": 0,
        "current_phase": "input_validation",
        "started_unix_seconds": time.time(),
    }
    _atomic_json(report_path, report)

    def save_progress() -> None:
        report["total_wall_seconds"] = float(time.monotonic() - started)
        _atomic_json(report_path, report)

    def ensure_time_remaining(phase: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            report["current_phase"] = phase
            report["status"] = "timed_out_or_partial"
            report["error"] = f"Expanded probe total wall cap of {max_wall_seconds:.1f} seconds expired."
            save_progress()
            raise TimeoutError(report["error"])
        return remaining

    if any(not path.is_file() for path in stencil_paths):
        raise FileNotFoundError("An expanded train response atlas is missing.")
    loaded = [load_response_atlas_stencil(path) for path in stencil_paths]
    raw_stencils = tuple(stencil for stencil, _metadata in loaded)
    if any(stencil.split is not EvidenceSplit.TRAIN for stencil in raw_stencils):
        raise ValueError("Expanded fit-capability probe accepts train-split response stencils only.")
    family_ids = tuple(sorted({stencil.physical_family_id for stencil in raw_stencils}))
    if len(family_ids) != 8:
        raise ValueError("Expanded fit-capability probe requires eight distinct train response families.")
    report["train_family_ids"] = list(family_ids)
    report["train_atlas_hashes"] = [
        {
            "path": str(path.expanduser().resolve()),
            "sha256": _checkpoint_digest(path),
            "metadata_sha256": _checkpoint_digest(path.with_suffix(".json")),
        }
        for path in stencil_paths
    ]
    report["module_counts"] = [len(stencil.baseline.design.active_modules) for stencil in raw_stencils]
    source_model, checkpoint = load_model(checkpoint_path, device)
    if int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))) != 4738:
        raise ValueError("Expanded fit-capability probe requires the intact Run1804 e4738 checkpoint.")
    if str(source_model.config.core_honf.forward_architecture) != "dense_pairwise_field":
        raise ValueError("Expanded fit-capability probe requires the Run1804 dense native architecture.")
    source_model.eval()
    dataset_root = _resolve_dataset_path(checkpoint, str(dataset_path) if dataset_path else None)
    train_config = checkpoint.get("train_config", {})
    dataset_config = train_config.get("dataset", {})
    raw_dataset = GlobalChannelThermalDataset(
        dataset_root,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    historical_sampling = replace(sampling, max_fluid_queries=max(3072, sampling.max_fluid_queries))
    historical_source = HistoricalValueSource.from_dataset(raw_dataset, sampling=historical_sampling)
    required_historical_count = int(recipe.get("historical_train_case_count", -1))
    if len(historical_source.case_ids) != required_historical_count:
        raise ValueError(
            "Expanded probe requires the fully reconciled packed train cohort: "
            f"expected {required_historical_count} converged cases, got {len(historical_source.case_ids)}."
        )
    sampled_panel = sample_training_panel(raw_stencils, config=sampling)
    training_stencils = tuple(item.stencil for item in sampled_panel)
    fixed_heat_controls, fixed_heat_provenance, _review_controls = _load_fixed_heat_null_controls(
        recipe,
        raw_stencils,
        sampled_panel,
        stencil_paths=stencil_paths,
    )
    historical_case_by_family = _historical_calibration_case_map(family_ids, historical_source)
    scales = derive_training_scales(
        training_stencils,
        smooth_peak_beta=smooth_peak_beta,
        historical_value_source=historical_source,
    )
    template = _make_input_template(raw_dataset)
    normalization_stats = checkpoint.get("global_normalization_stats", {})
    source_operator = DifferentiableThermalOperator(
        source_model,
        template,
        dataset_config=dataset_config,
        normalization_stats=normalization_stats,
        query_batch_size=query_batch_size,
    )
    target_model, _refit_config = _native_checkpoint_initialization(source_model)
    target_operator = DifferentiableThermalOperator(
        target_model,
        template,
        dataset_config=dataset_config,
        normalization_stats=normalization_stats,
        query_batch_size=query_batch_size,
    )
    report["current_phase"] = "native_update_zero_parity"
    parity_rows: dict[str, float] = {}
    with torch.no_grad():
        for stencil in training_stencils:
            queries = role_queries_from_stencil(stencil, device=device)
            for label, record in (("baseline", stencil.baseline), *stencil.variants.items()):
                design = DesignInput.from_state(record.design, device=device)
                context = context_inputs(record.context)
                reference = source_operator(design, context, queries)
                target = target_operator(design, context, queries)
                for role_name in target.role_values:
                    difference = float((reference.role_values[role_name] - target.role_values[role_name]).abs().max().cpu())
                    parity_rows[f"{stencil.physical_family_id}/{label}/{role_name}"] = difference
    if any(value != 0.0 for value in parity_rows.values()):
        raise AssertionError(f"Expanded native scope changed Run1804 update-zero outputs: {parity_rows}.")
    native_scope = _configure_native_expanded_response_interface_scope(target_model)
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in target_model.named_parameters()
    }
    enabled_terms = (
        "value", "finite", "finite_peak", "pressure_value", "pressure_response",
        "fixed_heat_null", "fixed_heat_thermal",
    )
    calibration_diagnostics: dict[str, Any] = {}
    weights = dict(calibrate_operator_weights(
        target_operator,
        training_stencils,
        scales=scales,
        historical_value_source=historical_source,
        parameters=target_model.parameters(),
        device=device,
        enabled_terms=enabled_terms,
        include_feasibility_bce=False,
        diagnostic_sink=calibration_diagnostics,
        fixed_heat_controls=fixed_heat_controls,
    ))
    required_terms = tuple(recipe.get("required_response_terms", ()))
    required_controls = tuple(recipe.get("required_control_terms", ()))
    missing_weights = sorted((set(required_terms) | set(required_controls)) - set(weights))
    if missing_weights:
        raise ValueError(f"Expanded response path has no finite nonzero train gradient for {missing_weights}.")
    initial_metrics = _evaluate_expanded_train_objective(
        target_operator,
        training_stencils,
        historical_source,
        historical_case_by_family,
        scales,
        weights,
        fixed_heat_controls,
        device=device,
    )
    current_parameters = dict(target_model.named_parameters())
    changed_during_calibration = [
        name for name, value in initial_parameters.items()
        if not torch.equal(value, current_parameters[name].detach())
    ]
    if changed_during_calibration:
        raise AssertionError(
            f"Expanded response calibration changed model parameters before update one: {changed_during_calibration}."
        )
    # Lazy buffer materialization is complete only after the calibration and
    # baseline objective have exercised stencils, historical cases and controls.
    frozen_buffers = {
        name: value.detach().clone()
        for name, value in target_model.named_buffers()
    }
    initial_model_state = {
        name: value.detach().clone()
        for name, value in target_model.state_dict().items()
    }
    report.update({
        "current_phase": "bounded_optimizer_probe",
        "native_trainable_scope": native_scope,
        "native_parameter_inventory": _parameter_inventory(target_model),
        "update_zero_max_abs_output_difference_by_family_state_role": parity_rows,
        "loss_scale_calibration": {
            "scope": "eight_train_families_plus_one_historical_train_case_per_family",
            "historical_train_examples_by_family": historical_case_by_family,
            "loss_scales": _dataclass_record(scales),
        },
        "fixed_heat_control_provenance": fixed_heat_provenance,
        "gradient_calibration": {
            "weights": weights,
            "diagnostics": calibration_diagnostics,
            "training_families_only": True,
            "development_families_loaded": False,
        },
        "initial_train_objective": initial_metrics,
        "reference_solver_calls": 0,
    })
    save_progress()
    optimizer_updates = int(recipe["probe_optimizer_updates"])
    stages = tuple(
        TrainingStage(
            name=stage.name,
            start_update=stage.start_update,
            stop_update=min(stage.stop_update, optimizer_updates),
            active_terms=stage.active_terms,
        )
        for stage in load_staged_training_config(str(recipe_path), arm="R_response").stages
        if stage.start_update < optimizer_updates
    )
    pair_config = load_staged_training_config(str(recipe_path), arm="R_response")
    probe_config = StagedTrainingConfig(
        arm="R_response",
        max_optimizer_updates=optimizer_updates,
        max_epochs=(optimizer_updates + len(training_stencils) - 1) // len(training_stencils),
        total_optimizer_update_ceiling=int(recipe["shared_remedy_optimizer_call_ceiling"]),
        checkpoint_every_updates=int(recipe.get("probe_checkpoint_every_updates", 20)),
        max_wall_seconds=ensure_time_remaining("probe_training_start"),
        review_updates=(optimizer_updates,),
        random_seed=pair_config.random_seed,
        deterministic_eval_mode=True,
        deterministic_algorithms=True,
        project_response_gradient_blockwise=False,
        response_ramp_start_update=pair_config.response_ramp_start_update,
        response_ramp_end_update=pair_config.response_ramp_end_update,
        response_ramp_terms=pair_config.response_ramp_terms,
        required_response_terms=pair_config.required_response_terms,
        required_control_terms=pair_config.required_control_terms,
        include_feasibility_bce=False,
        stages=stages,
    )
    learning_rate = float(recipe.get("learning_rate", 1.0e-5))
    weight_decay = float(recipe.get("weight_decay", 1.0e-5))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in target_model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    probe_checkpoint_paths: list[str] = []
    for name, value in initial_model_state.items():
        if not torch.equal(target_model.state_dict()[name], value):
            raise AssertionError(f"Expanded probe changed native starting tensor before update one: {name}.")

    def save_probe_checkpoint(payload: Mapping[str, Any], label: str) -> None:
        update = int(payload["actual_optimizer_updates"])
        persisted = dict(payload)
        persisted["response_control_run_provenance"] = {
            "source_checkpoint": str(checkpoint_path.expanduser().resolve()),
            "source_checkpoint_sha256": source_digest,
            "training_recipe": str(recipe_path),
            "training_recipe_sha256": _checkpoint_digest(recipe_path),
            "active_scope_name": native_scope["name"],
            "active_scope_inventory": native_scope,
            "training_atlas_paths": [str(path.expanduser().resolve()) for path in stencil_paths],
            "training_atlas_hashes": report["train_atlas_hashes"],
            "training_family_ids": list(family_ids),
            "reference_solver_calls": 0,
        }
        path = output_dir / f"expanded_response_probe_{label}_u{update:05d}.pt"
        _atomic_torch_save(path, persisted)
        probe_checkpoint_paths.append(str(path.resolve()))
        report["probe_checkpoint_paths"] = list(probe_checkpoint_paths)
        save_progress()

    def record_attempt(completed: int, attempted: int) -> None:
        report["optimizer_updates_completed"] = int(completed)
        report["optimizer_updates_attempted"] = int(attempted)
        report["optimizer_calls_charged_to_shared_remedy_ledger"] = int(attempted)
        save_progress()

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    fit_started = time.perf_counter()
    fit = run_staged_fit(
        target_operator,
        target_model,
        optimizer,
        training_stencils,
        scales=scales,
        loss_weights=weights,
        historical_value_source=historical_source,
        fixed_heat_controls=fixed_heat_controls,
        config=probe_config,
        device=device,
        on_checkpoint=save_probe_checkpoint,
        on_optimizer_attempt=record_attempt,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    fit_wall_seconds = time.perf_counter() - fit_started
    final_metrics = _evaluate_expanded_train_objective(
        target_operator,
        training_stencils,
        historical_source,
        historical_case_by_family,
        scales,
        weights,
        fixed_heat_controls,
        device=device,
    )
    changed_by_block: dict[str, dict[str, int]] = {}
    for label, module_names in native_scope["trainable_module_names_by_block"].items():
        prefix_tuple = tuple(f"{name}." for name in module_names)
        parameter_names = [name for name in initial_parameters if name.startswith(prefix_tuple)]
        changed_names = [
            name for name in parameter_names
            if not torch.equal(initial_parameters[name], dict(target_model.named_parameters())[name].detach())
        ]
        changed_by_block[label] = {
            "parameter_count": len(parameter_names),
            "changed_parameter_count": len(changed_names),
            "changed_numel": sum(
                int((initial_parameters[name] != dict(target_model.named_parameters())[name].detach()).sum().item())
                for name in changed_names
            ),
        }
    changed_frozen_parameters = [
        name for name, parameter in target_model.named_parameters()
        if not parameter.requires_grad
        and (name not in initial_parameters or not torch.equal(initial_parameters[name], parameter.detach()))
    ]
    changed_buffers = [
        name for name, value in target_model.named_buffers()
        if name not in frozen_buffers or not torch.equal(frozen_buffers[name], value.detach())
    ]
    if changed_frozen_parameters or changed_buffers:
        raise AssertionError(
            "Expanded deterministic probe changed frozen model state: "
            f"parameters={changed_frozen_parameters}, buffers={changed_buffers}."
        )
    value_initial = float(initial_metrics["equal_family_mean"]["weighted_value_objective"])
    value_final = float(final_metrics["equal_family_mean"]["weighted_value_objective"])
    response_initial = float(initial_metrics["equal_family_mean"]["weighted_response_objective"])
    response_final = float(final_metrics["equal_family_mean"]["weighted_response_objective"])
    response_reduction = 1.0 - response_final / max(response_initial, 1.0e-12)
    value_regression = value_final / max(value_initial, 1.0e-12) - 1.0
    value_guard = float(recipe.get("probe_max_value_regression_fraction", 0.02))
    expanded_path_changed = any(
        changed_by_block[label]["changed_numel"] > 0
        for label in ("native_module_environment_updates", "native_fine_receiver_reads")
    )
    updates_complete = fit.final_update == optimizer_updates and fit.actual_optimizer_updates == optimizer_updates
    capability_gate = {
        "response_objective_reduced": bool(response_reduction > 0.0),
        "response_objective_reduction_fraction": response_reduction,
        "value_guard_fraction": value_guard,
        "value_guard_passed": bool(value_regression <= value_guard),
        "value_objective_regression_fraction": value_regression,
        "expanded_native_interaction_parameters_changed": expanded_path_changed,
        "updates_complete": updates_complete,
        "supports_matched_trial": bool(
            updates_complete and response_reduction > 0.0
            and value_regression <= value_guard and expanded_path_changed
        ),
    }
    replay_segment = tuple(step.historical_case_id for step in fit.history)
    if any(case_id is None for case_id in replay_segment):
        raise RuntimeError("Expanded probe fit history omitted a historical case ID.")
    replay_coverage = historical_replay_coverage(
        historical_source.case_ids,
        initial_update=fit.initial_update,
        segment_case_ids=tuple(str(case_id) for case_id in replay_segment),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        cuda_record = {
            "logical_device": str(device),
            "physical_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    else:
        raise RuntimeError("The expanded probe cannot report or use CPU fallback.")
    report.update({
        "status": "passed" if updates_complete else "timed_out_or_partial",
        "current_phase": "complete" if updates_complete else "partial_fit_saved",
        "optimizer_updates_completed": fit.actual_optimizer_updates,
        "optimizer_updates_attempted": fit.attempted_optimizer_steps,
        "optimizer_calls_charged_to_shared_remedy_ledger": fit.attempted_optimizer_steps,
        "optimizer_updates_through_gate": fit.final_update,
        "attempted_optimizer_steps_cumulative": fit.total_attempted_optimizer_steps,
        "fit_wall_seconds": fit_wall_seconds,
        "total_wall_seconds": float(time.monotonic() - started),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "actual_update_history": [_dataclass_record(step) for step in fit.history],
        "historical_training_replay_coverage": replay_coverage,
        "final_train_objective": final_metrics,
        "response_objective_reduction_fraction": response_reduction,
        "value_objective_regression_fraction": value_regression,
        "trainable_parameter_change_by_block": changed_by_block,
        "frozen_parameter_values_unchanged": True,
        "frozen_buffer_values_unchanged": True,
        "capability_gate": capability_gate,
        "reference_solver_calls": 0,
        "cuda": cuda_record,
        "cublas_workspace_config": cublas_workspace,
        "finished_unix_seconds": time.time(),
    })
    save_progress()
    return report


def _load_expanded_probe_evidence(
    path: Path,
    *,
    checkpoint_path: Path,
    recipe_path: Path,
    train_atlas_paths: Sequence[Path],
    expected_family_ids: Sequence[str],
) -> dict[str, Any]:
    """Require passed train-only evidence and allow only a review-schedule delta."""

    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("mode") != "expanded_response_fit_capability_probe":
        raise ValueError("Expanded matched fitting requires a completed expanded train-only capability probe.")
    if payload.get("checkpoint_sha256") != _checkpoint_digest(checkpoint_path):
        raise ValueError("Expanded fit-capability probe belongs to a different source checkpoint.")
    paired_recipe_path = recipe_path.expanduser().resolve()
    probe_recipe_path = (paired_recipe_path.parent / "response_control_native_expanded_interface.json").resolve()
    if Path(str(payload.get("recipe", ""))).expanduser().resolve() != probe_recipe_path:
        raise ValueError("Expanded probe evidence does not name the pinned base recipe in this config directory.")
    if not probe_recipe_path.is_file() or payload.get("recipe_sha256") != _checkpoint_digest(probe_recipe_path):
        raise ValueError("The probe-pinned expanded response recipe is missing or no longer matches its SHA256.")
    if not paired_recipe_path.is_file():
        raise FileNotFoundError(f"The matched expanded response recipe is missing: {paired_recipe_path}")
    probe_recipe = json.loads(probe_recipe_path.read_text(encoding="utf-8"))
    paired_recipe = json.loads(paired_recipe_path.read_text(encoding="utf-8"))
    probe_review_updates = probe_recipe.get("review_updates")
    paired_review_updates = paired_recipe.get("review_updates")
    if probe_review_updates != [200, 600] or paired_review_updates != [200, 500, 600]:
        raise ValueError(
            "Expanded matched fitting requires the probe's [200, 600] recipe and the paired [200, 500, 600] review schedule."
        )
    probe_recipe_without_reviews = dict(probe_recipe)
    paired_recipe_without_reviews = dict(paired_recipe)
    probe_recipe_without_reviews.pop("review_updates", None)
    paired_recipe_without_reviews.pop("review_updates", None)
    if probe_recipe_without_reviews != paired_recipe_without_reviews:
        raise ValueError(
            "The paired expanded recipe may differ from the probe-pinned recipe only in review_updates."
        )
    paired_recipe_digest = _checkpoint_digest(paired_recipe_path)
    if payload.get("native_trainable_scope", {}).get("name") != "native_expanded_response_interface":
        raise ValueError("Expanded fit-capability probe did not exercise the requested native scope.")
    if int(payload.get("optimizer_updates_completed", -1)) != 80:
        raise ValueError("Expanded matched fitting requires all 80 bounded probe updates.")
    if int(payload.get("optimizer_updates_attempted", -1)) != 80:
        raise ValueError("Expanded probe attempted-update ledger does not equal the 80-update cap.")
    ledger_calls = int(payload.get("optimizer_calls_charged_to_shared_remedy_ledger", -1))
    if ledger_calls != 80:
        raise ValueError("Expanded probe shared-remedy ledger charge must equal its 80 completed attempts.")
    if ledger_calls > 600:
        raise ValueError("Expanded probe exceeds the shared short-remedy optimizer-call ceiling.")
    if int(payload.get("reference_solver_calls", -1)) != 0:
        raise ValueError("Expanded fit-capability probe must use stored physical evidence only.")
    if payload.get("development_stencil_count") != 0:
        raise ValueError("Expanded fit-capability probe must not use development stencils.")
    gate = payload.get("capability_gate")
    if not isinstance(gate, Mapping) or gate.get("supports_matched_trial") is not True:
        raise ValueError("Train-only expanded fit-capability evidence does not justify a matched trial.")
    report_atlas_hashes = payload.get("train_atlas_hashes", ())
    expected_hashes = {
        (str(row["sha256"]), str(row["metadata_sha256"]))
        for row in report_atlas_hashes
        if isinstance(row, Mapping) and "sha256" in row and "metadata_sha256" in row
    }
    supplied_hashes = {
        (_checkpoint_digest(item), _checkpoint_digest(item.with_suffix(".json")))
        for item in train_atlas_paths
    }
    if len(expected_hashes) != 8 or expected_hashes != supplied_hashes:
        raise ValueError("Expanded fit-capability probe does not cover these exact eight train response atlases.")
    if set(payload.get("train_family_ids", ())) != set(expected_family_ids):
        raise ValueError("Expanded fit-capability probe train family IDs differ from the matched panel.")
    return {
        "path": str(resolved),
        "sha256": _checkpoint_digest(resolved),
        "probe_recipe_path": str(probe_recipe_path),
        "probe_recipe_sha256": str(payload["recipe_sha256"]),
        "paired_recipe_path": str(paired_recipe_path),
        "paired_recipe_sha256": paired_recipe_digest,
        "recipe_schedule_difference": {
            "field": "review_updates",
            "probe": probe_review_updates,
            "paired": paired_review_updates,
        },
        "supports_matched_trial": True,
        "optimizer_updates_completed": 80,
        "optimizer_updates_attempted": 80,
        "optimizer_calls_charged_to_shared_remedy_ledger": ledger_calls,
        "capability_gate": dict(gate),
    }


def run_read_only_full_grid_train_replay(
    *,
    checkpoint_path: Path,
    stencil_paths: Sequence[Path],
    dataset_path: Path | None,
    recipe_config_path: Path,
    fit_manifest_path: Path,
    value_checkpoint_path: Path,
    response_checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    query_batch_size: int,
    max_wall_seconds: float,
) -> dict[str, Any]:
    """Replay a passed native expanded-response gate without optimizer or solver calls."""

    started = time.monotonic()
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Read-only full-grid train replay requires the allocated CUDA device.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("Read-only full-grid train replay requires CUDA_VISIBLE_DEVICES=2.")
    if len(stencil_paths) != 8:
        raise ValueError("Expanded native resume replay requires exactly eight train atlases.")
    if max_wall_seconds <= 0.0:
        raise ValueError("max_wall_seconds must be positive for read-only train replay.")
    recipe_path = recipe_config_path.expanduser().resolve()
    fit_path = fit_manifest_path.expanduser().resolve()
    source_path = checkpoint_path.expanduser().resolve()
    value_path = value_checkpoint_path.expanduser().resolve()
    response_path = response_checkpoint_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "read_only_full_grid_train_replay_manifest.json"
    if manifest_path == fit_path:
        raise ValueError("Replay output must not overwrite its source paired-fit manifest.")

    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if (
        recipe.get("name") != "native_expanded_response_interface"
        or recipe.get("initialization_mode") != "native_checkpoint"
        or recipe.get("review_updates") != [200, 500, 600]
    ):
        raise ValueError("Read-only expanded replay requires the paired [200, 500, 600] recipe.")
    cublas_workspace = (
        _enable_deterministic_algorithms(device)
        if bool(recipe.get("deterministic_algorithms", False))
        else None
    )
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if fit.get("status") != "passed" or fit.get("mode") != "paired_staged_fit":
        raise ValueError("Read-only replay requires a passed paired staged-fit manifest.")
    gate = int(fit.get("review_cap", -1))
    if gate not in recipe["review_updates"] or fit.get("review_decisions", {}).get(str(gate)) != "stop":
        raise ValueError("Read-only replay requires a passed fit stopped at a configured review gate.")
    if (
        Path(str(fit.get("checkpoint", ""))).expanduser().resolve() != source_path
        or fit.get("checkpoint_sha256") != _checkpoint_digest(source_path)
    ):
        raise ValueError("Read-only replay source checkpoint differs from the passed fit.")
    if (
        Path(str(fit.get("staged_recipe", ""))).expanduser().resolve() != recipe_path
        or fit.get("staged_recipe_name") != recipe["name"]
        or fit.get("read_only_comparator") != recipe.get("read_only_comparator")
    ):
        raise ValueError("Read-only replay recipe or pinned comparator differs from the passed fit.")

    resolved_stencils = [path.expanduser().resolve() for path in stencil_paths]
    loaded = [load_response_atlas_stencil(path) for path in resolved_stencils]
    stencils = tuple(item[0] for item in loaded)
    if any(stencil.split is not EvidenceSplit.TRAIN for stencil in stencils):
        raise ValueError("Read-only full-grid train replay rejects non-train atlases.")
    family_ids = [stencil.physical_family_id for stencil in stencils]
    if len(set(family_ids)) != 8:
        raise ValueError("Read-only expanded replay requires eight distinct train families.")
    expected_paths = [str(path) for path in resolved_stencils]
    if fit.get("train_atlas_paths") != expected_paths:
        raise ValueError("Read-only replay train atlas path/order differs from the passed fit.")
    if set(fit.get("train_family_ids", ())) != set(family_ids):
        raise ValueError("Read-only replay train family IDs differ from the passed fit.")
    probe_evidence = fit.get("expanded_fit_capability_probe")
    if not isinstance(probe_evidence, Mapping):
        raise ValueError("Passed expanded fit has no probe-pinned capability evidence.")  # noqa: TRY004
    verified_probe = _load_expanded_probe_evidence(
        Path(str(probe_evidence.get("path", ""))),
        checkpoint_path=source_path,
        recipe_path=recipe_path,
        train_atlas_paths=resolved_stencils,
        expected_family_ids=family_ids,
    )
    for key in ("sha256", "probe_recipe_sha256", "paired_recipe_sha256"):
        if probe_evidence.get(key) != verified_probe.get(key):
            raise ValueError(f"Passed fit capability evidence changed at {key}.")

    source_model, checkpoint = load_model(source_path, device)
    if int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))) != 4738:
        raise ValueError("Read-only expanded replay requires intact Run1804 e4738.")
    if str(source_model.config.core_honf.forward_architecture) != "dense_pairwise_field":
        raise ValueError("Read-only expanded replay requires the Run1804 dense native architecture.")
    source_model.eval()
    target_model, refit_config = _native_checkpoint_initialization(source_model)
    native_scope = _configure_native_expanded_response_interface_scope(target_model)
    if fit.get("initialization_mode") != "native_checkpoint" or fit.get("refit_config") != refit_config:
        raise ValueError("Read-only replay initialization differs from the passed native fit.")

    historical_replay = fit.get("historical_value_replay")
    if not isinstance(historical_replay, Mapping):
        raise ValueError("Passed native fit is missing its historical train dataset identity.")  # noqa: TRY004
    dataset_root = _resolve_dataset_path(checkpoint, str(dataset_path) if dataset_path else None)
    if Path(str(historical_replay.get("dataset", ""))).expanduser().resolve() != dataset_root.resolve():
        raise ValueError("Read-only replay historical train dataset differs from the passed fit.")
    dataset_stat = dataset_root.stat()
    if (
        int(historical_replay.get("dataset_size_bytes", -1)) != dataset_stat.st_size
        or int(historical_replay.get("dataset_mtime_ns", -1)) != dataset_stat.st_mtime_ns
    ):
        raise ValueError("Read-only replay historical train dataset bytes changed since the fit.")
    raw_dataset = GlobalChannelThermalDataset(
        dataset_root,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    template = _make_input_template(raw_dataset)
    operator = DifferentiableThermalOperator(
        target_model,
        template,
        dataset_config=checkpoint.get("train_config", {}).get("dataset", {}),
        normalization_stats=checkpoint.get("global_normalization_stats", {}),
        query_batch_size=query_batch_size,
    )
    initial_queries = role_queries_from_stencil(stencils[0], device=device)
    initial_design = DesignInput.from_state(stencils[0].baseline.design, device=device)
    with torch.no_grad():
        operator(initial_design, context_inputs(stencils[0].baseline.context), initial_queries)
    materialized_buffer_names = sorted(name for name, _value in target_model.named_buffers())
    scope_buffer_names = set(native_scope.get("frozen_buffer_names", ()))
    native_scope["buffers_initialized_by_materialization"] = sorted(
        set(materialized_buffer_names) - scope_buffer_names
    )
    native_scope["frozen_buffer_names"] = materialized_buffer_names
    if fit.get("native_trainable_scope") != native_scope:
        raise ValueError("Read-only replay expanded trainable scope differs from the passed fit.")

    scales = fit.get("loss_scales")
    if not isinstance(scales, Mapping) or not isinstance(scales.get("pressure_limit_by_family"), Mapping):
        raise ValueError("Passed native fit is missing frozen per-family pressure limits.")  # noqa: TRY004
    pressure_limits = dict(scales["pressure_limit_by_family"])
    if set(pressure_limits) != set(family_ids):
        raise ValueError("Frozen train pressure-limit families differ from the replay panel.")
    smooth_peak_beta = float(scales.get("smooth_peak_beta", 1.0))
    arms = {"R_value": value_path, "R_response": response_path}
    fit_arms = fit.get("arms")
    fit_checkpoint_paths = fit.get("checkpoint_paths")
    if not isinstance(fit_arms, Mapping) or not isinstance(fit_checkpoint_paths, Mapping):
        raise TypeError("Passed paired fit is missing its arm/checkpoint inventory.")
    arm_checkpoint_evidence: dict[str, dict[str, str]] = {}
    arm_payloads: dict[str, Mapping[str, Any]] = {}
    expected_training_config = load_staged_training_config(str(recipe_path), arm="R_response")
    recorded_fit_wall = fit.get("max_wall_seconds")
    for arm, path in arms.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _checkpoint_digest(path)
        recorded_paths = fit_checkpoint_paths.get(arm)
        fit_arm = fit_arms.get(arm)
        if not isinstance(recorded_paths, list) or str(path) not in [
            str(Path(item).expanduser().resolve()) for item in recorded_paths
        ]:
            raise ValueError(f"Passed fit manifest does not record the selected {arm} checkpoint.")
        if not isinstance(fit_arm, Mapping):
            raise TypeError(f"Passed fit manifest has no {arm} accounting.")
        _validate_fit_arm_gate_accounting(fit_arm, arm=arm, required_update=gate)
        if int(fit_arm.get("total_attempted_optimizer_steps", -1)) < gate:
            raise ValueError(f"Passed {arm} fit attempt total is below its cumulative update gate.")
        payload = _load_safe_response_checkpoint(path)
        if (
            payload.get("arm") != arm
            or int(payload.get("actual_optimizer_updates", -1)) != gate
            or int(payload.get("attempted_optimizer_steps", -1))
            != int(fit_arm.get("total_attempted_optimizer_steps", -2))
        ):
            raise ValueError(f"Passed {arm} checkpoint update/attempt accounting differs from its fit manifest.")
        saved_training_config = payload.get("training_config")
        if not isinstance(saved_training_config, Mapping):
            raise ValueError(f"Passed {arm} checkpoint has no pinned training config.")  # noqa: TRY004
        checkpoint_wall = saved_training_config.get("max_wall_seconds")
        if not isinstance(checkpoint_wall, (int, float)) or checkpoint_wall <= 0.0:
            raise ValueError(f"Passed {arm} checkpoint has no positive max-wall setting.")
        if recorded_fit_wall is not None and float(recorded_fit_wall) != float(checkpoint_wall):
            raise ValueError(f"Passed {arm} checkpoint max-wall differs from the fit manifest.")
        expected_config = training_config_mapping(
            replace(expected_training_config, max_wall_seconds=float(checkpoint_wall)),
            arm=arm,
        )
        if dict(saved_training_config) != expected_config:
            raise ValueError(f"Passed {arm} checkpoint training config differs from the pinned recipe.")
        if not isinstance(payload.get("model"), Mapping) or not payload["model"]:
            raise ValueError(f"Passed {arm} checkpoint has no model state for replay.")
        optimizer_state = payload.get("optimizer")
        if not isinstance(optimizer_state, Mapping) or not isinstance(
            optimizer_state.get("param_groups"), list
        ):
            raise ValueError(f"Passed {arm} checkpoint has no resumable optimizer state.")  # noqa: TRY004
        required_rng = (
            "python_rng_state", "numpy_rng_state", "torch_rng_state",
            "sampler_rng_state", "sampler_remaining_order",
            "historical_case_order", "historical_next_index",
        )
        if any(payload.get(name) is None for name in required_rng):
            raise ValueError(f"Passed {arm} checkpoint is missing resumable RNG/sampler state.")
        cuda_rng = payload.get("cuda_rng_state_by_model_device")
        if not isinstance(cuda_rng, Mapping) or not cuda_rng:
            raise ValueError(f"Passed {arm} checkpoint is missing device-specific CUDA RNG state.")
        arm_payloads[arm] = payload
        arm_checkpoint_evidence[arm] = {"path": str(path), "sha256": digest}
    if not _same(
        arm_payloads["R_value"].get("sampler_rng_state"),
        arm_payloads["R_response"].get("sampler_rng_state"),
    ):
        raise ValueError("Read-only replay arm checkpoints have different sampler RNG states.")
    if not _same(
        arm_payloads["R_value"].get("sampler_remaining_order"),
        arm_payloads["R_response"].get("sampler_remaining_order"),
    ):
        raise ValueError("Read-only replay arm checkpoints have different remaining sampler order.")
    historical_case_order = list(historical_replay.get("train_case_order", ()))
    if (
        not historical_case_order
        or len(set(historical_case_order)) != len(historical_case_order)
        or len(historical_case_order) != int(recipe.get("historical_train_case_count", -1))
    ):
        raise ValueError("Passed fit historical train-case order is missing, duplicated, or the wrong size.")
    for arm, payload in arm_payloads.items():
        if (
            list(payload["historical_case_order"]) != historical_case_order
            or int(payload["historical_next_index"]) != gate % len(historical_case_order)
        ):
            raise ValueError(f"Passed {arm} checkpoint historical replay cursor differs from the fit gate.")
    response_weights = fit.get("calibrated_response_weights")
    if not isinstance(response_weights, Mapping) or dict(
        arm_payloads["R_response"].get("calibrated_loss_weights", {})
    ) != dict(response_weights):
        raise ValueError("Passed response checkpoint weights differ from the frozen paired-fit weights.")

    train_entries = [
        {"path": str(path), "sha256": _checkpoint_digest(path), "json_sha256": _checkpoint_digest(path.with_suffix(".json"))}
        for path in resolved_stencils
    ]
    result: dict[str, Any] = {
        "status": "running",
        "mode": "read_only_full_grid_train_replay",
        "review_gate_update": gate,
        "source_fit_manifest": {"path": str(fit_path), "sha256": _checkpoint_digest(fit_path)},
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _checkpoint_digest(source_path),
        "refit_config": refit_config,
        "native_trainable_scope": native_scope,
        "train_stencils": train_entries,
        "train_family_ids": family_ids,
        "per_stencil_reference_states": {stencil.physical_family_id: len(stencil.records) for stencil in stencils},
        "arm_checkpoints": arm_checkpoint_evidence,
        "optimizer_instances_created": 0,
        "optimizer_calls": 0,
        "optimizer_updates": 0,
        "reference_solver_calls": 0,
        "reference_solves": 0,
        "optimizer_accounting_basis": "Replay constructs no optimizer and only evaluates stored train atlas states.",
        "deterministic_algorithms_enabled": bool(recipe.get("deterministic_algorithms", False)),
        "cublas_workspace_config": cublas_workspace,
        "evaluations": {arm: [] for arm in arms},
        "started_unix_seconds": time.time(),
    }
    progress_path = output_dir / "read_only_full_grid_train_replay_progress.json"
    _atomic_json(progress_path, result)
    active_arm = {"name": "setup"}
    forward_calls = {arm: 0 for arm in arms}

    def count_forward(_module: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        name = str(active_arm["name"])
        if name in forward_calls:
            forward_calls[name] += 1

    hook = target_model.register_forward_hook(count_forward)
    try:
        for arm, path in arms.items():
            if time.monotonic() - started >= max_wall_seconds:
                raise TimeoutError("Read-only full-grid train replay reached its wall cap.")
            target_model.load_state_dict(arm_payloads[arm]["model"], strict=True)
            target_model.eval()
            active_arm["name"] = arm
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            arm_started = time.perf_counter()
            rows: list[dict[str, Any]] = []
            for stencil in stencils:
                if time.monotonic() - started >= max_wall_seconds:
                    raise TimeoutError("Read-only full-grid train replay reached its wall cap.")
                family_id = stencil.physical_family_id
                with torch.no_grad():
                    metrics = evaluate_stencil(
                        operator,
                        stencil,
                        pressure_limit={family_id: float(pressure_limits[family_id])},
                        mixed_specs=_mixed_specs(stencil),
                        smooth_peak_beta=smooth_peak_beta,
                        device=device,
                    )
                rows.append({
                    "family_id": family_id,
                    "split": stencil.split.value,
                    "source": stencil.source.value,
                    "record_count": len(stencil.records),
                    "pressure_limit_frozen_from_fit": float(pressure_limits[family_id]),
                    "metrics": metrics,
                })
                result["evaluations"][arm] = list(rows)
                result["native_model_forward_calls_by_arm"] = dict(forward_calls)
                result["elapsed_wall_seconds"] = time.monotonic() - started
                _atomic_json(progress_path, result)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            else:
                peak = None
            result.setdefault("arm_wall_seconds", {})[arm] = time.perf_counter() - arm_started
            result.setdefault("cuda_peak_bytes_by_arm", {})[arm] = peak
        result.update({
            "status": "passed",
            "native_model_forward_calls_by_arm": dict(forward_calls),
            "total_wall_seconds": time.monotonic() - started,
            "finished_unix_seconds": time.time(),
            "cuda": _cuda_evidence(device),
        })
        _atomic_json(progress_path, result)
        return result
    except Exception as exc:
        result.update({
            "status": "timed_out" if isinstance(exc, TimeoutError) else "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "native_model_forward_calls_by_arm": dict(forward_calls),
            "total_wall_seconds": time.monotonic() - started,
            "finished_unix_seconds": time.time(),
        })
        _atomic_json(progress_path, result)
        raise
    finally:
        hook.remove()


def run_paired_fit(
    *,
    checkpoint_path: Path,
    stencil_paths: Sequence[Path],
    development_paths: Sequence[Path],
    dataset_path: Path | None,
    output_dir: Path,
    device: torch.device,
    sampling: ReceiverSamplingConfig,
    max_wall_seconds: float,
    initialization_mode: str = "three_term_conversion",
    recipe_config_path: Path | None = None,
    r0_diagnostic_path: Path | None = None,
    fit_capability_probe_path: Path | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    review_cap: int,
    review_continuations: Sequence[int],
    smooth_peak_beta: float,
    query_batch_size: int,
    resume_value_path: Path | None = None,
    resume_response_path: Path | None = None,
    frozen_loss_scales_path: Path | None = None,
    frozen_response_weights_path: Path | None = None,
    resume_fit_manifest_path: Path | None = None,
    resume_replay_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Run matched value/response arms through one explicit review gate."""

    total_started = time.monotonic()
    if not stencil_paths:
        raise ValueError("Paired fitting needs at least one train stencil.")
    if (resume_value_path is None) != (resume_response_path is None):
        raise ValueError("Resume requires both arm checkpoints from the same paired gate.")
    resuming = resume_value_path is not None
    if resuming:
        if frozen_loss_scales_path is None or frozen_response_weights_path is not None:
            raise ValueError(
                "A paired resume requires frozen loss scales but takes response weights from its checkpoints."
            )
        if resume_fit_manifest_path is None or resume_replay_manifest_path is None:
            raise ValueError("A paired resume requires both source fit and read-only replay manifests.")
    else:
        if (frozen_loss_scales_path is None) != (frozen_response_weights_path is None):
            raise ValueError(
                "A fresh fit requires both frozen scale and response-weight JSON files, or neither."
            )
        if resume_fit_manifest_path is not None or resume_replay_manifest_path is not None:
            raise ValueError("Resume manifests can be supplied only with a paired checkpoint resume.")
    if recipe_config_path is not None and initialization_mode != "native_checkpoint":
        raise ValueError("Named native response recipes require native_checkpoint initialization.")
    native_recipe = (
        recipe_config_path.expanduser().resolve()
        if recipe_config_path is not None
        else Path(__file__).resolve().parents[3] / "configs" / "response_control_native_staged.json"
        if initialization_mode == "native_checkpoint"
        else None
    )
    recipe_payload = (
        json.loads(native_recipe.read_text(encoding="utf-8"))
        if native_recipe is not None
        else {}
    )
    recipe_name = recipe_payload.get("name")
    expanded_recipe = recipe_name == "native_expanded_response_interface"
    nonlinear_recipe = recipe_name in {
        "native_nonlinear_interface",
        "native_expanded_response_interface",
    }
    expanded_probe_evidence: dict[str, Any] | None = None
    if expanded_recipe:
        if native_recipe is None or fit_capability_probe_path is None:
            raise ValueError(
                "Expanded native fitting requires its named recipe and a passed train-only capability probe."
            )
        if r0_diagnostic_path is not None:
            raise ValueError("The previous nonlinear-head R0 projection report does not apply to expanded scope.")
    elif fit_capability_probe_path is not None:
        raise ValueError("The expanded fit-capability report applies only to its separately named recipe.")
    if bool(recipe_payload.get("deterministic_algorithms", False)):
        _enable_deterministic_algorithms(device)
    source_model, checkpoint = load_model(checkpoint_path, device)
    source_model.eval()
    loaded_stencils = [load_response_atlas_stencil(path) for path in stencil_paths]
    raw_stencils = [stencil for stencil, _ in loaded_stencils]
    if any(stencil.split is not EvidenceSplit.TRAIN for stencil in raw_stencils):
        raise ValueError("Paired fitting accepts EvidenceSplit.TRAIN stencils only.")
    raw_family_ids = sorted({stencil.physical_family_id for stencil in raw_stencils})
    if nonlinear_recipe and len(raw_family_ids) != 8:
        raise ValueError(
            "Native response recipes require all eight distinct train families; "
            f"received {len(raw_family_ids)}: {raw_family_ids}."
        )
    if expanded_recipe:
        assert native_recipe is not None and fit_capability_probe_path is not None
        expanded_probe_evidence = _load_expanded_probe_evidence(
            fit_capability_probe_path,
            checkpoint_path=checkpoint_path,
            recipe_path=native_recipe,
            train_atlas_paths=stencil_paths,
            expected_family_ids=raw_family_ids,
        )
    loaded_development = [
        (path, *load_response_atlas_stencil(path)) for path in development_paths
    ]
    if nonlinear_recipe:
        if len(loaded_development) != 4:
            raise ValueError(
                "Native response recipes require all four stored Re90 development stencils for review."
            )
        invalid_dev = [
            str(path)
            for path, stencil, _ in loaded_development
            if stencil.split is EvidenceSplit.TRAIN
            or not np.isclose(float(stencil.baseline.context.values.get("re", np.nan)), 90.0)
        ]
        if invalid_dev:
            raise ValueError(
                "The nonlinear-interface development panel must contain only stored Re90 non-train stencils: "
                f"{invalid_dev}."
            )
    response_arm = "R_response" if nonlinear_recipe else "B_response"
    value_arm = f"{response_arm[0]}_value"
    config = replace(
        load_staged_training_config(
            path=None if native_recipe is None else str(native_recipe), arm=response_arm
        ),
        max_wall_seconds=max_wall_seconds,
    )
    projection_evidence: dict[str, Any] | None = None
    if nonlinear_recipe:
        if int(checkpoint.get("epoch", checkpoint.get("current_epoch", -1))) != 4738:
            raise ValueError("Native response recipes require the intact Run1804 e4738 checkpoint.")
        if str(source_model.config.core_honf.forward_architecture) != "dense_pairwise_field":
            raise ValueError("Native response recipes require the Run1804 dense native architecture.")
        required = set(config.required_response_terms)
        active_at_u50 = set(config.active_terms(49))
        if not required.issubset(active_at_u50) or any(
            config.term_multiplier(term, 49) != 1.0 for term in required
        ):
            raise ValueError("Native response and pressure objectives must be fully active by update 50.")
        required_controls = set(config.required_control_terms)
        if not required_controls.issubset(active_at_u50) or any(
            config.term_multiplier(term, 49) != 1.0 for term in required_controls
        ):
            raise ValueError("Native fixed-heat control objectives must be fully active by update 50.")
        if expanded_recipe and config.project_response_gradient_blockwise:
            raise ValueError("Expanded response fitting does not use the previous three-head gradient projection.")
        if not expanded_recipe and config.project_response_gradient_blockwise:
            if r0_diagnostic_path is None:
                raise ValueError(
                    "The nonlinear-interface recipe requires its train-only R0 diagnostic to justify projection."
                )
            r0_path = r0_diagnostic_path.expanduser().resolve()
            projection_evidence = _load_r0_projection_evidence(r0_path, checkpoint_path)
            justified = bool(projection_evidence["justified_by_measured_m10_combined_value_conflict"])
            config = replace(config, project_response_gradient_blockwise=justified)
    elif r0_diagnostic_path is not None:
        raise ValueError("R0 projection evidence applies only to native_nonlinear_interface.")
    effective_update_cap = _validate_review_gate(
        review_cap, config, len(raw_stencils)
    )
    sampled_panel = sample_training_panel(raw_stencils, config=sampling)
    training_stencils = tuple(item.stencil for item in sampled_panel)
    fixed_heat_controls: tuple[FixedHeatNullControl, ...] = ()
    fixed_heat_provenance: dict[str, Any] | None = None
    fixed_heat_review_stencils: tuple[tuple[str, ResponseStencil], ...] = ()
    if nonlinear_recipe:
        fixed_heat_controls, fixed_heat_provenance, fixed_heat_review_stencils = _load_fixed_heat_null_controls(
            recipe_payload,
            raw_stencils,
            sampled_panel,
            stencil_paths=stencil_paths,
        )
    dataset_root = _resolve_dataset_path(checkpoint, str(dataset_path) if dataset_path else None)
    train_config = checkpoint.get("train_config", {})
    dataset_config = train_config.get("dataset", {})
    raw_dataset = GlobalChannelThermalDataset(
        dataset_root,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    broad_eval_cases = (
        select_broad_evaluation_cases(raw_dataset, requested=30)
        if nonlinear_recipe
        else ()
    )
    template = _make_input_template(raw_dataset)
    # Pressure sections alone occupy 1,280 rows on the stored 64x128 grid.
    # Keep interior fluid support in the broad historical replay even when a
    # small stencil-only pilot requests fewer ordinary fluid queries.
    historical_sampling = replace(sampling, max_fluid_queries=max(3072, sampling.max_fluid_queries))
    historical_value_source = (
        HistoricalValueSource.from_dataset(raw_dataset, sampling=historical_sampling)
        if initialization_mode == "native_checkpoint"
        else None
    )
    if nonlinear_recipe and historical_value_source is None:
        raise ValueError("Native response recipes require the packed historical train cohort.")
    if expanded_recipe and historical_value_source is not None:
        expected_historical_count = int(recipe_payload.get("historical_train_case_count", -1))
        if len(historical_value_source.case_ids) != expected_historical_count:
            raise ValueError(
                "Expanded response fitting requires the complete packed train cohort: "
                f"expected {expected_historical_count}, got {len(historical_value_source.case_ids)}."
            )
    historical_calibration_case_map = (
        _historical_calibration_case_map(raw_family_ids, historical_value_source)
        if nonlinear_recipe and historical_value_source is not None
        else {}
    )
    scales_provenance: dict[str, Any] | None = None
    response_weight_provenance: dict[str, Any] | None = None
    fixed_response_weights: dict[str, float] | None = None
    if frozen_loss_scales_path is None:
        scales = derive_training_scales(
            training_stencils,
            smooth_peak_beta=smooth_peak_beta,
            historical_value_source=(historical_value_source if nonlinear_recipe else None),
        )
    else:
        scales, scales_provenance = _load_frozen_loss_scales(
            frozen_loss_scales_path,
            raw_stencils,
            smooth_peak_beta=smooth_peak_beta,
            expected_calibration_scope=(
                "eight_train_families_plus_one_historical_train_case_per_family"
                if nonlinear_recipe
                else None
            ),
            expected_family_ids=(raw_family_ids if nonlinear_recipe else None),
            expected_historical_case_map=(
                historical_calibration_case_map if nonlinear_recipe else None
            ),
        )
        if not resuming:
            fixed_response_weights, response_weight_provenance = _load_frozen_response_weights(
                frozen_response_weights_path,  # type: ignore[arg-type]
                required_terms=(
                    config.required_response_terms if response_arm.startswith("R_") else None
                ),
            )
    loss_scales_snapshot = {
        "value": dict(scales.value),
        "finite": dict(scales.finite),
        "mixed": dict(scales.mixed),
        "pressure_value": scales.pressure_value,
        "pressure_response": scales.pressure_response,
        "pressure_limit": scales.pressure_limit,
        "pressure_boundary": scales.pressure_boundary,
        "solid_temperature": scales.solid_temperature,
        "smooth_peak_beta": scales.smooth_peak_beta,
        "near_limit_band": scales.near_limit_band,
        "near_limit_multiplier": scales.near_limit_multiplier,
        "pressure_limit_by_family": dict(scales.pressure_limit_by_family or {}),
    }
    if frozen_loss_scales_path is None and initialization_mode == "native_checkpoint":
        # A fresh native u100 gate must leave the exact train-derived scale
        # source needed by the strict u300 resume check. Keep it beside the
        # ignored run outputs, before any optimizer update, and never replace
        # a different prior calibration in the same directory.
        derived_scale_path = output_dir / "native_train_loss_scales.json"
        derived_payload = {
            "frozen_scales": loss_scales_snapshot,
            "train_family_ids": raw_family_ids,
            "historical_train_case_ids": (
                list(historical_calibration_case_map.values())
            ),
            "historical_train_examples_by_family": historical_calibration_case_map,
            "equal_family_weighting": True,
            "calibration_scope": (
                "eight_train_families_plus_one_historical_train_case_per_family"
                if nonlinear_recipe
                else "train_stencils_before_native_fit"
            ),
            "source": "derived_from_train_stencils_before_native_fit",
        }
        comparable_payload = json.loads(json.dumps(derived_payload, default=_json_default))
        if derived_scale_path.exists():
            if json.loads(derived_scale_path.read_text(encoding="utf-8")) != comparable_payload:
                raise FileExistsError("The native fit output directory contains a different train-scale calibration.")
        else:
            _atomic_json(derived_scale_path, derived_payload)
        scales_provenance = {
            "path": str(derived_scale_path.resolve()),
            "sha256": _checkpoint_digest(derived_scale_path),
            "scope": derived_payload["calibration_scope"],
            "family_ids": raw_family_ids,
            "historical_train_case_ids": derived_payload["historical_train_case_ids"],
            "equal_family_weighting": True,
        }
    resume_payloads = None
    if resuming:
        resume_payloads = {
            value_arm: _load_safe_response_checkpoint(resume_value_path),
            response_arm: _load_safe_response_checkpoint(resume_response_path),
        }
    native_scope = None
    if initialization_mode == "native_checkpoint":
        target_model, refit_config = _native_checkpoint_initialization(source_model)
        requested_scope = recipe_payload.get("native_trainable_scope", "native_output_heads")
        if requested_scope == "native_nonlinear_interface":
            native_scope = _configure_native_nonlinear_interface_scope(target_model)
        elif requested_scope == "native_expanded_response_interface":
            native_scope = _configure_native_expanded_response_interface_scope(target_model)
        elif requested_scope == "native_output_heads":
            native_scope = _configure_native_output_head_scope(target_model)
        else:
            raise ValueError(f"Unknown native trainable scope {requested_scope!r}.")
    elif initialization_mode == "three_term_conversion":
        target_model, refit_config = _make_refit_model(source_model, checkpoint, device=device)
    else:
        raise ValueError(f"Unsupported initialization_mode {initialization_mode!r}.")
    resume_provenance = None
    if resuming:
        resume_provenance = validate_paired_resume_provenance(
            fit_manifest_path=resume_fit_manifest_path,  # type: ignore[arg-type]
            replay_manifest_path=resume_replay_manifest_path,  # type: ignore[arg-type]
            source_checkpoint_path=checkpoint_path,
            train_atlas_paths=stencil_paths,
            development_atlas_paths=development_paths,
            train_family_ids=[stencil.physical_family_id for stencil in raw_stencils],
            train_sampling=[_sampling_summary_mapping(item.summary) for item in sampled_panel],
            frozen_scales_path=frozen_loss_scales_path,  # type: ignore[arg-type]
            loss_scales=loss_scales_snapshot,
            refit_config=refit_config,
            native_trainable_scope=native_scope,
            native_parameter_inventory=_parameter_inventory(target_model) if native_scope is not None else None,
            historical_case_order=(historical_value_source.case_ids if historical_value_source is not None else ()),
            historical_dataset_path=(historical_value_source.dataset_path if historical_value_source is not None else None),
            resume_checkpoint_paths={
                value_arm: resume_value_path,  # type: ignore[dict-item]
                response_arm: resume_response_path,  # type: ignore[dict-item]
            },
            resume_payloads=resume_payloads,  # type: ignore[arg-type]
            training_config=config,
            required_update=int(resume_payloads[value_arm]["actual_optimizer_updates"]),
        )
        response_weight_provenance = {
            "source": f"verified_{response_arm}_u{int(resume_payloads[value_arm]['actual_optimizer_updates'])}_checkpoint",
            "fit_manifest_sha256": resume_provenance["fit_manifest"]["sha256"],
        }
    operator = DifferentiableThermalOperator(
        target_model,
        template,
        dataset_config=dataset_config,
        normalization_stats=checkpoint.get("global_normalization_stats", {}),
        query_batch_size=query_batch_size,
    )
    if initialization_mode == "native_checkpoint":
        initial_queries = role_queries_from_stencil(training_stencils[0], device=device)
        initial_design = DesignInput.from_state(training_stencils[0].baseline.design, device=device)
        with torch.no_grad():
            operator(
                initial_design,
                context_inputs(training_stencils[0].baseline.context),
                initial_queries,
            )
        transfer = dict(refit_config)
        transfer["native_output_scope"] = native_scope
    else:
        transfer = _materialize_and_warm_start(
            source_model, target_model, operator, training_stencils[0], device=device
        )
    incumbent_operator: DifferentiableThermalOperator | None = None
    if nonlinear_recipe:
        # Keep the intact e4738 incumbent on its original checkpoint state.
        # The paired target is a separate native copy and is the only model
        # passed to either optimizer.
        incumbent_operator = DifferentiableThermalOperator(
            source_model,
            template,
            dataset_config=dataset_config,
            normalization_stats=checkpoint.get("global_normalization_stats", {}),
            query_batch_size=query_batch_size,
        )
        incumbent_model_state_before = {
            name: value.detach().cpu().clone()
            for name, value in source_model.state_dict().items()
        }
        initial_queries = role_queries_from_stencil(training_stencils[0], device=device)
        initial_design = DesignInput.from_state(training_stencils[0].baseline.design, device=device)
        with torch.no_grad():
            incumbent_operator(
                initial_design,
                context_inputs(training_stencils[0].baseline.context),
                initial_queries,
            )
        changed_incumbent_state = [
            name
            for name, value in source_model.state_dict().items()
            if name in incumbent_model_state_before
            and not torch.equal(value.detach().cpu(), incumbent_model_state_before[name])
        ]
        missing_incumbent_state = sorted(
            set(incumbent_model_state_before) - set(source_model.state_dict())
        )
        if changed_incumbent_state or missing_incumbent_state:
            raise RuntimeError(
                "Materializing incumbent review inputs changed persistent checkpoint state: "
                f"changed={changed_incumbent_state}, missing={missing_incumbent_state}."
            )
        source_model.eval()
    parameter_inventory = _parameter_inventory(target_model)
    target_model.eval()
    derivative_started = time.perf_counter()
    derivative = check_pressure_peak_ad_fd(
        operator,
        training_stencils[0],
        variable="heating",
        module_slot=0,
        step=max(abs(float(training_stencils[0].baseline.design.modules[0].heating)) * 1.0e-2, 1.0e-5),
        relative_tolerance=0.35,
        absolute_tolerance=1.0e-4,
        device=device,
    )
    if not derivative["passed"]:
        raise RuntimeError(f"Whole-wrapper pressure/peak AD-FD check failed: {derivative}")
    derivative_wall_seconds = time.perf_counter() - derivative_started

    forward_phase = {"name": "paired_response_weight_calibration"}
    native_forward_call_counts: dict[str, dict[str, int]] = {
        "target_model": {},
        "incumbent_e4738": {},
    }

    def _forward_counter(model_label: str):
        def count_forward(_module: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
            phase = str(forward_phase["name"])
            counts = native_forward_call_counts[model_label]
            counts[phase] = counts.get(phase, 0) + 1

        return count_forward

    target_forward_hook = target_model.register_forward_hook(_forward_counter("target_model"))
    incumbent_forward_hook = (
        source_model.register_forward_hook(_forward_counter("incumbent_e4738"))
        if nonlinear_recipe
        else None
    )

    frozen_parameter_snapshot: dict[str, torch.Tensor] = {}
    frozen_buffer_snapshot: dict[str, torch.Tensor] = {}
    frozen_buffer_checkpoint_audit: list[dict[str, Any]] = []
    if native_scope is not None:
        frozen_parameter_snapshot = {
            name: parameter.detach().cpu().clone()
            for name, parameter in target_model.named_parameters()
            if not parameter.requires_grad
        }
        frozen_buffer_snapshot = {
            name: value.detach().cpu().clone()
            for name, value in target_model.named_buffers()
        }
        scope_buffer_names = set(native_scope.get("frozen_buffer_names", ()))
        native_scope["buffers_initialized_by_materialization"] = sorted(
            set(frozen_buffer_snapshot) - scope_buffer_names
        )
        # Scope provenance is refreshed after lazy positional encoders have
        # seen the complete warm-up/AD-FD input path.
        native_scope["frozen_buffer_names"] = sorted(frozen_buffer_snapshot)

    optimizer_config = checkpoint.get("train_config", {}).get("training", {})
    lr = float(
        learning_rate
        if learning_rate is not None
        else recipe_payload.get("learning_rate", 1.0e-5)
        if expanded_recipe
        else 1.0e-5
        if initialization_mode == "native_checkpoint"
        else optimizer_config.get("learning_rate", 3.0e-4)
    )
    decay = float(
        weight_decay
        if weight_decay is not None
        else recipe_payload.get("weight_decay", 1.0e-5)
        if expanded_recipe
        else optimizer_config.get("weight_decay", 1.0e-5)
    )
    if expanded_recipe and (
        not np.isclose(lr, float(recipe_payload["learning_rate"]), rtol=0.0, atol=1.0e-15)
        or not np.isclose(decay, float(recipe_payload["weight_decay"]), rtol=0.0, atol=1.0e-15)
    ):
        raise ValueError("Expanded matched fitting uses only its pinned learning rate and weight decay.")

    def optimizer_factory(model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=lr,
            weight_decay=decay,
        )

    latest_payloads: dict[str, Mapping[str, Any]] = {}
    checkpoint_paths: dict[str, list[str]] = {value_arm: [], response_arm: []}

    def save_checkpoint(arm: str, payload: Mapping[str, Any], label: str) -> None:
        arm_config = StagedTrainingConfig.from_mapping(payload["training_config"])
        safe_label = "".join(character if character.isalnum() else "_" for character in label)
        update = int(payload["actual_optimizer_updates"])
        path = output_dir / f"response_control_{arm}_{safe_label}_u{update:05d}.pt"
        checkpoint_buffer_audit: dict[str, Any] | None = None
        if native_scope is not None:
            state = payload.get("model")
            if not isinstance(state, Mapping):
                raise TypeError("Native checkpoint payload has no named model state.")
            current_parameters = dict(target_model.named_parameters())
            missing_parameters = sorted(
                set(frozen_parameter_snapshot) - set(current_parameters)
            )
            changed_parameters = sorted(
                name
                for name, reference in frozen_parameter_snapshot.items()
                if name in current_parameters
                and not torch.equal(current_parameters[name].detach().cpu(), reference)
            )
            missing_checkpoint_parameters = sorted(
                name for name in frozen_parameter_snapshot if state.get(name) is None
            )
            changed_checkpoint_parameters = sorted(
                name
                for name, reference in frozen_parameter_snapshot.items()
                if state.get(name) is not None
                and not torch.equal(state[name].detach().cpu(), reference)
            )
            if (
                missing_parameters
                or changed_parameters
                or missing_checkpoint_parameters
                or changed_checkpoint_parameters
            ):
                raise RuntimeError(
                    f"Frozen native parameters changed at {arm} update {update}: "
                    f"missing={missing_parameters}, changed={changed_parameters}, "
                    f"checkpoint_missing={missing_checkpoint_parameters}, "
                    f"checkpoint_changed={changed_checkpoint_parameters}."
                )
            checkpoint_buffer_audit = _audit_named_buffers(
                target_model, frozen_buffer_snapshot
            )
            if not checkpoint_buffer_audit["passed"]:
                raise RuntimeError(
                    f"Frozen native buffers changed at {arm} update {update}: "
                    f"{checkpoint_buffer_audit}."
                )
            for name in checkpoint_buffer_audit["added_names"]:
                frozen_buffer_snapshot[name] = (
                    dict(target_model.named_buffers())[name].detach().cpu().clone()
                )
            frozen_buffer_checkpoint_audit.append({
                "arm": arm,
                "label": label,
                "update": update,
                **checkpoint_buffer_audit,
            })
        persisted_payload = dict(payload)
        persisted_payload["response_control_calibration_provenance"] = {
            "loss_scales": loss_scales_snapshot,
            "loss_scales_source": scales_provenance,
            "response_weight_source": response_weight_provenance,
            "resume_provenance": resume_provenance,
            "source_checkpoint": str(checkpoint_path.expanduser().resolve()),
            "source_checkpoint_sha256": _checkpoint_digest(checkpoint_path),
            "training_recipe": None if native_recipe is None else str(native_recipe),
            "training_recipe_sha256": (
                None if native_recipe is None else _checkpoint_digest(native_recipe)
            ),
            "plan_schema": 1,
            "active_scope": native_scope,
            "expanded_fit_capability_probe": expanded_probe_evidence,
            "frozen_buffer_checkpoint_audit": checkpoint_buffer_audit,
            "native_forward_call_counts_by_phase": {
                name: dict(counts) for name, counts in native_forward_call_counts.items()
            },
        }
        _atomic_torch_save(path, persisted_payload)
        loaded = _load_safe_response_checkpoint(path)
        restore_checkpoint_payload(target_model, active_optimizers[arm], loaded, config=arm_config)
        latest_payloads[arm] = loaded
        checkpoint_paths[arm].append(str(path))

    active_optimizers: dict[str, torch.optim.Optimizer] = {}

    def optimizer_factory_tracked(model: torch.nn.Module) -> torch.optim.Optimizer:
        optimizer = optimizer_factory(model)
        _verify_optimizer_inventory(model, optimizer)
        _optimizer_hyperparameter_inventory(
            optimizer, learning_rate=lr, weight_decay=decay
        )
        arm = value_arm if len(active_optimizers) == 0 else response_arm
        forward_phase["name"] = f"{arm}_training"
        active_optimizers[arm] = optimizer
        return optimizer

    all_mixed_specs: list[MixedResponseSpec] = []
    seen_specs: set[tuple[str, str, str, str]] = set()
    for stencil in training_stencils:
        for spec in _mixed_specs(stencil):
            key = (spec.role, spec.joint_variant, spec.first_variant, spec.second_variant)
            if key not in seen_specs:
                seen_specs.add(key)
                all_mixed_specs.append(spec)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    fit_started = time.perf_counter()
    paired = run_paired_staged_fits(
        target_model,
        lambda model: DifferentiableThermalOperator(
            model,
            template,
            dataset_config=dataset_config,
            normalization_stats=checkpoint.get("global_normalization_stats", {}),
            query_batch_size=query_batch_size,
        ),
        optimizer_factory_tracked,
        training_stencils,
        historical_value_source=historical_value_source,
        scales=scales,
        mixed_specs=tuple(all_mixed_specs),
        config=config,
        device=device,
        on_checkpoint=save_checkpoint,
        stop_at_update=review_cap,
        review_continuations=review_continuations,
        resume_payloads=resume_payloads,
        fixed_response_weights=fixed_response_weights,
        fixed_heat_controls=fixed_heat_controls,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        train_cuda_peak = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        torch.cuda.reset_peak_memory_stats(device)
    else:
        train_cuda_peak = None
    fit_seconds = time.perf_counter() - fit_started
    curve_path = write_paired_training_curves(output_dir / "paired_training_curves.csv", paired)
    reached_updates = {
        arm: result.final_update for arm, result in paired.arms.items()
    }
    if any(update != review_cap for update in reached_updates.values()):
        raise TimeoutError(
            f"Paired arms did not reach requested review gate {review_cap}: {reached_updates}. "
            "Any numbered checkpoints and partial loss curve were preserved."
        )

    development_results: dict[str, list[dict[str, Any]]] = {value_arm: [], response_arm: []}
    review_evaluation: dict[str, Any] | None = None
    development_started = time.perf_counter()
    if nonlinear_recipe:
        if incumbent_operator is None:
            raise RuntimeError("The intact incumbent operator was not prepared for nonlinear review.")
        panel_stencils: dict[str, list[tuple[str, ResponseStencil, float]]] = {
            "train_eight_families": [],
            "re90_four_development": [],
            "fixed_heat_four_controls": [],
        }
        family_pressure_limits = dict(scales.pressure_limit_by_family or {})
        for stencil in raw_stencils:
            if stencil.physical_family_id not in family_pressure_limits:
                raise ValueError(
                    f"Train review is missing the original fixed pressure limit for {stencil.physical_family_id!r}."
                )
            panel_stencils["train_eight_families"].append((
                stencil.physical_family_id,
                stencil,
                float(family_pressure_limits[stencil.physical_family_id]),
            ))
        for path, stencil, _ in loaded_development:
            del path
            pressure = stencil.baseline.output.quantities["pressure_drop"]  # type: ignore[union-attr]
            if not pressure.resolved:
                raise ValueError("Re90 review requires its resolved original baseline pressure.")
            dev_limit = float(pressure.value) * 1.05
            panel_stencils["re90_four_development"].append((
                stencil.physical_family_id, stencil, dev_limit
            ))
        for control_id, review_stencil in fixed_heat_review_stencils:
            family_id = review_stencil.physical_family_id
            if family_id not in family_pressure_limits:
                raise ValueError(
                    f"Fixed-heat review has no frozen family pressure limit for {family_id!r}."
                )
            panel_stencils["fixed_heat_four_controls"].append((
                control_id,
                review_stencil,
                float(family_pressure_limits[family_id]),
            ))
        if len(panel_stencils["train_eight_families"]) != 8:
            raise ValueError("The u200 train review panel must contain eight family stencils.")
        if len(panel_stencils["fixed_heat_four_controls"]) != 4:
            raise ValueError("The u200 fixed-heat review panel must contain all four controls.")
        broad_records = [
            (
                row,
                load_stored_reference_case(dataset_root, str(row["case_id"])),
            )
            for row in broad_eval_cases
        ]
        if len(broad_records) != 30 or len({row["case_id"] for row, _ in broad_records}) != 30:
            raise ValueError("The u200 broad train review panel must contain 30 unique stored cases.")
        for row, record in broad_records:
            if record.output is None or record.design.split is not EvidenceSplit.TRAIN:
                raise ValueError(f"Broad review case {row['case_id']!r} is not a solved train record.")
            pressure_target = record.output.quantities["pressure_drop"]
            if not pressure_target.resolved or not np.isfinite(float(pressure_target.value)):
                raise ValueError("Broad train review pressure target is non-finite.")

        models: dict[str, DifferentiableThermalOperator] = {
            "incumbent_e4738": incumbent_operator,
            value_arm: operator,
            response_arm: operator,
        }
        for arm in (value_arm, response_arm):
            if not latest_payloads.get(arm):
                raise RuntimeError(f"The paired u{review_cap} checkpoint for {arm} is missing.")

        review_models: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for model_name, review_operator in models.items():
            if model_name in latest_payloads:
                # The operator closes over the shared target module. Restore
                # the requested arm immediately before its complete panel so
                # both names cannot accidentally replay the last loaded arm.
                _restore_review_arm_state(target_model, model_name, latest_payloads)
            model_panels: dict[str, list[dict[str, Any]]] = {}
            for panel_name, items in panel_stencils.items():
                forward_phase["name"] = f"review_{model_name}_{panel_name}"
                result_rows: list[dict[str, Any]] = []
                for case_id, stencil, pressure_limit in items:
                    with torch.no_grad():
                        metrics = evaluate_stencil(
                            review_operator,
                            stencil,
                            pressure_limit={stencil.physical_family_id: pressure_limit},
                            mixed_specs=_mixed_specs(stencil),
                            smooth_peak_beta=smooth_peak_beta,
                            device=device,
                        )
                    result_rows.append({
                        "case_id": case_id,
                        "family_id": stencil.physical_family_id,
                        "split": stencil.split.value,
                        "source": stencil.source.value,
                        "pressure_limit_original": pressure_limit,
                        "metrics": metrics,
                    })
                model_panels[panel_name] = result_rows
            broad_rows: list[dict[str, Any]] = []
            forward_phase["name"] = f"review_{model_name}_broad_train_historical_30"
            for row, record in broad_records:
                pressure_target = record.output.quantities["pressure_drop"]  # type: ignore[union-attr]
                if not pressure_target.resolved or not np.isfinite(float(pressure_target.value)):
                    raise ValueError(f"Broad case {row['case_id']!r} has unresolved original pressure.")
                fixed_limit = float(pressure_target.value) * 1.05
                with torch.no_grad():
                    metrics = evaluate_absolute_record(
                        review_operator,
                        record,
                        pressure_limit=fixed_limit,
                        device=device,
                    )
                broad_rows.append({
                    **dict(row),
                    "split": record.design.split.value,
                    "source": record.source.value,
                    "pressure_limit_original": fixed_limit,
                    "pressure_limit_rule": "1.05x stored original train-record pressure baseline",
                    "metrics": metrics,
                })
            model_panels["broad_train_historical_30"] = broad_rows
            review_models[model_name] = model_panels
            if model_name in development_results:
                development_results[model_name] = model_panels["re90_four_development"]
        review_evaluation = {
            "review_update": review_cap,
            "models": review_models,
            "panel_counts": {
                "train_eight_families": 8,
                "re90_four_development": 4,
                "fixed_heat_four_controls": 4,
                "broad_train_historical_30": 30,
            },
            "broad_train_selection": {
                "rule": "minimum, median, and maximum Reynolds cases per active-module-count stratum; deterministic fill if needed",
                "cases": [dict(row) for row in broad_eval_cases],
                "pressure_limit_rule": "1.05x each stored original train-record pressure baseline, frozen before model prediction",
            },
            "temperature_units": "Each metric row carries the source role's declared channel_units; no SI relabeling is inferred.",
            "fixed_heat_control_provenance": fixed_heat_provenance,
        }
    else:
        for arm in (value_arm, response_arm):
            if not latest_payloads.get(arm):
                continue
            target_model.load_state_dict(latest_payloads[arm]["model"], strict=True)
            target_model.eval()
            forward_phase["name"] = f"development_evaluation_{arm}"
            for path, dev_stencil, _ in loaded_development:
                if dev_stencil.split is EvidenceSplit.TRAIN:
                    raise ValueError("Development evaluation paths must not carry the train split label.")
                dev_limit = float(dev_stencil.baseline.output.quantities["pressure_drop"].value) * 1.05  # type: ignore[union-attr]
                with torch.no_grad():
                    metrics = evaluate_stencil(
                        operator,
                        dev_stencil,
                        pressure_limit={dev_stencil.physical_family_id: dev_limit},
                        mixed_specs=_mixed_specs(dev_stencil),
                        smooth_peak_beta=smooth_peak_beta,
                        device=device,
                    )
                development_results[arm].append(
                    {
                        "family_id": dev_stencil.physical_family_id,
                        "split": dev_stencil.split.value,
                        "source": dev_stencil.source.value,
                        "pressure_limit_from_original_start": dev_limit,
                        "metrics": metrics,
                    }
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        development_cuda_peak = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    else:
        development_cuda_peak = None
    development_wall_seconds = time.perf_counter() - development_started
    target_forward_hook.remove()
    if incumbent_forward_hook is not None:
        incumbent_forward_hook.remove()
    historical_replay_coverage_by_arm: dict[str, Any] = {}
    paired_historical_case_sequence_equal: bool | None = None
    if historical_value_source is not None:
        segment_ids = {
            arm: tuple(step.historical_case_id for step in fit.history)
            for arm, fit in paired.arms.items()
        }
        if any(any(case_id is None for case_id in values) for values in segment_ids.values()):
            raise RuntimeError("Native paired-fit history omitted historical train case IDs.")
        paired_historical_case_sequence_equal = len(set(segment_ids.values())) == 1
        if not paired_historical_case_sequence_equal:
            raise RuntimeError("Matched native arms consumed different historical train case sequences.")
        historical_replay_coverage_by_arm = {
            arm: historical_replay_coverage(
                historical_value_source.case_ids,
                initial_update=fit.initial_update,
                segment_case_ids=tuple(str(case_id) for case_id in segment_ids[arm]),
            )
            for arm, fit in paired.arms.items()
        }
    return {
        "status": "passed",
        "mode": "paired_staged_fit",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _checkpoint_digest(checkpoint_path),
        "selected_epoch": checkpoint.get("epoch", checkpoint.get("current_epoch")),
        "initialization_mode": initialization_mode,
        "staged_recipe": str(native_recipe) if native_recipe is not None else "historical_default",
        "staged_recipe_name": recipe_payload.get("name"),
        "expanded_fit_capability_probe": expanded_probe_evidence,
        "read_only_comparator": recipe_payload.get("read_only_comparator") if expanded_recipe else None,
        "train_atlas_paths": [str(path.resolve()) for path in stencil_paths],
        "train_family_ids": [stencil.physical_family_id for stencil in training_stencils],
        "train_contexts": [dict(stencil.baseline.context.values) for stencil in training_stencils],
        "train_sampling": [_sampling_summary_mapping(item.summary) for item in sampled_panel],
        "historical_value_replay": (
            None if historical_value_source is None else {
                "dataset": str(historical_value_source.dataset_path),
                "dataset_size_bytes": historical_value_source.dataset_path.stat().st_size,
                "dataset_mtime_ns": historical_value_source.dataset_path.stat().st_mtime_ns,
                "train_case_count": len(historical_value_source.case_ids),
                "train_case_order": list(historical_value_source.case_ids),
                "sampling": asdict(historical_sampling),
                "one_case_per_response_update": True,
                "realized_case_coverage": historical_value_source.realized_coverage,
            }
        ),
        "historical_replay_coverage_by_arm": historical_replay_coverage_by_arm,
        "paired_historical_case_sequence_equal": paired_historical_case_sequence_equal,
        "development_paths": [str(path.resolve()) for path in development_paths],
        "refit_config": refit_config,
        "warm_start": transfer,
        "parameter_inventory": parameter_inventory,
        "native_trainable_scope": native_scope,
        "projection_evidence": projection_evidence,
        "projection_active": bool(config.project_response_gradient_blockwise),
        "resolved_optimizer": {
            "name": "AdamW",
            "learning_rate": lr,
            "weight_decay": decay,
            "scope": native_scope["name"] if native_scope is not None else "all_trainable_parameters",
        },
        "whole_wrapper_ad_fd": derivative,
        "whole_wrapper_ad_fd_wall_seconds": derivative_wall_seconds,
        "gradient_calibration": {
            "scope": (
                "resume_checkpoint"
                if resume_payloads is not None
                else "frozen_recipe_calibration"
                if fixed_response_weights is not None
                else "eight_train_families_plus_one_historical_train_case_per_family"
                if nonlinear_recipe and historical_value_source is not None
                else "balanced_train_families_plus_historical_train_cases"
                if historical_value_source is not None
                else "balanced_train_families_only"
            ),
            "train_family_ids": sorted({stencil.physical_family_id for stencil in training_stencils}),
            "historical_train_case_ids": (
                list(historical_calibration_case_map.values())
                if historical_value_source is not None and resume_payloads is None
                and fixed_response_weights is None
                else []
            ),
            "historical_train_examples_by_family": historical_calibration_case_map,
            "weights_source": response_weight_provenance,
            "weights": dict(paired.calibrated_response_weights),
            "gradient_diagnostics": dict(paired.gradient_calibration),
        },
        "loss_scales_source": scales_provenance,
        "loss_scales": loss_scales_snapshot,
        "resume_provenance": resume_provenance,
        "review_decisions": dict(paired.review_decisions),
        "review_cap": review_cap,
        "effective_update_cap": effective_update_cap,
        "review_continuations": list(review_continuations),
        "review_stage_cap": _review_stage_cap_record(
            review_cap=review_cap,
            resume_payloads=resume_payloads,
            review_continuations=review_continuations,
        ),
        "arms": {
            arm: {
                "initial_update": result.initial_update,
                "actual_optimizer_updates": result.actual_optimizer_updates,
                "attempted_optimizer_steps": result.attempted_optimizer_steps,
                "final_update": result.final_update,
                "total_attempted_optimizer_steps": result.total_attempted_optimizer_steps,
                "stopped_at_review": result.stopped_at_review,
                "stopped_for_wall_time": result.stopped_for_wall_time,
                "wall_seconds": result.wall_seconds,
            }
            for arm, result in paired.arms.items()
        },
        "calibrated_response_weights": dict(paired.calibrated_response_weights),
        "fixed_heat_control_provenance": fixed_heat_provenance,
        "frozen_buffer_integrity": dict(paired.buffer_integrity),
        "frozen_parameter_snapshot_count": len(frozen_parameter_snapshot),
        "frozen_buffer_checkpoint_audit": frozen_buffer_checkpoint_audit,
        "curve_path": str(curve_path),
        "checkpoint_paths": checkpoint_paths,
        "development_evaluation": development_results,
        "review_evaluation": review_evaluation,
        "development_evaluation_wall_seconds": development_wall_seconds,
        "native_forward_call_counts": {
            name: dict(counts) for name, counts in native_forward_call_counts.items()
        },
        "native_forward_call_count_definition": (
            "Root ChannelThermalHONFModel forward-hook invocations by named calibration, arm-training, and review panel phase; "
            "excludes pre-fit input materialization and AD-FD checks before the hook was attached."
        ),
        "checkpoint_selection_rule": (
            f"No automatic selection. Review matched {value_arm}/{response_arm} at this gate; "
            "feasibility and critical responses first, then decision and role-separated receiver quality."
        ),
        "fit_wall_seconds": fit_seconds,
        "total_wall_seconds": float(time.monotonic() - total_started),
        "cuda": _cuda_evidence(device),
        "cuda_phase_peaks": {
            "paired_fit": train_cuda_peak,
            "development_replay": development_cuda_peak,
        },
    }


def run_checkpoint_only_review(
    *,
    checkpoint_path: Path,
    source_manifest_path: Path,
    value_checkpoint_path: Path,
    response_checkpoint_path: Path,
    stencil_paths: Sequence[Path],
    development_paths: Sequence[Path],
    dataset_path: Path | None,
    recipe_config_path: Path,
    r0_diagnostic_path: Path,
    frozen_loss_scales_path: Path,
    output_dir: Path,
    device: torch.device,
    sampling: ReceiverSamplingConfig,
    max_wall_seconds: float,
    query_batch_size: int,
) -> dict[str, Any]:
    """Evaluate the exact saved R1 u200 pair without creating an optimizer."""

    started = time.monotonic()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    recipe_payload = json.loads(recipe_config_path.read_text(encoding="utf-8"))
    cublas_workspace = (
        _enable_deterministic_algorithms(device)
        if bool(recipe_payload.get("deterministic_algorithms", False))
        else None
    )
    source_model, source_checkpoint = load_model(checkpoint_path, device)
    if int(source_checkpoint.get("epoch", source_checkpoint.get("current_epoch", -1))) != 4738:
        raise ValueError("Checkpoint-only R1 review requires the intact Run1804 e4738 source.")
    source_model.eval()

    train_loaded = [load_response_atlas_stencil(path) for path in stencil_paths]
    train_stencils = [stencil for stencil, _ in train_loaded]
    if len(train_stencils) != 8 or len({stencil.physical_family_id for stencil in train_stencils}) != 8:
        raise ValueError("Checkpoint-only R1 review requires eight unique train families.")
    if any(stencil.split is not EvidenceSplit.TRAIN for stencil in train_stencils):
        raise ValueError("Checkpoint-only train review received a non-train response atlas.")
    development_loaded = [
        (path, *load_response_atlas_stencil(path)) for path in development_paths
    ]
    if len(development_loaded) != 4 or any(
        stencil.split is EvidenceSplit.TRAIN
        or not np.isclose(float(stencil.baseline.context.values.get("re", np.nan)), 90.0)
        for _, stencil, _ in development_loaded
    ):
        raise ValueError("Checkpoint-only review requires the four stored non-train Re90 stencils.")

    config = load_staged_training_config(str(recipe_config_path), arm="R_response")
    config = replace(config, max_wall_seconds=float(source_manifest.get("max_wall_seconds", 1800.0)))
    projection_evidence: dict[str, Any] | None = None
    if config.project_response_gradient_blockwise:
        projection_evidence = _load_r0_projection_evidence(r0_diagnostic_path, checkpoint_path)
        config = replace(
            config,
            project_response_gradient_blockwise=bool(
                projection_evidence["justified_by_measured_m10_combined_value_conflict"]
            ),
        )

    value_payload = _load_safe_response_checkpoint(value_checkpoint_path)
    response_payload = _load_safe_response_checkpoint(response_checkpoint_path)
    arm_payloads = {"R_value": value_payload, "R_response": response_payload}
    scales_payload = json.loads(frozen_loss_scales_path.read_text(encoding="utf-8"))
    frozen_scales = scales_payload["frozen_scales"]
    provenance = validate_checkpoint_only_review_provenance(
        failed_manifest_path=source_manifest_path,
        source_checkpoint_path=checkpoint_path,
        recipe_path=recipe_config_path,
        r0_diagnostic_path=r0_diagnostic_path,
        frozen_scales_path=frozen_loss_scales_path,
        train_atlas_paths=stencil_paths,
        development_atlas_paths=development_paths,
        arm_checkpoint_paths={
            "R_value": value_checkpoint_path,
            "R_response": response_checkpoint_path,
        },
        arm_payloads=arm_payloads,
        training_config=config,
        review_update=200,
    )
    family_pressure_limits = {
        str(key): float(value)
        for key, value in frozen_scales.get("pressure_limit_by_family", {}).items()
    }
    if set(family_pressure_limits) != {stencil.physical_family_id for stencil in train_stencils}:
        raise ValueError("Frozen original pressure limits do not cover the eight train families exactly.")
    smooth_peak_beta = float(frozen_scales["smooth_peak_beta"])

    dataset_root = _resolve_dataset_path(
        source_checkpoint,
        str(dataset_path) if dataset_path is not None else None,
    )
    train_config = source_checkpoint.get("train_config", {})
    dataset_config = train_config.get("dataset", {})
    raw_dataset = GlobalChannelThermalDataset(
        dataset_root,
        split="train",
        points_per_case=1,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        include_grid=False,
        include_structure_targets=False,
    )
    broad_eval_cases = select_broad_evaluation_cases(raw_dataset, requested=30)
    if len(broad_eval_cases) != 30:
        raise ValueError("Checkpoint-only review requires exactly 30 selected broad train cases.")
    template = _make_input_template(raw_dataset)

    target_model, refit_config = _native_checkpoint_initialization(source_model)
    target_operator = DifferentiableThermalOperator(
        target_model,
        template,
        dataset_config=dataset_config,
        normalization_stats=source_checkpoint.get("global_normalization_stats", {}),
        query_batch_size=query_batch_size,
    )
    source_operator = DifferentiableThermalOperator(
        source_model,
        template,
        dataset_config=dataset_config,
        normalization_stats=source_checkpoint.get("global_normalization_stats", {}),
        query_batch_size=query_batch_size,
    )
    setup_forward_counts = {"incumbent_e4738": 0, "target_initialization": 0}

    def _setup_forward_counter(name: str):
        def count_forward(_module: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
            setup_forward_counts[name] += 1

        return count_forward

    source_setup_hook = source_model.register_forward_hook(
        _setup_forward_counter("incumbent_e4738")
    )
    target_setup_hook = target_model.register_forward_hook(
        _setup_forward_counter("target_initialization")
    )
    first_stencil = train_stencils[0]
    initial_design = DesignInput.from_state(first_stencil.baseline.design, device=device)
    initial_queries = role_queries_from_stencil(first_stencil, device=device)
    initial_context = context_inputs(first_stencil.baseline.context)
    try:
        with torch.no_grad():
            source_operator(initial_design, initial_context, initial_queries)
            target_operator(initial_design, initial_context, initial_queries)
    finally:
        source_setup_hook.remove()
        target_setup_hook.remove()
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    if set(source_state) != set(target_state) or any(
        not torch.equal(source_state[name].detach().cpu(), target_state[name].detach().cpu())
        for name in source_state
    ):
        raise RuntimeError("Deep-copied source and target differ after lazy-state materialization.")
    source_buffers = dict(source_model.named_buffers())
    target_buffers = dict(target_model.named_buffers())
    if set(source_buffers) != set(target_buffers) or any(
        not torch.equal(source_buffers[name].detach().cpu(), target_buffers[name].detach().cpu())
        for name in source_buffers
    ):
        raise RuntimeError("Deep-copied source and target buffers differ after materialization.")

    native_scope = _configure_native_nonlinear_interface_scope(target_model)
    trainable_names = set(native_scope["trainable_parameter_names"])
    source_parameters = dict(source_model.named_parameters())
    target_parameters = dict(target_model.named_parameters())
    if set(source_parameters) != set(target_parameters):
        raise RuntimeError("Source and target parameter inventories differ after materialization.")
    for arm, payload in arm_payloads.items():
        target_model.load_state_dict(payload["model"], strict=True)
        for name, parameter in target_model.named_parameters():
            if name not in trainable_names and not torch.equal(
                parameter.detach().cpu(), source_parameters[name].detach().cpu()
            ):
                raise RuntimeError(f"{arm} changed frozen source parameter {name!r}.")
        loaded_buffers = dict(target_model.named_buffers())
        if set(loaded_buffers) != set(source_buffers) or any(
            not torch.equal(loaded_buffers[name].detach().cpu(), source_buffers[name].detach().cpu())
            for name in source_buffers
        ):
            raise RuntimeError(f"{arm} changed a frozen or lazy materialized source buffer.")

    sampled_panel = sample_training_panel(train_stencils, config=sampling)
    fixed_heat_controls, fixed_heat_provenance, fixed_heat_review_stencils = (
        _load_fixed_heat_null_controls(
            recipe_payload,
            train_stencils,
            sampled_panel,
            stencil_paths=stencil_paths,
        )
    )
    del fixed_heat_controls
    if len(fixed_heat_review_stencils) != 4:
        raise ValueError("Checkpoint-only review requires four strict raw fixed-heat stencils.")

    panel_stencils: dict[str, list[tuple[str, ResponseStencil, float]]] = {
        "train_eight_families": [],
        "re90_four_development": [],
        "fixed_heat_four_controls": [],
    }
    for stencil in train_stencils:
        panel_stencils["train_eight_families"].append((
            stencil.physical_family_id,
            stencil,
            family_pressure_limits[stencil.physical_family_id],
        ))
    for _, stencil, _ in development_loaded:
        pressure = stencil.baseline.output.quantities["pressure_drop"]  # type: ignore[union-attr]
        if not pressure.resolved or not np.isfinite(float(pressure.value)):
            raise ValueError("Re90 review requires its resolved original baseline pressure.")
        panel_stencils["re90_four_development"].append((
            stencil.physical_family_id,
            stencil,
            float(pressure.value) * 1.05,
        ))
    for control_id, stencil in fixed_heat_review_stencils:
        panel_stencils["fixed_heat_four_controls"].append((
            control_id,
            stencil,
            family_pressure_limits[stencil.physical_family_id],
        ))
    broad_records = [
        (row, load_stored_reference_case(dataset_root, str(row["case_id"])))
        for row in broad_eval_cases
    ]
    if len(broad_records) != 30 or len({row["case_id"] for row, _ in broad_records}) != 30:
        raise ValueError("The broad train review panel must contain 30 unique stored cases.")

    forward_counts: dict[str, dict[str, int]] = {
        model_name: {} for model_name in ("incumbent_e4738", "R_value", "R_response")
    }
    active_forward_model = {"name": "setup"}

    def _count_forward(_module: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        name = str(active_forward_model["name"])
        if name not in forward_counts:
            return
        forward_counts[name]["native_model_forward_calls"] = (
            forward_counts[name].get("native_model_forward_calls", 0) + 1
        )

    source_hook = source_model.register_forward_hook(_count_forward)
    target_hook = target_model.register_forward_hook(_count_forward)
    progress_path = output_dir / "checkpoint_review_progress.json"
    result_path = output_dir / "checkpoint_review_result.json"
    review_models: dict[str, dict[str, list[dict[str, Any]]]] = {}
    result: dict[str, Any] = {
        **provenance,
        "status": "running",
        "optimizer_calls": 0,
        "optimizer_instances_created": 0,
        "reference_solver_calls": 0,
        "optimizer_accounting_basis": "checkpoint_review code constructs no optimizer and invokes no training/update function",
        "setup_native_forward_call_counts": setup_forward_counts,
        "deterministic_algorithms_enabled": bool(recipe_payload.get("deterministic_algorithms", False)),
        "cublas_workspace_config": cublas_workspace,
        "source_checkpoint_identity": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _checkpoint_digest(checkpoint_path),
        },
        "native_initialization": refit_config,
        "native_scope": native_scope,
        "materialized_initial_state_equal": True,
        "frozen_parameters_and_buffers_equal_source": True,
        "projection_evidence": (
            None if projection_evidence is None else {
                "justified_by_measured_m10_combined_value_conflict": bool(
                    projection_evidence["justified_by_measured_m10_combined_value_conflict"]
                ),
                "r0_content_hash_attested_by_source_checkpoint": False,
            }
        ),
        "training_config": config,
        "loss_scales": frozen_scales,
        "checkpoint_calibrated_loss_weights": {
            arm: dict(arm_payloads[arm]["calibrated_loss_weights"])
            for arm in ("R_value", "R_response")
        },
        "fixed_heat_control_provenance": fixed_heat_provenance,
        "panel_counts": {
            "train_eight_families": 8,
            "re90_four_development": 4,
            "fixed_heat_four_controls": 4,
            "broad_train_historical_30": 30,
        },
        "development_panel_attestation": provenance["development_panel_attestation"],
        "temperature_units": "Each metric row carries the source role's declared channel_units; no SI relabeling is inferred.",
        "models": review_models,
        "native_forward_call_counts": forward_counts,
        "started_unix_seconds": time.time(),
    }
    _atomic_json(progress_path, result)

    def save_progress() -> None:
        result["native_forward_call_counts"] = {
            name: dict(counts) for name, counts in forward_counts.items()
        }
        result["elapsed_wall_seconds"] = time.monotonic() - started
        _atomic_json(progress_path, result)

    try:
        for model_name in ("incumbent_e4738", "R_value", "R_response"):
            if time.monotonic() - started >= max_wall_seconds:
                raise TimeoutError("Checkpoint-only review reached its internal wall cap.")
            if model_name == "incumbent_e4738":
                review_operator = source_operator
                active_forward_model["name"] = model_name
                source_model.eval()
            else:
                target_model.load_state_dict(arm_payloads[model_name]["model"], strict=True)
                target_model.eval()
                review_operator = target_operator
                active_forward_model["name"] = model_name
            model_panels: dict[str, list[dict[str, Any]]] = {}
            for panel_name, items in panel_stencils.items():
                panel_rows: list[dict[str, Any]] = []
                for case_id, stencil, pressure_limit in items:
                    if time.monotonic() - started >= max_wall_seconds:
                        raise TimeoutError("Checkpoint-only review reached its internal wall cap.")
                    with torch.no_grad():
                        metrics = evaluate_stencil(
                            review_operator,
                            stencil,
                            pressure_limit={stencil.physical_family_id: pressure_limit},
                            mixed_specs=_mixed_specs(stencil),
                            smooth_peak_beta=smooth_peak_beta,
                            device=device,
                        )
                    panel_rows.append({
                        "case_id": case_id,
                        "family_id": stencil.physical_family_id,
                        "split": stencil.split.value,
                        "source": stencil.source.value,
                        "pressure_limit_original": pressure_limit,
                        "metrics": metrics,
                    })
                    model_panels[panel_name] = list(panel_rows)
                    review_models[model_name] = model_panels
                    result["models"] = review_models
                    save_progress()

            broad_rows: list[dict[str, Any]] = []
            for row, record in broad_records:
                if time.monotonic() - started >= max_wall_seconds:
                    raise TimeoutError("Checkpoint-only review reached its internal wall cap.")
                pressure = record.output.quantities["pressure_drop"]  # type: ignore[union-attr]
                if not pressure.resolved or not np.isfinite(float(pressure.value)):
                    raise ValueError(f"Broad case {row['case_id']!r} has unresolved original pressure.")
                fixed_limit = float(pressure.value) * 1.05
                with torch.no_grad():
                    metrics = evaluate_absolute_record(
                        review_operator,
                        record,
                        pressure_limit=fixed_limit,
                        device=device,
                    )
                broad_rows.append({
                    **dict(row),
                    "split": record.design.split.value,
                    "source": record.source.value,
                    "pressure_limit_original": fixed_limit,
                    "pressure_limit_rule": "1.05x each stored original train-record pressure baseline",
                    "metrics": metrics,
                })
                model_panels["broad_train_historical_30"] = list(broad_rows)
                review_models[model_name] = model_panels
                result["models"] = review_models
                save_progress()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result.update({
            "status": "passed",
            "finished_unix_seconds": time.time(),
            "total_wall_seconds": time.monotonic() - started,
            "native_forward_call_counts": {
                name: dict(counts) for name, counts in forward_counts.items()
            },
        })
        _atomic_json(result_path, result)
        _atomic_json(progress_path, result)
        return result
    except Exception as exc:
        result.update({
            "status": "timed_out" if isinstance(exc, TimeoutError) else "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "finished_unix_seconds": time.time(),
            "total_wall_seconds": time.monotonic() - started,
            "native_forward_call_counts": {
                name: dict(counts) for name, counts in forward_counts.items()
            },
        })
        _atomic_json(progress_path, result)
        raise
    finally:
        source_hook.remove()
        target_hook.remove()


def _cuda_evidence(
    device: torch.device,
    *,
    free_bytes_before_fit: int | None = None,
    preflight_peak_allocated_bytes: int | None = None,
    preflight_peak_reserved_bytes: int | None = None,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"available": False}
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "available": True,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device),
        "name": properties.name,
        "uuid": str(getattr(properties, "uuid", "unavailable")),
        "free_bytes_before_fit": (
            int(free_bytes_before_fit) if free_bytes_before_fit is not None else None
        ),
        "free_bytes_after_run": int(free_bytes),
        "total_bytes": int(total_bytes),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "preflight_peak_allocated_bytes": preflight_peak_allocated_bytes,
        "preflight_peak_reserved_bytes": preflight_peak_reserved_bytes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preflight", "r0", "expanded_probe", "paired", "checkpoint_review", "train_replay"),
        default="preflight",
    )
    parser.add_argument(
        "--initialization-mode",
        choices=("native_checkpoint", "three_term_conversion"),
        default="three_term_conversion",
        help="Use the intact selected checkpoint or the preserved historical full-access conversion.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-stencil", required=True, type=Path, action="append")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--recipe-config", type=Path, default=None)
    parser.add_argument("--r0-diagnostic-json", type=Path, default=None)
    parser.add_argument("--fit-capability-probe-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-wall-seconds", type=float, default=1800.0)
    parser.add_argument("--max-fluid-queries", type=int, default=3072)
    parser.add_argument("--solid-queries-per-module", type=int, default=128)
    parser.add_argument("--hot-solid-points-per-module", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=2317)
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--smooth-peak-beta", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--review-cap", type=int, choices=(100, 200, 300, 500, 600, 1000, 2000), default=100)
    parser.add_argument("--continue-after", type=int, choices=(100, 200, 300, 500, 600, 1000), action="append", default=[])
    parser.add_argument("--development-stencil", type=Path, action="append", default=[])
    parser.add_argument("--resume-value", "--resume-b-value", dest="resume_b_value", type=Path, default=None)
    parser.add_argument("--resume-response", "--resume-b-response", dest="resume_b_response", type=Path, default=None)
    parser.add_argument("--resume-fit-manifest-json", type=Path, default=None)
    parser.add_argument("--resume-replay-manifest-json", type=Path, default=None)
    parser.add_argument("--frozen-loss-scales-json", type=Path, default=None)
    parser.add_argument("--frozen-response-weights-json", type=Path, default=None)
    parser.add_argument("--review-source-manifest", type=Path, default=None)
    parser.add_argument("--review-value-checkpoint", type=Path, default=None)
    parser.add_argument("--review-response-checkpoint", type=Path, default=None)
    parser.add_argument("--review-update", type=int, choices=(200,), default=200)
    parser.add_argument("--replay-fit-manifest-json", type=Path, default=None)
    parser.add_argument("--replay-value-checkpoint", type=Path, default=None)
    parser.add_argument("--replay-response-checkpoint", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    path_parts = output_dir.parts
    if not any(
        path_parts[index] == "diagnostics" and index + 1 < len(path_parts)
        and path_parts[index + 1] == "generated"
        for index in range(len(path_parts) - 1)
    ):
        raise ValueError("Run outputs must be placed under an ignored diagnostics/generated directory.")
    if args.mode in {"checkpoint_review", "train_replay"} and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Read-only review modes require a fresh empty ignored output directory.")
    replay_only_args = (
        args.replay_fit_manifest_json,
        args.replay_value_checkpoint,
        args.replay_response_checkpoint,
    )
    if args.mode != "train_replay" and any(value is not None for value in replay_only_args):
        raise ValueError("Train replay source arguments apply only to --mode train_replay.")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / (
        "read_only_full_grid_train_replay_manifest.json"
        if args.mode == "train_replay"
        else "response_control_manifest.json"
    )
    manifest: dict[str, Any] = {
        "status": "running",
        "mode": args.mode,
        "initialization_mode": args.initialization_mode,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "train_stencils": [str(path.expanduser().resolve()) for path in args.train_stencil],
        "development_stencils": [
            str(path.expanduser().resolve()) for path in args.development_stencil
        ],
        "dataset": None if args.dataset is None else str(args.dataset.expanduser().resolve()),
        "device": args.device,
        "max_wall_seconds": args.max_wall_seconds,
        "review_cap": args.review_cap,
        "random_seed": args.random_seed,
        "recipe_config": (
            None if args.recipe_config is None
            else str(args.recipe_config.expanduser().resolve())
        ),
        "r0_diagnostic_json": (
            None if args.r0_diagnostic_json is None
            else str(args.r0_diagnostic_json.expanduser().resolve())
        ),
        "fit_capability_probe_json": (
            None if args.fit_capability_probe_json is None
            else str(args.fit_capability_probe_json.expanduser().resolve())
        ),
        "frozen_loss_scales_json": (
            None
            if args.frozen_loss_scales_json is None
            else str(args.frozen_loss_scales_json.expanduser().resolve())
        ),
        "frozen_response_weights_json": (
            None
            if args.frozen_response_weights_json is None
            else str(args.frozen_response_weights_json.expanduser().resolve())
        ),
        "review_source_manifest": (
            None if args.review_source_manifest is None
            else str(args.review_source_manifest.expanduser().resolve())
        ),
        "review_value_checkpoint": (
            None if args.review_value_checkpoint is None
            else str(args.review_value_checkpoint.expanduser().resolve())
        ),
        "review_response_checkpoint": (
            None if args.review_response_checkpoint is None
            else str(args.review_response_checkpoint.expanduser().resolve())
        ),
        "replay_fit_manifest_json": (
            None if args.replay_fit_manifest_json is None
            else str(args.replay_fit_manifest_json.expanduser().resolve())
        ),
        "replay_value_checkpoint": (
            None if args.replay_value_checkpoint is None
            else str(args.replay_value_checkpoint.expanduser().resolve())
        ),
        "replay_response_checkpoint": (
            None if args.replay_response_checkpoint is None
            else str(args.replay_response_checkpoint.expanduser().resolve())
        ),
        "review_update": args.review_update,
        "resume_fit_manifest_json": (
            None
            if args.resume_fit_manifest_json is None
            else str(args.resume_fit_manifest_json.expanduser().resolve())
        ),
        "resume_replay_manifest_json": (
            None
            if args.resume_replay_manifest_json is None
            else str(args.resume_replay_manifest_json.expanduser().resolve())
        ),
        "started_unix_seconds": time.time(),
    }
    _atomic_json(manifest_path, manifest)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def timeout_handler(signum: int, _frame: Any) -> None:
        raise TimeoutError(f"Received signal {signum} before the bounded run completed.")

    signal.signal(signal.SIGTERM, timeout_handler)
    try:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        for stencil_path in args.train_stencil + args.development_stencil:
            if not stencil_path.is_file():
                raise FileNotFoundError(stencil_path)
        if args.max_wall_seconds <= 0.0:
            raise ValueError("max_wall_seconds must be positive.")
        device = torch.device(args.device)
        if device.type == "cuda" and (not torch.cuda.is_available() or device.index not in {None, 0}):
            raise RuntimeError("Use CUDA_VISIBLE_DEVICES=2 with logical --device cuda:0 for the allocated GPU.")
        sampling = ReceiverSamplingConfig(
            max_fluid_queries=args.max_fluid_queries,
            solid_queries_per_module=args.solid_queries_per_module,
            hot_solid_points_per_module=args.hot_solid_points_per_module,
            random_seed=args.random_seed,
        )
        if args.mode == "checkpoint_review":
            required_paths = {
                "source failed manifest": args.review_source_manifest,
                "R_value u200 checkpoint": args.review_value_checkpoint,
                "R_response u200 checkpoint": args.review_response_checkpoint,
                "named recipe": args.recipe_config,
                "R0 diagnostic": args.r0_diagnostic_json,
                "frozen scales": args.frozen_loss_scales_json,
            }
            missing = [name for name, path in required_paths.items() if path is None]
            if missing:
                raise ValueError(f"Checkpoint-only review is missing required inputs: {missing}.")
            if args.initialization_mode != "native_checkpoint":
                raise ValueError("Checkpoint-only R1 review requires native_checkpoint initialization.")
            if len(args.train_stencil) != 8 or len(args.development_stencil) != 4:
                raise ValueError("Checkpoint-only R1 review requires exactly eight train and four Re90 stencils.")
            if args.review_update != 200:
                raise ValueError("Checkpoint-only review is restricted to the saved u200 gate.")
            forbidden = (
                args.resume_b_value,
                args.resume_b_response,
                args.resume_fit_manifest_json,
                args.resume_replay_manifest_json,
                args.frozen_response_weights_json,
                args.learning_rate,
                args.weight_decay,
                bool(args.continue_after),
            )
            if any(value is not None and value is not False for value in forbidden):
                raise ValueError("Checkpoint-only review does not accept resume, optimizer, or continuation controls.")
            for name, path in required_paths.items():
                assert path is not None
                if not path.is_file():
                    raise FileNotFoundError(f"{name} is missing: {path}")
            manifest["planned_optimizer_updates"] = 0
            manifest["optimizer_calls"] = 0
            manifest["optimizer_instances_created"] = 0
            manifest["reference_solver_calls"] = 0
            _atomic_json(manifest_path, manifest)
            result = run_checkpoint_only_review(
                checkpoint_path=args.checkpoint.expanduser().resolve(),
                source_manifest_path=args.review_source_manifest.expanduser().resolve(),  # type: ignore[union-attr]
                value_checkpoint_path=args.review_value_checkpoint.expanduser().resolve(),  # type: ignore[union-attr]
                response_checkpoint_path=args.review_response_checkpoint.expanduser().resolve(),  # type: ignore[union-attr]
                stencil_paths=[path.expanduser().resolve() for path in args.train_stencil],
                development_paths=[path.expanduser().resolve() for path in args.development_stencil],
                dataset_path=args.dataset,
                recipe_config_path=args.recipe_config.expanduser().resolve(),  # type: ignore[union-attr]
                r0_diagnostic_path=args.r0_diagnostic_json.expanduser().resolve(),  # type: ignore[union-attr]
                frozen_loss_scales_path=args.frozen_loss_scales_json.expanduser().resolve(),  # type: ignore[union-attr]
                output_dir=output_dir,
                device=device,
                sampling=sampling,
                max_wall_seconds=args.max_wall_seconds,
                query_batch_size=args.query_batch_size,
            )
        elif args.mode == "train_replay":
            if args.initialization_mode != "native_checkpoint":
                raise ValueError("Read-only expanded train replay requires native_checkpoint initialization.")
            required_paths = {
                "named expanded recipe": args.recipe_config,
                "passed paired-fit manifest": args.replay_fit_manifest_json,
                "R_value gate checkpoint": args.replay_value_checkpoint,
                "R_response gate checkpoint": args.replay_response_checkpoint,
            }
            missing = [name for name, path in required_paths.items() if path is None]
            if missing:
                raise ValueError(f"Read-only train replay is missing required paths: {missing}.")
            if args.development_stencil:
                raise ValueError("Read-only full-grid train replay rejects development stencils.")
            forbidden = (
                args.r0_diagnostic_json,
                args.fit_capability_probe_json,
                args.learning_rate,
                args.weight_decay,
                args.resume_b_value,
                args.resume_b_response,
                args.resume_fit_manifest_json,
                args.resume_replay_manifest_json,
                args.frozen_loss_scales_json,
                args.frozen_response_weights_json,
                args.review_source_manifest,
                args.review_value_checkpoint,
                args.review_response_checkpoint,
                bool(args.continue_after),
            )
            if any(value is not None and value is not False for value in forbidden):
                raise ValueError("Read-only train replay does not accept optimizer, fit, or review overrides.")
            result = run_read_only_full_grid_train_replay(
                checkpoint_path=args.checkpoint.expanduser().resolve(),
                stencil_paths=[path.expanduser().resolve() for path in args.train_stencil],
                dataset_path=args.dataset,
                recipe_config_path=args.recipe_config.expanduser().resolve(),  # type: ignore[union-attr]
                fit_manifest_path=args.replay_fit_manifest_json.expanduser().resolve(),  # type: ignore[union-attr]
                value_checkpoint_path=args.replay_value_checkpoint.expanduser().resolve(),  # type: ignore[union-attr]
                response_checkpoint_path=args.replay_response_checkpoint.expanduser().resolve(),  # type: ignore[union-attr]
                output_dir=output_dir,
                device=device,
                query_batch_size=args.query_batch_size,
                max_wall_seconds=args.max_wall_seconds,
            )
        elif args.mode == "preflight":
            if args.recipe_config is not None:
                raise ValueError("A named staged recipe applies only to paired-fit mode.")
            if any(
                value is not None
                for value in (
                    args.frozen_loss_scales_json,
                    args.frozen_response_weights_json,
                    args.resume_b_value,
                    args.resume_b_response,
                    args.resume_fit_manifest_json,
                    args.resume_replay_manifest_json,
                    args.r0_diagnostic_json,
                    args.fit_capability_probe_json,
                )
            ):
                raise ValueError("Frozen calibration and resume overrides apply only to paired-fit mode.")
            if len(args.train_stencil) != 1:
                raise ValueError("The first launch is restricted to exactly one train stencil and one update.")
            result = run_one_update_preflight(
                checkpoint_path=args.checkpoint.expanduser().resolve(),
                stencil_path=args.train_stencil[0].expanduser().resolve(),
                dataset_path=args.dataset,
                output_dir=output_dir,
                device=device,
                sampling=sampling,
                max_wall_seconds=args.max_wall_seconds,
                initialization_mode=args.initialization_mode,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                smooth_peak_beta=args.smooth_peak_beta,
                query_batch_size=args.query_batch_size,
            )
        elif args.mode == "r0":
            if args.development_stencil:
                raise ValueError("R0 is train-only and does not accept development stencil paths.")
            if any(
                value is not None
                for value in (
                    args.recipe_config,
                    args.r0_diagnostic_json,
                    args.frozen_loss_scales_json,
                    args.frozen_response_weights_json,
                    args.resume_b_value,
                    args.resume_b_response,
                    args.resume_fit_manifest_json,
                    args.resume_replay_manifest_json,
                    args.fit_capability_probe_json,
                )
            ):
                raise ValueError("R0 does not accept paired-fit recipes, frozen calibration, or resume overrides.")
            if len(args.train_stencil) != 4:
                raise ValueError("R0 requires exactly four train stencils spanning M=3,5,7,10.")
            manifest["planned_optimizer_updates_max"] = 180
            manifest["reference_solver_calls"] = 0
            _atomic_json(manifest_path, manifest)
            result = run_train_only_scope_diagnostic(
                checkpoint_path=args.checkpoint.expanduser().resolve(),
                stencil_paths=[path.expanduser().resolve() for path in args.train_stencil],
                dataset_path=args.dataset,
                output_dir=output_dir,
                device=device,
                sampling=sampling,
                query_batch_size=args.query_batch_size,
                max_wall_seconds=args.max_wall_seconds,
            )
        elif args.mode == "expanded_probe":
            if args.initialization_mode != "native_checkpoint":
                raise ValueError("The expanded fit-capability probe requires native_checkpoint initialization.")
            if args.recipe_config is None or not args.recipe_config.is_file():
                raise ValueError("The expanded fit-capability probe requires its checked-in named recipe.")
            if args.development_stencil:
                raise ValueError("The expanded fit-capability probe is train-only and rejects development stencils.")
            if len(args.train_stencil) != 8:
                raise ValueError("The expanded fit-capability probe requires exactly eight train stencils.")
            forbidden = (
                args.r0_diagnostic_json,
                args.fit_capability_probe_json,
                args.learning_rate,
                args.weight_decay,
                args.resume_b_value,
                args.resume_b_response,
                args.resume_fit_manifest_json,
                args.resume_replay_manifest_json,
                args.frozen_loss_scales_json,
                args.frozen_response_weights_json,
                args.review_source_manifest,
                args.review_value_checkpoint,
                args.review_response_checkpoint,
                bool(args.continue_after),
            )
            if any(value is not None and value is not False for value in forbidden):
                raise ValueError("The expanded fit-capability probe does not accept paired/resume overrides.")
            manifest["planned_optimizer_updates"] = 80
            manifest["shared_remedy_optimizer_call_ceiling"] = 600
            manifest["reference_solver_calls"] = 0
            _atomic_json(manifest_path, manifest)
            result = run_expanded_response_fit_capability_probe(
                checkpoint_path=args.checkpoint.expanduser().resolve(),
                stencil_paths=[path.expanduser().resolve() for path in args.train_stencil],
                dataset_path=args.dataset,
                recipe_config_path=args.recipe_config.expanduser().resolve(),
                output_dir=output_dir,
                device=device,
                sampling=sampling,
                query_batch_size=args.query_batch_size,
                max_wall_seconds=args.max_wall_seconds,
                smooth_peak_beta=args.smooth_peak_beta,
            )
        else:
            resuming = args.resume_b_value is not None or args.resume_b_response is not None
            if (args.resume_b_value is None) != (args.resume_b_response is None):
                raise ValueError("Resume requires both paired arm checkpoints.")
            if resuming:
                if args.frozen_loss_scales_json is None or args.frozen_response_weights_json is not None:
                    raise ValueError(
                        "A resume requires --frozen-loss-scales-json and takes response weights from the checkpoints."
                    )
                if args.resume_fit_manifest_json is None or args.resume_replay_manifest_json is None:
                    raise ValueError("A resume requires both source provenance manifest paths.")
            else:
                if (args.frozen_loss_scales_json is None) != (args.frozen_response_weights_json is None):
                    raise ValueError("Supply both frozen scale and response-weight JSON files, or neither.")
                if args.resume_fit_manifest_json is not None or args.resume_replay_manifest_json is not None:
                    raise ValueError("Resume manifests require both --resume-b-value and --resume-b-response.")
            if args.resume_b_value is not None and not args.resume_b_value.is_file():
                raise FileNotFoundError(args.resume_b_value)
            if args.resume_b_response is not None and not args.resume_b_response.is_file():
                raise FileNotFoundError(args.resume_b_response)
            if args.frozen_loss_scales_json is not None and not args.frozen_loss_scales_json.is_file():
                raise FileNotFoundError(args.frozen_loss_scales_json)
            if args.frozen_response_weights_json is not None and not args.frozen_response_weights_json.is_file():
                raise FileNotFoundError(args.frozen_response_weights_json)
            if args.resume_fit_manifest_json is not None and not args.resume_fit_manifest_json.is_file():
                raise FileNotFoundError(args.resume_fit_manifest_json)
            if args.resume_replay_manifest_json is not None and not args.resume_replay_manifest_json.is_file():
                raise FileNotFoundError(args.resume_replay_manifest_json)
            if args.r0_diagnostic_json is not None and not args.r0_diagnostic_json.is_file():
                raise FileNotFoundError(args.r0_diagnostic_json)
            result = run_paired_fit(
                checkpoint_path=args.checkpoint.expanduser().resolve(),
                stencil_paths=[path.expanduser().resolve() for path in args.train_stencil],
                development_paths=[path.expanduser().resolve() for path in args.development_stencil],
                dataset_path=args.dataset,
                output_dir=output_dir,
                device=device,
                sampling=sampling,
                max_wall_seconds=args.max_wall_seconds,
                initialization_mode=args.initialization_mode,
                recipe_config_path=args.recipe_config,
                r0_diagnostic_path=args.r0_diagnostic_json,
                fit_capability_probe_path=args.fit_capability_probe_json,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                review_cap=args.review_cap,
                review_continuations=args.continue_after,
                smooth_peak_beta=args.smooth_peak_beta,
                query_batch_size=args.query_batch_size,
                resume_value_path=args.resume_b_value,
                resume_response_path=args.resume_b_response,
                frozen_loss_scales_path=args.frozen_loss_scales_json,
                frozen_response_weights_path=args.frozen_response_weights_json,
                resume_fit_manifest_path=args.resume_fit_manifest_json,
                resume_replay_manifest_path=args.resume_replay_manifest_json,
            )
        manifest.update(result)
        manifest["finished_unix_seconds"] = time.time()
        _atomic_json(manifest_path, manifest)
        return 0
    except Exception as exc:
        if args.mode == "r0":
            r0_path = output_dir / "r0_train_only_scope_diagnostic.json"
            if r0_path.is_file():
                r0_payload = json.loads(r0_path.read_text(encoding="utf-8"))
                timed_out = isinstance(exc, TimeoutError)
                trial_status = r0_payload.get("optimizer_trial_status", [])
                for trial in trial_status:
                    if trial.get("status") == "running":
                        trial["status"] = "timed_out" if timed_out else "failed"
                r0_payload.update({
                    "status": (
                        r0_payload.get("status")
                        if r0_payload.get("status") in {"timed_out", "timed_out_or_partial"}
                        else "timed_out" if timed_out else "failed"
                    ),
                    "optimizer_trial_status": trial_status,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "finished_unix_seconds": time.time(),
                    "reference_solver_calls": 0,
                })
                _atomic_json(r0_path, r0_payload)
        if args.mode == "expanded_probe":
            probe_path = output_dir / "expanded_response_fit_capability_probe.json"
            if probe_path.is_file():
                probe_payload = json.loads(probe_path.read_text(encoding="utf-8"))
                timed_out = isinstance(exc, TimeoutError)
                attempted = int(probe_payload.get("optimizer_updates_attempted", 0))
                probe_payload.update({
                    "status": (
                        "timed_out_or_partial" if timed_out and attempted
                        else "timed_out" if timed_out
                        else "failed"
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "finished_unix_seconds": time.time(),
                    "reference_solver_calls": 0,
                    "optimizer_calls_charged_to_shared_remedy_ledger": attempted,
                })
                _atomic_json(probe_path, probe_payload)
                manifest["optimizer_updates_attempted"] = attempted
                manifest["optimizer_updates_completed"] = int(
                    probe_payload.get("optimizer_updates_completed", 0)
                )
                manifest["optimizer_calls_charged_to_shared_remedy_ledger"] = attempted
        manifest.update(
            {
                "status": "timed_out" if isinstance(exc, TimeoutError) else "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "total_wall_seconds": time.time() - float(manifest["started_unix_seconds"]),
                "finished_unix_seconds": time.time(),
            }
        )
        _atomic_json(manifest_path, manifest)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI
    raise SystemExit(main())
