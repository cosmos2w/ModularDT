"""Fixed25_v1 Thermal adapter for the common interaction-refinement engine.

Thermal keeps its affine heat law and native shared-grid role extraction.
Geometry/prescribed context and receiver coordinates form the inference path;
heating is applied separately. Stored TRAIN velocity is exposed only to the
qualified discrete-operator residual supervision term, never to the scene,
context, approximation, router, or inference preparation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from honf_forward_core.interface_fields.interaction_refinement import (
    RefinedSourceResponseOperator,
)
from honf_runtime.unified_training import (
    LossTerm,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    TaskBatch,
    restore_rng_state,
)

from ..source_response import CONTEXT_KEYS, ThermalSourceResponse

DEFAULT_PARENT_CHECKPOINT = Path(
    "/data/wanglz/ModularDT/thermal_development/response_operator_20261006/thermal_pair/R-direct/epoch_2500_model.pt"
)
DEFAULT_PARENT_SHA256 = "05974d2fbc367bad8f2092063818ce9073c21783131753b3648fb3ea11a870aa"
DEFAULT_FLOW_CHECKPOINT = Path(
    "/data/wanglz/ModularDT/thermal_development/response_operator_20261006/D-sep2500/epoch_2500_model.pt"
)
DEFAULT_FLOW_SHA256 = "914fe0b4e07c2b805a4f1df1e0d1e54a53180165d61349278acff05c335abdda"
DEFAULT_ATLAS_DIRECTORY = Path(
    __file__
).resolve().parents[4] / "diagnostics/generated/interactions/physical_response_atlas_20260926/families"
FIXED25_FINGERPRINT = "933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044"
TRAIN_RESPONSE_IDS = ("0001", "0318", "0333", "0348")
DEVELOPMENT_RESPONSE_IDS = ("0304", "0320", "0335", "0350")
NEAR_RESPONSE_IDS = ("0277", "0291", "0294", "0687")
Q_PROXY_COEFFICIENT = 0.05
BASE_APPROXIMATION_COEFFICIENT = 0.1
ROUTER_IMPORTANCE_COEFFICIENT = 0.01
EXPECTED_WORK_TARGET_SHARE = 0.05
EXPECTED_WORK_WEIGHT_CAP = 0.1
RESPONSE_SURFACE_STRIDE = 4
OPERATOR_ROWS_PER_CASE = 128
THERMAL_GATE_HARD_VERSION = "hard_v1"
THERMAL_GATE_COMPACT_C1_VERSION = "compact_c1_v1"
THERMAL_GATE_COMPACT_C1_TRANSITION = (0.35, 0.65)
THERMAL_TRANSFER_EPOCHS = 500
THERMAL_TRANSFER_LR = 3.0e-6
THERMAL_QUERY_BUDGET_OVERRIDE_KEYS = frozenset({"fluid_queries", "material_queries_per_module"})


def _resolve_query_budget(
    profile_budget: Mapping[str, Any], override: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, int] | None]:
    """Apply only native query-count overrides; physical and batch guards stay fixed."""
    budget = dict(profile_budget)
    if override is None:
        return budget, None
    if not isinstance(override, Mapping) or not override:
        raise ValueError("Thermal query-budget override must be a nonempty object.")
    keys = set(override)
    if keys - THERMAL_QUERY_BUDGET_OVERRIDE_KEYS:
        raise ValueError("Thermal query-budget overrides may change only fluid/material query counts.")
    resolved: dict[str, int] = {}
    for name, value in override.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"Thermal {name} override must be a positive integer.")
        resolved[name] = value
    budget.update(resolved)
    return budget, resolved


def _resolve_gate_binding(
    gate_version: str = THERMAL_GATE_HARD_VERSION,
    gate_transition: Sequence[float] | None = None,
) -> tuple[str, tuple[float, float]]:
    """Validate the explicit C1 option while preserving legacy hard defaults."""

    version = str(gate_version)
    if version == THERMAL_GATE_HARD_VERSION:
        if gate_transition is not None:
            raise ValueError("Thermal hard_v1 does not bind a compact gate transition.")
        return version, THERMAL_GATE_COMPACT_C1_TRANSITION
    if version != THERMAL_GATE_COMPACT_C1_VERSION:
        raise ValueError("Thermal gate version must be hard_v1 or compact_c1_v1.")
    if gate_transition is None:
        raise ValueError("Thermal compact_c1_v1 requires its explicit gate_transition interval.")
    transition = tuple(float(value) for value in gate_transition)
    if transition != THERMAL_GATE_COMPACT_C1_TRANSITION:
        raise ValueError("Thermal compact_c1_v1 is sealed to [0.35, 0.65].")
    return version, transition


@dataclass(frozen=True)
class ThermalSceneInputs:
    """Whitelisted prescribed geometry/context only."""

    structure: Mapping[str, Any]
    response_structure: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ThermalInferenceScene:
    structure: Mapping[str, torch.Tensor]
    response_structure: Mapping[str, torch.Tensor] | None


@dataclass(frozen=True)
class ThermalReceivers:
    fluid_xy: torch.Tensor
    local_query_points: torch.Tensor
    heat: torch.Tensor
    response_fluid_xy: torch.Tensor | None = None
    response_local_query_points: torch.Tensor | None = None
    response_heat: torch.Tensor | None = None
    native_solid_mask: torch.Tensor | None = None
    response_native_solid_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class ThermalResponseTargets:
    family_id: str
    fluid: torch.Tensor
    surface: torch.Tensor
    surface_mask: torch.Tensor
    material: torch.Tensor
    material_mask: torch.Tensor


@dataclass(frozen=True)
class ThermalTargets:
    case_ids: tuple[str, ...]
    case_indices: tuple[int, ...]
    epoch: int
    field_targets: torch.Tensor
    point_weights: torch.Tensor
    interface_target: torch.Tensor
    interface_valid_mask: torch.Tensor
    material_targets: torch.Tensor
    response: ThermalResponseTargets | None = None


@dataclass
class ThermalPredictions:
    model: ThermalSourceResponse
    execution_mode: str
    phase: str
    native_prepared: Any
    native_main: Mapping[str, torch.Tensor]
    native_full: Mapping[str, torch.Tensor] | None
    response_prepared: Any | None
    response_main: Mapping[str, torch.Tensor] | None
    response_full: Mapping[str, torch.Tensor] | None
    auxiliary: dict[str, Any]
    work: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_digest(named: Mapping[str, Any]) -> str:
    def update_nested(digest: Any, value: Any) -> None:
        """Hash optimizer-state trees without NumPy object-pointer bytes."""
        if isinstance(value, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                digest.update(type(key).__name__.encode("utf-8") + b":" + repr(key).encode("utf-8") + b"\0")
                update_nested(digest, value[key])
            return
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            return
        array = np.ascontiguousarray(value)
        digest.update(b"array\0")
        digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
        digest.update(array.tobytes())

    digest = hashlib.sha256()
    for name, value in sorted(named.items()):
        digest.update(name.encode("utf-8") + b"\0")
        if isinstance(value, Mapping):
            # Flat normalization-stat arrays retain their historical digest.
            # Only nested optimizer state uses the typed recursive encoding.
            update_nested(digest, value)
        elif torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _read_response_families(atlas_directory: Path, family_ids: Sequence[str]) -> tuple[dict[str, Any], ...]:
    """Read only the declared original-TRAIN atlas family IDs.

    This provider-owned filtered loader preserves the legacy fitter's default
    four-family behavior while allowing the already-existing development
    families to remain a separate, explicit validation cohort.
    """
    families = []
    for raw_id in family_ids:
        family_id = str(raw_id)
        path = atlas_directory / f"train_{family_id}_responses.npz"
        with np.load(path, allow_pickle=False) as handle:
            atlas = {name: handle[name].copy() for name in handle.files}
        metadata = json.loads(str(atlas["family_metadata_json"]))
        expected_family = "duplicate_family:0001+0273" if family_id == "0001" else f"stored_family:{family_id}"
        if (metadata.get("source_dataset_split") != "train" or metadata.get("anchor_id") != family_id
                or metadata.get("physical_family_id") != expected_family):
            raise ValueError(f"Thermal response family {family_id} is not the declared original-TRAIN record.")
        labels = atlas["all_labels"].tolist()
        if "baseline" not in labels or "heat_transfer_plus" not in labels:
            raise ValueError(f"Thermal response family {family_id} lacks its fixed baseline/plus pair.")
        baseline, plus = labels.index("baseline"), labels.index("heat_transfer_plus")
        if not np.array_equal(atlas["module_centers_xy"][baseline], atlas["module_centers_xy"][plus]):
            raise ValueError("Thermal response supervision must hold physical layout fixed.")
        case = metadata["case_config"]
        centers = atlas["module_centers_xy"][baseline].astype(np.float32)
        count = len(centers)
        module_ids = [f"{family_id}:module:{slot}" for slot in range(count)]
        if atlas["active_module_ids"].tolist() != module_ids:
            raise ValueError("Thermal response source IDs changed from their original module slots.")
        for role in ("interface", "solid_temperature"):
            role_ids = atlas[f"{role}_receiver_module_ids"]
            if len(role_ids) % count or not all(np.all(block == module_id) for block, module_id in
                    zip(np.split(role_ids, count), module_ids, strict=True)):
                raise ValueError("Thermal response role rows do not preserve physical module IDs.")
        material = np.asarray([
            case["runtime"]["nu"], case["thermal"]["solid_alpha"], case["thermal"]["fluid_alpha"],
            case["thermal"]["solid_k"], case["thermal"]["fluid_k"], case["domain"]["module_radius"],
        ], dtype=np.float32)
        structure = {
            "module_centers": centers,
            "module_present": np.ones(count, dtype=np.float32),
            "module_source_ids": np.arange(count, dtype=np.int64),
            "material_params": material,
            "re": np.asarray([case["flow"]["re"]], dtype=np.float32),
            "u_in": np.asarray([case["flow"]["u_in"]], dtype=np.float32),
            "domain_length_x": np.asarray([case["domain"]["lx"]], dtype=np.float32),
            "domain_length_y": np.asarray([case["domain"]["ly"]], dtype=np.float32),
        }
        deltas = {
            role: atlas[f"{role}_values"][plus].astype(np.float64)
            - atlas[f"{role}_values"][baseline].astype(np.float64)
            for role in ("fluid_fields", "interface", "solid_temperature")
        }
        local_flat = atlas["solid_temperature_query_features"]
        if count < 1 or len(local_flat) % count:
            raise ValueError("Thermal material response rows do not divide evenly by active modules.")
        local_count = len(local_flat) // count
        local = local_flat[:local_count, :2]
        if not all(np.array_equal(local, local_flat[slot * local_count:(slot + 1) * local_count, :2])
                   for slot in range(count)):
            raise ValueError("Thermal material response coordinates lost their per-module local join.")
        families.append({
            "family_id": family_id,
            "source": str(path),
            "source_sha256": _sha256(path),
            "structure": structure,
            "heat_increment": (atlas["heating"][plus].astype(np.float64)
                               - atlas["heating"][baseline].astype(np.float64)).astype(np.float32),
            "fluid_xy": atlas["fluid_fields_query_features"][:, :2].astype(np.float32),
            "fluid_valid": atlas["heat_common_fluid_mask"].astype(bool),
            "material_local": local.astype(np.float32),
            "deltas": deltas,
            "module_count": count,
            "interface_valid": atlas["heat_common_interface_mask"].astype(bool),
            "material_valid": atlas["heat_common_solid_mask"].reshape(count, local_count).astype(bool),
        })
    if tuple(str(family["family_id"]) for family in families) != tuple(map(str, family_ids)):
        raise ValueError("Thermal response family order differs from the sealed explicit membership.")
    return tuple(families)


def _parent_optimizer_seed(parent_model: nn.Module, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping) or len(optimizer_state.get("param_groups", ())) != 1:
        raise ValueError("Retained R-direct parent must have one audited AdamW group.")
    saved_ids = list(optimizer_state["param_groups"][0]["params"])
    named = list(parent_model.named_parameters())
    if len(saved_ids) != len(named):
        raise ValueError("R-direct optimizer IDs do not align with the retained named fine parameters.")
    state = optimizer_state["state"]
    by_name = {
        name: copy.deepcopy(state[saved_id])
        for (name, _parameter), saved_id in zip(named, saved_ids, strict=True)
        if saved_id in state
    }
    if set(by_name) != {name for name, _parameter in named}:
        raise ValueError("Retained R-direct parent is missing an AdamW moment state for a fine parameter.")
    return {
        "state_by_name": by_name,
        "groups": [{"name": "thermal_fine", "parameter_names": [name for name, _ in named]}],
    }


def _recipe_optimizer_identity(
    parent: Mapping[str, Any], optimizer_seed: Mapping[str, Any],
) -> dict[str, Any]:
    group = parent["optimizer_state_dict"]["param_groups"][0]
    return {
        "group": {key: copy.deepcopy(group[key]) for key in ("lr", "betas", "eps", "weight_decay", "amsgrad")
                  if key in group},
        "state_by_name_sha256": _tensor_digest(optimizer_seed["state_by_name"]),
        "parameter_count": len(optimizer_seed["state_by_name"]),
    }


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_development_refinement_parent_binding(
    config: Mapping[str, Any],
) -> tuple[Path | None, str | None]:
    path_value = config.get("development_refinement_parent")
    sha256_value = config.get("development_refinement_parent_sha256")
    if (path_value is None) != (sha256_value is None):
        raise ValueError(
            "Thermal development_refinement_parent and its required SHA-256 must be supplied together."
        )
    if path_value is None:
        return None, None
    if not isinstance(sha256_value, str) or len(sha256_value) != 64:
        raise ValueError("Thermal development parent binding requires a full 64-character SHA-256.")
    try:
        bytes.fromhex(sha256_value)
    except ValueError as error:
        raise ValueError("Thermal development parent SHA-256 must contain only hexadecimal digits.") from error
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("The explicitly bound Thermal development refinement parent does not exist.")
    return path, sha256_value


def _validate_development_refinement_parent_header(
    checkpoint: Mapping[str, Any], *, path: Path, expected_sha256: str,
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Reject formal, stale, wrong-arm, and wrong-task Thermal parent states."""

    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("Thermal development parent checkpoint SHA-256 does not match the required binding.")
    identity = checkpoint.get("experiment_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Thermal development parent lacks a sealed unified experiment identity mapping.")
    if (checkpoint.get("workflow") != "unified_interaction_refinement"
            or identity.get("workflow") != "unified_interaction_refinement"
            or identity.get("task") != "ThermalChannel"
            or identity.get("development_profile") != "fixed25_v1"
            or identity.get("run_id") != "thermal_adaptive_refine_20261007"
            or checkpoint.get("arm") != "adaptive_detail"
            or type(checkpoint.get("epoch")) is not int
            or checkpoint.get("epoch") != 2500
            or type(checkpoint.get("current_epoch")) is not int
            or checkpoint.get("current_epoch") != 2500):
        raise ValueError(
            "Thermal transfer parent must be the sealed fixed25_v1 epoch-2500 adaptive DEVELOPMENT checkpoint."
        )
    provider_identity = identity.get("provider_identity")
    if not isinstance(provider_identity, Mapping):
        raise TypeError("Thermal development parent lacks its provider/data/objective identity mapping.")
    if (provider_identity.get("task") != "ThermalChannel"
            or provider_identity.get("dataset_split") != "fixed25_v1"
            or provider_identity.get("manifest_fingerprint") != FIXED25_FINGERPRINT
            or provider_identity.get("gate_version", THERMAL_GATE_HARD_VERSION) != THERMAL_GATE_HARD_VERSION
            or provider_identity.get("gate_transition") is not None):
        raise ValueError("Thermal transfer parent provider binding is not the preserved fixed25 hard_v1 recipe.")
    if identity.get("engine_config", {}).get("total_epochs") != 2500:
        raise ValueError("Thermal transfer parent identity does not declare the 2500-epoch development horizon.")
    for key in ("model_state_dict", "optimizer_state_by_name", "provider_training_state", "rng_state"):
        if not isinstance(checkpoint.get(key), Mapping):
            raise TypeError(f"Thermal development parent is missing its exact {key} state mapping.")
    return dict(identity), provider_identity, checkpoint["provider_training_state"]


def _provider_identity_for_transfer_comparison(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the declared new-child lineage and gate-version fields."""

    result = copy.deepcopy(dict(value))
    for key in ("development_refinement_parent", "gate_version", "gate_transition"):
        result.pop(key, None)
    return result


def _validate_development_provider_identity(
    source: Mapping[str, Any], active: Mapping[str, Any],
) -> None:
    if (_provider_identity_for_transfer_comparison(source)
            != _provider_identity_for_transfer_comparison(active)):
        raise ValueError(
            "Thermal development parent provider/data/normalizer/response/objective binding differs from the active task."
        )


def _development_optimizer_seed(
    checkpoint: Mapping[str, Any], model: nn.Module, provider: ThermalRefinementTask,
) -> tuple[dict[str, Any], str]:
    """Validate and return all named AdamW state for the new age-1 child."""

    payload = checkpoint["optimizer_state_by_name"]
    state_by_name = payload.get("state_by_name")
    source_groups = payload.get("groups")
    specs = tuple(provider.optimizer_groups(model, "adaptive_detail", "warmup"))
    expected_groups = [{"name": spec.name, "parameter_names": list(spec.parameter_names)} for spec in specs]
    if source_groups != expected_groups:
        raise ValueError("Thermal development parent optimizer group membership differs from its child model.")
    if checkpoint.get("optimizer_group_names") != [spec.name for spec in specs]:
        raise ValueError("Thermal development parent optimizer group names/order are not the sealed fine/refinement pair.")
    if not isinstance(state_by_name, Mapping):
        raise TypeError("Thermal development parent named AdamW states must be a mapping.")
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if len(parameters) != 65 or set(state_by_name) != set(parameters):
        raise ValueError("Thermal development parent must provide all 65 trainable named AdamW states.")
    for name, parameter in parameters.items():
        state = state_by_name[name]
        if not isinstance(state, Mapping) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise ValueError(f"Thermal development parent moments are incomplete for {name!r}.")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = state[moment_name]
            if (not torch.is_tensor(moment) or tuple(moment.shape) != tuple(parameter.shape)
                    or not bool(torch.isfinite(moment).all())):
                raise ValueError(f"Thermal development parent {moment_name} is invalid for {name!r}.")
        step = state["step"]
        if torch.is_tensor(step):
            if step.numel() != 1 or not bool(torch.isfinite(step).all()) or float(step.item()) < 1:
                raise ValueError(f"Thermal development parent AdamW step is invalid for {name!r}.")
        elif not isinstance(step, (int, float)) or not math.isfinite(float(step)) or float(step) < 1:
            raise ValueError(f"Thermal development parent AdamW step is invalid for {name!r}.")
    optimizer_seed = copy.deepcopy(dict(payload))
    return optimizer_seed, _tensor_digest(state_by_name)


def _validate_inherited_provider_training_state(
    state: Mapping[str, Any], *, temperature_std: float, base_loss_weight: float,
) -> str:
    """Require the source-only work calibration and fixed loss scale to transfer intact."""

    calibration = state.get("expected_work_calibration")
    if not isinstance(calibration, Mapping):
        raise TypeError("Thermal development parent TRAIN-only expected-work calibration must be a mapping.")
    if (not math.isclose(float(state.get("temperature_std", float("nan"))), temperature_std,
                         rel_tol=0.0, abs_tol=0.0)
            or not math.isclose(float(state.get("base_loss_weight", float("nan"))), base_loss_weight,
                                rel_tol=0.0, abs_tol=0.0)
            or not math.isclose(float(state.get("expected_work_weight", float("nan"))),
                                float(calibration.get("coefficient", float("nan"))),
                                rel_tol=0.0, abs_tol=0.0)):
        raise ValueError("Thermal development parent provider training scales do not match its fixed TRAIN binding.")
    if (calibration.get("stage") != "thermal_expected_work_calibration"
            or calibration.get("absolute_epoch") != 601
            or calibration.get("arm") != "adaptive_detail"
            or calibration.get("validation_values_read") is not False
            or calibration.get("stored_uv_used_for_calibration") is not False):
        raise ValueError("Thermal development parent expected-work calibration is not the retained TRAIN-only receipt.")
    return _tensor_digest(state)


def _structure_only(structure: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: structure[key] for key in CONTEXT_KEYS if key in structure}
    required = {"module_centers", "module_present", "material_params"}
    if not required.issubset(result):
        raise ValueError("Thermal scene is missing prescribed geometry/material context.")
    forbidden = {"heat_powers", "steady_field", "field_targets", "interface_target", "module_internal_temperature_points"}
    if forbidden & set(result):
        raise ValueError("Thermal inference scene contains a control or stored target.")
    return result


def _move_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    return value


def _sample_primary(
    cases: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    epoch: int,
    key: SamplingKey | None,
    training: bool,
    budget: Mapping[str, Any],
    device: torch.device,
    validation_sampling_indices: Mapping[str, int] | None = None,
) -> tuple[Mapping[str, Any], str]:
    """Sample the maintained native roles from one provider-owned RNG key."""
    from channelthermal.data.collation import ChannelThermalBatchCollator

    samples = []
    sampling_hash = hashlib.sha256()
    for index in indices:
        original = cases[int(index)]
        case_id = str(original["case_id"])
        if training:
            if key is None:
                raise ValueError("Thermal TRAIN query sampling requires an engine-owned SamplingKey.")
            if key.epoch != int(epoch):
                raise ValueError("Thermal sample epoch differs from its engine-owned SamplingKey.")
            if key.sampling_version == SamplingKey.CASE_EPOCH_VERSION:
                fluid = key.native_indices(
                    len(original["query_xy"]), int(budget["fluid_queries"]),
                    case_id, "fluid", "primary_native_queries", replace=False,
                )
                material = key.native_indices(
                    len(original["module_internal_query_points"]),
                    int(budget["material_queries_per_module"]),
                    case_id, "material", "primary_native_queries", replace=False,
                )
            else:
                rng = key.numpy_rng("thermal_primary_native_queries", case_id)
                fluid = rng.choice(len(original["query_xy"]), int(budget["fluid_queries"]), replace=False)
                material = rng.choice(
                    len(original["module_internal_query_points"]),
                    int(budget["material_queries_per_module"]), replace=False,
                )
        else:
            sampling_index = int(index) if validation_sampling_indices is None else validation_sampling_indices[case_id]
            rng = np.random.default_rng(1000 + sampling_index * 104729)
            fluid = rng.choice(len(original["query_xy"]), int(budget["fluid_queries"]), replace=False)
            material = rng.choice(
                len(original["module_internal_query_points"]),
                int(budget["material_queries_per_module"]), replace=False,
            )
        surface_stride = int(budget.get("surface_stride", RESPONSE_SURFACE_STRIDE))
        samples.append({
            "structure": original["structure"],
            "query_xy": original["query_xy"][fluid],
            "field_targets": original["field_targets"][fluid],
            "point_weights": original["point_weights"][fluid],
            "interface_target": original["interface_target"][:, ::surface_stride],
            "interface_condition_valid_mask": original["interface_condition_valid_mask"][:, ::surface_stride],
            "module_internal_temperature_points": original["module_internal_temperature_points"][:, material],
            "module_internal_query_points": original["module_internal_query_points"][material],
        })
        sampling_hash.update(case_id.encode("utf-8") + b"\0")
        sampling_hash.update(np.ascontiguousarray(fluid, dtype=np.int64).tobytes())
        sampling_hash.update(np.ascontiguousarray(material, dtype=np.int64).tobytes())
    batch = _move_tree(ChannelThermalBatchCollator()(samples), device)
    return batch, sampling_hash.hexdigest()


def _native_receivers_for_counts(
    adapter: ThermalSourceResponse,
    structure: Mapping[str, Any],
    fluid_xy: torch.Tensor,
    local_query_points: torch.Tensor | None,
    *,
    ntheta: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce prepare_native's geometry-only deduplicated grid receiver set."""

    centers = torch.as_tensor(structure["module_centers"], dtype=torch.float32)
    if centers.ndim == 2:
        centers = centers[None]
    present = torch.as_tensor(structure["module_present"], device=centers.device, dtype=centers.dtype)
    if present.ndim == 1:
        present = present[None]
    lengths = torch.cat((
        adapter._column(structure, "domain_length_x", centers, 12.0),
        adapter._column(structure, "domain_length_y", centers, 6.0),
    ), -1)
    material = torch.as_tensor(structure["material_params"], device=centers.device, dtype=centers.dtype)
    if material.ndim == 1:
        material = material[None].expand(centers.shape[0], -1)
    radius = material[:, 5:6]
    delta = torch.minimum((lengths / lengths.new_tensor([adapter.nx, adapter.ny])).amin(-1, keepdim=True),
                          0.15 * radius)
    theta = torch.arange(ntheta, dtype=centers.dtype, device=centers.device) * (2 * torch.pi / ntheta)
    normals = torch.stack((theta.cos(), theta.sin()), -1)
    roles: dict[str, torch.Tensor] = {
        "fluid": torch.as_tensor(fluid_xy, device=centers.device, dtype=centers.dtype),
        "surface": centers[:, :, None] + radius[:, None, :, None] * normals[None, None],
        "outside": centers[:, :, None] + (radius + delta)[:, None, :, None] * normals[None, None],
    }
    if local_query_points is not None:
        local = torch.as_tensor(local_query_points, device=centers.device, dtype=centers.dtype)
        if local.ndim == 3:
            local = local[:, None].expand(-1, centers.shape[1], -1, -1)
        roles["material"] = centers[:, :, None] + radius[:, None, :, None] * local
    stencils = {name: adapter.native_stencil(points, lengths) for name, points in roles.items()}
    unions = []
    for batch_index in range(centers.shape[0]):
        selected = []
        for name, stencil in stencils.items():
            needed = stencil.weights[batch_index] != 0
            if name != "fluid":
                active = present[batch_index, :, None].expand(*roles[name].shape[1:-1]).reshape(-1)
                needed = needed & active[:, None].bool()
            needed = needed & stencil.valid[batch_index, :, None]
            selected.append(stencil.indices[batch_index][needed])
        union = torch.unique(torch.cat(selected), sorted=True)
        if union.numel() == 0:
            union = torch.zeros(1, dtype=torch.long, device=centers.device)
        unions.append(union)
    width = max(int(union.numel()) for union in unions)
    padded = torch.stack([torch.cat((union, union[:1].expand(width - union.numel()))) for union in unions])
    xy = torch.stack((padded % adapter.nx + 0.5, padded // adapter.nx + 0.5), -1).to(centers)
    query = xy * (lengths[:, None] / lengths.new_tensor([adapter.nx, adapter.ny]))
    return query, present


def _pair_counts(
    structure: Mapping[str, Any], queries: torch.Tensor, present: torch.Tensor,
) -> tuple[int, int]:
    centers = torch.as_tensor(structure["module_centers"], device=queries.device, dtype=queries.dtype)
    if centers.ndim == 2:
        centers = centers[None]
    present = torch.as_tensor(present, device=queries.device, dtype=queries.dtype)
    if present.ndim == 1:
        present = present[None]
    material = torch.as_tensor(structure["material_params"], device=queries.device, dtype=queries.dtype)
    if material.ndim == 1:
        material = material[None].expand(centers.shape[0], -1)
    radius = material[:, 5:6].expand(-1, centers.shape[1])
    distance = torch.linalg.vector_norm(queries[:, :, None] - centers[:, None], dim=-1)
    scaled = distance / radius[:, None].clamp_min(1e-12)
    transition = ((scaled - 2) / 2).clamp(0, 1)
    near_weight = (1 - transition.square() * (3 - 2 * transition)) * present[:, None]
    active = present[:, None] > 0
    active_pairs = int(active.expand_as(near_weight).sum().item())
    protected_pairs = int(((near_weight > 0) & active).sum().item())
    return active_pairs, protected_pairs


def _operator_query_receivers(
    adapter: ThermalSourceResponse,
    balances: Sequence[Any],
    case_indices: Sequence[int],
    epoch: int,
    structure: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the existing TRAIN-only four-stratum stencil row sampler."""
    centers = torch.as_tensor(structure["module_centers"], dtype=torch.float32)
    if centers.ndim == 2:
        centers = centers[None]
    present = torch.as_tensor(structure["module_present"], device=centers.device, dtype=centers.dtype)
    if present.ndim == 1:
        present = present[None]
    lengths = torch.cat((
        ThermalSourceResponse._column(structure, "domain_length_x", centers, 12.0),
        ThermalSourceResponse._column(structure, "domain_length_y", centers, 6.0),
    ), -1)
    nx, ny = adapter.nx, adapter.ny
    rows = _operator_row_indices(balances, case_indices, epoch, centers.device)
    ix, iy = rows % nx, rows // nx
    stencil = torch.stack((
        rows,
        iy * nx + (ix - 1).clamp_min(0),
        iy * nx + (ix + 1).clamp_max(nx - 1),
        (iy - 1).clamp_min(0) * nx + ix,
        (iy + 1).clamp_max(ny - 1) * nx + ix,
    ), -1)
    unions = [torch.unique(stencil[index], sorted=True) for index in range(stencil.shape[0])]
    width = max(int(union.numel()) for union in unions)
    padded = torch.stack([torch.cat((union, union[:1].expand(width - union.numel()))) for union in unions])
    xy = torch.stack((padded % nx + 0.5, padded // nx + 0.5), -1).to(lengths)
    return xy * (lengths[:, None] / lengths.new_tensor([nx, ny])), present


def _operator_row_indices(
    balances: Sequence[Any], case_indices: Sequence[int], epoch: int, device: torch.device,
) -> torch.Tensor:
    """Repeat the inherited fixed four-stratum TRAIN row sampler exactly."""
    rows_by_case = []
    for index in case_indices:
        source = balances[int(index)].source_slots[0].detach().cpu().numpy()
        boundary = np.zeros_like(source, dtype=bool)
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        near_interface = np.zeros_like(source, dtype=bool)
        near_interface[:, 1:] |= (source[:, 1:] >= 0) != (source[:, :-1] >= 0)
        near_interface[1:] |= (source[1:] >= 0) != (source[:-1] >= 0)
        rng = np.random.default_rng(int(index) * 104729 + int(epoch) * 1000003 + 17)
        categories = (
            boundary,
            (source >= 0) & ~boundary,
            near_interface & ~boundary,
            (source < 0) & ~boundary,
        )
        chosen = []
        rows_per_category = OPERATOR_ROWS_PER_CASE // len(categories)
        for category in categories:
            valid = np.flatnonzero(category)
            if valid.size == 0:
                raise ValueError("Thermal operator stratum is empty for a selected TRAIN case.")
            chosen.extend(rng.choice(valid, rows_per_category, replace=len(valid) < rows_per_category).tolist())
        rows_by_case.append(chosen)
    if not rows_by_case:
        raise ValueError("Thermal operator residual requires at least one TRAIN case.")
    return torch.as_tensor(rows_by_case, dtype=torch.long, device=device)


def _operator_stencil_receivers(
    balance: Any,
    row_indices: torch.Tensor,
    lengths: torch.Tensor,
    nx: int,
    ny: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, source_ny, source_nx = balance.source_slots.shape
    if (source_nx, source_ny) != (nx, ny):
        raise ValueError("Thermal residual grid dimensions differ from the retained operator grid.")
    ix, iy = row_indices % nx, row_indices // nx
    stencil = torch.stack((
        row_indices,
        iy * nx + (ix - 1).clamp_min(0),
        iy * nx + (ix + 1).clamp_max(nx - 1),
        (iy - 1).clamp_min(0) * nx + ix,
        (iy + 1).clamp_max(ny - 1) * nx + ix,
    ), -1)
    unions = [torch.unique(stencil[index], sorted=True) for index in range(batch)]
    width = max(int(union.numel()) for union in unions)
    receiver_ids = torch.stack([
        torch.cat((union, union[:1].expand(width - union.numel()))) for union in unions
    ])
    gather = torch.stack([torch.searchsorted(union, stencil[index]) for index, union in enumerate(unions)])
    native = torch.stack((receiver_ids % nx + 0.5, receiver_ids // nx + 0.5), -1).to(lengths)
    query = native * (lengths[:, None] / lengths.new_tensor([nx, ny]))
    return query, receiver_ids, gather, stencil, ix, iy


def _residual_from_stencil_kernel(
    grid_kernel: torch.Tensor,
    balance: Any,
    row_indices: torch.Tensor,
    gather: torch.Tensor,
    stencil: torch.Tensor,
    ix: torch.Tensor,
    iy: torch.Tensor,
) -> torch.Tensor:
    batch, _ny, nx = balance.source_slots.shape
    batch_indices = torch.arange(batch, device=grid_kernel.device)[:, None, None]
    kernel_rows = grid_kernel[batch_indices, gather]
    coefficients = balance.coefficients.reshape(batch, -1, 5).gather(
        1, row_indices[..., None].expand(-1, -1, 5))
    applied = (kernel_rows * coefficients[..., None]).sum(2)
    slots = balance.source_slots.reshape(batch, -1).gather(1, row_indices)
    modules = torch.arange(balance.present.shape[1], device=slots.device)
    forcing = ((slots[..., None] == modules) * balance.present[:, None]).to(applied)
    boundary = (ix == 0) | (ix == nx - 1) | (iy == 0) | (iy == balance.source_slots.shape[1] - 1)
    forcing = forcing * (~boundary[..., None])
    return (applied - forcing) / coefficients.abs().sum(-1).clamp_min(
        torch.finfo(applied.dtype).eps)[..., None]


def _merge_auxiliary(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if key in {"base_loss", "router_importance_loss", "expected_work"}:
            continue
        if key not in result:
            result[key] = value
        elif ((torch.is_tensor(result[key]) and torch.is_tensor(value))
              or (isinstance(result[key], (int, float)) and isinstance(value, (int, float)))):
            result[key] = result[key] + value
    for prefix, alias in (("base", "base_loss"), ("router_importance", "router_importance_loss"),
                          ("expected_work", "expected_work")):
        numerator = result.get(f"{prefix}_numerator")
        denominator = result.get(f"{prefix}_denominator")
        if torch.is_tensor(numerator) and torch.is_tensor(denominator):
            result[alias] = numerator / denominator.clamp_min(1)
    return result


class ThermalRefinementTask:
    """Fixed25 Thermal task provider; the shared engine owns every update."""

    def __init__(
        self,
        *,
        model: ThermalSourceResponse,
        parent_model: ThermalSourceResponse | None,
        parent: Mapping[str, Any] | None,
        parent_path: Path | None,
        flow_path: Path,
        atlas_directory: Path,
        training_cases: Sequence[Mapping[str, Any]],
        validation_cases: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
        train_families: Sequence[Mapping[str, Any]],
        development_families: Sequence[Mapping[str, Any]],
        balances: Sequence[Any],
        optimizer_seed: Mapping[str, Any] | None,
        device: torch.device | str,
        stats: Mapping[str, Any] | None = None,
        recipe: Mapping[str, Any] | None = None,
        validation_budget: Mapping[str, Any] | None = None,
        calibration_budget: Mapping[str, Any] | None = None,
        gate_version: str = THERMAL_GATE_HARD_VERSION,
        gate_transition: Sequence[float] | None = None,
        development_refinement_parent: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.parent_model = None if parent_model is None else parent_model.eval().requires_grad_(False)
        self.parent = parent
        self.parent_path = parent_path
        self.flow_path = flow_path
        self.atlas_directory = atlas_directory
        self.training_cases = tuple(training_cases)
        self.validation_cases = tuple(validation_cases)
        self.manifest = dict(manifest)
        self.train_families = tuple(train_families)
        self.development_families = tuple(development_families)
        self.balances = tuple(balances)
        self.optimizer_seed = copy.deepcopy(optimizer_seed)
        self.device = torch.device(device)
        if stats is None:
            if parent is None:
                raise ValueError("Thermal task requires explicit normalization stats when no parent is bound.")
            stats = parent["global_normalization_stats"]
        if recipe is None:
            if parent is None:
                raise ValueError("Thermal task requires an explicit recipe when no parent is bound.")
            recipe = parent["fit_identity"]["recipe"]
        self.stats = stats
        self.recipe = dict(recipe)
        self.gate_version, self.gate_transition = _resolve_gate_binding(gate_version, gate_transition)
        self.development_refinement_parent = (
            None if development_refinement_parent is None else copy.deepcopy(dict(development_refinement_parent))
        )
        self.development_optimizer_seed: Mapping[str, Any] | None = None
        self.development_parent_rng_state: Mapping[str, Any] | None = None
        self.budget = dict(self.recipe["budget"])
        self.validation_budget = self.budget if validation_budget is None else dict(validation_budget)
        self.calibration_budget = self.budget if calibration_budget is None else dict(calibration_budget)
        self._separate_query_budgets = validation_budget is not None or calibration_budget is not None
        self.response_scales = dict(self.recipe["calibration"]["response_scales"])
        self.response_coefficient = float(self.recipe["calibration"]["response_coefficient"])
        self.operator_coefficient = float(self.recipe["calibration"]["operator_coefficient"])
        self.use_operator = self.recipe["operator_decision"]["operator_constraint"] == "qualified"
        self.temperature_std = float(np.asarray(self.stats["field_std_by_channel"]).reshape(-1)[4])
        self.base_loss_weight = BASE_APPROXIMATION_COEFFICIENT / max(self.temperature_std**2, 1.0e-12)
        self._expected_work_weight = 0.0
        self._expected_work_calibration: Mapping[str, Any] | None = None
        self._training_samples = 0
        self._validation_samples = 0
        self._train_index_by_id = {str(case["case_id"]): index for index, case in enumerate(self.training_cases)}
        self._validation_ids = tuple(str(case["case_id"]) for case in self.validation_cases)
        self._train_ids = tuple(str(case["case_id"]) for case in self.training_cases)
        self._train_family_by_id = {str(item["family_id"]): item for item in self.train_families}
        self._development_response_inputs = tuple(self._prepare_response_sample(family, training=False, key=None)
                                                  for family in self.development_families)
        self._parent_response_baseline = (
            self._measure_response_set(self.parent_model, self._development_response_inputs, execution_mode="all_fine")
            if self.parent_model is not None and self._development_response_inputs else None
        )
        self._geometry_helper = self.model
        self._parent_moment_identity = (
            _recipe_optimizer_identity(self.parent, self.optimizer_seed)
            if self.parent is not None and self.optimizer_seed is not None else None
        )

    @property
    def parent_optimizer_seed(self) -> Mapping[str, Any] | None:
        return copy.deepcopy(self.optimizer_seed)

    @property
    def training_optimizer_seed(self) -> Mapping[str, Any] | None:
        """Return the inherited moments for this run's declared starting state."""

        seed = self.development_optimizer_seed if self.development_optimizer_seed is not None else self.optimizer_seed
        return copy.deepcopy(seed)

    def identity_payload(self) -> Mapping[str, Any]:
        payload = {
            "task": "ThermalChannel",
            "dataset_split": "fixed25_v1",
            "manifest_fingerprint": self.manifest["manifest_sha256"],
            "training_case_ids": list(self._train_ids),
            "validation_case_ids": list(self._validation_ids),
            "parent_checkpoint": str(self.parent_path),
            "parent_checkpoint_sha256": _sha256(self.parent_path),
            "parent_epoch": int(self.parent["epoch"]),
            "parent_model_mode": self.parent["fit_identity"]["mode"],
            "parent_optimizer_moments": self._parent_moment_identity,
            "flow_checkpoint": str(self.flow_path),
            "flow_checkpoint_sha256": _sha256(self.flow_path),
            "flow_parent_epoch": 2500,
            "flow_inference_inputs": "none; frozen D-sep identity only",
            "training_flow_target_policy": "stored TRAIN u/v only in the qualified discrete-operator residual supervision term",
            "normalization_stats_sha256": _tensor_digest(self.stats),
            "training_response_family_ids": [str(item["family_id"]) for item in self.train_families],
            "training_response_sources": [{"family_id": item["family_id"], "path": item["source"],
                                            "sha256": item["source_sha256"]} for item in self.train_families],
            "development_response_family_ids": [str(item["family_id"]) for item in self.development_families],
            "development_response_sources": [{"family_id": item["family_id"], "path": item["source"],
                                               "sha256": item["source_sha256"]} for item in self.development_families],
            "near_response_exposure": list(NEAR_RESPONSE_IDS),
            "native_query_budget": dict(self.budget),
            "q_proxy_coefficient": Q_PROXY_COEFFICIENT,
            "response_coefficient": self.response_coefficient,
            "response_scales": self.response_scales,
            "operator_coefficient": self.operator_coefficient,
            "operator_decision": self.recipe["operator_decision"],
            "base_approximation": {"weight": BASE_APPROXIMATION_COEFFICIENT,
                                    "fixed_temperature_std": self.temperature_std,
                                    "scaled_weight": self.base_loss_weight},
            "router_importance_weight": ROUTER_IMPORTANCE_COEFFICIENT,
            "expected_work_calibration": {"target_gradient_share": EXPECTED_WORK_TARGET_SHARE,
                                           "coefficient_cap": EXPECTED_WORK_WEIGHT_CAP,
                                           "panel_case_ids": list(self.recipe["calibration"]["calibration_case_ids"])},
            "refinement_architecture": dict(self.model.core.refinement_config),
            "output_law": "affine",
            "applicable_control": "heat; supplied only to affine application",
            "training_objective": "inherited native fluid/surface/material T + .05 q proxy + parent-calibrated TRAIN response + qualified operator residual; plus fixed-scale base approximation and staged adaptive terms",
            "solver_attempts": 0,
        }
        if self._separate_query_budgets:
            payload["validation_query_budget"] = dict(self.validation_budget)
            payload["calibration_query_budget"] = dict(self.calibration_budget)
        if self.gate_version == THERMAL_GATE_COMPACT_C1_VERSION:
            payload.update({
                "gate_version": self.gate_version,
                "gate_transition": list(self.gate_transition),
            })
        if self.development_refinement_parent is not None:
            payload["development_refinement_parent"] = copy.deepcopy(self.development_refinement_parent)
        return payload

    def epoch_cases(self, epoch: int, seed: int) -> Sequence[str]:
        del epoch, seed
        return self._train_ids

    def _response_sample(
        self, family: Mapping[str, Any], *, training: bool, key: SamplingKey | None,
        budget: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        sample_budget = (self.budget if training else self.validation_budget) if budget is None else budget
        if training:
            if key is None:
                raise ValueError("TRAIN response sampling requires its engine-owned SamplingKey.")
            family_id = str(family["family_id"])
            if key.sampling_version == SamplingKey.CASE_EPOCH_VERSION:
                valid_fluid = np.flatnonzero(np.asarray(family["fluid_valid"], dtype=bool))
                fluid_count = int(sample_budget["fluid_queries"])
                material_count = int(sample_budget["material_queries_per_module"])
                if len(valid_fluid) < fluid_count or len(family["material_local"]) < material_count:
                    raise ValueError("Thermal response family has insufficient fixed receiver coverage.")
                fluid_ids = valid_fluid[key.native_indices(
                    len(valid_fluid), fluid_count, family_id, "response_fluid", "response_addendum",
                )]
                material_ids = key.native_indices(
                    len(family["material_local"]), material_count,
                    family_id, "response_material", "response_addendum",
                )
                rng = None
            else:
                rng = key.numpy_rng("thermal_response_addendum", family_id)
        else:
            family_index = DEVELOPMENT_RESPONSE_IDS.index(str(family["family_id"]))
            rng = np.random.default_rng(0x5448524D + family_index * 104729)
        if not training or key is None or key.sampling_version != SamplingKey.CASE_EPOCH_VERSION:
            valid_fluid = np.flatnonzero(np.asarray(family["fluid_valid"], dtype=bool))
            fluid_count = int(sample_budget["fluid_queries"])
            material_count = int(sample_budget["material_queries_per_module"])
            if len(valid_fluid) < fluid_count or len(family["material_local"]) < material_count:
                raise ValueError("Thermal response family has insufficient fixed receiver coverage.")
            fluid_ids = rng.choice(valid_fluid, fluid_count, replace=False)
            material_ids = rng.choice(len(family["material_local"]), material_count, replace=False)
        surface_stride = int(sample_budget.get("surface_stride", RESPONSE_SURFACE_STRIDE))
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32)
        sample = {
            "family_id": str(family["family_id"]),
            "structure": {name: tensor(value)[None] for name, value in family["structure"].items()},
            "fluid_xy": tensor(family["fluid_xy"][fluid_ids])[None],
            "local": tensor(family["material_local"][material_ids])[None],
            "heat": tensor(family["heat_increment"])[None],
            "fluid_target": tensor(family["deltas"]["fluid_fields"][fluid_ids, 4])[None],
            "surface_target": tensor(family["deltas"]["interface"][:, 0].reshape(family["module_count"], -1)[:, ::surface_stride])[None],
            "surface_mask": tensor(family["interface_valid"][:, 0].reshape(family["module_count"], -1)[:, ::surface_stride])[None],
            "material_target": tensor(family["deltas"]["solid_temperature"].reshape(family["module_count"], -1)[:, material_ids])[None],
            "material_mask": tensor(family["material_valid"][:, material_ids])[None],
            "fluid_ids": np.asarray(fluid_ids, dtype=np.int64),
            "material_ids": np.asarray(material_ids, dtype=np.int64),
        }
        if "native_solid_mask" in family:
            mask = np.asarray(family["native_solid_mask"])
            if mask.dtype != np.bool_ or mask.ndim != 2:
                raise ValueError("Bound TRAIN response native_solid_mask must be a stored boolean [ny,nx] array.")
            sample["native_solid_mask"] = torch.as_tensor(mask.copy(), dtype=torch.bool)[None]
        return sample

    def _prepare_response_sample(
        self, family: Mapping[str, Any], *, training: bool, key: SamplingKey | None,
        budget: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._response_sample(family, training=training, key=key, budget=budget)

    def _batch_from_indices(
        self, case_indices: Sequence[int], epoch: int, *, key: SamplingKey | None,
        training: bool, include_response: bool, budget: Mapping[str, Any] | None = None,
    ) -> TaskBatch:
        indices = tuple(int(index) for index in case_indices)
        sample_budget = (self.budget if training else self.validation_budget) if budget is None else budget
        sample, primary_sampling_hash = _sample_primary(
            self.training_cases if training else self.validation_cases,
            indices,
            epoch=epoch,
            key=key,
            training=training,
            budget=sample_budget,
            device=self.device,
            validation_sampling_indices=getattr(self, "validation_sampling_indices", None),
        )
        selected_cases = (self.training_cases if training else self.validation_cases)
        selected_records = [selected_cases[index] for index in indices]
        primary_masks = [case.get("native_solid_mask") for case in selected_records]
        native_solid_mask = None
        if any(mask is not None for mask in primary_masks):
            if any(mask is None for mask in primary_masks):
                raise ValueError("Native curl requires one saved geometry mask for every primary case in a batch.")
            mask_arrays = [np.asarray(mask) for mask in primary_masks]
            if any(mask.dtype != np.bool_ or mask.ndim != 2 for mask in mask_arrays):
                raise ValueError("Primary native_solid_mask entries must be stored boolean [ny,nx] arrays.")
            if len({tuple(mask.shape) for mask in mask_arrays}) != 1:
                raise ValueError("Primary native_solid_mask grids differ within the physical batch.")
            native_solid_mask = torch.as_tensor(np.stack(mask_arrays), dtype=torch.bool, device=self.device)
        raw_structure = sample["structure"]
        scene_structure = _structure_only(raw_structure)
        response_sample = None
        if include_response and key is not None and key.microbatch_index == 0:
            family_index = (key.epoch + key.update_index) % len(TRAIN_RESPONSE_IDS)
            response_sample = self._prepare_response_sample(
                self._train_family_by_id[TRAIN_RESPONSE_IDS[family_index]], training=True, key=key,
                budget=sample_budget)
        response_structure = None if response_sample is None else response_sample["structure"]
        response_native_solid_mask = (
            None if response_sample is None else response_sample.get("native_solid_mask")
        )
        if response_native_solid_mask is not None:
            response_native_solid_mask = response_native_solid_mask.to(device=self.device, dtype=torch.bool)
        fluid_xy = sample["query_xy"]
        local = sample["module_internal_query_points"]
        receivers = ThermalReceivers(
            fluid_xy=fluid_xy.to(self.device),
            local_query_points=local.to(self.device),
            heat=raw_structure["heat_powers"].to(self.device),
            response_fluid_xy=None if response_sample is None else response_sample["fluid_xy"].to(self.device),
            response_local_query_points=None if response_sample is None else response_sample["local"].to(self.device),
            response_heat=None if response_sample is None else response_sample["heat"].to(self.device),
            native_solid_mask=native_solid_mask,
            response_native_solid_mask=response_native_solid_mask,
        )
        response_target = None
        if response_sample is not None:
            response_target = ThermalResponseTargets(
                family_id=response_sample["family_id"],
                fluid=response_sample["fluid_target"].to(self.device),
                surface=response_sample["surface_target"].to(self.device),
                surface_mask=response_sample["surface_mask"].to(self.device),
                material=response_sample["material_target"].to(self.device),
                material_mask=response_sample["material_mask"].to(self.device),
            )
        sampling_digest = hashlib.sha256(primary_sampling_hash.encode("ascii"))
        if response_sample is not None:
            sampling_digest.update(str(response_sample["family_id"]).encode("utf-8") + b"\0")
            sampling_digest.update(np.ascontiguousarray(response_sample["fluid_ids"], dtype=np.int64).tobytes())
            sampling_digest.update(np.ascontiguousarray(response_sample["material_ids"], dtype=np.int64).tobytes())
        case_set = self.training_cases if training else self.validation_cases
        case_ids = tuple(str(case_set[index]["case_id"]) for index in indices)
        targets = ThermalTargets(
            case_ids=case_ids,
            case_indices=indices if training else (),
            epoch=int(epoch),
            field_targets=sample["field_targets"].to(self.device),
            point_weights=sample["point_weights"].to(self.device),
            interface_target=sample["interface_target"].to(self.device),
            interface_valid_mask=sample["interface_condition_valid_mask"].to(self.device),
            material_targets=sample["module_internal_temperature_points"].to(self.device),
            response=response_target,
        )
        if training:
            self._training_samples += len(indices)
        else:
            self._validation_samples += len(indices)
        return TaskBatch(
            scene_inputs=ThermalSceneInputs(scene_structure, response_structure),
            receivers=receivers,
            targets=targets,
            auxiliary={"epoch": int(epoch),
                       "response_family_id": None if response_target is None else response_target.family_id,
                       "query_sampling_sha256": sampling_digest.hexdigest()},
            case_keys=case_ids,
        )

    def make_batch(self, case_keys: Sequence[Any], key: SamplingKey) -> TaskBatch:
        indices = []
        for case_id in case_keys:
            try:
                indices.append(self._train_index_by_id[str(case_id)])
            except KeyError as error:
                raise ValueError(f"Thermal TRAIN case {case_id!r} is outside fixed25_v1.") from error
        return self._batch_from_indices(indices, key.epoch, key=key, training=True, include_response=True)

    def make_scene(self, scene_inputs: ThermalSceneInputs) -> ThermalInferenceScene:
        if not isinstance(scene_inputs, ThermalSceneInputs):
            raise TypeError("Thermal scene construction requires the target-free scene record.")
        structure = _structure_only(scene_inputs.structure)
        response_structure = None if scene_inputs.response_structure is None else _structure_only(scene_inputs.response_structure)
        return ThermalInferenceScene(_move_tree(structure, self.device),
                                     None if response_structure is None else _move_tree(response_structure, self.device))

    def _set_execution(
        self, model: ThermalSourceResponse, mode: str, phase: str, temperature: float,
        training: bool, threshold: float = 0.5,
    ) -> None:
        aliases = {"warmup": "all_fine", "full_detail": "all_fine", "full": "all_fine",
                   "adaptive_detail": "adaptive"}
        core_mode = aliases.get(mode, mode)
        if core_mode not in {"all_fine", "adaptive", "all_base", "nearest", "upstream", "shuffle"}:
            raise ValueError(f"Unknown Thermal refinement evaluation mode {mode!r}.")
        core_phase = "open" if phase in ("warmup", "open") else phase
        setter = getattr(model.core, "set_execution", None)
        if callable(setter):
            setter(mode=core_mode, phase=core_phase, threshold=float(threshold),
                   temperature=float(temperature), training_signal=bool(training),
                   gate_version=getattr(self, "gate_version", THERMAL_GATE_HARD_VERSION),
                   gate_transition=getattr(self, "gate_transition", THERMAL_GATE_COMPACT_C1_TRANSITION))
        elif core_mode != "all_fine":
            raise TypeError("Thermal route controls require the opt-in refined source-response core.")

    @staticmethod
    def _full_replay(model: ThermalSourceResponse, prepared: Any, heat: torch.Tensor) -> Mapping[str, torch.Tensor]:
        full_response = getattr(prepared.response, "full_response", None)
        if full_response is None:
            if getattr(prepared.response, "mode", None) == "direct" and prepared.response.far_kernel is not None:
                full_response = prepared.response
            else:
                raise RuntimeError("Adaptive Thermal read did not retain its already-paid full fine response.")
        return model.apply_native(replace(prepared, response=full_response), heat)

    def predict_native(
        self,
        model: nn.Module,
        scene: ThermalInferenceScene,
        receivers: ThermalReceivers,
        execution_mode: str,
        phase: str,
        epoch: int = 0,
        temperature: float = 1.0,
        *,
        threshold: float = 0.5,
    ) -> tuple[ThermalPredictions, Mapping[str, Any]]:
        if not isinstance(model, ThermalSourceResponse):
            raise TypeError("Thermal provider requires the shared-grid affine source-response adapter.")
        from honf_runtime.compat import recursive_to_device

        self._set_execution(model, execution_mode, phase, temperature, model.training, threshold)
        model.core.reset_auxiliary()
        rx = recursive_to_device(receivers, self.device)
        native_prepared = model.prepare_native(scene.structure, rx.fluid_xy,
            local_query_points=rx.local_query_points, ntheta=16)
        native_main = model.apply_native(native_prepared, rx.heat)
        native_full = None
        response_prepared = response_main = response_full = None
        if execution_mode == "adaptive_detail" and model.training:
            native_full = self._full_replay(model, native_prepared, rx.heat)
        if rx.response_fluid_xy is not None:
            if scene.response_structure is None or rx.response_local_query_points is None or rx.response_heat is None:
                raise ValueError("Thermal TRAIN response addendum must bind geometry, receivers and heat separately.")
            response_prepared = model.prepare_native(scene.response_structure, rx.response_fluid_xy,
                local_query_points=rx.response_local_query_points, ntheta=16)
            response_main = model.apply_native(response_prepared, rx.response_heat, increment=True)
            if execution_mode == "adaptive_detail" and model.training:
                response_full = self._full_replay(model, response_prepared, rx.response_heat)
        auxiliary = model.core.auxiliary_terms()
        model.core.reset_auxiliary()
        work = self._read_work_counts(native_prepared, response_prepared, auxiliary)
        predictions = ThermalPredictions(
            model=model,
            execution_mode=execution_mode,
            phase=phase,
            native_prepared=native_prepared,
            native_main=native_main,
            native_full=native_full,
            response_prepared=response_prepared,
            response_main=response_main,
            response_full=response_full,
            auxiliary=dict(auxiliary),
            work=work,
        )
        state = {"predictions": predictions, "base_auxiliary": dict(auxiliary), "work": work}
        return predictions, state

    @staticmethod
    def _read_work_counts(native_prepared: Any, response_prepared: Any | None, auxiliary: Mapping[str, Any]) -> dict[str, int]:
        counts = {name: int(auxiliary.get(name, 0)) for name in (
            "cheap_rows", "fine_rows", "near_rows", "gate_rows", "active_pairs", "padded_fine_capacity",
            "selected_detail_rows")}
        for prefix, prepared in (("native", native_prepared), ("response", response_prepared)):
            if prepared is None:
                continue
            present = prepared.source_present > 0.5
            fluid = prepared.stencils["fluid"].valid
            surface = prepared.stencils["surface"].valid.reshape(
                prepared.source_present.shape[0], *prepared.role_shapes["surface"])
            material = prepared.stencils["material"].valid.reshape(
                prepared.source_present.shape[0], *prepared.role_shapes["material"])
            counts[f"{prefix}_fluid_queries"] = int(fluid.numel())
            counts[f"{prefix}_fluid_valid_queries"] = int(fluid.sum().item())
            counts[f"{prefix}_surface_queries"] = int(present.sum().item() * surface.shape[-1])
            counts[f"{prefix}_surface_valid_queries"] = int((surface & present[..., None]).sum().item())
            counts[f"{prefix}_material_queries"] = int(present.sum().item() * material.shape[-1])
            counts[f"{prefix}_material_valid_queries"] = int((material & present[..., None]).sum().item())
            counts[f"{prefix}_grid_rows"] = int(prepared.grid_indices.numel())
        if not counts["active_pairs"]:
            for prepared in (native_prepared, response_prepared):
                if prepared is None:
                    continue
                refinement_work = getattr(prepared.response, "refinement_aux", None)
                if isinstance(refinement_work, Mapping):
                    for name in counts:
                        counts[name] += int(refinement_work.get(name, 0))
                    if "padded_fine_capacity" not in refinement_work:
                        counts["padded_fine_capacity"] += int(
                            prepared.response.receivers.shape[1] * prepared.source_present.shape[1])
                    continue
                receiver_count = int(prepared.response.receivers.shape[1])
                active_sources = int((prepared.source_present > 0).sum().item())
                padded_sources = int(prepared.source_present.shape[1])
                counts["active_pairs"] += receiver_count * active_sources
                # The retained dense fine reader evaluates padded slots before
                # applying the source-presence mask, so count its actual work.
                counts["fine_rows"] += receiver_count * padded_sources
                counts["padded_fine_capacity"] += receiver_count * padded_sources
        return counts

    def _case_reconstruction(self, output: Mapping[str, torch.Tensor], prepared: Any, targets: ThermalTargets) -> dict[str, torch.Tensor]:
        from thermal_source_response_fit import scalar_scale

        weight = targets.point_weights
        fluid_error = (output["fluid_temperature"][..., 0] - targets.field_targets[..., 4]) / scalar_scale(
            self.stats, "field_std_by_channel", 4)
        fluid_mse = (fluid_error.square() * weight).sum(1) / weight.sum(1).clamp_min(1e-12)
        present = prepared.source_present
        surface_shape = output["pred_interface"].shape[:-1]
        native_surface_valid = prepared.stencils["surface"].valid.reshape(surface_shape)
        native_outside_valid = prepared.stencils["outside"].valid.reshape(surface_shape)
        surface_mask = present[..., None] * native_surface_valid * torch.isfinite(targets.interface_target[..., 0])
        surface_error = (output["pred_interface"][..., 0] - targets.interface_target[..., 0]) / scalar_scale(
            self.stats, "interface_targets_std", 0)
        valid_module = present * (surface_mask.sum(-1) > 0)
        surface_per_module = (surface_error.square() * surface_mask).sum(-1) / surface_mask.sum(-1).clamp_min(1)
        surface_mse = (surface_per_module * valid_module).sum(-1) / valid_module.sum(-1).clamp_min(1)
        material_error = (output["pred_internal_temperature"][..., 0] - targets.material_targets) / scalar_scale(
            self.stats, "internal_temperature_std")
        material_mse = ((material_error.square() * present[..., None]).sum((1, 2)) /
                        (present.sum(1) * material_error.shape[-1]).clamp_min(1))
        q_error = (output["pred_interface"][..., 1] - targets.interface_target[..., 1]) / scalar_scale(
            self.stats, "interface_targets_std", 1)
        q_mask = present[..., None] * native_surface_valid * native_outside_valid * torch.isfinite(
            targets.interface_target[..., 1])
        q_modules = present * (q_mask.sum(-1) > 0)
        q_per_module = (q_error.square() * q_mask).sum(-1) / q_mask.sum(-1).clamp_min(1)
        q_mse = (q_per_module * q_modules).sum(-1) / q_modules.sum(-1).clamp_min(1)
        thermal = (fluid_mse + surface_mse + material_mse) / 3
        return {"reconstruction": thermal + Q_PROXY_COEFFICIENT * q_mse,
                "thermal_selector": thermal, "fluid": fluid_mse, "surface": surface_mse,
                "material": material_mse, "q_proxy": q_mse}

    def _response_loss(self, output: Mapping[str, torch.Tensor], targets: ThermalResponseTargets) -> torch.Tensor:
        fluid = ((output["fluid_temperature"][0, :, 0] - targets.fluid[0]) / self.response_scales["fluid"]).square().mean()
        surface = (((output["pred_interface"][0, ..., 0] - targets.surface[0]) / self.response_scales["surface"]).square()
                   * targets.surface_mask[0]).sum() / targets.surface_mask[0].sum().clamp_min(1)
        material = (
            ((output["pred_internal_temperature"][0, ..., 0] - targets.material[0]) /
             self.response_scales["material"]).square() * targets.material_mask[0]
        ).sum() / targets.material_mask[0].sum().clamp_min(1)
        return (fluid + surface + material) / 3

    def _operator_loss(self, predictions: ThermalPredictions, targets: ThermalTargets) -> tuple[torch.Tensor | None, Mapping[str, Any]]:
        if not self.use_operator:
            return None, {}
        from thermal_source_response_fit import scalar_scale

        from channelthermal.source_response_residual import DiscreteThermalBalance

        prepared = predictions.native_prepared
        device = prepared.context.centers.device
        coefficient = torch.cat([self.balances[index].coefficients for index in targets.case_indices]).to(device)
        slots = torch.cat([self.balances[index].source_slots for index in targets.case_indices]).to(device)
        balance = DiscreteThermalBalance(coefficient, slots, prepared.source_present)
        row_indices = _operator_row_indices(self.balances, targets.case_indices, targets.epoch, device)
        query, receiver_ids, gather, stencil, ix, iy = _operator_stencil_receivers(
            balance, row_indices, prepared.context.lengths, predictions.model.nx,
            predictions.model.ny)
        # One stencil read retains both route-composed and paid full-fine
        # kernels. The same exact source-column stencil is used for both
        # operator terms, avoiding a duplicate fine read for replay.
        stencil_read = predictions.model.core.prepare_receivers(
            prepared.context, query, receiver_ids=receiver_ids)
        main_grid_kernel = stencil_read.dense_kernel()[..., 0]
        full_response = getattr(stencil_read, "full_response", None)
        if full_response is None:
            raise RuntimeError("Thermal operator replay did not retain its paid full-fine stencil response.")
        full_grid_kernel = full_response.dense_kernel()[..., 0]
        main_residual = _residual_from_stencil_kernel(
            main_grid_kernel, balance, row_indices, gather, stencil, ix, iy)
        full_residual = _residual_from_stencil_kernel(
            full_grid_kernel, balance, row_indices, gather, stencil, ix, iy)
        temperature_std = scalar_scale(self.stats, "field_std_by_channel", 4)
        main_residual = main_residual / temperature_std
        full_residual = full_residual / temperature_std
        present = prepared.source_present
        row_denominator = (present.sum(-1) * OPERATOR_ROWS_PER_CASE).clamp_min(1)
        main_per_case = (main_residual.square() * present[:, None]).sum((1, 2)) / row_denominator
        full_per_case = (full_residual.square() * present[:, None]).sum((1, 2)) / row_denominator
        replay_weight = 0.25 if predictions.execution_mode == "adaptive_detail" else 0.0
        value = ((1.0 - replay_weight) * main_per_case + replay_weight * full_per_case).mean()
        receipt = {
            "operator_rows": int(row_indices.numel()),
            "kernel_columns": int(balance.present.sum()),
            "neural_stencil_receiver_rows": int(query.shape[0] * query.shape[1]),
            "unique_stencil_receiver_rows": sum(int(torch.unique(stencil[index]).numel())
                                                 for index in range(stencil.shape[0])),
            "per_case_unique_receiver_rows": [int(torch.unique(stencil[index]).numel())
                                               for index in range(stencil.shape[0])],
            "physical_solves": 0,
            "adaptive_operator_replay_weight": replay_weight,
            "main_operator_loss": float(main_per_case.detach().mean()),
            "full_operator_loss": float(full_per_case.detach().mean()),
        }
        operator_aux = predictions.model.core.auxiliary_terms()
        predictions.model.core.reset_auxiliary()
        predictions.auxiliary = _merge_auxiliary(predictions.auxiliary, operator_aux)
        predictions.work = {**predictions.work,
            **{name: int(predictions.work.get(name, 0)) + int(operator_aux.get(name, 0)) for name in (
                "cheap_rows", "fine_rows", "near_rows", "gate_rows", "active_pairs", "padded_fine_capacity", "selected_detail_rows")}}
        predictions.work["operator_rows"] = int(receipt["operator_rows"])
        predictions.work["operator_neural_stencil_receiver_rows"] = int(receipt["neural_stencil_receiver_rows"])
        predictions.work["operator_unique_stencil_receiver_rows"] = int(receipt["unique_stencil_receiver_rows"])
        return value, receipt

    def loss_denominators(self, batches: Sequence[TaskBatch], phase: str, arm: str) -> Mapping[str, float]:
        totals: dict[str, float] = {}
        for batch in batches:
            targets = batch.targets
            scene = batch.scene_inputs
            receivers = batch.receivers
            main_queries, main_present = _native_receivers_for_counts(
                self._geometry_helper, scene.structure, receivers.fluid_xy, receivers.local_query_points)
            main_active, main_protected = _pair_counts(scene.structure, main_queries, main_present)
            active_pairs, protected_pairs = main_active, main_protected
            if targets.response is not None:
                response_queries, response_present = _native_receivers_for_counts(
                    self._geometry_helper, scene.response_structure, receivers.response_fluid_xy,
                    receivers.response_local_query_points)
                response_active, response_protected = _pair_counts(scene.response_structure, response_queries,
                                                                   response_present)
                active_pairs += response_active
                protected_pairs += response_protected
            if self.use_operator:
                op_query, op_present = _operator_query_receivers(
                    self._geometry_helper, self.balances, targets.case_indices, targets.epoch,
                    scene.structure)
                op_active, op_protected = _pair_counts(scene.structure, op_query, op_present)
                active_pairs += op_active
                protected_pairs += op_protected
            per_batch: dict[str, float] = {
                "reconstruction": float(len(targets.case_ids)),
                "base_loss": float(active_pairs),
            }
            if self.use_operator:
                per_batch["operator_residual"] = float(len(targets.case_ids))
            if targets.response is not None:
                per_batch["response"] = 1.0
            if arm == "adaptive_detail" and phase == "open":
                per_batch["router_importance_loss"] = float(active_pairs - protected_pairs)
            if arm == "adaptive_detail" and phase in ("soft", "hard"):
                per_batch["expected_work"] = float(active_pairs)
            for name, denominator in per_batch.items():
                if denominator <= 0:
                    raise ValueError(f"Thermal target-only denominator for {name!r} is empty.")
                totals[name] = totals.get(name, 0.0) + denominator
        return totals

    def loss_terms(
        self,
        predictions: ThermalPredictions,
        targets: ThermalTargets,
        phase: str,
        auxiliary_state: Mapping[str, Any],
    ) -> Mapping[str, LossTerm]:
        main = self._case_reconstruction(predictions.native_main, predictions.native_prepared, targets)
        adaptive = predictions.execution_mode == "adaptive_detail"
        replay_weight = 0.25 if adaptive else 0.0
        reconstruction_numerator = main["reconstruction"].sum() * (1.0 - replay_weight)
        if replay_weight:
            if predictions.native_full is None:
                raise RuntimeError("Adaptive Thermal training requires the already-paid Full-detail replay.")
            full = self._case_reconstruction(predictions.native_full, predictions.native_prepared, targets)
            reconstruction_numerator = reconstruction_numerator + replay_weight * full["reconstruction"].sum()
        result: dict[str, LossTerm] = {
            "reconstruction": LossTerm(reconstruction_numerator, len(targets.case_ids))
        }
        operator_value, operator_receipt = self._operator_loss(predictions, targets)
        if operator_value is not None:
            result["operator_residual"] = LossTerm(
                operator_value * len(targets.case_ids), len(targets.case_ids), self.operator_coefficient)
            auxiliary_state["operator_work"] = dict(operator_receipt)
        if targets.response is not None:
            if predictions.response_main is None or predictions.response_prepared is None:
                raise RuntimeError("Thermal TRAIN response addendum lost its native prediction.")
            response_value = self._response_loss(predictions.response_main, targets.response)
            if replay_weight:
                if predictions.response_full is None:
                    raise RuntimeError("Adaptive Thermal response training requires the paid full replay.")
                response_value = (1.0 - replay_weight) * response_value + replay_weight * self._response_loss(
                    predictions.response_full, targets.response)
            result["response"] = LossTerm(response_value, 1.0, self.response_coefficient)
        aux = predictions.auxiliary
        if "base_numerator" not in aux:
            raise RuntimeError("Thermal refinement core omitted cheap-path approximation supervision.")
        result["base_loss"] = LossTerm(aux["base_numerator"], aux["base_denominator"], self.base_loss_weight)
        if adaptive and phase == "open":
            result["router_importance_loss"] = LossTerm(
                aux["router_importance_numerator"], aux["router_importance_denominator"],
                ROUTER_IMPORTANCE_COEFFICIENT)
        if adaptive and phase in ("soft", "hard"):
            if self._expected_work_calibration is None:
                raise RuntimeError("Thermal expected-work pressure was not sealed by the TRAIN calibration hook.")
            result["expected_work"] = LossTerm(
                aux["expected_work_numerator"], aux["expected_work_denominator"], self._expected_work_weight)
        predictions.work = dict(predictions.work)
        auxiliary_state["work"] = predictions.work
        auxiliary_state["base_auxiliary"] = aux
        return result

    def validation_loss_terms(
        self,
        predictions: ThermalPredictions,
        targets: ThermalTargets,
        auxiliary_state: Mapping[str, Any],
        *,
        batch: TaskBatch,
        arm: str,
    ) -> Mapping[str, LossTerm]:
        """Measure validation terms available from the deployed hard route only."""
        del auxiliary_state, batch
        reconstruction = self._case_reconstruction(
            predictions.native_main, predictions.native_prepared, targets)["reconstruction"]
        result: dict[str, LossTerm] = {
            "reconstruction": LossTerm(reconstruction.sum(), len(targets.case_ids), 1.0),
        }
        if arm == "adaptive_detail" and self._expected_work_calibration is not None:
            response = getattr(predictions.native_prepared, "response", None)
            route = getattr(response, "refinement_aux", {})
            probability = route.get("probability") if isinstance(route, Mapping) else None
            protected = route.get("protected") if isinstance(route, Mapping) else None
            present = predictions.native_prepared.source_present
            if torch.is_tensor(probability) and torch.is_tensor(protected) and torch.is_tensor(present):
                active = (present[:, None, :] > 0).expand_as(probability)
                protected_active = protected.bool() & active
                eligible = active & ~protected_active
                numerator = (probability * eligible).sum() + protected_active.sum()
                result["expected_work"] = LossTerm(
                    numerator, active.sum(), float(self._expected_work_weight))
        return result

    def validation_batches(self):
        validation_microbatch = int(self.validation_budget["microbatch_cases"])
        for start in range(0, len(self.validation_cases), validation_microbatch):
            indices = range(start, min(start + validation_microbatch, len(self.validation_cases)))
            yield self._batch_from_indices(indices, 0, key=None, training=False, include_response=False,
                                           budget=self.validation_budget)

    @staticmethod
    def _per_case_native_metrics(
        predictions: ThermalPredictions, targets: ThermalTargets, stats: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        from thermal_source_response_fit import scalar_scale

        output = predictions.native_main
        prepared = predictions.native_prepared
        present = prepared.source_present > 0.5
        rows = []
        surface_valid = prepared.stencils["surface"].valid.reshape(output["pred_interface"].shape[:-1])
        outside_valid = prepared.stencils["outside"].valid.reshape(output["pred_interface"].shape[:-1])
        for index, case_id in enumerate(targets.case_ids):
            weight = targets.point_weights[index]
            fluid_error = output["fluid_temperature"][index, :, 0] - targets.field_targets[index, :, 4]
            fluid_mse = (fluid_error.square() * weight).sum() / weight.sum().clamp_min(1e-12)
            fluid_std = scalar_scale(stats, "field_std_by_channel", 4)
            fluid_valid = weight > 0
            fluid_rmse = torch.sqrt(fluid_mse.clamp_min(0))
            surface_mask = present[index, :, None] & surface_valid[index] & torch.isfinite(
                targets.interface_target[index, ..., 0])
            surface_error = output["pred_interface"][index, ..., 0] - targets.interface_target[index, ..., 0]
            surface_mse = (surface_error.square() * surface_mask).sum() / surface_mask.sum().clamp_min(1)
            material_mask = present[index, :, None].expand_as(targets.material_targets[index])
            material_error = output["pred_internal_temperature"][index, ..., 0] - targets.material_targets[index]
            material_mse = (material_error.square() * material_mask).sum() / material_mask.sum().clamp_min(1)
            q_mask = (present[index, :, None] & surface_valid[index] & outside_valid[index]
                      & torch.isfinite(targets.interface_target[index, ..., 1]))
            q_error = output["pred_interface"][index, ..., 1] - targets.interface_target[index, ..., 1]
            q_mse = (q_error.square() * q_mask).sum() / q_mask.sum().clamp_min(1)
            module_peak_errors = []
            for module in range(present.shape[1]):
                if not bool(present[index, module]):
                    continue
                target_peak = targets.material_targets[index, module].max()
                pred_peak = output["pred_internal_temperature"][index, module, :, 0].max()
                module_peak_errors.append((pred_peak - target_peak).square())
            peak_mse = torch.stack(module_peak_errors).mean() if module_peak_errors else fluid_mse.new_zeros(())
            standardized = ((fluid_mse / fluid_std**2) +
                            (surface_mse / scalar_scale(stats, "interface_targets_std", 0)**2) +
                            (material_mse / scalar_scale(stats, "internal_temperature_std")**2)) / 3
            rows.append({
                "case_id": case_id,
                "module_count": int(present[index].sum().item()),
                "fluid_temperature_rmse": float(fluid_rmse.detach()),
                "surface_temperature_rmse": float(torch.sqrt(surface_mse.clamp_min(0)).detach()),
                "material_temperature_rmse": float(torch.sqrt(material_mse.clamp_min(0)).detach()),
                "module_peak_rmse": float(torch.sqrt(peak_mse.clamp_min(0)).detach()),
                "q_proxy_rmse": float(torch.sqrt(q_mse.clamp_min(0)).detach()),
                "fluid_temperature_mse_standardized": float((fluid_mse / fluid_std**2).detach()),
                "surface_temperature_mse_standardized": float((surface_mse / scalar_scale(stats, "interface_targets_std", 0)**2).detach()),
                "material_temperature_mse_standardized": float((material_mse / scalar_scale(stats, "internal_temperature_std")**2).detach()),
                "field_score": float(standardized.detach()),
                "fluid_valid_rows": int(fluid_valid.sum().item()),
                "surface_valid_rows": int(surface_mask.sum().item()),
                "material_valid_rows": int(material_mask.sum().item()),
                "temperature_unit": "packed_dataset_native_temperature",
            })
        return rows

    def validation_metrics(self, predictions, targets, auxiliary_state):
        return {"case_rows": self._per_case_native_metrics(predictions, targets, self.stats)}

    def reduce_native_metrics(self, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        rows = [row for record in records for row in record["case_rows"]]
        ids = [row["case_id"] for row in rows]
        if len(rows) != 22 or len(set(ids)) != 22 or set(ids) != set(self._validation_ids):
            raise ValueError("Thermal validation must expose the exact fixed25_v1 DEV22 membership once each.")
        result: dict[str, Any] = {"scope": "fixed25_v1 exposed DEV22; equal-case sampled native roles",
                                  "case_count": 22, "case_ids": ids,
                                  "temperature_unit": "packed_dataset_native_temperature",
                                  "field_score": float(np.mean([row["field_score"] for row in rows])),
                                  "per_case": rows}
        for metric in ("fluid_temperature_rmse", "surface_temperature_rmse", "material_temperature_rmse",
                       "module_peak_rmse", "q_proxy_rmse"):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_p90"] = float(np.quantile(values, 0.9))
        for metric in ("fluid_temperature_mse_standardized", "surface_temperature_mse_standardized",
                       "material_temperature_mse_standardized"):
            result[f"{metric}_mean"] = float(np.mean([row[metric] for row in rows]))
        return result

    def _measure_response_set(
        self,
        model: ThermalSourceResponse,
        response_inputs: Sequence[Mapping[str, Any]],
        *,
        execution_mode: str,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        original_training = model.training
        model.eval()
        if hasattr(model.core, "set_execution"):
            model.core.set_execution(mode=execution_mode, phase="hard", temperature=temperature,
                                     threshold=0.5, training_signal=False,
                                     gate_version=self.gate_version,
                                     gate_transition=self.gate_transition)
        role_values = {name: [] for name in ("fluid", "surface", "material")}
        family_rows = []
        try:
            with torch.no_grad():
                for sample in response_inputs:
                    device = next(model.parameters()).device
                    structure = _move_tree(sample["structure"], device)
                    prepared = model.prepare_native(structure, sample["fluid_xy"].to(device),
                        local_query_points=sample["local"].to(device), ntheta=16)
                    prediction = model.apply_native(prepared, sample["heat"].to(device), increment=True)
                    fluid_error = prediction["fluid_temperature"][0, :, 0] - sample["fluid_target"][0].to(device)
                    surface_error = prediction["pred_interface"][0, ..., 0] - sample["surface_target"][0].to(device)
                    material_error = prediction["pred_internal_temperature"][0, ..., 0] - sample["material_target"][0].to(device)
                    fluid_rmse = float(fluid_error.square().mean().sqrt().cpu())
                    surface_mask = sample["surface_mask"][0].to(device)
                    material_mask = sample["material_mask"][0].to(device)
                    surface_rmse = float(((surface_error.square() * surface_mask).sum() /
                                          surface_mask.sum().clamp_min(1)).sqrt().cpu())
                    material_rmse = float(((material_error.square() * material_mask).sum() /
                                           material_mask.sum().clamp_min(1)).sqrt().cpu())
                    family_rows.append({"family_id": sample["family_id"], "fluid_rmse": fluid_rmse,
                                        "surface_rmse": surface_rmse, "material_rmse": material_rmse})
                    role_values["fluid"].append(fluid_rmse)
                    role_values["surface"].append(surface_rmse)
                    role_values["material"].append(material_rmse)
        finally:
            model.train(original_training)
            if hasattr(model.core, "reset_auxiliary"):
                model.core.reset_auxiliary()
        means = {role: float(np.mean(values)) for role, values in role_values.items()}
        floors = {role: max(0.01 * float(self.response_scales[role]), 1.0e-12) for role in means}
        ratios = {role: means[role] / max(self._parent_response_baseline.get("means", means)[role], floors[role])
                  for role in means} if hasattr(self, "_parent_response_baseline") else {role: 1.0 for role in means}
        return {"means": means, "family_rows": family_rows, "ratio_floors": floors,
                "ratios": ratios, "max_ratio": float(max(ratios.values()))}

    def extra_validation_metrics(self, model: nn.Module, arm: str, phase: str) -> Mapping[str, Any]:
        if not isinstance(model, ThermalSourceResponse):
            raise TypeError("Thermal response guard received another task model.")
        if self._parent_response_baseline is None:
            return {}
        current = self._measure_response_set(model, self._development_response_inputs,
                                             execution_mode="all_fine" if arm in ("warmup", "full_detail") else "adaptive")
        parent_means = self._parent_response_baseline["means"]
        floors = {role: max(0.01 * float(self.response_scales[role]), 1.0e-12) for role in current["means"]}
        ratios = {role: current["means"][role] / max(parent_means[role], floors[role]) for role in current["means"]}
        return {
            "response_development_scope": "four existing TRAIN response-development families; sampled fixed receivers",
            "response_development_family_ids": list(DEVELOPMENT_RESPONSE_IDS),
            "response_development_rmse_mean": current["means"],
            "response_development_rmse_parent": parent_means,
            "response_development_ratio": ratios,
            "response_guard_max_ratio": float(max(ratios.values())),
            "response_guard_limit": 1.10,
            "response_guard_floors_from_train_scales": floors,
            "response_development_family_rows": current["family_rows"],
            "validation_route_phase": "hard",
            "validation_execution_mode": arm,
            "response_reference": "existing stored analytic-wake/shared-grid records; no new solves",
        }

    def optimizer_groups(self, model: nn.Module, arm: str, stage: str) -> Sequence[OptimizerGroupSpec]:
        del arm, stage
        fine_names, refinement_names = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            (refinement_names if name.startswith("core.refinement.") else fine_names).append(name)
        if not fine_names or not refinement_names:
            raise ValueError("Thermal task must retain the parent fine parameters and add one refinement group.")
        if self.development_refinement_parent is not None:
            # This is a fresh 500-epoch child clock with inherited moments.
            # Both parameter groups use the same constant LR by design.
            fine_schedule = refinement_schedule = ScheduleSpec(
                peak_lr=THERMAL_TRANSFER_LR, warmup_start_lr=THERMAL_TRANSFER_LR,
                warmup_epochs=0, hold_through_epoch=THERMAL_TRANSFER_EPOCHS - 1,
                total_epochs=THERMAL_TRANSFER_EPOCHS, final_lr=THERMAL_TRANSFER_LR)
        else:
            # Preserve the sealed default identity and schedule for every
            # non-transfer Thermal run.
            fine_schedule = ScheduleSpec(
                peak_lr=5.0e-5, warmup_start_lr=3.0e-6, warmup_epochs=20,
                hold_through_epoch=1000, total_epochs=2500, final_lr=3.0e-6)
            refinement_schedule = ScheduleSpec(
                peak_lr=3.0e-4, warmup_start_lr=3.0e-4, warmup_epochs=0,
                hold_through_epoch=1000, total_epochs=2500, final_lr=3.0e-6)
        weight_decay = float(self.recipe["weight_decay"])
        return (
            OptimizerGroupSpec("thermal_fine", tuple(fine_names), fine_schedule, weight_decay=weight_decay),
            OptimizerGroupSpec("thermal_refinement", tuple(refinement_names), refinement_schedule,
                               weight_decay=weight_decay),
        )

    def work_counts(self, batch: TaskBatch, predictions: ThermalPredictions, auxiliary_state: Mapping[str, Any]) -> Mapping[str, int | float]:
        del batch
        return {name: int(auxiliary_state.get("work", predictions.work).get(name, 0)) for name in (
            "cheap_rows", "fine_rows", "near_rows", "gate_rows", "active_pairs", "padded_fine_capacity",
            "selected_detail_rows", "native_fluid_queries", "native_fluid_valid_queries",
            "native_surface_queries", "native_surface_valid_queries", "native_material_queries",
            "native_material_valid_queries", "native_grid_rows", "response_fluid_queries",
            "response_fluid_valid_queries", "response_surface_queries", "response_surface_valid_queries",
            "response_material_queries", "response_material_valid_queries", "response_grid_rows",
            "operator_rows", "operator_neural_stencil_receiver_rows", "operator_unique_stencil_receiver_rows")}

    def _calibration_batches(self) -> TaskBatch:
        ids = [str(value) for value in self.recipe["calibration"]["calibration_case_ids"]]
        indices = [self._train_index_by_id[value] for value in ids]
        key = SamplingKey(0, 600, 0, 0, "thermal_cost_calibration", "adaptive_detail")
        return self._batch_from_indices(indices, 600, key=key, training=True, include_response=False,
                                        budget=self.calibration_budget)

    def calibrate_expected_work_weight(self, model: ThermalSourceResponse) -> Mapping[str, Any]:
        if self._expected_work_calibration is not None:
            raise RuntimeError("Thermal expected-work coefficient was already calibrated.")
        batch = self._calibration_batches()
        scene = self.make_scene(batch.scene_inputs)
        model.train(True)
        model.core.reset_auxiliary()
        self._set_execution(model, "adaptive_detail", "open", 1.0, True)
        rx = _move_tree(batch.receivers, self.device)
        targets = batch.targets
        prepared = model.prepare_native(scene.structure, rx.fluid_xy, local_query_points=rx.local_query_points, ntheta=16)
        output = model.apply_native(prepared, rx.heat)
        native = self._case_reconstruction(output, prepared, targets)["reconstruction"].mean()
        auxiliary = model.core.auxiliary_terms()
        model.core.reset_auxiliary()
        router = tuple(model.core.refinement.router.parameters())
        if not router:
            raise ValueError("Thermal cost calibration found no trainable router parameters.")
        native_grads = torch.autograd.grad(native, router, retain_graph=True, allow_unused=True)
        if not torch.is_tensor(auxiliary.get("expected_work_numerator")):
            raise ValueError("Thermal TRAIN expected-work numerator is missing at calibration.")
        expected = auxiliary["expected_work_numerator"] / auxiliary["expected_work_denominator"].clamp_min(1)
        work_grads = torch.autograd.grad(expected, router, allow_unused=True)
        norm = lambda values: math.sqrt(sum(float(item.detach().double().square().sum().cpu())
                                            for item in values if item is not None))
        native_norm, work_norm = norm(native_grads), norm(work_grads)
        if not math.isfinite(native_norm) or not math.isfinite(work_norm):
            raise FloatingPointError("Thermal TRAIN expected-work calibration produced nonfinite router gradients.")
        if native_norm <= 0 or work_norm <= 0:
            coefficient = 0.0
            reason = "native or expected-work router gradient is zero; no division performed"
        else:
            coefficient = min(EXPECTED_WORK_WEIGHT_CAP, EXPECTED_WORK_TARGET_SHARE * native_norm / work_norm)
            reason = "single post-open TRAIN router-gradient ratio calibration"
        self._expected_work_weight = float(coefficient)
        receipt = {
            "stage": "thermal_expected_work_calibration",
            "absolute_epoch": 601,
            "arm": "adaptive_detail",
            "panel_case_ids": list(batch.targets.case_ids),
            "query_seed": {"seed": 0, "epoch": 600, "update": 0, "microbatch": 0,
                            "arm_excluded": True},
            "router_parameter_count": sum(parameter.numel() for parameter in router),
            "native_router_gradient_norm": native_norm,
            "unit_expected_work_router_gradient_norm": work_norm,
            "target_gradient_share": EXPECTED_WORK_TARGET_SHARE,
            "coefficient_cap": EXPECTED_WORK_WEIGHT_CAP,
            "coefficient_uncapped": None if native_norm <= 0 or work_norm <= 0 else EXPECTED_WORK_TARGET_SHARE * native_norm / work_norm,
            "coefficient": coefficient,
            "achieved_gradient_share": 0.0 if native_norm <= 0 else coefficient * work_norm / native_norm,
            "reason": reason,
            "stored_uv_used_for_calibration": False,
            "validation_values_read": False,
        }
        self._expected_work_calibration = receipt
        return receipt

    def on_phase_start(self, *, model: nn.Module, arm: str, epoch: int, phase: str, temperature: float):
        receipt = {"arm": arm, "epoch": int(epoch), "phase": phase, "temperature": float(temperature)}
        if arm == "adaptive_detail" and int(epoch) == 601 and phase == "soft":
            receipt["expected_work_calibration"] = dict(self.calibrate_expected_work_weight(model))
        elif self._expected_work_calibration is not None:
            receipt["expected_work_coefficient"] = float(self._expected_work_weight)
        return receipt

    def training_state_dict(self) -> Mapping[str, Any]:
        return {"expected_work_weight": self._expected_work_weight,
                "expected_work_calibration": copy.deepcopy(self._expected_work_calibration),
                "temperature_std": self.temperature_std,
                "base_loss_weight": self.base_loss_weight}

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        if not math.isclose(float(state.get("temperature_std", self.temperature_std)), self.temperature_std,
                            rel_tol=0.0, abs_tol=0.0):
            raise ValueError("Saved Thermal fixed TRAIN temperature scale changed.")
        weight = float(state.get("expected_work_weight", 0.0))
        if not math.isfinite(weight) or not 0 <= weight <= EXPECTED_WORK_WEIGHT_CAP:
            raise ValueError("Saved Thermal expected-work coefficient exceeds its sealed cap.")
        self._expected_work_weight = weight
        saved = state.get("expected_work_calibration")
        self._expected_work_calibration = None if saved is None else dict(saved)

    def preparation_summary(self) -> dict[str, Any]:
        return {
            "subset_id": "fixed25_v1",
            "manifest_fingerprint": self.manifest["manifest_sha256"],
            "training_cases": len(self.training_cases),
            "validation_cases": len(self.validation_cases),
            "training_response_families": list(TRAIN_RESPONSE_IDS),
            "development_response_families": list(DEVELOPMENT_RESPONSE_IDS),
            "operator_residual_enabled": self.use_operator,
            "operator_residual_training_rows_per_case": OPERATOR_ROWS_PER_CASE if self.use_operator else 0,
            "inference_scene_keys": list(CONTEXT_KEYS),
            "heat_and_stored_uv_in_scene": False,
            "stored_uv_supervision_only": self.use_operator,
            "solver_attempts": 0,
        }


def create_task(config: Mapping[str, Any]) -> tuple[nn.Module, ThermalRefinementTask, Mapping[str, Any]]:
    """Load exact retained parent bindings and make the refined child model."""

    config = dict(config)
    development_parent_path, development_parent_sha256 = _resolve_development_refinement_parent_binding(config)
    from thermal_source_response_fit import build_balances, read_primary

    from honf_runtime.compat import load_trusted_checkpoint, set_seed

    parent_path = Path(config.get("parent_checkpoint", DEFAULT_PARENT_CHECKPOINT)).expanduser().resolve()
    flow_path = Path(config.get("flow_checkpoint", DEFAULT_FLOW_CHECKPOINT)).expanduser().resolve()
    atlas_directory = Path(config.get("atlas_directory", DEFAULT_ATLAS_DIRECTORY)).expanduser().resolve()
    if not parent_path.is_file() or not flow_path.is_file():
        raise FileNotFoundError("Retained R-direct2500 and D-sep2500 parents must both exist.")
    parent_sha = _sha256(parent_path)
    flow_sha = _sha256(flow_path)
    if (config.get("require_canonical_parent", True)
            and (parent_sha != DEFAULT_PARENT_SHA256 or flow_sha != DEFAULT_FLOW_SHA256)):
        raise ValueError("Thermal refinement must use the exact retained R-direct2500/D-sep2500 parents.")
    parent = load_trusted_checkpoint(parent_path, map_location="cpu")
    flow = load_trusted_checkpoint(flow_path, map_location="cpu")
    development_parent = None
    development_identity = None
    development_source_provider_identity = None
    if development_parent_path is not None:
        development_parent = load_trusted_checkpoint(development_parent_path, map_location="cpu")
        development_identity, development_source_provider_identity, _ = (
            _validate_development_refinement_parent_header(
                development_parent,
                path=development_parent_path,
                expected_sha256=development_parent_sha256,
            )
        )
    if (parent.get("epoch") != 2500 or parent.get("fit_identity", {}).get("mode") != "direct"
            or flow.get("epoch") != 2500 or flow.get("dependency_policy") != "D-sep"):
        raise ValueError("Thermal refinement parents must be the literal R-direct2500 and D-sep2500 development endpoints.")
    recipe = parent["fit_identity"]["recipe"]
    if (recipe.get("manifest_sha256") != FIXED25_FINGERPRINT
            or recipe.get("flow_checkpoint_sha256") != flow_sha
            or recipe.get("operator_decision", {}).get("operator_constraint") != "qualified"
            or recipe.get("calibration", {}).get("operator_constraint") != "qualified"):
        raise ValueError("Retained parent does not bind the expected fixed25 qualified operator objective and flow partner.")
    if flow.get("dependency_identity") != "thermal_dependency_flow_v1":
        raise ValueError("Retained D-sep endpoint has an unexpected Thermal flow identity.")
    training_budget, query_budget_override = _resolve_query_budget(
        recipe["budget"], config.get("query_budget_override"))
    provider_recipe = dict(recipe)
    provider_recipe["budget"] = training_budget
    validation_budget = dict(recipe["budget"]) if query_budget_override is not None else None
    calibration_budget = dict(recipe["budget"]) if query_budget_override is not None else None
    training_cases, manifest = read_primary(parent, "train")
    validation_cases, validation_manifest = read_primary(parent, "test")
    if manifest["manifest_sha256"] != validation_manifest["manifest_sha256"] or manifest["manifest_sha256"] != FIXED25_FINGERPRINT:
        raise ValueError("Thermal training and exposed validation must share fixed25_v1 membership.")
    train_families = _read_response_families(atlas_directory, TRAIN_RESPONSE_IDS)
    development_families = _read_response_families(atlas_directory, DEVELOPMENT_RESPONSE_IDS)
    set_seed(int(config.get("seed", 0)))
    parent_model = ThermalSourceResponse(parent["source_response_config"]["core"],
        **parent["source_response_config"]["adapter"])
    parent_model.load_state_dict(parent["thermal_state_dict"], strict=True)
    optimizer_seed = _parent_optimizer_seed(parent_model, parent)
    child_model = copy.deepcopy(parent_model)
    temperature_std = float(np.asarray(parent["global_normalization_stats"]["field_std_by_channel"]).reshape(-1)[4])
    refined = RefinedSourceResponseOperator.from_fine(
        child_model.core, base_width=16, router_hidden=32, residual_scale=temperature_std)
    refined.refinement.residual_scale.fill_(temperature_std)
    child_model.core = refined
    child_model.core_config = copy.deepcopy(refined.config)
    parent_names = [name for name, _ in parent_model.named_parameters()]
    child_names = [name for name, _ in child_model.named_parameters()]
    if child_names[:len(parent_names)] != parent_names or len(child_names) <= len(parent_names):
        raise ValueError("Refinement must preserve every parent parameter name and append new small parameters.")
    if development_parent is not None:
        # This is a fresh 500-epoch child age. The exact source model is loaded
        # strictly; its historical epoch count is provenance, not the child's
        # optimizer or sampler clock.
        child_model.load_state_dict(development_parent["model_state_dict"], strict=True)
    balances = build_balances(training_cases) if recipe["operator_decision"]["operator_constraint"] == "qualified" else []
    provider = ThermalRefinementTask(
        model=child_model,
        parent_model=parent_model,
        parent=parent,
        parent_path=parent_path,
        flow_path=flow_path,
        atlas_directory=atlas_directory,
        training_cases=training_cases,
        validation_cases=validation_cases,
        manifest=manifest,
        train_families=train_families,
        development_families=development_families,
        balances=balances,
        optimizer_seed=optimizer_seed,
        device=config.get("device", "cpu"),
        recipe=provider_recipe,
        validation_budget=validation_budget,
        calibration_budget=calibration_budget,
        gate_version=str(config.get("gate_version", THERMAL_GATE_HARD_VERSION)),
        gate_transition=config.get("gate_transition"),
        development_refinement_parent=(
            None if development_parent is None else {
                "path": str(development_parent_path),
                "sha256": str(development_parent_sha256),
                "epoch": int(development_parent["epoch"]),
                "arm": str(development_parent["arm"]),
                "run_id": str(development_identity["run_id"]),
                "development_profile": str(development_identity["development_profile"]),
                "source_provider_identity_sha256": _json_sha256(development_source_provider_identity),
            }
        ),
    )
    if development_parent is not None:
        development_optimizer_seed, moments_sha256 = _development_optimizer_seed(
            development_parent, child_model, provider)
        development_provider_state = development_parent["provider_training_state"]
        provider_state_sha256 = _validate_inherited_provider_training_state(
            development_provider_state,
            temperature_std=provider.temperature_std,
            base_loss_weight=provider.base_loss_weight,
        )
        source_normalization_sha256 = development_source_provider_identity.get("normalization_stats_sha256")
        if source_normalization_sha256 != _tensor_digest(provider.stats):
            raise ValueError("Thermal development parent normalization differs from the active fixed25 provider.")
        provider.development_optimizer_seed = development_optimizer_seed
        provider.development_parent_rng_state = copy.deepcopy(development_parent["rng_state"])
        provider.development_refinement_parent.update({
            "optimizer_moment_state_by_name_sha256": moments_sha256,
            "optimizer_moment_parameter_count": len(development_optimizer_seed["state_by_name"]),
            "provider_training_state_sha256": provider_state_sha256,
            "inherited_expected_work_weight": float(development_provider_state["expected_work_weight"]),
            "inherited_calibration_epoch": int(
                development_provider_state["expected_work_calibration"]["absolute_epoch"]
            ),
            "child_schedule": {
                "age_origin": 1,
                "epochs": THERMAL_TRANSFER_EPOCHS,
                "learning_rate": THERMAL_TRANSFER_LR,
                "both_parameter_groups": True,
            },
        })
        provider.load_training_state_dict(development_provider_state)
        active_provider_identity = provider.identity_payload()
        _validate_development_provider_identity(
            development_source_provider_identity, active_provider_identity)
        if ("python" not in provider.development_parent_rng_state
                or "numpy" not in provider.development_parent_rng_state
                or "torch_cpu" not in provider.development_parent_rng_state):
            raise ValueError("Thermal development parent RNG state is incomplete for exact matched initialization.")
        restore_rng_state(provider.development_parent_rng_state)
    return child_model, provider, provider.training_optimizer_seed
