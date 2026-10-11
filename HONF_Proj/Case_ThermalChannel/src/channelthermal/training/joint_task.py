"""Fixed-cohort Thermal provider for fresh joint regional field learning."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from honf_runtime.unified_training import LossTerm, OptimizerGroupSpec, SamplingKey, ScheduleSpec, TaskBatch

from ..data.datasets import GlobalChannelThermalDataset, H5Normalizer, fit_global_normalizer
from ..data.development_split import (
    development_case_ids,
    read_case_catalog,
    validate_development_manifest,
)
from ..flow_curl import NATIVE_CURL_READOUT_LAW
from ..joint_regional import (
    INTERACTION_PRESERVING_THERMAL_MODES,
    JOINT_THERMAL_CHANNEL_ORDER,
    JOINT_THERMAL_ID,
    JOINT_THERMAL_MODES,
    InteractionPreservingThermalAdapter,
    JointThermalRegionalAdapter,
)
from .unified_task import (
    DEFAULT_ATLAS_DIRECTORY,
    FIXED25_FINGERPRINT,
    OPERATOR_ROWS_PER_CASE,
    Q_PROXY_COEFFICIENT,
    RESPONSE_SURFACE_STRIDE,
    TRAIN_RESPONSE_IDS,
    ThermalPredictions,
    ThermalRefinementTask,
    _operator_row_indices,
    _operator_stencil_receivers,
    _read_response_families,
    _residual_from_stencil_kernel,
    _tensor_digest,
)

JOINT_TRAINING_MODE = "joint"
THERMAL_RECOVERY_MODES = tuple(JOINT_THERMAL_MODES) + tuple(INTERACTION_PRESERVING_THERMAL_MODES)
JOINT_MICROBATCH_DEFAULT = 4
JOINT_EFFECTIVE_BATCH_DEFAULT = 48
JOINT_PRIMARY_QUERY_DEFAULT = 1024
JOINT_MANIFEST_DEFAULT = Path("/data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json")
JOINT_SOURCE_ROOT = Path(__file__).resolve().parents[4]
JOINT_TOOLS = JOINT_SOURCE_ROOT / "tools"
JOINT_NEAR_RADIUS_MULTIPLE = 2.0
FULL_TRAIN_FOLLOWUP_PROTOCOL = "interaction_preserving_full_train_followup1000_v1"
FULL_TRAIN_FOLLOWUP_DATASET = "thermal_original600_train_canonical89_validation_followup_v1"
FULL_TRAIN_FOLLOWUP_TRAIN_SHA256 = "3e255541359ec6551863bbcd122591b782d075b4477145c63d4d5f4f7adb3b25"
FULL_TRAIN_FOLLOWUP_VALIDATION_SHA256 = "1c33b4cc5ebddb1720a6ba6adcd9ad88beb988302008c58623cf55b2fdea41ab"


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _case_ids_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8") + b"\0")
    return digest.hexdigest()


def validate_full_train_followup_membership(
    training_ids: Sequence[str], validation_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate the full-follow-up split using IDs only, before target loading or norm fitting."""
    train = tuple(map(str, training_ids))
    validation = tuple(map(str, validation_ids))
    if len(train) != 600 or len(set(train)) != 600:
        raise ValueError("Thermal full follow-up requires exactly 600 unique original TRAIN IDs.")
    if len(validation) != 89 or len(set(validation)) != 89 or "0273" in validation:
        raise ValueError("Thermal full follow-up requires the canonical89 validation IDs without 0273.")
    if set(train).intersection(validation):
        raise ValueError("Thermal full follow-up TRAIN and exposed validation IDs must be disjoint.")
    train_hash, validation_hash = _case_ids_hash(train), _case_ids_hash(validation)
    if (train_hash != FULL_TRAIN_FOLLOWUP_TRAIN_SHA256
            or validation_hash != FULL_TRAIN_FOLLOWUP_VALIDATION_SHA256):
        raise ValueError("Thermal follow-up TRAIN/validation IDs differ from the sealed full-population contract.")
    return {
        "training_case_count": len(train),
        "training_membership_sha256": train_hash,
        "validation_case_count": len(validation),
        "validation_membership_sha256": validation_hash,
        "validation_targets_are_original_h5_test_split_exposed": True,
        "unexposed_original_test_targets_read": False,
    }


def _shared_direct_state_sha256(model: torch.nn.Module) -> str:
    """Hash only the common pair/source-read path shared by P, P-G and P-H."""
    mode_specific = (
        "collective_", "source_membership_score", "environment_membership_score",
        "receiver_query", "edge_key", "access_geometry",
    )
    digest = hashlib.sha256()
    names = [name for name in model.state_dict() if not any(part in name for part in mode_specific)]
    if not names or any(not name.startswith("core.") for name in names):
        raise ValueError("Thermal fresh-P calibration could not identify the common direct-core state.")
    for name in sorted(names):
        tensor = model.state_dict()[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stored_binary_mask(value: Any, *, expected_shape: tuple[int, int], source: str) -> tuple[np.ndarray, str, str]:
    """Validate one stored geometry mask and hash its logical 0/1 cells."""
    raw = np.asarray(value)
    if raw.ndim != 2 or tuple(raw.shape) != tuple(expected_shape):
        raise ValueError(f"{source} must have the declared native grid shape {expected_shape}.")
    if raw.dtype.kind not in "bui" or not np.isin(raw, (0, 1)).all():
        raise ValueError(f"{source} must contain only stored boolean/0/1 geometry values.")
    mask = np.ascontiguousarray(raw.astype(bool, copy=True))
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(mask.shape), "logical_dtype": "bool"},
                             sort_keys=True, separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    return mask, digest.hexdigest(), str(raw.dtype)


def _read_fixed_native_mask_catalog(
    data_path: Path, training_ids: Sequence[str], validation_ids: Sequence[str], *,
    expected_shape: tuple[int, int] = (64, 128), full_train_followup1000: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read only stored geometry masks for the selected sealed population."""
    expected_counts = (600, 89) if full_train_followup1000 else (150, 22)
    if (len(training_ids), len(validation_ids)) != expected_counts:
        scope = "full TRAIN600/canonical89" if full_train_followup1000 else "fixed25 TRAIN150/DEV22"
        raise ValueError(f"The native-curl mask catalog requires exactly the sealed {scope} IDs.")
    ordered_ids = tuple(map(str, training_ids)) + tuple(map(str, validation_ids))
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("Fixed25 native mask IDs must be unique across TRAIN and DEV.")
    masks: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    stored_dtypes: dict[str, str] = {}
    with h5py.File(data_path, "r") as packed:
        cases = packed["cases"]
        for case_id in ordered_ids:
            if case_id not in cases or "module_mask" not in cases[case_id]:
                raise ValueError(f"Bound H5 is missing geometry-only cases/{case_id}/module_mask.")
            mask, mask_sha256, stored_dtype = _stored_binary_mask(
                cases[case_id]["module_mask"][...], expected_shape=expected_shape,
                source=f"cases/{case_id}/module_mask",
            )
            masks[case_id] = mask
            hashes[case_id] = mask_sha256
            stored_dtypes[case_id] = stored_dtype
    split_records = {
        "training_case_ids": list(map(str, training_ids)),
        "validation_case_ids": list(map(str, validation_ids)),
        "mask_sha256_by_case_id": hashes,
        "stored_dtype_by_case_id": stored_dtypes,
    }
    binding = {
        "source": "bound packed H5 cases/<case_id>/module_mask only",
        "target_data_keys_read": [],
        "mask_shape_ny_nx": list(expected_shape),
        **split_records,
        "catalog_sha256": _json_hash(split_records),
    }
    if full_train_followup1000:
        binding["dataset_protocol"] = FULL_TRAIN_FOLLOWUP_DATASET
        binding["training_membership_sha256"] = _case_ids_hash(training_ids)
        binding["validation_membership_sha256"] = _case_ids_hash(validation_ids)
    return masks, binding


def _last_indexed_frame_name(index_path: Path) -> str:
    """Read only the final indexed filename; do not load frame outputs or row metrics."""
    with index_path.open("rb") as stream:
        header = stream.readline().decode("utf-8").rstrip("\r\n").split(",")
        try:
            filename_index = header.index("file")
        except ValueError as error:
            raise ValueError(f"Solver frame index lacks its filename column: {index_path}") from error
        stream.seek(0, 2)
        position = stream.tell()
        if position <= 0:
            raise ValueError(f"Solver frame index is empty: {index_path}")
        stream.seek(position - 1)
        if stream.read(1) == b"\n":
            position -= 1
        while position > 0:
            stream.seek(position - 1)
            if stream.read(1) == b"\n":
                break
            position -= 1
        stream.seek(position)
        last = stream.readline().decode("utf-8").rstrip("\r\n").split(",")
    if filename_index >= len(last):
        raise ValueError(f"Final solver frame index row is malformed: {index_path}")
    filename = last[filename_index]
    if not filename or Path(filename).name != filename or not filename.endswith(".npz"):
        raise ValueError(f"Final solver frame index contains an unsafe filename: {filename!r}")
    return filename


def _read_train_response_native_masks(
    atlas_directory: Path, families: Sequence[Mapping[str, Any]], *,
    expected_shape: tuple[int, int] = (64, 128),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read geometry masks only from fixed TRAIN response-family final solver frames."""
    masks: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    atlas_root = atlas_directory.resolve().parent
    expected_ids = tuple(TRAIN_RESPONSE_IDS)
    if tuple(str(item["family_id"]) for item in families) != expected_ids:
        raise ValueError("Response mask catalog requires the four ordered original-TRAIN families.")
    for family in families:
        family_id = str(family["family_id"])
        atlas_path = atlas_directory / f"train_{family_id}_responses.npz"
        with np.load(atlas_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["family_metadata_json"]))
        if metadata.get("source_dataset_split") != "train" or metadata.get("anchor_id") != family_id:
            raise ValueError(f"Response mask metadata is not the fixed original-TRAIN family {family_id}.")
        raw_root = (atlas_root / "raw" / "train" / family_id).resolve()
        tag = "reference_solver"
        variants = {}
        variant_masks = []
        for label in ("baseline", "heat_transfer_plus", "heat_transfer_minus"):
            pattern = f"case_atlas_{family_id}_{label}_*_{tag}"
            record_dirs = [path for path in raw_root.glob(pattern) if path.is_dir()]
            if len(record_dirs) != 1:
                raise ValueError(f"Expected exactly one stored TRAIN {family_id}/{label} solver record.")
            record_dir = record_dirs[0].resolve()
            config_path = record_dir / "case_config.json"
            index_path = record_dir / "frame_index.csv"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config_save = config.get("save", {})
            if (config_save.get("case_id") != f"atlas_{family_id}_{label}"
                    or config_save.get("tag") != tag
                    or Path(str(config_save.get("root_dir", ""))).expanduser().resolve() != raw_root):
                raise ValueError(f"Stored response mask record changed identity for {family_id}/{label}.")
            runtime = config.get("runtime", {})
            if runtime.get("converged") is not True:
                raise ValueError(f"Stored response mask record is not converged for {family_id}/{label}.")
            frame_name = _last_indexed_frame_name(index_path)
            frame_path = (record_dir / "scene" / frame_name).resolve()
            if frame_path.parent != (record_dir / "scene").resolve() or not frame_path.is_file():
                raise ValueError(f"Stored response mask frame is missing for {family_id}/{label}.")
            with np.load(frame_path, allow_pickle=False) as frame:
                if "module_mask" not in frame.files:
                    raise ValueError(f"Stored response frame lacks geometry-only module_mask for {family_id}/{label}.")
                mask, mask_sha256, stored_dtype = _stored_binary_mask(
                    frame["module_mask"], expected_shape=expected_shape,
                    source=f"{family_id}/{label}/module_mask",
                )
            variant_masks.append(mask)
            variants[label] = {
                "record_directory": str(record_dir),
                "case_config_sha256": _sha256_file(config_path),
                "frame_index_path": str(index_path),
                "final_frame_filename": frame_name,
                "final_frame_step": int(runtime.get("final_step", -1)),
                "module_mask_sha256": mask_sha256,
                "module_mask_stored_dtype": stored_dtype,
                "module_mask_shape_ny_nx": list(mask.shape),
            }
        if any(not np.array_equal(variant_masks[0], mask) for mask in variant_masks[1:]):
            raise ValueError(f"Stored baseline/plus/minus geometry masks differ for TRAIN family {family_id}.")
        fluid_valid = np.asarray(family["fluid_valid"], dtype=bool).reshape(-1)
        if not np.array_equal(~variant_masks[0].reshape(-1), fluid_valid):
            raise ValueError(f"Stored TRAIN family {family_id} response grid does not match its solver-frame mask.")
        masks[family_id] = variant_masks[0]
        provenance[family_id] = {
            "partition": "original TRAIN response family",
            "mask_source": "stored final baseline/heat-transfer-plus/heat-transfer-minus solver-frame module_mask",
            "mask_sha256": variants["baseline"]["module_mask_sha256"],
            "solver_record_variants": variants,
            "all_three_geometry_masks_bit_identical": True,
            "atlas_fluid_valid_equals_complement": True,
        }
    return masks, {
        "source": "existing TRAIN atlas raw solver frames; module_mask arrays only",
        "target_arrays_read_from_frames": [],
        "family_order": list(expected_ids),
        "families": provenance,
        "catalog_sha256": _json_hash(provenance),
    }


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str((tuple(value.shape), str(value.dtype))).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _load_data_binding(manifest: Mapping[str, Any] | str | Path | None, data_path: str | Path | None):
    if manifest is None:
        manifest_path = JOINT_MANIFEST_DEFAULT
        if not manifest_path.is_file():
            raise FileNotFoundError(f"The sealed fixed25_v1 manifest is unavailable: {manifest_path}")
        loaded = json.loads(manifest_path.read_text())
    elif isinstance(manifest, Mapping):
        manifest_path = None
        loaded = copy.deepcopy(dict(manifest))
    else:
        manifest_path = Path(manifest).expanduser().resolve()
        loaded = json.loads(manifest_path.read_text())

    bound_data_path = Path(data_path).expanduser().resolve() if data_path is not None else Path(
        loaded.get("source", {}).get("dataset_path", "")
    ).expanduser().resolve()
    if not str(bound_data_path) or not bound_data_path.is_file():
        raise FileNotFoundError(f"The fixed Thermal packed dataset is unavailable: {bound_data_path}")
    verified = validate_development_manifest(
        loaded,
        bound_data_path,
        expected_fingerprint=FIXED25_FINGERPRINT,
    )
    return verified, bound_data_path, manifest_path


def _read_selected_cases(dataset_path: Path, case_ids: Sequence[str], *, split: str, normalizer: H5Normalizer,
                         include_grid: bool) -> list[dict[str, Any]]:
    dataset = GlobalChannelThermalDataset(
        dataset_path,
        split=split,
        points_per_case=None,
        normalize_inputs=False,
        normalize_targets=False,
        random_point_sampling=False,
        seed=0,
        include_grid=include_grid,
        normalizer=normalizer,
        case_ids=case_ids,
    )
    if dataset.selected_case_ids != list(case_ids):
        raise ValueError("Joint Thermal loader changed the fixed source-case ordering.")
    cases = [dataset[index] for index in range(len(dataset))]
    for case in cases:
        present = np.asarray(case["structure"]["module_present"]) > 0.5
        if not np.array_equal(np.flatnonzero(present), np.arange(int(present.sum()))):
            raise ValueError("Joint Thermal source columns require contiguous original module slots.")
        case["structure"]["module_source_ids"] = np.arange(len(present), dtype=np.int64)
    return cases


def _load_recipe_helpers():
    if str(JOINT_TOOLS) not in sys.path:
        sys.path.insert(0, str(JOINT_TOOLS))
    from thermal_source_response_fit import build_balances, response_scales

    return build_balances, response_scales


def _gradient_norm(loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter], *, retain_graph=False) -> float:
    gradients = torch.autograd.grad(loss, tuple(parameters), allow_unused=True, retain_graph=retain_graph)
    squared = sum(float(gradient.detach().double().square().sum().cpu()) for gradient in gradients if gradient is not None)
    value = math.sqrt(squared)
    if not math.isfinite(value):
        raise FloatingPointError("Joint Thermal auxiliary calibration produced a nonfinite gradient norm.")
    return value


class JointThermalTask(ThermalRefinementTask):
    """Thermal native training records with a genuinely joint flow/T objective.

    The base class is used only for its fixed25 primary/response sampling and
    native batch records. Every refinement execution hook and loss path is
    replaced here; no parent state or frozen flow partner is accepted.
    """

    def __init__(
        self,
        *,
        model: JointThermalRegionalAdapter,
        training_cases: Sequence[Mapping[str, Any]],
        validation_cases: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
        data_path: Path,
        source_binding: Mapping[str, Any],
        train_families: Sequence[Mapping[str, Any]],
        balances: Sequence[Any],
        normalization_stats: Mapping[str, Any],
        mode: str,
        device: torch.device | str,
        budget: Mapping[str, Any],
        effective_batch_size: int,
        total_epochs: int,
        formal_full: bool,
        response_coefficient: float,
        operator_coefficient: float,
        operator_rows_per_case: int = OPERATOR_ROWS_PER_CASE,
        atlas_directory: Path = DEFAULT_ATLAS_DIRECTORY,
        auxiliary_calibration: Mapping[str, Any] | None = None,
        validation_scope: str | None = None,
        optimizer_schedule: ScheduleSpec | None = None,
        weight_decay: float | None = None,
        native_sampling_protocol: str | None = None,
        flow_readout_law: str | None = None,
        full_train_followup1000: bool = False,
    ) -> None:
        if mode not in THERMAL_RECOVERY_MODES or model.mode != mode:
            raise ValueError("Joint Thermal task mode must match its freshly initialized adapter.")
        if operator_rows_per_case != OPERATOR_ROWS_PER_CASE:
            raise ValueError("The sealed Thermal joint operator uses exactly128 rows per TRAIN case.")
        for name, value in (("response", response_coefficient), ("operator", operator_coefficient)):
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"Joint Thermal {name} coefficient must be finite and lie in [0,1].")
        if total_epochs < 2:
            raise ValueError("Joint Thermal schedule horizon must be at least two epochs.")

        self.joint_mode = mode
        self.flow_readout_law = flow_readout_law
        if self.flow_readout_law not in (None, NATIVE_CURL_READOUT_LAW):
            raise ValueError("Joint Thermal provider received an unsupported flow readout law.")
        if self.flow_readout_law and mode not in INTERACTION_PRESERVING_THERMAL_MODES:
            raise ValueError("The native-curl readout belongs only to fresh P-family providers.")
        if getattr(model, "flow_readout_law", None) != self.flow_readout_law:
            raise ValueError("Joint Thermal provider and adapter flow readout laws must match exactly.")
        self.data_path = Path(data_path).resolve()
        self.source_binding = copy.deepcopy(dict(source_binding))
        self.formal_full = bool(formal_full)
        self.full_train_followup1000 = bool(full_train_followup1000)
        self.total_epochs = int(total_epochs)
        if self.formal_full and self.full_train_followup1000:
            raise ValueError("Thermal legacy formal_full and bounded full-TRAIN follow-up identities are exclusive.")
        if self.full_train_followup1000 and (
                mode not in INTERACTION_PRESERVING_THERMAL_MODES or self.total_epochs != 5000):
            raise ValueError("Thermal full-TRAIN follow-up is a fresh P-family 5000-horizon identity.")
        self.formal_validation_scope = validation_scope
        self.formal_optimizer_schedule = optimizer_schedule
        self.formal_weight_decay = weight_decay
        self.native_sampling_protocol = native_sampling_protocol
        self.effective_batch_size = int(effective_batch_size)
        self.operator_rows_per_case = int(operator_rows_per_case)
        self.auxiliary_calibration = None if auxiliary_calibration is None else copy.deepcopy(dict(auxiliary_calibration))
        self.flow_role_weights = {name: 0.5 / 4 for name in ("u", "v", "p", "omega")}
        self.thermal_role_weights = {name: 0.5 / 3 for name in ("fluid", "surface", "material")}
        self.response_weight = float(response_coefficient)
        self.operator_weight = float(operator_coefficient)
        self.q_proxy_weight = float(Q_PROXY_COEFFICIENT)
        self.use_operator = True

        stats = copy.deepcopy(dict(normalization_stats))
        if "interface_targets_std" not in stats and "interface_target_std" in stats:
            stats["interface_targets_std"] = np.asarray(stats["interface_target_std"], dtype=np.float32).copy()
        if "interface_targets_mean" not in stats and "interface_target_mean" in stats:
            stats["interface_targets_mean"] = np.asarray(stats["interface_target_mean"], dtype=np.float32).copy()
        # The legacy task initializer remains a narrow reusable owner of the
        # native role samplers/record types; all of its learned-parent and
        # refinement behavior is bypassed by overrides below.
        recipe = {
            "budget": dict(budget),
            "calibration": {
                "response_scales": _load_recipe_helpers()[1](train_families),
                "response_coefficient": self.response_weight,
                "operator_coefficient": self.operator_weight,
            },
            "operator_decision": {
                "operator_constraint": "qualified",
                "reason": "new joint Thermal family retains the established TRAIN-only native discrete operator",
            },
            "weight_decay": 1.0e-4 if weight_decay is None else float(weight_decay),
            "manifest_sha256": manifest["manifest_sha256"],
        }
        super().__init__(
            model=model,
            parent_model=None,
            parent=None,
            parent_path=None,
            flow_path=None,
            atlas_directory=atlas_directory,
            training_cases=training_cases,
            validation_cases=validation_cases,
            manifest=manifest,
            train_families=train_families,
            development_families=(),
            balances=balances,
            optimizer_seed=None,
            device=device,
            stats=stats,
            recipe=recipe,
        )
        self.response_scales = recipe["calibration"]["response_scales"]
        self.response_coefficient = self.response_weight
        self.operator_coefficient = self.operator_weight
        self._geometry_helper = model
        self._joint_loss_metadata = self._build_loss_metadata()
        if self.joint_mode in INTERACTION_PRESERVING_THERMAL_MODES and self.auxiliary_calibration is not None:
            self._validate_auxiliary_calibration(self.auxiliary_calibration)

    def identity_payload(self) -> Mapping[str, Any]:
        train_ids = list(self._train_ids)
        validation_ids = list(self._validation_ids)
        training_hash = _case_ids_hash(train_ids)
        fixed25_train_hash = _case_ids_hash(development_case_ids(self.manifest, "train"))
        if (self.formal_full or self.full_train_followup1000) and training_hash == fixed25_train_hash:
            raise ValueError("Manual fullTRAIN identity cannot reuse fixed25_v1 quarter TRAIN membership.")
        normalization_stats = {
            str(name): np.asarray(value).tolist() for name, value in self.stats.items()
        }
        payload = {
            "task": "ThermalChannel",
            "task_provider": "thermal_joint_regional_v1",
            "training_mode": JOINT_TRAINING_MODE,
            "arm": self.joint_mode,
            "dataset_split": ("original_train_full_followup1000" if self.full_train_followup1000 else
                              "original_train_full_manual_only" if self.formal_full else "fixed25_v1"),
            "manifest_fingerprint": self.manifest["manifest_sha256"],
            "manifest_binding_scope": ("full_followup1000_full_split_and_exposed_validation" if self.full_train_followup1000 else
                                        "fixed25_v1_validation_sanity_panel_only" if self.formal_full else "fixed25_v1_training_and_validation"),
            "training_membership_scope": ("all_original_train_600_followup1000" if self.full_train_followup1000 else
                                          "all_original_train_600_manual_only" if self.formal_full else "fixed25_v1_selected_train_150"),
            "training_membership_sha256": training_hash,
            "fixed25_v1_train_membership_sha256": fixed25_train_hash,
            "training_case_ids": train_ids,
            "training_case_ids_sha256": training_hash,
            "training_case_count": len(train_ids),
            "validation_case_ids": validation_ids,
            "validation_case_ids_sha256": _case_ids_hash(validation_ids),
            "validation_case_count": len(validation_ids),
            "validation_scope": ("canonical89_exposed_validation_followup1000" if self.full_train_followup1000 else
                                 "fixed25_v1_exposed_DEV22_sanity_only" if self.formal_full else "fixed25_v1_exposed_DEV22"),
            "source_binding": copy.deepcopy(self.source_binding),
            "model_family": getattr(self.model, "family_id", getattr(self.model, "FAMILY", JOINT_THERMAL_ID)),
            "model_config": self.model.model_config(),
            "adapter_config": self.model.adapter_config(),
            "normalization_scope": ("exact_original600_TRAIN_only_followup1000" if self.full_train_followup1000 else
                                    "all_original_train_only" if self.formal_full else "selected_fixed25_train_only"),
            "normalization_stats_sha256": _tensor_digest(self.stats),
            "normalization_stats": normalization_stats,
            "native_query_budget": dict(self.budget),
            "native_metric_supports": {
                "fluid_temperature_near": f"valid fixed primary receivers within {JOINT_NEAR_RADIUS_MULTIPLE:g} local physical source radii of the nearest active module",
                "fluid_temperature_far": f"valid fixed primary receivers beyond {JOINT_NEAR_RADIUS_MULTIPLE:g} local physical source radii of every active module",
                "material_peak": "maximum over finite, in-domain sampled native material receivers for each active module",
            },
            "effective_batch_size": self.effective_batch_size,
            "operator_rows_per_case": self.operator_rows_per_case,
            "response_family_ids": [str(item["family_id"]) for item in self.train_families],
            "response_scales": dict(self.response_scales),
            "loss_weights": {
                "flow_group": {"group_weight": 0.5, "per_role": dict(self.flow_role_weights)},
                "thermal_group": {"group_weight": 0.5, "per_role": dict(self.thermal_role_weights)},
                "q_proxy": {"coefficient": self.q_proxy_weight, "inside_balanced_thermal_group": False},
                "response": {"coefficient": self.response_weight},
                "operator_residual": {"coefficient": self.operator_weight},
            },
            "auxiliary_calibration": copy.deepcopy(self.auxiliary_calibration),
            "schedule": {
                "peak_lr": 3.0e-4,
                "warmup_start_lr": 3.0e-5,
                "warmup_epochs": 20,
                "hold_through_epoch": min(1000, self.total_epochs - 1),
                "total_epochs": self.total_epochs,
                "final_lr": 3.0e-6,
                "optimizer": "AdamW",
                "gradient_clip_norm": 1.0,
            },
            "external_learned_model_dependencies": [],
            "formal_full_run_policy": "manual_only_not_launched" if self.formal_full else None,
            "solver_attempts": 0,
        }
        if self.full_train_followup1000:
            payload.update({
                "execution_protocol": FULL_TRAIN_FOLLOWUP_PROTOCOL,
                "dataset_protocol": FULL_TRAIN_FOLLOWUP_DATASET,
                "full_train_followup1000": True,
                "approved_stop_after": 1000,
                "normalization_training_membership_sha256": training_hash,
                "validation_targets_are_original_h5_test_split_exposed": True,
                "unexposed_test_targets_read": False,
            })
        if getattr(self, "formal_validation_scope", None) == "canonical89":
            payload["manifest_binding_scope"] = ("full_followup1000_full_split_and_source_metadata" if self.full_train_followup1000
                                                  else "fixed25_v1_source_metadata_only")
            payload["validation_scope"] = ("canonical89_exposed_validation_followup1000" if self.full_train_followup1000
                                           else "formal_canonical89_exposed_validation")
        schedule = getattr(self, "formal_optimizer_schedule", None)
        weight_decay = getattr(self, "formal_weight_decay", None)
        if schedule is not None:
            payload["schedule"].update(asdict(schedule))
        if schedule is not None or weight_decay is not None:
            payload["schedule"].update({"weight_decay": 1.0e-4 if weight_decay is None else weight_decay,
                                        "betas": [0.9, 0.999], "eps": 1.0e-8})
        if getattr(self, "native_sampling_protocol", None) == "baseline_formal_v1":
            payload["native_sampling_identity"] = {
                "protocol": "baseline_formal_v1", "dataset": "ThermalChannel",
                "train_membership_fingerprint": _json_hash(train_ids),
                "primary_stream": "primary_native_queries",
                "response_stream": "response_addendum",
                "validation_sampling_indices": dict(self.validation_sampling_indices),
                "validation_seed_rule": "1000 + original90_index * 104729",
                "operator_seed_rule": "originalTRAIN_index * 104729 + epoch * 1000003 + 17",
            }
        flow_readout_law = getattr(self, "flow_readout_law", None)
        if flow_readout_law is not None:
            payload["flow_readout_law"] = flow_readout_law
        return payload

    def _build_loss_metadata(self) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for role in ("u", "v", "p", "omega"):
            result[f"flow/{role}"] = {
                "panel": "native_prediction",
                "label": f"Flow {role}",
                "unit": "dimensionless scaled squared error",
                "formula": f"Per-case weighted native query MSE of {role}, divided by its selected-TRAIN standard deviation squared.",
                "weight": "0.5/4 equal flow-family share",
            }
        for role in ("fluid", "surface", "material"):
            result[f"temperature/{role}"] = {
                "panel": "native_prediction",
                "label": f"Temperature {role}",
                "unit": "dimensionless scaled squared error",
                "formula": f"Per-case native {role} MSE, divided by its selected-TRAIN standard deviation squared and its native valid-role denominator.",
                "weight": "0.5/3 equal thermal-family share",
            }
        result["q_proxy"] = {
            "panel": "native_prediction",
            "label": "Native q proxy",
            "unit": "dimensionless scaled squared error",
            "formula": "Per-case native surface-flux proxy MSE divided by its selected-TRAIN standard deviation squared.",
            "weight": "0.05 outside the equal flow/thermal group balance",
        }
        result["response"] = {
            "panel": "physics",
            "label": "TRAIN heat response",
            "unit": "dimensionless scaled squared error",
            "formula": "Equal mean of fluid, surface and material temperature-increment MSE on the existing original-TRAIN response addendum.",
            "weight": f"TRAIN-only calibrated coefficient {self.response_weight:.9g}",
        }
        result["operator_residual"] = {
            "panel": "physics",
            "label": "Native discrete thermal operator",
            "unit": "dimensionless row-normalized squared residual",
            "formula": "Per-case mean A K - B residual over 128 fixed-stratum native rows using stored TRAIN velocity only in the supervision-side operator coefficient.",
            "weight": f"TRAIN-only calibrated coefficient {self.operator_weight:.9g}",
        }
        return result

    def loss_metadata(self) -> Mapping[str, Mapping[str, str]]:
        return copy.deepcopy(self._joint_loss_metadata)

    def _prepare_response_sample(
        self, family: Mapping[str, Any], *, training: bool, key: SamplingKey | None,
        budget: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Keep physical source IDs integral through the legacy native sampler."""
        sample = super()._prepare_response_sample(family, training=training, key=key, budget=budget)
        source_ids = family["structure"].get("module_source_ids")
        if source_ids is not None:
            source_ids = np.asarray(source_ids)
            if source_ids.ndim != 1 or not np.issubdtype(source_ids.dtype, np.integer):
                raise ValueError("TRAIN response addenda require ordered integer physical source identities.")
            sample["structure"]["module_source_ids"] = torch.as_tensor(
                source_ids.copy(), dtype=torch.long
            )[None]
        return sample

    def phase_reporting(self, *, arm: str, epoch: int, phase: str) -> Mapping[str, Any]:
        if arm != self.joint_mode:
            raise ValueError("Joint Thermal provider was invoked under another experiment arm.")
        return {
            "training_mode": JOINT_TRAINING_MODE,
            "phase": str(phase),
            "epoch": int(epoch),
            "all_joint_parameters_active": True,
            "flow_and_temperature_supervised_together": True,
        }

    def loss_denominators(self, batches: Sequence[TaskBatch], phase: str, arm: str) -> Mapping[str, float]:
        del phase
        if arm != self.joint_mode:
            raise ValueError("Joint Thermal loss denominator request used the wrong arm label.")
        totals: dict[str, float] = {}
        for batch in batches:
            count = float(len(batch.targets.case_ids))
            if count <= 0:
                raise ValueError("Joint Thermal macrobatch cannot contain an empty microbatch.")
            for role in ("u", "v", "p", "omega"):
                name = f"flow/{role}"
                totals[name] = totals.get(name, 0.0) + count
            for role in ("fluid", "surface", "material"):
                name = f"temperature/{role}"
                totals[name] = totals.get(name, 0.0) + count
            totals["q_proxy"] = totals.get("q_proxy", 0.0) + count
            if self.use_operator and batch.targets.case_indices:
                totals["operator_residual"] = totals.get("operator_residual", 0.0) + count
            if batch.targets.response is not None:
                totals["response"] = totals.get("response", 0.0) + 1.0
        return totals

    def predict_native(
        self,
        model,
        scene,
        receivers,
        execution_mode,
        phase,
        epoch=0,
        temperature=1.0,
        *,
        threshold=0.5,
    ):
        del phase, epoch, temperature, threshold
        if model is not self.model or execution_mode != self.joint_mode:
            raise ValueError("Joint Thermal prediction requires its own candidate and declared mode label.")
        from honf_runtime.compat import recursive_to_device

        rx = recursive_to_device(receivers, self.device)
        main_mask_options = {}
        if self.flow_readout_law == NATIVE_CURL_READOUT_LAW:
            if rx.native_solid_mask is None:
                raise ValueError("Native-curl primary flow requires its saved geometry-only mask.")
            main_mask_options["native_solid_mask"] = rx.native_solid_mask
        main_prepared = model.prepare_native(
            scene.structure,
            rx.fluid_xy,
            local_query_points=rx.local_query_points,
            ntheta=16,
            chunk_size=model.receiver_tile,
            **main_mask_options,
        )
        main = model.apply_native(main_prepared, rx.heat)
        response_prepared = response_main = None
        if rx.response_fluid_xy is not None:
            if scene.response_structure is None or rx.response_local_query_points is None or rx.response_heat is None:
                raise ValueError("Joint Thermal response addendum is missing native input receivers or heat increment.")
            response_mask_options = {}
            if self.flow_readout_law == NATIVE_CURL_READOUT_LAW:
                if rx.response_native_solid_mask is None:
                    raise ValueError("Native-curl TRAIN response requires its saved solver-frame geometry mask.")
                response_mask_options["native_solid_mask"] = rx.response_native_solid_mask
            response_prepared = model.prepare_native(
                scene.response_structure,
                rx.response_fluid_xy,
                local_query_points=rx.response_local_query_points,
                ntheta=16,
                chunk_size=model.receiver_tile,
                **response_mask_options,
            )
            response_main = model.apply_native(response_prepared, rx.response_heat, increment=True)
        work = {
            "native_fluid_queries": int(rx.fluid_xy.shape[0] * rx.fluid_xy.shape[1]),
            "native_operator_rows": 0,
        }
        work["native_valid_fluid_queries"] = int(main_prepared.stencils["fluid"].valid.sum().item())
        work["native_surface_queries"] = int(main_prepared.stencils["surface"].valid.numel())
        work["native_material_queries"] = int(main_prepared.stencils["material"].valid.numel())
        predictions = ThermalPredictions(
            model=model,
            execution_mode=execution_mode,
            phase="joint",
            native_prepared=main_prepared,
            native_main=main,
            native_full=None,
            response_prepared=response_prepared,
            response_main=response_main,
            response_full=None,
            auxiliary={},
            work=work,
        )
        return predictions, {"work": work}

    def _native_role_losses(self, output: Mapping[str, torch.Tensor], prepared, targets):
        stats = self.stats
        field_std = output["pred_field"].new_tensor(np.asarray(stats["field_std_by_channel"], dtype=np.float32))
        query_weight = targets.point_weights.to(output["pred_field"])
        query_weight = query_weight * prepared.stencils["fluid"].valid.to(query_weight)
        predicted = output["pred_field"]
        target_fields = targets.field_targets.to(predicted)
        flow_finite = torch.isfinite(target_fields[..., :4])
        flow_errors = (predicted[..., :4] - torch.nan_to_num(target_fields[..., :4])) / field_std[:4]
        flow_weights = query_weight[..., None] * flow_finite
        flow_per_channel = (flow_errors.square() * flow_weights).sum(1) / flow_weights.sum(1).clamp_min(1.0e-12)
        result: dict[str, torch.Tensor] = {
            f"flow/{role}": flow_per_channel[:, index]
            for index, role in enumerate(("u", "v", "p", "omega"))
        }

        fluid_weight = query_weight * torch.isfinite(target_fields[..., 4])
        fluid_error = (predicted[..., 4] - torch.nan_to_num(target_fields[..., 4])) / field_std[4]
        result["temperature/fluid"] = (
            (fluid_error.square() * fluid_weight).sum(1) / fluid_weight.sum(1).clamp_min(1.0e-12)
        )

        present = prepared.source_present.to(predicted)
        surface_shape = output["pred_interface"].shape[:-1]
        surface_valid = prepared.stencils["surface"].valid.reshape(surface_shape)
        outside_valid = prepared.stencils["outside"].valid.reshape(surface_shape)
        surface_target = targets.interface_target[..., 0].to(predicted)
        surface_mask = present[..., None] * surface_valid * torch.isfinite(surface_target)
        surface_scale = max(float(np.asarray(stats["interface_targets_std"]).reshape(-1)[0]), 1.0e-6)
        surface_error = (output["pred_interface"][..., 0] - torch.nan_to_num(surface_target)) / surface_scale
        surface_per_module = (surface_error.square() * surface_mask).sum(-1) / surface_mask.sum(-1).clamp_min(1)
        active_surface_module = present * (surface_mask.sum(-1) > 0)
        result["temperature/surface"] = (
            surface_per_module * active_surface_module
        ).sum(-1) / active_surface_module.sum(-1).clamp_min(1)

        material_target = targets.material_targets.to(predicted)
        material_error = (
            output["pred_internal_temperature"][..., 0] - torch.nan_to_num(material_target)
        ) / max(float(np.asarray(stats["internal_temperature_std"]).reshape(-1)[0]), 1.0e-6)
        material_valid = (
            present[..., None]
            * prepared.stencils["material"].valid.reshape(material_target.shape).to(present.dtype)
            * torch.isfinite(material_target)
        )
        result["temperature/material"] = (
            material_error.square() * material_valid
        ).sum((1, 2)) / material_valid.sum((1, 2)).clamp_min(1)

        q_target = targets.interface_target[..., 1].to(predicted)
        q_scale = max(float(np.asarray(stats["interface_targets_std"]).reshape(-1)[1]), 1.0e-6)
        q_mask = present[..., None] * surface_valid * outside_valid * torch.isfinite(q_target)
        q_error = (output["pred_interface"][..., 1] - torch.nan_to_num(q_target)) / q_scale
        q_per_module = (q_error.square() * q_mask).sum(-1) / q_mask.sum(-1).clamp_min(1)
        q_modules = present * (q_mask.sum(-1) > 0)
        result["q_proxy"] = (q_per_module * q_modules).sum(-1) / q_modules.sum(-1).clamp_min(1)
        result["flow_group"] = torch.stack([result[f"flow/{role}"] for role in ("u", "v", "p", "omega")]).mean(0)
        result["thermal_group"] = torch.stack(
            [result[f"temperature/{role}"] for role in ("fluid", "surface", "material")]
        ).mean(0)
        return result

    def _operator_loss(self, predictions: ThermalPredictions, targets) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if not targets.case_indices:
            return None, {}
        from channelthermal.source_response_residual import DiscreteThermalBalance

        prepared = predictions.native_prepared
        device = prepared.context.centers.device
        coefficients = torch.cat([self.balances[index].coefficients for index in targets.case_indices]).to(device)
        slots = torch.cat([self.balances[index].source_slots for index in targets.case_indices]).to(device)
        balance = DiscreteThermalBalance(coefficients, slots, prepared.source_present)
        rows = _operator_row_indices(self.balances, targets.case_indices, targets.epoch, device)
        query, receiver_ids, gather, stencil, ix, iy = _operator_stencil_receivers(
            balance, rows, prepared.context.lengths, self.model.nx, self.model.ny
        )
        prepared_read = self.model.core.prepare_receivers(
            prepared.context,
            query,
            receiver_ids=receiver_ids,
            chunk_size=self.model.receiver_tile,
        )
        kernel = prepared_read.dense_kernel()[..., 0]
        residual = _residual_from_stencil_kernel(kernel, balance, rows, gather, stencil, ix, iy)
        temperature_scale = max(float(np.asarray(self.stats["field_std_by_channel"]).reshape(-1)[4]), 1.0e-6)
        residual = residual / temperature_scale
        denominator = (prepared.source_present.sum(-1) * self.operator_rows_per_case).clamp_min(1)
        per_case = (residual.square() * prepared.source_present[:, None]).sum((1, 2)) / denominator
        receipt = {
            "operator_rows": int(rows.numel()),
            "rows_per_case": self.operator_rows_per_case,
            "kernel_source_columns": int(balance.present.sum().item()),
            "unique_stencil_receiver_rows": int(sum(torch.unique(stencil[index]).numel() for index in range(stencil.shape[0]))),
            "physical_solves": 0,
            "uses_stored_velocity_only_for_supervision_coefficients": True,
        }
        return per_case, receipt

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        del phase
        if predictions.model is not self.model or predictions.execution_mode != self.joint_mode:
            raise ValueError("Joint Thermal objective received predictions from another mode/model.")
        values = self._native_role_losses(predictions.native_main, predictions.native_prepared, targets)
        terms = {
            f"flow/{role}": LossTerm(values[f"flow/{role}"].sum(), len(targets.case_ids), self.flow_role_weights[role])
            for role in ("u", "v", "p", "omega")
        }
        terms.update({
            f"temperature/{role}": LossTerm(
                values[f"temperature/{role}"].sum(), len(targets.case_ids), self.thermal_role_weights[role]
            )
            for role in ("fluid", "surface", "material")
        })
        terms["q_proxy"] = LossTerm(
            values["q_proxy"].sum(), len(targets.case_ids), self.q_proxy_weight
        )
        operator_values, operator_receipt = self._operator_loss(predictions, targets)
        if operator_values is not None:
            terms["operator_residual"] = LossTerm(
                operator_values.sum(), len(targets.case_ids), self.operator_weight
            )
            auxiliary_state["operator"] = operator_receipt
            predictions.work.update({
                "operator_rows": operator_receipt["operator_rows"],
                "operator_unique_stencil_receiver_rows": operator_receipt["unique_stencil_receiver_rows"],
            })
        if targets.response is not None:
            if predictions.response_main is None or predictions.response_prepared is None:
                raise RuntimeError("Joint Thermal TRAIN response addendum lost its native affine output.")
            response = self._response_loss(predictions.response_main, targets.response)
            terms["response"] = LossTerm(response, 1.0, self.response_weight)
        auxiliary_state["work"] = dict(predictions.work)
        return terms

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch, arm):
        del auxiliary_state, batch
        if arm != self.joint_mode:
            raise ValueError("Joint Thermal validation objective received a mismatched arm.")
        values = self._native_role_losses(predictions.native_main, predictions.native_prepared, targets)
        terms = {
            f"flow/{role}": LossTerm(values[f"flow/{role}"].sum(), len(targets.case_ids), self.flow_role_weights[role])
            for role in ("u", "v", "p", "omega")
        }
        terms.update({
            f"temperature/{role}": LossTerm(
                values[f"temperature/{role}"].sum(), len(targets.case_ids), self.thermal_role_weights[role]
            )
            for role in ("fluid", "surface", "material")
        })
        terms["q_proxy"] = LossTerm(values["q_proxy"].sum(), len(targets.case_ids), self.q_proxy_weight)
        return terms

    def validation_metrics(self, predictions, targets, auxiliary_state):
        del auxiliary_state
        values = self._native_role_losses(predictions.native_main, predictions.native_prepared, targets)
        pred = predictions.native_main["pred_field"]
        target = targets.field_targets.to(pred)
        weights = targets.point_weights.to(pred) * predictions.native_prepared.stencils["fluid"].valid.to(pred)
        rows = []
        for index, case_id in enumerate(targets.case_ids):
            prepared = predictions.native_prepared
            present = prepared.source_present[index] > 0.5
            flow_rmse = {}
            for channel, role in enumerate(("u", "v", "p", "omega")):
                role_weight = weights[index] * torch.isfinite(target[index, :, channel])
                error = pred[index, :, channel] - torch.nan_to_num(target[index, :, channel])
                mse = (error.square() * role_weight).sum() / role_weight.sum().clamp_min(1.0e-12)
                flow_rmse[role] = float(torch.sqrt(mse.clamp_min(0)).detach())
            fluid_weight = weights[index] * torch.isfinite(target[index, :, 4])
            fluid_error = pred[index, :, 4] - torch.nan_to_num(target[index, :, 4])
            fluid_mse = (fluid_error.square() * fluid_weight).sum() / fluid_weight.sum().clamp_min(1.0e-12)
            fluid_rmse = torch.sqrt(fluid_mse.clamp_min(0))
            fluid_receiver_valid = prepared.stencils["fluid"].valid[index]
            center_delta = prepared.flow_receivers[index, :, None, :] - prepared.context.centers[index, None, :, :]
            normalized_distance = torch.linalg.vector_norm(center_delta, dim=-1) / prepared.context.source_lengths[
                index, None, :
            ].clamp_min(1.0e-12)
            normalized_distance = normalized_distance.masked_fill(~present[None], float("inf"))
            nearest_source_distance = normalized_distance.amin(-1)
            finite_fluid = torch.isfinite(target[index, :, 4]) & torch.isfinite(pred[index, :, 4])
            support_valid = (weights[index] > 0) & fluid_receiver_valid & finite_fluid
            near_mask = support_valid & (nearest_source_distance <= JOINT_NEAR_RADIUS_MULTIPLE)
            far_mask = support_valid & (nearest_source_distance > JOINT_NEAR_RADIUS_MULTIPLE)

            def support_rmse(mask, error):
                count = int(mask.sum().item())
                if count == 0:
                    return None, 0, 0.0
                squared_sum = float(error[mask].double().square().sum().item())
                return math.sqrt(squared_sum / count), count, squared_sum

            near_rmse, near_count, near_squared_sum = support_rmse(near_mask, fluid_error)
            far_rmse, far_count, far_squared_sum = support_rmse(far_mask, fluid_error)
            interface_target = targets.interface_target[index]
            surface_valid = predictions.native_prepared.stencils["surface"].valid[index].reshape(
                predictions.native_main["pred_interface"].shape[1:-1]
            )
            outside_valid = predictions.native_prepared.stencils["outside"].valid[index].reshape_as(surface_valid)
            surface_mask = present[:, None] & surface_valid & torch.isfinite(interface_target[..., 0])
            surface_error = predictions.native_main["pred_interface"][index, ..., 0] - torch.nan_to_num(interface_target[..., 0])
            surface_mse = (surface_error.square() * surface_mask).sum() / surface_mask.sum().clamp_min(1)
            material_target = targets.material_targets[index]
            material_valid = prepared.stencils["material"].valid.reshape(
                predictions.native_main["pred_internal_temperature"].shape[:-1]
            )[index]
            material_mask = present[:, None] & material_valid & torch.isfinite(material_target)
            material_error = predictions.native_main["pred_internal_temperature"][index, ..., 0] - torch.nan_to_num(material_target)
            material_mse = (material_error.square() * material_mask).sum() / material_mask.sum().clamp_min(1)
            peak_valid = material_mask & torch.isfinite(
                predictions.native_main["pred_internal_temperature"][index, ..., 0]
            )
            has_peak = peak_valid.any(-1) & present
            predicted_peak = predictions.native_main["pred_internal_temperature"][index, ..., 0].masked_fill(
                ~peak_valid, -torch.inf
            ).amax(-1)
            target_peak = material_target.masked_fill(~peak_valid, -torch.inf).amax(-1)
            peak_error = predicted_peak[has_peak] - target_peak[has_peak]
            peak_rmse = torch.sqrt(peak_error.square().mean().clamp_min(0)) if peak_error.numel() else None
            source_ids = prepared.context.source_ids[index]
            peak_rows = [
                {
                    "source_id": int(source_ids[module_index].item()),
                    "absolute_error": float(value.abs().detach()),
                }
                for module_index, value in zip(torch.nonzero(has_peak, as_tuple=True)[0].tolist(), peak_error)
            ]
            q_target = interface_target[..., 1]
            q_mask = present[:, None] & surface_valid & outside_valid & torch.isfinite(q_target)
            q_error = predictions.native_main["pred_interface"][index, ..., 1] - torch.nan_to_num(q_target)
            q_mse = (q_error.square() * q_mask).sum() / q_mask.sum().clamp_min(1)
            flow_group = values["flow_group"][index]
            thermal_group = values["thermal_group"][index]
            row = {
                "case_id": str(case_id),
                "module_count": int(present.sum().item()),
                "flow_group_standardized_mse": float(flow_group.detach()),
                "thermal_group_standardized_mse": float(thermal_group.detach()),
                "field_score": float((0.5 * flow_group + 0.5 * thermal_group).detach()),
                "fluid_temperature_rmse": float(fluid_rmse.detach()),
                "near_fluid_temperature_rmse": near_rmse,
                "near_fluid_temperature_query_count": near_count,
                "near_fluid_temperature_squared_error_sum": near_squared_sum,
                "far_fluid_temperature_rmse": far_rmse,
                "far_fluid_temperature_query_count": far_count,
                "far_fluid_temperature_squared_error_sum": far_squared_sum,
                "surface_temperature_rmse": float(torch.sqrt(surface_mse.clamp_min(0)).detach()),
                "material_temperature_rmse": float(torch.sqrt(material_mse.clamp_min(0)).detach()),
                "sampled_material_peak_rmse": None if peak_rmse is None else float(peak_rmse.detach()),
                "material_peak_abs_error_by_source": peak_rows,
                "q_proxy_rmse": float(torch.sqrt(q_mse.clamp_min(0)).detach()),
                "flow_rmse": flow_rmse,
                "flow_rmse_unit_by_channel": ["native dataset unit"] * 4,
                "temperature_unit": "packed_dataset_native_temperature",
                "channel_order": list(JOINT_THERMAL_CHANNEL_ORDER),
            }
            rows.append(row)
        return {"case_rows": rows}

    def reduce_native_metrics(self, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        rows = [row for record in records for row in record["case_rows"]]
        ids = [str(row["case_id"]) for row in rows]
        if len(rows) != len(self._validation_ids) or len(set(ids)) != len(ids) or set(ids) != set(self._validation_ids):
            raise ValueError("Joint Thermal validation must visit the exact exposed fixed cohort once per case.")
        result: dict[str, Any] = {
            "scope": (
                "formal canonical89 primary; exposed validation; equal-case native measurements"
                if getattr(self, "formal_validation_scope", None) == "canonical89" else
                "manual originalTRAIN600 preparation; fixed25_v1 exposed DEV22 sanity panel only"
                if self.formal_full else "fixed25_v1 exposed DEV22; equal-case native measurements"
            ),
            "case_count": len(rows),
            "case_ids": ids,
            "channel_order": list(JOINT_THERMAL_CHANNEL_ORDER),
            "field_score": float(np.mean([row["field_score"] for row in rows])),
            "flow_group_standardized_mse": float(np.mean([row["flow_group_standardized_mse"] for row in rows])),
            "thermal_group_standardized_mse": float(np.mean([row["thermal_group_standardized_mse"] for row in rows])),
            "per_case": rows,
        }
        for role in ("u", "v", "p", "omega"):
            values = np.asarray([row["flow_rmse"][role] for row in rows], dtype=np.float64)
            result[f"flow_{role}_rmse_mean"] = float(values.mean())
            result[f"flow_{role}_rmse_p90"] = float(np.quantile(values, 0.9))
        for metric in ("fluid_temperature_rmse", "surface_temperature_rmse", "material_temperature_rmse", "q_proxy_rmse"):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_p90"] = float(np.quantile(values, 0.9))
        peak_values = np.asarray([
            row["sampled_material_peak_rmse"] for row in rows
            if row["sampled_material_peak_rmse"] is not None
        ], dtype=np.float64)
        result["sampled_material_peak_rmse_mean"] = float(peak_values.mean()) if peak_values.size else None
        result["sampled_material_peak_rmse_p90"] = float(np.quantile(peak_values, 0.9)) if peak_values.size else None
        peak_errors = [item["absolute_error"] for row in rows for item in row["material_peak_abs_error_by_source"]]
        result["material_peak_absolute_error_mean"] = float(np.mean(peak_errors)) if peak_errors else None
        for support in ("near", "far"):
            count_key = f"{support}_fluid_temperature_query_count"
            sum_key = f"{support}_fluid_temperature_squared_error_sum"
            count = int(sum(row[count_key] for row in rows))
            squared_error_sum = float(sum(row[sum_key] for row in rows))
            result[f"{support}_fluid_temperature_query_count"] = count
            result[f"{support}_fluid_temperature_rmse"] = (
                math.sqrt(squared_error_sum / count) if count else None
            )
            supported_case_rmse = [
                row[f"{support}_fluid_temperature_rmse"] for row in rows
                if row[count_key] > 0 and row[f"{support}_fluid_temperature_rmse"] is not None
            ]
            result[f"{support}_fluid_temperature_case_rmse_mean"] = (
                float(np.mean(supported_case_rmse)) if supported_case_rmse else None
            )
        return result

    def optimizer_groups(self, model, arm: str, stage: str) -> Sequence[OptimizerGroupSpec]:
        del stage
        if model is not self.model or arm != self.joint_mode:
            raise ValueError("Joint Thermal optimizer request has a mismatched model or arm.")
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if not names or any(not name.startswith("core.") for name in names):
            raise ValueError("Every active Thermal joint parameter must belong to the one fresh core.")
        schedule = getattr(self, "formal_optimizer_schedule", None) or ScheduleSpec(
            peak_lr=3.0e-4,
            warmup_start_lr=3.0e-5,
            warmup_epochs=min(20, self.total_epochs - 1),
            hold_through_epoch=min(1000, self.total_epochs - 1),
            total_epochs=self.total_epochs,
            final_lr=3.0e-6,
        )
        weight_decay = getattr(self, "formal_weight_decay", None)
        return (OptimizerGroupSpec("joint", names, schedule,
                                   weight_decay=1.0e-4 if weight_decay is None else weight_decay),)

    def work_counts(self, batch: TaskBatch, predictions, auxiliary_state) -> Mapping[str, int | float]:
        result: dict[str, int | float] = {
            name: float(value) for name, value in auxiliary_state.get("work", predictions.work).items()
        }
        if self.joint_mode not in INTERACTION_PRESERVING_THERMAL_MODES:
            return result
        contexts = []
        main_native = predictions.native_prepared
        main_flow_receivers = getattr(main_native, "flow_query_receivers", main_native.flow_receivers)
        contexts.append(("main", main_native.context, main_native,
                         int(main_flow_receivers.shape[1]), int(main_native.grid_indices.shape[1])))
        if predictions.response_prepared is not None:
            response_native = predictions.response_prepared
            # Increment application has no nonlinear flow read. The response
            # context still executes its pair rounds and affine heat reader.
            contexts.append(("response", response_native.context, response_native,
                             0, int(response_native.grid_indices.shape[1])))

        pair_rounds = 2
        pair_mm = pair_me = pair_em = 0
        pair_mm_logical_no_self = pair_mm_logical_with_self = pair_me_logical = pair_em_logical = 0
        padding_source_slots = 0
        collective_groups = 0
        collective_source_score_slots = collective_environment_score_slots = 0
        collective_source_edges = collective_environment_edges = 0
        collective_receiver_score_slots = 0
        flow_read_pairs = main_temp_read_pairs = response_temp_read_pairs = 0
        active_source_receiver_pairs = active_flow_read_pairs = 0
        active_main_temp_read_pairs = active_response_temp_read_pairs = 0
        native_flow_queries = native_temp_union_queries = response_temp_union_queries = 0
        native_primary_label_queries = native_primary_unique_queries = native_flow_logical_unique_queries = 0
        native_curl_neighbor_unique_queries = native_curl_padding_receiver_slots = 0
        for name, prepared, native_prepared, flow_queries, union_queries in contexts:
            batch_count, source_capacity = prepared.present.shape
            environment_capacity = prepared.environment_coords.shape[1]
            active_by_scene = (prepared.present > 0).sum(dim=1).to(torch.int64)
            active_total = int(active_by_scene.sum().item())
            source_capacity = int(source_capacity)
            environment_capacity = int(environment_capacity)
            pair_mm += pair_rounds * int(batch_count) * source_capacity * source_capacity
            pair_me += pair_rounds * int(batch_count) * source_capacity * environment_capacity
            pair_em += pair_rounds * int(batch_count) * environment_capacity * source_capacity
            pair_mm_logical_no_self += pair_rounds * int((active_by_scene * (active_by_scene - 1)).sum().item())
            pair_mm_logical_with_self += pair_rounds * int((active_by_scene * active_by_scene).sum().item())
            pair_me_logical += pair_rounds * active_total * environment_capacity
            pair_em_logical += pair_rounds * active_total * environment_capacity
            padding_source_slots += int(batch_count * source_capacity - active_total)

            affine_pairs = int(batch_count * union_queries * source_capacity)
            active_affine_pairs = int(union_queries * active_total)
            if name == "main":
                stencil = getattr(native_prepared, "native_curl_stencil", None)
                label_queries = int(native_prepared.flow_receivers.shape[1])
                if stencil is not None and torch.is_tensor(getattr(stencil, "unique_counts", None)):
                    logical_counts = stencil.unique_counts.to(device=active_by_scene.device, dtype=torch.int64)
                    gather_indices = getattr(stencil, "gather_indices", None)
                    if (not torch.is_tensor(gather_indices) or gather_indices.ndim != 3
                            or gather_indices.shape[:2] != (batch_count, label_queries)
                            or gather_indices.shape[-1] != 5):
                        raise ValueError("Native-curl receiver union is missing its per-label five-point gather map.")
                    primary_counts = torch.tensor(
                        [torch.unique(gather_indices[row, :, 0]).numel() for row in range(batch_count)],
                        device=active_by_scene.device,
                        dtype=torch.int64,
                    )
                    if (logical_counts.shape != (batch_count,) or primary_counts.shape != (batch_count,)
                            or bool((logical_counts < primary_counts).any())
                            or bool((logical_counts > flow_queries).any())):
                        raise ValueError("Native-curl receiver union counts do not match the primary batch.")
                    logical_total = int(logical_counts.sum().item())
                    active_flow_pairs = int((logical_counts * active_by_scene).sum().item())
                    native_primary_unique_queries += int(primary_counts.sum().item())
                    native_curl_neighbor_unique_queries += int((logical_counts - primary_counts).sum().item())
                    native_curl_padding_receiver_slots += int(batch_count * flow_queries - logical_total)
                else:
                    logical_total = int(batch_count * flow_queries)
                    active_flow_pairs = int(flow_queries * active_total)
                    native_primary_unique_queries += batch_count * label_queries
                flow_pairs = int(batch_count * flow_queries * source_capacity)
                flow_read_pairs += flow_pairs
                active_flow_read_pairs += active_flow_pairs
                active_source_receiver_pairs += active_flow_pairs
                main_temp_read_pairs += affine_pairs
                active_main_temp_read_pairs += active_affine_pairs
                native_flow_queries += batch_count * flow_queries
                native_primary_label_queries += batch_count * label_queries
                native_flow_logical_unique_queries += logical_total
                native_temp_union_queries += batch_count * union_queries
            else:
                response_temp_read_pairs += affine_pairs
                active_response_temp_read_pairs += active_affine_pairs
                response_temp_union_queries += batch_count * union_queries

            groups = prepared.group_states
            group_count = int(groups.shape[1]) if torch.is_tensor(groups) else 0
            collective_groups += group_count
            if group_count:
                collective_source_score_slots += int(batch_count * group_count * source_capacity)
                collective_environment_score_slots += int(batch_count * group_count * environment_capacity)
                source_membership = getattr(prepared, "source_membership", None)
                environment_membership = getattr(prepared, "environment_membership", None)
                if torch.is_tensor(source_membership):
                    collective_source_edges += int(torch.count_nonzero(source_membership > 0).item())
                if torch.is_tensor(environment_membership):
                    collective_environment_edges += int(torch.count_nonzero(environment_membership > 0).item())
                # Main flow plus affine reads for this context. Response
                # increments have only the affine read; their flow_queries=0.
                collective_receiver_score_slots += int(batch_count * (flow_queries + union_queries) * group_count)

        batch_cases = len(getattr(batch.targets, "case_ids", ()))
        operator_rows = batch_cases * int(self.operator_rows_per_case) if self.use_operator else 0
        result.update({
            "pair_contexts_prepared": len(contexts),
            "pair_rounds_per_context": pair_rounds,
            "pair_MM_executed_slots_including_self": pair_mm,
            "pair_MM_active_logical_pairs_excluding_self": pair_mm_logical_no_self,
            "pair_MM_active_logical_pairs_including_self": pair_mm_logical_with_self,
            "pair_ME_executed_slots": pair_me,
            "pair_ME_active_logical_pairs": pair_me_logical,
            "pair_EM_executed_slots": pair_em,
            "pair_EM_active_logical_pairs": pair_em_logical,
            "pair_padding_source_slots_per_context": padding_source_slots,
            "native_flow_receivers": native_flow_queries,
            "native_primary_label_receivers": native_primary_label_queries,
            "native_primary_unique_receivers": native_primary_unique_queries,
            "native_flow_logical_unique_receivers": native_flow_logical_unique_queries,
            "native_curl_neighbor_added_unique_receivers": native_curl_neighbor_unique_queries,
            "native_curl_union_padding_receiver_slots": native_curl_padding_receiver_slots,
            "native_temperature_union_receivers": native_temp_union_queries,
            "response_temperature_union_receivers": response_temp_union_queries,
            "source_conditioned_flow_read_executed_pairs": flow_read_pairs,
            "source_conditioned_flow_read_active_pairs": active_flow_read_pairs,
            "temperature_affine_union_read_executed_pairs": main_temp_read_pairs,
            "temperature_affine_union_read_active_pairs": active_main_temp_read_pairs,
            "response_affine_union_read_executed_pairs": response_temp_read_pairs,
            "response_affine_union_read_active_pairs": active_response_temp_read_pairs,
            "response_increment_flow_read_executed_receivers": 0,
            "response_increment_flow_read_executed_pairs": 0,
            "active_physical_source_receiver_pairs": active_source_receiver_pairs,
            "collective_groups_across_prepared_contexts": collective_groups,
            "collective_source_membership_score_slots": collective_source_score_slots,
            "collective_environment_membership_score_slots": collective_environment_score_slots,
            "collective_source_memberships_nonzero": collective_source_edges,
            "collective_environment_memberships_nonzero": collective_environment_edges,
            "collective_receiver_access_score_slots": collective_receiver_score_slots,
            "native_operator_rows_supervised": operator_rows,
            "full_access_fallback": 0,
            "sparse_executor_savings_measured": 0,
        })
        return result

    def training_state_dict(self) -> Mapping[str, Any]:
        return {
            "training_mode": JOINT_TRAINING_MODE,
            "arm": self.joint_mode,
            "response_coefficient": self.response_weight,
            "operator_coefficient": self.operator_weight,
            "auxiliary_calibration": copy.deepcopy(self.auxiliary_calibration),
        }

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("training_mode") != JOINT_TRAINING_MODE or state.get("arm") != self.joint_mode:
            raise ValueError("Saved Thermal task state belongs to another joint mode.")
        for name, current in (("response_coefficient", self.response_weight), ("operator_coefficient", self.operator_weight)):
            if float(state.get(name, float("nan"))) != current:
                raise ValueError(f"Saved joint Thermal {name} differs from the active recipe.")
        saved = state.get("auxiliary_calibration")
        saved_receipt = None if saved is None else copy.deepcopy(dict(saved))
        if self.joint_mode in INTERACTION_PRESERVING_THERMAL_MODES:
            if self.auxiliary_calibration is None or saved_receipt != self.auxiliary_calibration:
                raise ValueError("Saved P-family task state has no sealed fresh-P calibration receipt.")
        self.auxiliary_calibration = saved_receipt

    def calibrate_auxiliary_coefficients(self) -> Mapping[str, Any]:
        """Keep the historical J calibration contract and seal fresh P separately."""
        if self.joint_mode in JOINT_THERMAL_MODES:
            return self._calibrate_legacy_auxiliary_coefficients()
        if self.joint_mode in INTERACTION_PRESERVING_THERMAL_MODES:
            return self._calibrate_interaction_preserving_auxiliary_coefficients()
        raise ValueError("Unsupported Thermal auxiliary-calibration family.")

    def _calibrate_legacy_auxiliary_coefficients(self) -> Mapping[str, Any]:
        """Retain the pre-recovery J-family TRAIN calibration schema verbatim."""
        if self.auxiliary_calibration is not None:
            raise RuntimeError("Joint Thermal auxiliary coefficients are already calibrated and sealed.")
        module_counts = [int(np.asarray(case["structure"]["module_present"]).sum()) for case in self.training_cases]
        selected_indices = [next((index for index, count in enumerate(module_counts) if count == desired), None)
                            for desired in (1, 3, 10, 12)]
        if any(index is None for index in selected_indices):
            raise ValueError("Joint Thermal calibration requires TRAIN examples at1,3,10,12 modules.")
        parameters = tuple(parameter for parameter in self.model.parameters() if parameter.requires_grad)
        field_gradient_norms: list[float] = []
        flow_gradient_norms: list[float] = []
        thermal_gradient_norms: list[float] = []
        operator_gradient_norms: list[float] = []
        for update_index, index in enumerate(selected_indices):
            key = SamplingKey(
                0, 1, update_index, 0, "joint_auxiliary_calibration", self.joint_mode,
                sampling_version=SamplingKey.CASE_EPOCH_VERSION,
                dataset_id=("thermal_full_followup1000_v1" if self.full_train_followup1000 else "thermal_fixed25_v1"),
            )
            batch = self._batch_from_indices([int(index)], 1, key=key, training=True,
                                             include_response=False, budget=self.budget)
            scene = self.make_scene(batch.scene_inputs)
            predictions, _ = self.predict_native(self.model, scene, batch.receivers, self.joint_mode, "joint")
            values = self._native_role_losses(predictions.native_main, predictions.native_prepared, batch.targets)
            flow_obj = 0.5 * values["flow_group"].mean()
            thermal_obj = 0.5 * values["thermal_group"].mean()
            base_obj = flow_obj + thermal_obj
            flow_gradient_norms.append(_gradient_norm(flow_obj, parameters, retain_graph=True))
            thermal_gradient_norms.append(_gradient_norm(thermal_obj, parameters, retain_graph=True))
            field_gradient_norms.append(_gradient_norm(base_obj, parameters, retain_graph=True))
            operator_values, _receipt = self._operator_loss(predictions, batch.targets)
            if operator_values is not None:
                operator_gradient_norms.append(_gradient_norm(operator_values.mean(), parameters))

        response_gradient_norms = []
        for update_index, family in enumerate(self.train_families):
            key = SamplingKey(
                0, 1, update_index, 0, "joint_response_calibration", self.joint_mode,
                sampling_version=SamplingKey.CASE_EPOCH_VERSION,
                dataset_id=("thermal_full_followup1000_v1" if self.full_train_followup1000 else "thermal_fixed25_v1"),
            )
            response_target = self._prepare_response_sample(family, training=True, key=key, budget=self.budget)
            structure = {name: value.to(self.device) for name, value in response_target["structure"].items()}
            response_mask_options = {}
            if self.flow_readout_law == NATIVE_CURL_READOUT_LAW:
                if "native_solid_mask" not in response_target:
                    raise ValueError("Native-curl calibration response is missing its saved TRAIN mask.")
                response_mask_options["native_solid_mask"] = response_target["native_solid_mask"].to(self.device)
            prepared = self.model.prepare_native(
                structure, response_target["fluid_xy"].to(self.device),
                local_query_points=response_target["local"].to(self.device), ntheta=16,
                chunk_size=self.model.receiver_tile,
                **response_mask_options,
            )
            output = self.model.apply_native(
                prepared, response_target["heat"].to(self.device), increment=True,
            )
            targets = self._response_targets_from_sample(response_target, self.device)
            response_loss = self._response_loss(output, targets)
            response_gradient_norms.append(_gradient_norm(response_loss, parameters))

        field_norm = float(np.mean(field_gradient_norms))
        response_norm = float(np.mean(response_gradient_norms))
        operator_norm = float(np.mean(operator_gradient_norms)) if operator_gradient_norms else 0.0
        target_added_gradient_ratio = 0.5
        response_coefficient = min(1.0, target_added_gradient_ratio * field_norm / max(response_norm, 1.0e-12))
        operator_coefficient = min(1.0, target_added_gradient_ratio * field_norm / max(operator_norm, 1.0e-12))
        receipt = {
            "method": "fresh_joint_init_fixed_train_gradient_ratio_v1",
            "calibration_mode": self.joint_mode,
            "calibration_case_ids": [self._train_ids[int(index)] for index in selected_indices],
            "calibration_module_counts": [1, 3, 10, 12],
            "response_family_ids": [str(item["family_id"]) for item in self.train_families],
            "flow_group_gradient_norms": flow_gradient_norms,
            "thermal_group_gradient_norms": thermal_gradient_norms,
            "balanced_field_gradient_norms": field_gradient_norms,
            "response_gradient_norms": response_gradient_norms,
            "operator_gradient_norms": operator_gradient_norms,
            "target_added_gradient_ratio": target_added_gradient_ratio,
            "response_coefficient": float(response_coefficient),
            "operator_coefficient": float(operator_coefficient),
            "q_proxy_coefficient": self.q_proxy_weight,
            "validation_values_read": False,
            "optimizer_steps": 0,
            "stored_velocity_used_only_for_operator_supervision": True,
        }
        self.response_weight = float(response_coefficient)
        self.operator_weight = float(operator_coefficient)
        self.response_coefficient = self.response_weight
        self.operator_coefficient = self.operator_weight
        self.auxiliary_calibration = receipt
        self._joint_loss_metadata = self._build_loss_metadata()
        return copy.deepcopy(receipt)

    def _calibrate_interaction_preserving_auxiliary_coefficients(self) -> Mapping[str, Any]:
        """Measure one fresh TRAIN-only response/operator gradient ratio.

        The root campaign calls this on its designated calibration arm/device,
        then passes the sealed coefficients unchanged to sibling modes.
        """
        if self.joint_mode != "P":
            raise ValueError("Thermal auxiliary calibration must run on the fresh direct P model.")
        if self._validation_ids:
            raise ValueError("Fresh-P calibration must use calibration_only provider construction with no DEV cases loaded.")
        if self.auxiliary_calibration is not None:
            raise RuntimeError("Joint Thermal auxiliary coefficients are already calibrated and sealed.")
        expected_training_count = 600 if getattr(self, "full_train_followup1000", False) else 150
        if len(self._train_ids) != expected_training_count:
            raise ValueError("Fresh-P auxiliary calibration is bound to the active sealed TRAIN population.")
        initialization_sha256 = _state_sha256(self.model)
        direct_path_sha256 = _shared_direct_state_sha256(self.model)
        module_counts = [int(np.asarray(case["structure"]["module_present"]).sum()) for case in self.training_cases]
        selected_indices = [next((index for index, count in enumerate(module_counts) if count == desired), None)
                            for desired in (1, 3, 10, 12)]
        if any(index is None for index in selected_indices):
            raise ValueError("Joint Thermal calibration requires TRAIN examples at1,3,10,12 modules.")
        parameters = tuple(parameter for parameter in self.model.parameters() if parameter.requires_grad)
        field_gradient_norms: list[float] = []
        flow_gradient_norms: list[float] = []
        thermal_gradient_norms: list[float] = []
        operator_gradient_norms: list[float] = []
        for update_index, index in enumerate(selected_indices):
            key = SamplingKey(
                0,
                1,
                update_index,
                0,
                "joint_auxiliary_calibration",
                self.joint_mode,
                sampling_version=SamplingKey.CASE_EPOCH_VERSION,
                dataset_id=("thermal_full_followup1000_v1" if getattr(self, "full_train_followup1000", False) else "thermal_fixed25_v1"),
            )
            batch = self._batch_from_indices([int(index)], 1, key=key, training=True,
                                             include_response=False, budget=self.budget)
            scene = self.make_scene(batch.scene_inputs)
            predictions, _ = self.predict_native(self.model, scene, batch.receivers, self.joint_mode, "joint")
            values = self._native_role_losses(predictions.native_main, predictions.native_prepared, batch.targets)
            flow_obj = 0.5 * values["flow_group"].mean()
            thermal_obj = 0.5 * values["thermal_group"].mean()
            base_obj = flow_obj + thermal_obj
            flow_gradient_norms.append(_gradient_norm(flow_obj, parameters, retain_graph=True))
            thermal_gradient_norms.append(_gradient_norm(thermal_obj, parameters, retain_graph=True))
            # The qualified operator penalty reads this same prepared affine
            # graph, so keep it alive until its gradient norm is measured.
            field_gradient_norms.append(_gradient_norm(base_obj, parameters, retain_graph=True))
            operator_values, _receipt = self._operator_loss(predictions, batch.targets)
            if operator_values is not None:
                operator_gradient_norms.append(_gradient_norm(operator_values.mean(), parameters))

        response_gradient_norms = []
        for update_index, family in enumerate(self.train_families):
            key = SamplingKey(
                0,
                1,
                update_index,
                0,
                "joint_response_calibration",
                self.joint_mode,
                sampling_version=SamplingKey.CASE_EPOCH_VERSION,
                dataset_id=("thermal_full_followup1000_v1" if getattr(self, "full_train_followup1000", False) else "thermal_fixed25_v1"),
            )
            response_target = self._prepare_response_sample(family, training=True, key=key, budget=self.budget)
            structure = {key: value.to(self.device) for key, value in response_target["structure"].items()}
            response_mask_options = {}
            if self.flow_readout_law == NATIVE_CURL_READOUT_LAW:
                if "native_solid_mask" not in response_target:
                    raise ValueError("Native-curl calibration response is missing its saved TRAIN mask.")
                response_mask_options["native_solid_mask"] = response_target["native_solid_mask"].to(self.device)
            prepared = self.model.prepare_native(
                structure,
                response_target["fluid_xy"].to(self.device),
                local_query_points=response_target["local"].to(self.device),
                ntheta=16,
                chunk_size=self.model.receiver_tile,
                **response_mask_options,
            )
            output = self.model.apply_native(
                prepared,
                response_target["heat"].to(self.device),
                increment=True,
            )
            # _response_sample materializes exactly the already declared TRAIN
            # addendum and preserves its original native IDs and masks.
            targets = self._response_targets_from_sample(response_target, self.device)
            response_loss = self._response_loss(output, targets)
            response_gradient_norms.append(_gradient_norm(response_loss, parameters))

        field_norm = float(np.mean(field_gradient_norms))
        response_norm = float(np.mean(response_gradient_norms))
        operator_norm = float(np.mean(operator_gradient_norms)) if operator_gradient_norms else 0.0
        target_added_gradient_ratio = 0.5
        response_coefficient = min(1.0, target_added_gradient_ratio * field_norm / max(response_norm, 1.0e-12))
        operator_coefficient = min(1.0, target_added_gradient_ratio * field_norm / max(operator_norm, 1.0e-12))
        family_hashes = {
            str(item["family_id"]): _sha256_file(self.atlas_directory / f"train_{item['family_id']}_responses.npz")
            for item in self.train_families
        }
        if tuple(family_hashes) != tuple(TRAIN_RESPONSE_IDS):
            raise ValueError("Fresh-P calibration did not preserve the four declared TRAIN response families in order.")
        receipt = {
            "schema_version": 1,
            "method": "fresh_joint_init_fixed_train_gradient_ratio_v1",
            "calibration_mode": "P",
            "compatible_modes": list(INTERACTION_PRESERVING_THERMAL_MODES),
            "initial_model_state_sha256": initialization_sha256,
            "shared_direct_path_sha256": direct_path_sha256,
            "initialization_seed": int(getattr(self.model, "seed", -1)),
            "dataset_protocol": (FULL_TRAIN_FOLLOWUP_DATASET if getattr(self, "full_train_followup1000", False) else "fixed25_v1"),
            "manifest_sha256": str(self.manifest["manifest_sha256"]),
            "training_membership_sha256": _case_ids_hash(self._train_ids),
            "training_case_ids": list(self._train_ids),
            "source_binding": copy.deepcopy(self.source_binding),
            "normalization_stats_sha256": _tensor_digest(self.stats),
            "primary_query_count": int(self.budget["fluid_queries"]),
            "operator_rows_per_case": int(self.operator_rows_per_case),
            "response_surface_stride": int(self.budget["surface_stride"]),
            "response_material_queries_per_module": int(self.budget["material_queries_per_module"]),
            "calibration_case_ids": [self._train_ids[int(index)] for index in selected_indices],
            "calibration_module_counts": [1, 3, 10, 12],
            "response_family_ids": [str(item["family_id"]) for item in self.train_families],
            "response_family_source_sha256": family_hashes,
            "response_family_partition": "original TRAIN response addendum; all four fixed families",
            "primary_partition": ("original600 TRAIN only" if getattr(self, "full_train_followup1000", False) else "fixed25_v1 TRAIN only"),
            "flow_group_gradient_norms": flow_gradient_norms,
            "thermal_group_gradient_norms": thermal_gradient_norms,
            "balanced_field_gradient_norms": field_gradient_norms,
            "response_gradient_norms": response_gradient_norms,
            "operator_gradient_norms": operator_gradient_norms,
            "target_added_gradient_ratio": target_added_gradient_ratio,
            "response_coefficient": float(response_coefficient),
            "operator_coefficient": float(operator_coefficient),
            "q_proxy_coefficient": self.q_proxy_weight,
            "validation_values_read": False,
            "validation_cases_materialized": 0,
            "development_response_families_loaded": False,
            "optimizer_steps": 0,
            "stored_velocity_used_only_for_operator_supervision": True,
        }
        if getattr(self, "flow_readout_law", None) is not None:
            receipt["flow_readout_law"] = self.flow_readout_law
        self.response_weight = float(response_coefficient)
        self.operator_weight = float(operator_coefficient)
        self.response_coefficient = self.response_weight
        self.operator_coefficient = self.operator_weight
        self.auxiliary_calibration = receipt
        self._joint_loss_metadata = self._build_loss_metadata()
        self._validate_auxiliary_calibration(receipt)
        return copy.deepcopy(receipt)

    def _validate_auxiliary_calibration(self, receipt: Mapping[str, Any]) -> None:
        """Reject calibration from J/H, another cohort, or another direct initialization."""
        if self.joint_mode not in INTERACTION_PRESERVING_THERMAL_MODES:
            raise ValueError("Only P-family Thermal tasks accept a fresh-P auxiliary calibration receipt.")
        active_flow_readout_law = getattr(self, "flow_readout_law", None)
        receipt_flow_readout_law = receipt.get("flow_readout_law")
        if active_flow_readout_law is not None:
            if receipt_flow_readout_law != active_flow_readout_law:
                raise ValueError("Thermal auxiliary calibration receipt belongs to a different flow readout law.")
        elif receipt_flow_readout_law is not None:
            raise ValueError("Legacy Thermal auxiliary calibration cannot accept a changed flow readout law.")
        module_counts = [int(np.asarray(case["structure"]["module_present"]).sum()) for case in self.training_cases]
        expected_case_ids = [
            self._train_ids[next(index for index, count in enumerate(module_counts) if count == desired)]
            for desired in (1, 3, 10, 12)
        ]
        if (
            receipt.get("schema_version") != 1
            or receipt.get("method") != "fresh_joint_init_fixed_train_gradient_ratio_v1"
            or receipt.get("calibration_mode") != "P"
            or receipt.get("compatible_modes") != list(INTERACTION_PRESERVING_THERMAL_MODES)
            or receipt.get("dataset_protocol") != (
                FULL_TRAIN_FOLLOWUP_DATASET if getattr(self, "full_train_followup1000", False) else "fixed25_v1")
            or receipt.get("training_membership_sha256") != _case_ids_hash(self._train_ids)
            or receipt.get("training_case_ids") != list(self._train_ids)
            or receipt.get("manifest_sha256") != str(self.manifest["manifest_sha256"])
            or receipt.get("source_binding") != self.source_binding
            or receipt.get("normalization_stats_sha256") != _tensor_digest(self.stats)
            or receipt.get("primary_query_count") != int(self.budget["fluid_queries"])
            or receipt.get("operator_rows_per_case") != int(self.operator_rows_per_case)
            or receipt.get("response_surface_stride") != int(self.budget["surface_stride"])
            or receipt.get("response_material_queries_per_module") != int(self.budget["material_queries_per_module"])
            or receipt.get("response_family_ids") != list(TRAIN_RESPONSE_IDS)
            or receipt.get("response_family_partition") != "original TRAIN response addendum; all four fixed families"
            or receipt.get("primary_partition") != (
                "original600 TRAIN only" if getattr(self, "full_train_followup1000", False) else "fixed25_v1 TRAIN only")
            or receipt.get("validation_values_read") is not False
            or receipt.get("validation_cases_materialized") != 0
            or receipt.get("development_response_families_loaded") is not False
            or receipt.get("optimizer_steps") != 0
            or receipt.get("stored_velocity_used_only_for_operator_supervision") is not True
            or receipt.get("target_added_gradient_ratio") != 0.5
            or receipt.get("q_proxy_coefficient") != float(Q_PROXY_COEFFICIENT)
            or receipt.get("initialization_seed") != int(getattr(self.model, "seed", -1))
            or not isinstance(receipt.get("initial_model_state_sha256"), str)
            or len(receipt["initial_model_state_sha256"]) != 64
            or receipt.get("shared_direct_path_sha256") != _shared_direct_state_sha256(self.model)
            or tuple(receipt.get("calibration_module_counts", ())) != (1, 3, 10, 12)
            or receipt.get("calibration_case_ids") != expected_case_ids
            or receipt.get("response_family_source_sha256") != {
                str(item["family_id"]): _sha256_file(
                    self.atlas_directory / f"train_{item['family_id']}_responses.npz"
                ) for item in self.train_families
            }
        ):
            raise ValueError("Thermal auxiliary calibration receipt is not the current fresh-P, fixed25 TRAIN artifact.")
        for name in ("response_coefficient", "operator_coefficient"):
            value = receipt.get(name)
            if type(value) not in (int, float) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"Thermal calibration receipt {name} is invalid.")
        if (float(receipt["response_coefficient"]) != self.response_weight
                or float(receipt["operator_coefficient"]) != self.operator_weight):
            raise ValueError("Thermal auxiliary coefficients differ from their sealed fresh-P receipt.")


    @staticmethod
    def _response_targets_from_sample(sample: Mapping[str, Any], device: torch.device):
        from .unified_task import ThermalResponseTargets

        return ThermalResponseTargets(
            family_id=str(sample["family_id"]),
            fluid=sample["fluid_target"].to(device=device, dtype=torch.float32),
            surface=sample["surface_target"].to(device=device, dtype=torch.float32),
            surface_mask=sample["surface_mask"].to(device=device, dtype=torch.float32),
            material=sample["material_target"].to(device=device, dtype=torch.float32),
            material_mask=sample["material_mask"].to(device=device, dtype=torch.float32),
        )


def build_thermal_joint_task(
    mode: str = "J-H",
    device: torch.device | str = "cpu",
    hidden: int = 128,
    message: int = 128,
    microbatch_size: int = JOINT_MICROBATCH_DEFAULT,
    effective_batch_size: int = JOINT_EFFECTIVE_BATCH_DEFAULT,
    primary_queries: int = JOINT_PRIMARY_QUERY_DEFAULT,
    manifest: Mapping[str, Any] | str | Path | None = None,
    formal_full: bool = False,
    regional_anchors: int = 16,
    locality_prior_strength: float = 0.0,
    receiver_tile: int = 512,
    seed: int = 0,
    total_epochs: int = 2500,
    operator_rows_per_case: int = OPERATOR_ROWS_PER_CASE,
    response_coefficient: float = 0.1,
    operator_coefficient: float = 0.1,
    data_path: str | Path | None = None,
    atlas_directory: str | Path = DEFAULT_ATLAS_DIRECTORY,
    depth: int = 2,
    validation_queries: int | None = None,
    auxiliary_calibration: Mapping[str, Any] | None = None,
    validation_scope: str | None = None,
    optimizer_schedule: ScheduleSpec | None = None,
    weight_decay: float | None = None,
    native_sampling_protocol: str | None = None,
    collective_width: int = 64,
    max_sources: int = 12,
    calibration_only: bool = False,
    flow_readout_law: str | None = None,
    full_train_followup1000: bool = False,
) -> tuple[JointThermalRegionalAdapter, JointThermalTask]:
    """Construct a fresh joint candidate and its fixed25_v1 provider.

    ``formal_full=True`` prepares the original600 TRAIN cases and fits all
    normalizers on those cases, but only returns a manual-only recipe/provider;
    this factory never starts or advances training.
    """
    if mode not in THERMAL_RECOVERY_MODES:
        raise ValueError(f"Joint Thermal mode must be one of {THERMAL_RECOVERY_MODES}.")
    if flow_readout_law not in (None, NATIVE_CURL_READOUT_LAW):
        raise ValueError("Unsupported opt-in Thermal flow readout law.")
    if flow_readout_law is not None and mode not in INTERACTION_PRESERVING_THERMAL_MODES:
        raise ValueError("The native-curl readout is a P-family-only model identity.")
    if flow_readout_law is not None and formal_full and not full_train_followup1000:
        raise ValueError("Native-curl full-TRAIN recipes require a separately authorized campaign.")
    if type(full_train_followup1000) is not bool:
        raise TypeError("Thermal full_train_followup1000 must be boolean.")
    if full_train_followup1000 and (formal_full or mode not in INTERACTION_PRESERVING_THERMAL_MODES
                                    or total_epochs != 5000):
        raise ValueError("Thermal full follow-up requires non-formal P-family identity and 5000-epoch horizon.")
    if isinstance(locality_prior_strength, (bool, np.bool_)):
        raise TypeError("Joint Thermal locality prior strength must be numeric, not boolean.")
    try:
        locality_prior_strength = float(locality_prior_strength)
    except (TypeError, ValueError) as error:
        raise ValueError("Joint Thermal locality prior strength must be finite and nonnegative.") from error
    if not math.isfinite(locality_prior_strength) or locality_prior_strength < 0:
        raise ValueError("Joint Thermal locality prior strength must be finite and nonnegative.")
    if mode in JOINT_THERMAL_MODES and locality_prior_strength and mode != "J-H":
        raise ValueError("Joint Thermal locality prior is only supported by J-H.")
    if mode == "P" and locality_prior_strength != 0:
        raise ValueError("Thermal P has no collective locality prior.")
    if mode in ("P-G", "P-H") and locality_prior_strength != 1.0:
        raise ValueError("The recovery Thermal P-G/P-H geometric locality prior is sealed to 1.0.")
    if mode in INTERACTION_PRESERVING_THERMAL_MODES and formal_full and not full_train_followup1000:
        raise ValueError("P-family full-TRAIN recipes remain unavailable until separately justified and authorized.")
    if calibration_only and (mode != "P" or (formal_full and not full_train_followup1000)
                             or auxiliary_calibration is not None):
        raise ValueError("Calibration-only construction is reserved for a fresh P model with no supplied receipt.")
    if mode in INTERACTION_PRESERVING_THERMAL_MODES and not calibration_only and auxiliary_calibration is None:
        raise ValueError("Every P-family Thermal comparison requires the sealed fresh-P TRAIN calibration receipt.")
    if microbatch_size < 1 or effective_batch_size < 1 or microbatch_size > effective_batch_size:
        raise ValueError("Joint Thermal microbatch must be positive and no larger than its effective batch.")
    if primary_queries < 1 or receiver_tile < 1:
        raise ValueError("Joint Thermal primary query count and receiver tile must be positive.")
    if operator_rows_per_case != OPERATOR_ROWS_PER_CASE:
        raise ValueError("The declared joint Thermal objective requires exactly128 operator rows per case.")
    if (formal_full or full_train_followup1000) and total_epochs != 5000:
        raise ValueError("The manual fullTRAIN joint recipe is bound to a5000-epoch horizon.")
    if any(option is not None for option in (validation_scope, native_sampling_protocol)) \
            and not (formal_full or full_train_followup1000):
        raise ValueError("Formal validation and sampling controls require the separate fullTRAIN identity.")
    if (optimizer_schedule is not None or weight_decay is not None) and not formal_full \
            and mode not in INTERACTION_PRESERVING_THERMAL_MODES:
        raise ValueError("Explicit development optimizer controls are reserved for the P-family recovery recipes.")
    if validation_scope not in (None, "canonical89"):
        raise ValueError("Joint Thermal formal validation scope must be canonical89.")
    if optimizer_schedule is not None and (not isinstance(optimizer_schedule, ScheduleSpec)
                                          or optimizer_schedule.total_epochs != total_epochs):
        raise ValueError("Formal optimizer schedule must match the declared horizon.")
    if weight_decay is not None and (type(weight_decay) not in (int, float)
                                    or not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError("Formal weight decay must be finite and nonnegative.")
    if native_sampling_protocol not in (None, "baseline_formal_v1"):
        raise ValueError("Unsupported formal native sampling protocol.")
    if native_sampling_protocol is not None and validation_scope != "canonical89":
        raise ValueError("Baseline formal sampling requires the canonical89 primary panel.")

    resolved_manifest, resolved_data_path, manifest_path = _load_data_binding(manifest, data_path)
    base_development_manifest_sha256 = str(resolved_manifest["manifest_sha256"])
    if formal_full or full_train_followup1000:
        _source, catalog = read_case_catalog(resolved_data_path)
        train_ids = tuple(record["case_id"] for record in catalog if record["split"] == "train")
        validation_ids = development_case_ids(resolved_manifest, "test")
        if len(train_ids) != 600:
            raise ValueError("Full-population Thermal preparation requires the original600 TRAIN cases.")
        if validation_scope == "canonical89":
            original90 = tuple(record["case_id"] for record in catalog if record["split"] == "test")
            validation_ids = tuple(case_id for case_id in original90 if case_id != "0273")
            if (len(original90) != 90 or original90.count("0273") != 1
                    or len(validation_ids) != 89 or len(set(validation_ids)) != 89
                    or set(train_ids).intersection(validation_ids)):
                raise ValueError("Formal validation requires original90 with duplicate0273 excluded from canonical89.")
        if full_train_followup1000:
            followup_membership = validate_full_train_followup_membership(train_ids, validation_ids)
            followup_manifest = {
                "execution_protocol": FULL_TRAIN_FOLLOWUP_PROTOCOL,
                "dataset_protocol": FULL_TRAIN_FOLLOWUP_DATASET,
                "base_manifest_sha256": resolved_manifest["manifest_sha256"],
                "training_case_ids": list(train_ids),
                "training_membership_sha256": followup_membership["training_membership_sha256"],
                "validation_case_ids": list(validation_ids),
                "validation_membership_sha256": followup_membership["validation_membership_sha256"],
                "excluded_validation_id": "0273",
                "excluded_validation_reason": "historical duplicate in original90 validation; preserve canonical89",
            }
            resolved_manifest = {
                **resolved_manifest,
                "full_train_followup1000": followup_manifest,
                "manifest_sha256": _json_hash(followup_manifest),
            }
    else:
        train_ids = development_case_ids(resolved_manifest, "train")
        validation_ids = development_case_ids(resolved_manifest, "test")
        if len(train_ids) != 150 or len(validation_ids) != 22:
            raise ValueError("Joint Thermal development requires literal fixed25_v1 150/22 membership.")

    normalizer = fit_global_normalizer(resolved_data_path, train_ids)
    training_cases = _read_selected_cases(
        resolved_data_path,
        train_ids,
        split="train",
        normalizer=normalizer,
        include_grid=True,
    )
    validation_cases = [] if calibration_only else _read_selected_cases(
        resolved_data_path,
        validation_ids,
        split="test",
        normalizer=normalizer,
        include_grid=False,
    )
    build_balances, _response_scales = _load_recipe_helpers()
    balances = build_balances(training_cases)
    families = _read_response_families(Path(atlas_directory).expanduser().resolve(), TRAIN_RESPONSE_IDS)
    native_mask_binding = None
    response_mask_binding = None
    if flow_readout_law == NATIVE_CURL_READOUT_LAW:
        native_masks, native_mask_binding = _read_fixed_native_mask_catalog(
            resolved_data_path, train_ids, validation_ids,
            full_train_followup1000=full_train_followup1000,
        )
        for case in (*training_cases, *validation_cases):
            case["native_solid_mask"] = native_masks[str(case["case_id"])]
        response_masks, response_mask_binding = _read_train_response_native_masks(
            Path(atlas_directory).expanduser().resolve(), families,
        )
        for family in families:
            family["native_solid_mask"] = response_masks[str(family["family_id"])]

    from honf_runtime.compat import set_seed

    set_seed(int(seed))
    model_factory = (
        InteractionPreservingThermalAdapter
        if mode in INTERACTION_PRESERVING_THERMAL_MODES
        else JointThermalRegionalAdapter
    )
    model_options = {
        "mode": mode,
        "normalization_stats": normalizer.stats,
        "hidden": hidden,
        "message": message,
        "regional_anchors": regional_anchors,
        "depth": depth,
        "receiver_tile": receiver_tile,
        "locality_prior_strength": locality_prior_strength,
        "seed": seed,
    }
    if mode in INTERACTION_PRESERVING_THERMAL_MODES:
        model_options.update({"collective_width": collective_width, "max_sources": max_sources})
    if flow_readout_law is not None:
        model_options["flow_readout_law"] = flow_readout_law
    model = model_factory(**model_options).to(device)
    budget = {
        "fluid_queries": int(primary_queries),
        "material_queries_per_module": 32,
        "surface_stride": RESPONSE_SURFACE_STRIDE,
        "microbatch_cases": int(microbatch_size),
        "effective_cases": int(effective_batch_size),
        "operator_rows_per_case": int(operator_rows_per_case),
        "receiver_tile": int(receiver_tile),
    }
    if validation_queries is not None:
        if validation_queries < 1:
            raise ValueError("Joint Thermal validation query count must be positive.")
        validation_budget = {**budget, "fluid_queries": int(validation_queries)}
    else:
        validation_budget = None
    stat = resolved_data_path.stat()
    source_binding = {
        "dataset_path": str(resolved_data_path),
        "dataset_id": "thermal_channel_global_v1",
        "dataset_file_size_bytes": int(stat.st_size),
        "dataset_file_mtime_ns": int(stat.st_mtime_ns),
        "dataset_catalog_metadata_sha256": resolved_manifest["source"]["metadata_sha256"],
        "development_manifest_path": None if manifest_path is None else str(manifest_path),
        "development_manifest_sha256": base_development_manifest_sha256,
        "formal_scope": ("original600_train_full_followup1000" if full_train_followup1000 else
                         "original_train_600" if formal_full else "fixed25_v1_train_150"),
    }
    if full_train_followup1000:
        source_binding.update({
            "execution_protocol": FULL_TRAIN_FOLLOWUP_PROTOCOL,
            "dataset_protocol": FULL_TRAIN_FOLLOWUP_DATASET,
            "normalization_fit_scope": "exact original600 TRAIN cases only",
            "validation_scope": "canonical89 exposed original-H5 split=test; 0273 excluded by historical duplicate rule",
            "unexposed_original_test_targets_read": False,
            "full_followup_manifest_sha256": resolved_manifest["manifest_sha256"],
        })
    if flow_readout_law is not None:
        source_binding.update({
            "flow_readout_law": flow_readout_law,
            "native_geometry_mask_catalog": native_mask_binding,
            "response_geometry_mask_catalog": response_mask_binding,
        })
    provider = JointThermalTask(
        model=model,
        training_cases=training_cases,
        validation_cases=validation_cases,
        manifest=resolved_manifest,
        data_path=resolved_data_path,
        source_binding=source_binding,
        train_families=families,
        balances=balances,
        normalization_stats=normalizer.stats,
        mode=mode,
        device=device,
        budget=budget,
        effective_batch_size=effective_batch_size,
        total_epochs=total_epochs,
        formal_full=formal_full,
        response_coefficient=response_coefficient,
        operator_coefficient=operator_coefficient,
        operator_rows_per_case=operator_rows_per_case,
        atlas_directory=Path(atlas_directory).expanduser().resolve(),
        auxiliary_calibration=auxiliary_calibration,
        validation_scope=validation_scope,
        optimizer_schedule=optimizer_schedule,
        weight_decay=weight_decay,
        native_sampling_protocol=native_sampling_protocol,
        flow_readout_law=flow_readout_law,
        full_train_followup1000=full_train_followup1000,
    )
    if validation_budget is not None:
        provider.validation_budget = validation_budget
    if native_sampling_protocol == "baseline_formal_v1":
        provider.validation_sampling_indices = {case_id: index for index, case_id in enumerate(original90)
                                                if case_id in set(validation_ids)}
    return model, provider
