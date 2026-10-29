"""Manual Thermal full-data recipe for the shared interaction trainer.

The formal factory binds the completed Run3901 D-sep endpoint and the original
full-data split, then initializes a new affine refinement model from the
preserved Run3902 architecture metadata. It never imports development model
weights or optimizer moments. All optimization remains in ``TrainingEngine``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
for _source in (PROJECT_ROOT / "src", PROJECT_ROOT / "tools", PROJECT_ROOT / "Case_ThermalChannel/src"):
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

FORMAL_RUN_ROOT = Path("/data/wanglz/ModularDT/thermal_formal/rdirect_full5000_20261007_034358")
DEFAULT_FLOW_CHECKPOINT = FORMAL_RUN_ROOT / (
    "Run_3901_20261007_034358_ordinary_dsep_full5000_v1/epoch_5000_model.pt")
DEFAULT_RESPONSE_REFERENCE = FORMAL_RUN_ROOT / (
    "Run_3902_20261007_034358_r_direct_full5000_v1/epoch_5000_model.pt")
DEFAULT_FLOW_SHA256 = "aa751d6c51660b4c889f80da842a4e190e1ea80434adf96587312fd1dd769751"
DEFAULT_PROFILE = PROJECT_ROOT / "src/config_core/forward/thermal_source_response/r-direct_full5000.json"
DEFAULT_ATLAS_DIRECTORY = PROJECT_ROOT / "diagnostics/generated/interactions/physical_response_atlas_20260926/families"
FORMAL_DUPLICATE_ID = "0273"
FORMAL_SCHEMA_VERSION = 1
FORMAL_DATASET_SCOPE = "formal_full_train_v1"
FORMAL_ARCHITECTURE = {"hidden": 64, "message": 64, "background_mode": True}
REFINEMENT_ARCHITECTURE = {"base_width": 16, "router_hidden": 32}
SUPPORTED_FORMAL_SAMPLING_VERSION = "case_epoch_v1"
SUPPORTED_FORMAL_SAMPLING_VERSIONS = ("legacy_packed_v1", SUPPORTED_FORMAL_SAMPLING_VERSION)

from channelthermal.training.unified_task import ThermalRefinementTask, _resolve_query_budget


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _json_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {name: np.asarray(value, dtype=np.float32).tolist() for name, value in sorted(stats.items())}


def _validate_formal_sampling_version(value: Any) -> str:
    if value not in SUPPORTED_FORMAL_SAMPLING_VERSIONS:
        raise ValueError("Thermal formal sampling version must be legacy_packed_v1 or case_epoch_v1.")
    return str(value)


def _load_formal_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    from honf_runtime.compat import load_trusted_checkpoint

    flow_path = Path(config.get("flow_checkpoint", DEFAULT_FLOW_CHECKPOINT)).expanduser().resolve()
    reference_path = Path(config.get("response_reference_checkpoint", DEFAULT_RESPONSE_REFERENCE)).expanduser().resolve()
    if not flow_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError("The completed formal3901 flow and formal3902 architecture/reference checkpoints are required.")
    flow_sha = _sha256_file(flow_path)
    if flow_sha != DEFAULT_FLOW_SHA256:
        raise ValueError("Formal Thermal requires the exact immutable Run3901 D-sep e5000 checkpoint.")
    flow = load_trusted_checkpoint(flow_path, map_location="cpu")
    response_reference = load_trusted_checkpoint(reference_path, map_location="cpu")
    if (int(flow.get("epoch", -1)) != 5000 or flow.get("dependency_policy") != "D-sep"
            or int(response_reference.get("epoch", -1)) != 5000
            or response_reference.get("fit_identity", {}).get("mode") != "direct"):
        raise ValueError("Formal bindings require completed Run3901 D-sep and Run3902 R-direct e5000 references.")
    reference_recipe = response_reference.get("fit_identity", {}).get("recipe", {})
    if reference_recipe.get("flow_checkpoint_sha256") != flow_sha:
        raise ValueError("Run3902 architecture reference is not bound to the exact Run3901 e5000 flow.")
    if (response_reference.get("formal_dataset_binding") != flow.get("formal_dataset_binding")
            or response_reference.get("formal_normalization_binding") != flow.get("formal_normalization_binding")
            or response_reference.get("formal_validation_binding") != flow.get("formal_validation_binding")):
        raise ValueError("Completed Run3901 and Run3902 formal data/normalization/validation identities differ.")
    source_config = flow.get("train_config", {}).get("dataset", {})
    dataset_path = Path(config.get("dataset_path", source_config.get("packed_h5_path", ""))).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError("The formally bound packed Thermal H5 dataset is unavailable.")
    if Path(source_config.get("packed_h5_path", "")).expanduser().resolve() != dataset_path:
        raise ValueError("Explicit packed-H5 path differs from the immutable Run3901 dataset source.")
    model_config = response_reference.get("source_response_config")
    if not isinstance(model_config, Mapping):
        raise TypeError("Run3902 is missing the source-response architecture metadata.")
    core = dict(model_config.get("core", {}))
    adapter = dict(model_config.get("adapter", {}))
    if core.get("mode") != "direct" or int(core.get("hidden", -1)) != FORMAL_ARCHITECTURE["hidden"] \
            or int(core.get("message", -1)) != FORMAL_ARCHITECTURE["message"]:
        raise ValueError("Run3902 architecture does not match the sealed fresh H64/message64 R-direct profile.")
    if adapter.get("environment_flow_context", False):
        raise ValueError("Formal Thermal remains geometry-only; the optional flow-context branch is disabled.")
    return {
        "flow_path": flow_path,
        "flow_sha256": flow_sha,
        "flow": flow,
        "response_reference_path": reference_path,
        "response_reference_sha256": _sha256_file(reference_path),
        "response_reference": response_reference,
        "dataset_path": dataset_path,
        "source_dataset_config": source_config,
        "core_config": core,
        "adapter_config": adapter,
        "response_recipe": reference_recipe,
    }


def _bind_data(context: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str], list[str]]:
    from thermal_formal_profile import bind_formal_validation, bind_original_train

    profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
    data = profile["data"]
    train_binding, train_ids = bind_original_train(
        context["dataset_path"], expected_count=int(data["expected_train_case_count"]),
        dataset_id=context["source_dataset_config"].get("dataset_id"))
    _validation_binding, canonical_ids, compatibility_ids = bind_formal_validation(
        context["dataset_path"],
        expected_primary_count=int(data["formal_validation"]["expected_primary_case_count"]),
        expected_compatibility_count=int(data["formal_validation"]["expected_compatibility_case_count"]),
        duplicate_case_id=str(data["formal_validation"]["excluded_training_duplicate_case_id"]))
    if FORMAL_DUPLICATE_ID != data["formal_validation"]["excluded_training_duplicate_case_id"]:
        raise ValueError("The formal duplicate-exclusion rule differs from the preserved profile.")
    return profile, train_binding, canonical_ids, compatibility_ids, [*train_ids]


def _response_source_receipts(atlas_directory: Path, family_ids: Sequence[str]) -> list[dict[str, str]]:
    receipts = []
    for family_id in family_ids:
        path = (atlas_directory / f"train_{family_id}_responses.npz").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Declared original-TRAIN response source is missing: {path}")
        receipts.append({"family_id": str(family_id), "path": str(path), "sha256": _sha256_file(path)})
    return receipts


def _startup_validation_binding(context: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    from channelthermal.training.unified_task import DEFAULT_PARENT_CHECKPOINT, DEFAULT_PARENT_SHA256
    from honf_runtime.compat import load_trusted_checkpoint

    startup_profile = profile["data"]["startup_validation"]
    dataset = context["response_reference"].get("train_config", {}).get("dataset", {})
    embedded = dataset.get("development_subset")
    manifest_sha = str(startup_profile["manifest_sha256"])
    ids = []
    source = {}
    if isinstance(embedded, Mapping):
        embedded_manifest_sha = embedded.get("manifest_sha256")
        embedded_ids = embedded.get("partitions", {}).get("test", {}).get("case_ids", [])
        if embedded_manifest_sha == manifest_sha and embedded_ids:
            ids = list(map(str, embedded_ids))
            source = {"kind": "formal3902_embedded_fixed25_manifest",
                      "architecture_reference_checkpoint": str(context["response_reference_path"]),
                      "architecture_reference_checkpoint_sha256": context["response_reference_sha256"]}
    if not ids:
        parent_path = DEFAULT_PARENT_CHECKPOINT.expanduser().resolve()
        if not parent_path.is_file() or _sha256_file(parent_path) != DEFAULT_PARENT_SHA256:
            raise ValueError("Fixed25 startup fallback requires the exact retained R-direct2500 metadata checkpoint.")
        parent = load_trusted_checkpoint(parent_path, map_location="cpu")
        parent_recipe = parent.get("fit_identity", {}).get("recipe", {})
        if parent_recipe.get("manifest_sha256") != manifest_sha:
            raise ValueError("Retained R-direct parent metadata does not bind the sealed fixed25_v1 manifest.")
        ids = list(map(str, parent_recipe.get("validation_case_ids", [])))
        source = {"kind": "retained_fixed25_v1_manifest_metadata",
                  "checkpoint": str(parent_path), "checkpoint_sha256": DEFAULT_PARENT_SHA256}
    if (len(ids) != int(startup_profile["expected_case_count"])
            or len(ids) != len(set(ids))):
        raise ValueError("Startup validation IDs must be the exact unique sealed fixed25_v1 DEV22 panel.")
    return {"manifest_sha256": manifest_sha, "case_ids": ids, "source": source}


def _validated_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    from thermal_formal_profile import validate_formal_bindings

    context = _load_formal_sources(config)
    profile, dataset_binding, canonical_ids, compatibility_ids, train_ids = _bind_data(context)
    flow = context["flow"]
    validation_binding = flow.get("formal_validation_binding")
    normalization_binding = flow.get("formal_normalization_binding")
    stats = flow.get("global_normalization_stats")
    if not isinstance(stats, Mapping) or not isinstance(normalization_binding, Mapping):
        raise TypeError("Run3901 is missing its full-TRAIN transform values or binding.")
    validate_formal_bindings(flow, dataset_binding, normalization_binding, validation_binding, stats)
    if (flow.get("formal_dataset_binding") != dataset_binding
            or list(validation_binding.get("primary_case_ids", [])) != canonical_ids
            or list(validation_binding.get("compatibility_case_ids", [])) != compatibility_ids):
        raise ValueError("Current source metadata differs from the completed formal3901/3902 data identities.")
    if len(train_ids) != 600 or len(canonical_ids) != 89 or len(compatibility_ids) != 90:
        raise ValueError("Thermal formal membership must remain 600 TRAIN, canonical89 and original90.")
    if context["response_recipe"].get("operator_decision", {}).get("operator_constraint") != "qualified":
        raise ValueError("Formal R-direct must retain the previously qualified TRAIN-only discrete operator.")
    startup_binding = _startup_validation_binding(context, profile)
    return {"context": context, "profile": profile, "dataset_binding": dataset_binding,
            "validation_binding": validation_binding, "normalization_binding": normalization_binding,
            "saved_normalization_stats": stats, "training_case_ids": train_ids,
            "canonical_validation_ids": canonical_ids, "compatibility_validation_ids": compatibility_ids,
            "startup_validation_binding": startup_binding}


def _make_fresh_model(context: Mapping[str, Any], seed: int, temperature_std: float):
    from channelthermal.source_response import ThermalSourceResponse
    from honf_forward_core.interface_fields.interaction_refinement import RefinedSourceResponseOperator
    from honf_runtime.compat import set_seed

    set_seed(int(seed))
    model = ThermalSourceResponse(context["core_config"], **context["adapter_config"])
    fresh_core_hash = _tensor_state_sha256(model)
    refined = RefinedSourceResponseOperator.from_fine(
        model.core, **REFINEMENT_ARCHITECTURE, residual_scale=float(temperature_std))
    refined.refinement.residual_scale.fill_(float(temperature_std))
    model.core = refined
    model.core_config = copy.deepcopy(refined.config)
    return model, fresh_core_hash, _tensor_state_sha256(model)


def prepare_recipe(config: Mapping[str, Any], *, metadata_only: bool = False) -> dict[str, Any]:
    """Bind formal metadata, or fit the full-TRAIN transforms/calibration for later manual use.

    Metadata-only preparation reads packed-H5 catalog metadata and the exact
    completed Run3901/3902 checkpoint bindings. It never loads field arrays,
    constructs an optimizer, or marks the recipe ready to train.
    """
    from thermal_formal_profile import fit_formal_normalizer, stats_sha256, validate_formal_bindings

    from channelthermal.training.unified_task import (
        DEFAULT_ATLAS_DIRECTORY,
        TRAIN_RESPONSE_IDS,
        _read_response_families,
    )
    from honf_runtime.compat import set_seed

    config = dict(config)
    metadata = _validated_metadata(config)
    context = metadata["context"]
    flow = context["flow"]
    profile = metadata["profile"]
    training_budget, query_budget_override = _resolve_query_budget(
        profile["budget"], config.get("query_budget_override"))
    sampling_version = config.get("sampling_version")
    if sampling_version is not None:
        sampling_version = _validate_formal_sampling_version(sampling_version)
    dataset_path = context["dataset_path"]
    stats = metadata["saved_normalization_stats"]
    normalization_binding = dict(metadata["normalization_binding"])
    atlas_directory = Path(config.get("atlas_directory", DEFAULT_ATLAS_DIRECTORY)).expanduser().resolve()
    response_sources = _response_source_receipts(atlas_directory, TRAIN_RESPONSE_IDS)
    reference_sources = context["response_recipe"].get("training_response_sources", [])
    reference_ids = tuple(str(row.get("family_id")) for row in reference_sources)
    if reference_ids and reference_ids != TRAIN_RESPONSE_IDS:
        raise ValueError("Run3902 response calibration must retain the four original-TRAIN family IDs.")
    if reference_sources:
        current_by_id = {row["family_id"]: row for row in response_sources}
        for row in reference_sources:
            family_id = str(row["family_id"])
            current = current_by_id[family_id]
            if (str(row.get("sha256")) != current["sha256"]
                    or Path(str(row.get("path", ""))).expanduser().resolve() != Path(current["path"])):
                raise ValueError(f"Formal response source {family_id} differs from the Run3902 TRAIN calibration.")

    calibration = None
    fresh_core_hash = None
    fresh_model_hash = None
    ready = False
    if not metadata_only:
        normalizer, fresh_normalization_binding = fit_formal_normalizer(dataset_path, metadata["dataset_binding"])
        stats = normalizer.stats
        if fresh_normalization_binding != normalization_binding:
            raise ValueError("Fresh full-TRAIN normalization differs from the exact completed Run3901 transform binding.")
        validate_formal_bindings(flow, metadata["dataset_binding"], normalization_binding,
                                 metadata["validation_binding"], stats)
        from thermal_source_response_fit import build_balances, calibrate, read_scoped_cases

        dataset_config = dict(context["source_dataset_config"])
        dataset_config["packed_h5_path"] = str(dataset_path)
        training_cases = read_scoped_cases(
            dataset_config, "train", metadata["training_case_ids"], stats, include_grid=True)
        if [str(case["case_id"]) for case in training_cases] != metadata["training_case_ids"]:
            raise ValueError("Full-TRAIN field loader changed the sealed source case order.")
        balances = build_balances(training_cases)
        from channelthermal.source_response import ThermalSourceResponse

        train_families = _read_response_families(atlas_directory, TRAIN_RESPONSE_IDS)
        actual_sources = [{"family_id": str(family["family_id"]),
                           "path": str(Path(family["source"]).resolve()),
                           "sha256": str(family["source_sha256"])} for family in train_families]
        if actual_sources != response_sources:
            raise ValueError("Loaded original-TRAIN response arrays differ from the metadata-only source catalog.")
        operator_decision = context["response_recipe"]["operator_decision"]
        if operator_decision.get("operator_constraint") != "qualified":
            raise ValueError("The fresh formal recipe requires the existing qualified TRAIN operator constraint.")
        set_seed(int(profile["seed"]))
        calibration_model = ThermalSourceResponse(context["core_config"], **context["adapter_config"])
        calibration = calibrate(
            {"direct": calibration_model}, training_cases, train_families, stats, balances, True,
            profile["budget"], float(profile["loss"]["q_proxy_coefficient"]))
        saved_response_calibration = context["response_recipe"].get("calibration", {})
        if calibration.get("response_scales") != saved_response_calibration.get("response_scales"):
            raise ValueError("The existing four-family TRAIN response scales differ from the formal3902 reference.")
        calibration["response_coefficient"] = float(saved_response_calibration["response_coefficient"])
        calibration["response_scales"] = dict(saved_response_calibration["response_scales"])
        calibration["response_scale_source"] = "preserved formal3902 four-family TRAIN calibration"
        calibration["operator_calibration_source"] = "fresh full-TRAIN Run3901-bound balance calibration"
        if calibration["operator_coefficient"] < 0 or not np.isfinite(calibration["operator_coefficient"]):
            raise ValueError("Fresh formal TRAIN operator calibration produced an invalid coefficient.")
        # The calibration is gradient-only. Rebuild under the declared seed to
        # seal the exact fresh e0 state that the manual factory will construct.
        fresh_model, fresh_core_hash, fresh_model_hash = _make_fresh_model(
            context, int(profile["seed"]), float(np.asarray(stats["field_std_by_channel"]).reshape(-1)[4]))
        del fresh_model
        ready = True

    reference_recipe = context["response_recipe"]
    operator_decision = copy.deepcopy(reference_recipe["operator_decision"])
    startup_binding = metadata["startup_validation_binding"]
    result = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "task": "thermal",
        "identity": "thermal_affine_interaction_refinement_formal_v1",
        "ready_for_training": ready,
        "metadata_only": bool(metadata_only),
        "dataset_scope": FORMAL_DATASET_SCOPE,
        "dataset_path": str(dataset_path),
        "atlas_directory": str(atlas_directory),
        "dataset_binding": metadata["dataset_binding"],
        "validation_binding": metadata["validation_binding"],
        "normalization_binding": normalization_binding,
        "normalization_stats": None if metadata_only else _json_stats(stats),
        "normalization_stats_sha256": stats_sha256(stats),
        "training_case_ids": metadata["training_case_ids"],
        "canonical_validation_ids": metadata["canonical_validation_ids"],
        "compatibility_validation_ids": metadata["compatibility_validation_ids"],
        "startup_validation_ids": startup_binding["case_ids"],
        "startup_validation_manifest_sha256": startup_binding["manifest_sha256"],
        "startup_validation_source": startup_binding["source"],
        "flow_checkpoint": str(context["flow_path"]),
        "flow_checkpoint_sha256": context["flow_sha256"],
        "flow_component_age": int(flow["epoch"]),
        "flow_inference_inputs": "none; frozen formal3901 lineage only",
        "architecture_reference_checkpoint": str(context["response_reference_path"]),
        "architecture_reference_checkpoint_sha256": context["response_reference_sha256"],
        "architecture_reference_weights_loaded": False,
        "fresh_model_seed": int(profile["seed"]),
        "fresh_fine_state_sha256": fresh_core_hash,
        "fresh_refined_state_sha256": fresh_model_hash,
        "architecture": {"core": context["core_config"], "adapter": context["adapter_config"],
                         "refinement": REFINEMENT_ARCHITECTURE},
        "budget": training_budget,
        "schedule": {"total_epochs": 5000, "hold_through_epoch": 2000, "final_lr": 3.0e-6,
                     "common_stages": {"warmup_epochs": 500, "open_through_epoch": 600,
                                       "soft_through_epoch": 800},
                     "monitor_every": 100},
        "optimizer": dict(profile["optimizer"]),
        "q_proxy_coefficient": float(profile["loss"]["q_proxy_coefficient"]),
        "training_response_families": list(TRAIN_RESPONSE_IDS),
        "training_response_sources": response_sources,
        "response_reference_scales": dict(reference_recipe["calibration"]["response_scales"]),
        "operator_decision": operator_decision,
        "calibration": calibration,
        "optimizer_moments": "fresh AdamW; no Run3901/3902 or development moments",
        "execution_arms": ["full_detail", "adaptive_detail"],
        "selection": {"field_metric": "canonical89 field_score", "response_guard": None,
                      "response_reference_context": "formal3902 only; not used as an independent selector"},
        "solver_attempts": 0,
    }
    if query_budget_override is not None:
        # Keep coefficient calibration and all exposed validation queries on
        # the established Q1024 profile while TRAIN alone uses the override.
        result["training_query_budget_override"] = query_budget_override
        result["calibration_budget"] = dict(profile["budget"])
        result["validation_budget"] = dict(profile["budget"])
    if sampling_version is not None:
        result["sampling_version"] = sampling_version
    validate_recipe(result)
    return result


def validate_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate source, flow, full-data split and normalization identities without fitting."""
    recipe = dict(recipe)
    if recipe.get("schema_version") != FORMAL_SCHEMA_VERSION or recipe.get("dataset_scope") != FORMAL_DATASET_SCOPE:
        raise ValueError("Thermal formal recipe has an unknown schema or dataset scope.")
    metadata = _validated_metadata(recipe)
    profile = metadata["profile"]
    expected_budget, query_budget_override = _resolve_query_budget(
        profile["budget"], recipe.get("training_query_budget_override"))
    if recipe.get("budget") != expected_budget:
        raise ValueError("Thermal formal training budget differs from the permitted profile/query override.")
    if query_budget_override is None:
        if "calibration_budget" in recipe or "validation_budget" in recipe:
            raise ValueError("Default Thermal formal recipes must retain the single established query budget.")
    elif (recipe.get("calibration_budget") != dict(profile["budget"])
          or recipe.get("validation_budget") != dict(profile["budget"])):
        raise ValueError("Thermal calibration and exposed validation budgets must remain at the baseline profile.")
    if "sampling_version" in recipe:
        _validate_formal_sampling_version(recipe["sampling_version"])
    for field, expected in (("dataset_binding", metadata["dataset_binding"]),
                            ("validation_binding", metadata["validation_binding"]),
                            ("training_case_ids", metadata["training_case_ids"]),
                            ("canonical_validation_ids", metadata["canonical_validation_ids"]),
                            ("compatibility_validation_ids", metadata["compatibility_validation_ids"]),
                            ("startup_validation_ids", metadata["startup_validation_binding"]["case_ids"]),
                            ("startup_validation_manifest_sha256", metadata["startup_validation_binding"]["manifest_sha256"]),
                            ("startup_validation_source", metadata["startup_validation_binding"]["source"])):
        if recipe.get(field) != expected:
            raise ValueError(f"Thermal formal recipe changed its {field}.")
    if (Path(recipe.get("dataset_path", "")).expanduser().resolve() != metadata["context"]["dataset_path"]
            or recipe.get("flow_checkpoint_sha256") != metadata["context"]["flow_sha256"]
            or recipe.get("flow_checkpoint") != str(metadata["context"]["flow_path"])):
        raise ValueError("Thermal formal recipe changed its packed-H5 or frozen Run3901 lineage.")
    if (recipe.get("architecture_reference_checkpoint") != str(metadata["context"]["response_reference_path"])
            or recipe.get("architecture_reference_checkpoint_sha256")
            != metadata["context"]["response_reference_sha256"]):
        raise ValueError("Thermal formal recipe changed its Run3902 architecture metadata source.")
    from channelthermal.training.unified_task import DEFAULT_ATLAS_DIRECTORY, TRAIN_RESPONSE_IDS

    atlas_directory = Path(recipe.get("atlas_directory", DEFAULT_ATLAS_DIRECTORY)).expanduser().resolve()
    response_sources = recipe.get("training_response_sources", [])
    if ([str(row.get("family_id")) for row in response_sources] != list(TRAIN_RESPONSE_IDS)
            or recipe.get("training_response_families") != list(TRAIN_RESPONSE_IDS)):
        raise ValueError("Thermal formal recipe changed the four original-TRAIN response families.")
    for row in response_sources:
        source = Path(str(row.get("path", ""))).expanduser().resolve()
        expected_source = (atlas_directory / f"train_{row['family_id']}_responses.npz").resolve()
        if (source != expected_source or not source.is_file()
                or _sha256_file(source) != row.get("sha256")):
            raise ValueError(f"Thermal formal response source {row.get('family_id')} changed after preparation.")
    ready = bool(recipe.get("ready_for_training"))
    if ready:
        from thermal_formal_profile import stats_sha256, validate_formal_bindings

        stats = recipe.get("normalization_stats")
        if not isinstance(stats, Mapping) or recipe.get("metadata_only"):
            raise ValueError("Training-ready Thermal formal recipe requires fresh full-TRAIN transform values.")
        if stats_sha256(stats) != recipe.get("normalization_stats_sha256"):
            raise ValueError("Thermal formal normalization values differ from their sealed hash.")
        if recipe.get("normalization_binding") != metadata["normalization_binding"]:
            raise ValueError("Thermal formal normalization binding differs from immutable Run3901.")
        validate_formal_bindings(metadata["context"]["flow"], metadata["dataset_binding"],
            recipe["normalization_binding"], metadata["validation_binding"], stats)
        if not recipe.get("fresh_refined_state_sha256") or not recipe.get("calibration"):
            raise ValueError("Training-ready Thermal formal recipe is missing fresh initialization/calibration receipts.")
    return {
        "status": "formal_metadata_validated" if not ready else "formal_train_recipe_validated",
        "ready_for_training": ready,
        "training_case_count": len(metadata["training_case_ids"]),
        "canonical_validation_case_count": len(metadata["canonical_validation_ids"]),
        "compatibility_validation_case_count": len(metadata["compatibility_validation_ids"]),
        "duplicate_training_id_in_compatibility": FORMAL_DUPLICATE_ID in metadata["compatibility_validation_ids"],
        "flow_checkpoint_sha256": metadata["context"]["flow_sha256"],
        "solver_attempts": 0,
    }


class FormalThermalRefinementTask(ThermalRefinementTask):
    """Formal cohort wrapper around the maintained Thermal loss/sampling adapter."""

    def __init__(self, *args, formal_recipe: Mapping[str, Any], startup_benchmark: bool = False, **kwargs):
        self.formal_recipe = dict(formal_recipe)
        self.startup_benchmark = bool(startup_benchmark)
        super().__init__(*args, **kwargs)

    def identity_payload(self) -> Mapping[str, Any]:
        payload = {
            "task": "ThermalChannel",
            "dataset_split": FORMAL_DATASET_SCOPE,
            "dataset_binding": self.formal_recipe["dataset_binding"],
            "validation_binding": self.formal_recipe["validation_binding"],
            "normalization_binding": self.formal_recipe["normalization_binding"],
            "normalization_stats_sha256": self.formal_recipe["normalization_stats_sha256"],
            "flow_checkpoint": self.formal_recipe["flow_checkpoint"],
            "flow_checkpoint_sha256": self.formal_recipe["flow_checkpoint_sha256"],
            "flow_inference_inputs": "none; frozen formal3901 lineage only",
            "architecture_reference_checkpoint": self.formal_recipe["architecture_reference_checkpoint"],
            "architecture_reference_checkpoint_sha256": self.formal_recipe["architecture_reference_checkpoint_sha256"],
            "architecture_reference_weights_loaded": False,
            "training_response_families": self.formal_recipe["training_response_families"],
            "training_response_sources": self.formal_recipe["training_response_sources"],
            "response_reference_scales": self.formal_recipe["response_reference_scales"],
            "operator_decision": self.formal_recipe["operator_decision"],
            "fresh_fine_state_sha256": self.formal_recipe["fresh_fine_state_sha256"],
            "fresh_refined_state_sha256": self.formal_recipe["fresh_refined_state_sha256"],
            "fresh_adamw_moments": True,
            "startup_validation_manifest_sha256": self.formal_recipe["startup_validation_manifest_sha256"],
            "startup_validation_source": self.formal_recipe["startup_validation_source"],
            "startup_benchmark": self.startup_benchmark,
            "validation_scope": ("fixed25_v1_DEV22_exposed" if self.startup_benchmark
                                 else "canonical89 primary plus original90 compatibility"),
            "output_law": "affine",
            "applicable_control": "heat; supplied only to affine application",
            "inference_scene_keys": ["prescribed geometry/material/boundary context"],
            "heat_and_stored_uv_in_scene": False,
            "stored_uv_supervision_only": True,
            "solver_attempts": 0,
        }
        if "training_query_budget_override" in self.formal_recipe:
            payload["training_query_budget"] = dict(self.formal_recipe["budget"])
            payload["calibration_query_budget"] = dict(self.formal_recipe["calibration_budget"])
            payload["validation_query_budget"] = dict(self.formal_recipe["validation_budget"])
        if self.formal_recipe.get("sampling_version") == SUPPORTED_FORMAL_SAMPLING_VERSION:
            payload["manifest_fingerprint"] = str(
                self.formal_recipe["dataset_binding"]["training_case_ids_sha256"])
        return payload

    def reduce_native_metrics(self, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        rows = [row for record in records for row in record["case_rows"]]
        if self.startup_benchmark:
            expected_ids = self.formal_recipe["startup_validation_ids"]
            if len(rows) != len(expected_ids) or {row["case_id"] for row in rows} != set(expected_ids):
                raise ValueError("Formal startup review must visit the exact declared DEV22 rows once.")
            return _aggregate_native_rows(rows, scope="fixed25_v1 DEV22 exposed startup review")
        compatibility_ids = self.formal_recipe["compatibility_validation_ids"]
        canonical_ids = self.formal_recipe["canonical_validation_ids"]
        if len(rows) != 90 or {row["case_id"] for row in rows} != set(compatibility_ids):
            raise ValueError("Formal monitoring must visit all original90 compatibility rows exactly once.")
        by_id = {row["case_id"]: row for row in rows}
        canonical = [by_id[case_id] for case_id in canonical_ids]
        result = _aggregate_native_rows(canonical, scope="formal canonical89 primary; exposed validation")
        compat = _aggregate_native_rows(rows, scope="formal original90 compatibility; exposed validation")
        result.update({f"original90_{name}": value for name, value in compat.items()
                       if name in {"field_score", "case_count", "case_ids"}
                       or name.endswith(("_mean", "_p90"))})
        result["compatibility_duplicate_case_id"] = FORMAL_DUPLICATE_ID
        return result

    def extra_validation_metrics(self, model: Any, arm: str, phase: str) -> Mapping[str, Any]:
        del model, arm, phase
        return {"formal_response_guard": None,
                "response_reference_context": self.formal_recipe["selection"]["response_reference_context"]}

    def optimizer_groups(self, model: torch.nn.Module, arm: str, stage: str):
        del arm, stage
        from honf_runtime.unified_training import OptimizerGroupSpec, ScheduleSpec

        fine_names, refinement_names = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            (refinement_names if name.startswith("core.refinement.") else fine_names).append(name)
        schedule = ScheduleSpec(peak_lr=3.0e-4, warmup_start_lr=3.0e-4, warmup_epochs=0,
                                hold_through_epoch=2000, total_epochs=5000, final_lr=3.0e-6)
        weight_decay = float(self.formal_recipe["optimizer"]["weight_decay"])
        return (OptimizerGroupSpec("thermal_fine", tuple(fine_names), schedule,
                                   weight_decay=weight_decay),
                OptimizerGroupSpec("thermal_refinement", tuple(refinement_names), schedule,
                                   weight_decay=weight_decay))

    def preparation_summary(self) -> Mapping[str, Any]:
        return {"subset_id": FORMAL_DATASET_SCOPE,
                "training_cases": len(self.training_cases),
                "validation_cases": len(self.validation_cases),
                "canonical_validation_case_count": len(self.formal_recipe["canonical_validation_ids"]),
                "compatibility_validation_case_count": len(self.formal_recipe["compatibility_validation_ids"]),
                "startup_benchmark": self.startup_benchmark,
                "training_response_families": self.formal_recipe["training_response_families"],
                "operator_residual_enabled": self.use_operator,
                "operator_residual_training_rows_per_case": self.formal_recipe["budget"]["operator_rows_per_case"],
                "heat_and_stored_uv_in_scene": False,
                "stored_uv_supervision_only": True,
                "optimizer_seed": None,
                "solver_attempts": 0}


def _aggregate_native_rows(rows: Sequence[Mapping[str, Any]], *, scope: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("Thermal validation metrics require at least one native case row.")
    result: dict[str, Any] = {"scope": scope, "case_count": len(rows),
                              "case_ids": [str(row["case_id"]) for row in rows],
                              "temperature_unit": "packed_dataset_native_temperature",
                              "field_score": float(np.mean([row["field_score"] for row in rows]))}
    for metric in ("fluid_temperature_rmse", "surface_temperature_rmse", "material_temperature_rmse",
                   "module_peak_rmse", "q_proxy_rmse"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_p90"] = float(np.quantile(values, 0.9))
    for metric in ("fluid_temperature_mse_standardized", "surface_temperature_mse_standardized",
                   "material_temperature_mse_standardized"):
        result[f"{metric}_mean"] = float(np.mean([row[metric] for row in rows]))
    return result


def create_task(recipe: Mapping[str, Any]):
    """Construct a fresh formal Thermal model/provider; return no optimizer seed."""
    recipe = dict(recipe)
    validation = validate_recipe(recipe)
    if not recipe.get("ready_for_training"):
        raise ValueError("Metadata-only Thermal formal recipes cannot construct a training task.")
    context = _validated_metadata(recipe)["context"]
    from thermal_source_response_fit import build_balances, read_scoped_cases

    from channelthermal.training.unified_task import (
        DEFAULT_ATLAS_DIRECTORY,
        TRAIN_RESPONSE_IDS,
        _read_response_families,
    )
    from honf_runtime.compat import set_seed

    del validation
    stats = {name: np.asarray(value, dtype=np.float32) for name, value in recipe["normalization_stats"].items()}
    dataset_config = dict(context["source_dataset_config"])
    dataset_config["packed_h5_path"] = recipe["dataset_path"]
    training_cases = read_scoped_cases(dataset_config, "train", recipe["training_case_ids"], stats,
                                       include_grid=True)
    if [str(case["case_id"]) for case in training_cases] != recipe["training_case_ids"]:
        raise ValueError("Thermal formal loader changed the bound original600 TRAIN order.")
    startup_benchmark = bool(recipe.get("startup_benchmark", False))
    validation_ids = (recipe["startup_validation_ids"] if startup_benchmark
                      else recipe["compatibility_validation_ids"])
    validation_cases = read_scoped_cases(dataset_config, "test", validation_ids, stats, include_grid=False)
    if [str(case["case_id"]) for case in validation_cases] != list(validation_ids):
        raise ValueError("Thermal formal validation loader changed its bound source order.")
    atlas = Path(recipe.get("atlas_directory", DEFAULT_ATLAS_DIRECTORY)).expanduser().resolve()
    families = _read_response_families(atlas, TRAIN_RESPONSE_IDS)
    balances = build_balances(training_cases) if recipe["operator_decision"]["operator_constraint"] == "qualified" else []
    if not balances or len(balances) != 600:
        raise ValueError("Thermal formal operator supervision requires the 600 full-TRAIN balances.")
    set_seed(int(recipe["fresh_model_seed"]))
    from channelthermal.source_response import ThermalSourceResponse
    from honf_forward_core.interface_fields.interaction_refinement import RefinedSourceResponseOperator

    model = ThermalSourceResponse(context["core_config"], **context["adapter_config"])
    fresh_core_hash = _tensor_state_sha256(model)
    refined = RefinedSourceResponseOperator.from_fine(
        model.core, **REFINEMENT_ARCHITECTURE,
        residual_scale=float(np.asarray(stats["field_std_by_channel"]).reshape(-1)[4]))
    refined.refinement.residual_scale.fill_(float(np.asarray(stats["field_std_by_channel"]).reshape(-1)[4]))
    model.core = refined
    model.core_config = copy.deepcopy(refined.config)
    if (fresh_core_hash != recipe["fresh_fine_state_sha256"]
            or _tensor_state_sha256(model) != recipe["fresh_refined_state_sha256"]):
        raise ValueError("Fresh Thermal fine/base/router initialization differs from the fully prepared recipe.")
    if model.adapter_config().get("environment_flow_context", False):
        raise ValueError("Thermal formal training must preserve geometry-only inference context.")

    provider_recipe = {
        "budget": recipe["budget"],
        "weight_decay": recipe["optimizer"]["weight_decay"],
        "operator_decision": recipe["operator_decision"],
        "calibration": recipe["calibration"],
    }
    provider = FormalThermalRefinementTask(
        model=model, parent_model=None, parent=None, parent_path=None,
        flow_path=Path(recipe["flow_checkpoint"]), atlas_directory=atlas,
        training_cases=training_cases, validation_cases=validation_cases,
        manifest={"manifest_sha256": recipe["dataset_binding"]["training_case_ids_sha256"]},
        train_families=families, development_families=(), balances=balances,
        optimizer_seed=None, device=recipe.get("device", "cpu"), stats=stats, recipe=provider_recipe,
        validation_budget=recipe.get("validation_budget"),
        calibration_budget=recipe.get("calibration_budget"),
        formal_recipe=recipe, startup_benchmark=startup_benchmark,
    )
    return model, provider, None
