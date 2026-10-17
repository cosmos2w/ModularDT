"""Strict Round6 contracts for frozen receiver-readout experiments."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .inherited_attention_protocol import validate_inherited_attention_recipe
from .moment_hyperedge_protocol import (
    MOMENT_CORE_SCHEMA,
    MOMENT_FAMILY,
    MOMENT_PROTOCOL,
    MOMENT_WORKFLOW,
    canonical_sha256,
    file_sha256,
    native_recipe_for_moment_hyperedges,
    validate_moment_hyperedge_recipe,
)

READOUT_FAMILY = "hyperedge_readout_round6"
READOUT_WORKFLOW = "hyperedge_readout_fit_v1"
READOUT_PROTOCOL = "frozen_hyperedge_readout_diagnostic_v1"
READOUT_RECIPE_SCHEMA = "hyperedge_readout_recipe_v1"
READOUT_ATTACHMENT_SCHEMA = "hyperedge_readout_parent_binding_v1"
READOUT_CORE_SCHEMA = "hyperedge_readout_core_v1"
READOUT_SOURCE_EPOCH = 2500
READOUT_EXTENSION_SOURCE_EPOCH = 1000
READOUT_HEALTH_REVIEW_AGE = 100
READOUT_PRIMARY_FIT_AGE = 500
READOUT_MAX_FIT_AGE = 1000
READOUT_JOINT_FINAL_EPOCH = 2500
READOUT_SAMPLER_OFFSET = 2500
READOUT_VARIANT = "H"
READOUT_REDUCTIONS = frozenset(("pool_first", "edge_first"))
READOUT_ACCESS_MODES = frozenset(("local_cover", "full_bank_diagnostic"))
READOUT_INIT_SEED = 0
READOUT_EXTENSION_DECISION_SCHEMA = "hyperedge_readout_extension_decision_v1"

_RECIPE_FIELDS = frozenset({
    "schema", "family", "model_family", "workflow", "protocol", "dataset",
    "phase", "representation", "architecture", "variant", "source_checkpoint",
    "source_checkpoint_sha256", "source_recipe_sha256", "source_epoch",
    "source_provider_identity_sha256", "native_recipe", "native_recipe_sha256",
    "receiver_reduction", "receiver_access", "geometry_mode",
    "receiver_readout_init_seed", "core_config", "fit_age_ceiling",
    "initial_review_stop", "sampler_epoch_mapping", "frozen_parameter_policy",
    "joint_gate_binding",
})


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _canonical_checkpoint_path(value: Any) -> str:
    if (not isinstance(value, str) or not Path(value).is_absolute()
            or str(Path(value).resolve()) != value):
        raise ValueError("Round6 source_checkpoint must be an absolute canonical path.")
    return value


def _validate_joint_gate_binding(
    value: Any, *, dataset: str | None, verify_files: bool = False,
) -> dict[str, Any]:
    """Validate the three concrete receipts required before joint confirmation."""
    fields = {"schema", "dataset", "local_signal", "native_law", "resource_reservation"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Joint confirmation requires the exact TRAIN signal, native-law and resource gate binding.")
    binding = copy.deepcopy(dict(value))
    if (binding.get("schema") != "hyperedge_readout_joint_gate_binding_v1"
            or binding.get("dataset") not in ("thermal", "wind")
            or (dataset is not None and binding.get("dataset") != dataset)):
        raise ValueError("Joint gate binding has an unsupported schema or differs from its dataset.")
    receipt_fields = {
        "local_signal": ("useful_signal", "hyperedge_readout_local_signal_v1"),
        "native_law": ("passed", "hyperedge_readout_native_law_v1"),
        "resource_reservation": ("approved", "hyperedge_readout_resource_reservation_v1"),
    }
    for name, (decision_key, expected_schema) in receipt_fields.items():
        row = binding.get(name)
        if (not isinstance(row, Mapping) or set(row) != {"path", "sha256", decision_key}
                or not isinstance(row.get("path"), str)
                or not Path(row["path"]).is_absolute()
                or str(Path(row["path"]).resolve()) != row["path"]
                or not _is_sha256(row.get("sha256"))
                or row.get(decision_key) is not True):
            raise ValueError(f"Joint gate binding has an invalid {name} receipt reference.")
        if verify_files:
            receipt_path = Path(row["path"]).resolve(strict=True)
            if file_sha256(receipt_path) != row["sha256"]:
                raise ValueError(f"Joint gate receipt {name!r} changed after review.")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (not isinstance(receipt, Mapping)
                    or receipt.get("schema") != expected_schema
                    or receipt.get("dataset") != binding["dataset"]
                    or receipt.get(decision_key) is not True):
                raise ValueError(f"Joint gate receipt {name!r} does not carry its required positive decision.")
    return binding


def verify_joint_gate_binding(value: Any, *, dataset: str) -> dict[str, Any]:
    """Recheck all receipt files immediately before a joint branch is prepared."""
    return _validate_joint_gate_binding(value, dataset=dataset, verify_files=True)


def verify_frozen_extension_decision(
    path: str | Path, *, recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the one paired 500-to-1000 extension decision to its exact TRAIN evidence."""
    value = validate_hyperedge_readout_recipe(recipe)
    if value["phase"] != "frozen_probe":
        raise ValueError("A frozen extension decision cannot authorize another Round6 phase.")
    receipt_path = Path(path).expanduser().resolve(strict=True)
    digest = file_sha256(receipt_path)
    decision = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema", "dataset", "phase", "source_checkpoint_sha256", "audit_panel_sha256",
        "fit_age_from", "fit_age_to", "arms", "paired_resource_reserved", "approved",
        "both_arms_finite_gradients", "any_arm_train_signal",
    }
    if not isinstance(decision, Mapping) or set(decision) != expected_fields:
        raise ValueError("Round6 extension decision receipt fields differ from the sealed schema.")
    if (decision.get("schema") != READOUT_EXTENSION_DECISION_SCHEMA
            or decision.get("dataset") != value["dataset"]
            or decision.get("phase") != "frozen_probe"
            or decision.get("source_checkpoint_sha256") != value["source_checkpoint_sha256"]
            or not _is_sha256(decision.get("audit_panel_sha256"))
            or decision.get("fit_age_from") != READOUT_PRIMARY_FIT_AGE
            or decision.get("fit_age_to") != READOUT_MAX_FIT_AGE
            or decision.get("paired_resource_reserved") is not True
            or decision.get("approved") is not True
            or decision.get("both_arms_finite_gradients") is not True
            or decision.get("any_arm_train_signal") is not True):
        raise ValueError("Round6 extension decision does not approve this exact paired TRAIN horizon.")
    arms = decision.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"pool_first", "edge_first"}:
        raise ValueError("Round6 extension decision must bind both matched local arms.")
    for name, signal in arms.items():
        if (not isinstance(signal, Mapping)
                or signal.get("schema") != "hyperedge_readout_extension_signal_v1"
                or type(signal.get("train_signal")) is not bool
                or signal.get("finite_gradients") is not True):
            raise ValueError(f"Round6 extension signal for {name!r} is malformed or has nonfinite gradients.")
    if not any(arms[name]["train_signal"] is True for name in arms):
        raise ValueError("Round6 extension decision has no positive arm-specific TRAIN signal.")
    return {
        "schema": "round6_frozen_extension_receipt_binding_v1",
        "path": str(receipt_path),
        "sha256": digest,
        "decision": copy.deepcopy(dict(decision)),
    }


def frozen_extension_resume_amendment(
    source_payload: Mapping[str, Any], *, recipe: Mapping[str, Any],
    stop_after: int, decision_path: str | Path | None,
) -> dict[str, Any] | None:
    """Require a receipt-bound 500 review before any frozen fit extends to 1000."""
    value = validate_hyperedge_readout_recipe(recipe)
    if value["phase"] != "frozen_probe":
        raise ValueError("Only the frozen Round6 phase has a 500-to-1000 extension.")
    saved_epoch = source_payload.get("epoch")
    if type(saved_epoch) is not int:
        raise ValueError("Round6 frozen resume checkpoint has no exact fit age.")
    if stop_after <= READOUT_PRIMARY_FIT_AGE:
        if decision_path is not None:
            raise ValueError("An extension receipt cannot be attached before the fit-500 review boundary.")
        return None
    if stop_after != READOUT_MAX_FIT_AGE or saved_epoch >= READOUT_MAX_FIT_AGE:
        raise ValueError("The only optional frozen extension is the completed paired fit-500-to-1000 horizon.")
    prior_amendments = source_payload.get("resume_amendments", [])
    prior = [row for row in prior_amendments if isinstance(row, Mapping)
             and row.get("kind") == "round6_frozen_extension"] if isinstance(prior_amendments, list) else []
    if saved_epoch == READOUT_PRIMARY_FIT_AGE:
        if prior:
            raise ValueError("The first frozen extension receipt must attach to the fit-500 checkpoint only once.")
        if decision_path is None:
            raise ValueError("Round6 fit 1000 requires an explicit positive TRAIN extension decision receipt.")
        binding = verify_frozen_extension_decision(decision_path, recipe=value)
        return {
            "kind": "round6_frozen_extension",
            "source_fit_epoch": READOUT_PRIMARY_FIT_AGE,
            "target_fit_epoch": READOUT_MAX_FIT_AGE,
            "extension_decision": binding,
            "paired_completion_required": True,
        }
    if saved_epoch < READOUT_PRIMARY_FIT_AGE or len(prior) != 1:
        raise ValueError("Round6 extension continuation lacks its exact fit-500 receipt-bearing resume amendment.")
    saved_binding = prior[0].get("extension_decision")
    if not isinstance(saved_binding, Mapping):
        raise TypeError("Round6 extension resume amendment must retain a decision receipt mapping.")
    saved_path = saved_binding.get("path")
    verified = verify_frozen_extension_decision(saved_path, recipe=value)
    if dict(verified) != dict(saved_binding):
        raise ValueError("Round6 extension receipt changed after the first continuation checkpoint.")
    if decision_path is not None:
        supplied = verify_frozen_extension_decision(decision_path, recipe=value)
        if dict(supplied) != dict(saved_binding):
            raise ValueError("Round6 resumed extension decision differs from its saved receipt binding.")
    return None


def readout_core_config(
    native_recipe: Mapping[str, Any], *, receiver_reduction: str,
    receiver_access: str, geometry_mode: str, init_seed: int = READOUT_INIT_SEED,
) -> dict[str, Any]:
    """Construct the strict R6 core config over the native H recipe dimensions."""
    native = validate_inherited_attention_recipe(native_recipe)
    base = {
        "schema": MOMENT_CORE_SCHEMA,
        "architecture": "IAG-C",
        **copy.deepcopy(dict(native["model"])),
    }
    if base.get("moment_variant") != READOUT_VARIANT:
        raise ValueError("Round6 receiver readouts attach only to the completed H moment core.")
    if receiver_reduction not in READOUT_REDUCTIONS:
        raise ValueError("Round6 reduction must be pool_first or edge_first.")
    if receiver_access not in READOUT_ACCESS_MODES:
        raise ValueError("Round6 access must be local_cover or full_bank_diagnostic.")
    if receiver_access == "full_bank_diagnostic" and receiver_reduction != "edge_first":
        raise ValueError("The full-bank diagnostic uses only the edge_first reducer.")
    expected_geometry = "scene_frame_normalized" if native["dataset"] == "thermal" else "native_signed"
    if geometry_mode != expected_geometry:
        raise ValueError(f"{native['dataset']} Round6 geometry mode must be {expected_geometry!r}.")
    if type(init_seed) is not int or init_seed != READOUT_INIT_SEED:
        raise ValueError("Round6 fixes the matched receiver-readout initialization seed at zero.")
    base.update({
        "schema": READOUT_CORE_SCHEMA,
        "receiver_reduction": receiver_reduction,
        "receiver_access": receiver_access,
        "geometry_mode": geometry_mode,
        "receiver_readout_init_seed": init_seed,
    })
    return base


def make_hyperedge_readout_recipe(
    source_checkpoint: str | Path, *, receiver_reduction: str,
    receiver_access: str = "local_cover", phase: str = "frozen_probe",
    joint_gate_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pin one strict Round6 recipe to a same-dataset H e2500 checkpoint."""
    from .compat import load_trusted_checkpoint

    path = Path(source_checkpoint).expanduser().resolve(strict=True)
    checkpoint = load_trusted_checkpoint(path, map_location="cpu")
    identity = checkpoint.get("experiment_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Round6 H source experiment identity must be a sealed mapping.")
    expected_source_epoch = (
        READOUT_SOURCE_EPOCH if phase == "frozen_probe" else READOUT_EXTENSION_SOURCE_EPOCH
    )
    if phase not in ("frozen_probe", "joint_confirmation"):
        raise ValueError("Round6 phase must be frozen_probe or joint_confirmation.")
    if phase == "frozen_probe" and joint_gate_binding is not None:
        raise ValueError("Frozen probes do not take joint-confirmation gate bindings.")
    if phase == "joint_confirmation":
        _validate_joint_gate_binding(joint_gate_binding, dataset=None, verify_files=True)
    if (checkpoint.get("workflow") != MOMENT_WORKFLOW
            or identity.get("family") != MOMENT_FAMILY
            or identity.get("model_family") != MOMENT_FAMILY
            or identity.get("workflow") != MOMENT_WORKFLOW
            or identity.get("protocol") != MOMENT_PROTOCOL
            or checkpoint.get("arm") != "IAG-C"
            or checkpoint.get("epoch") != expected_source_epoch):
        raise ValueError(f"Round6 {phase} requires an exact native IAG-C Round5 H checkpoint at e{expected_source_epoch}.")
    source_recipe = validate_moment_hyperedge_recipe(identity.get("recipe", {}))
    if source_recipe.get("variant") != READOUT_VARIANT:
        raise ValueError("Round6 receiver readouts require the source run's H variant.")
    if identity.get("recipe_sha256") != canonical_sha256(source_recipe):
        raise ValueError("Round5 H source recipe checksum is inconsistent.")
    if source_recipe.get("source_checkpoint_sha256") != file_sha256(source_recipe["source_checkpoint"]):
        raise ValueError("Round5 H source's own e500 parent no longer matches its pinned digest.")
    provider_identity = identity.get("provider_identity")
    if not isinstance(provider_identity, Mapping):
        raise TypeError("Round5 H source native provider identity must be a mapping.")
    native_recipe = native_recipe_for_moment_hyperedges(source_recipe)
    if (native_recipe.get("dataset") not in ("thermal", "wind")
            or native_recipe.get("architecture") != "IAG-C"
            or native_recipe.get("normalization") != "soft"
            or native_recipe.get("trajectory") != "fresh_soft"
            or native_recipe.get("total_epochs") != READOUT_SOURCE_EPOCH):
        raise ValueError("Round6 preserves the source H run's exact native soft development recipe.")
    if (phase == "joint_confirmation"
            and (not isinstance(joint_gate_binding, Mapping)
                 or joint_gate_binding.get("dataset") != native_recipe["dataset"])):
        raise ValueError("Joint-confirmation gate binding must name the source H dataset.")
    geometry_mode = "scene_frame_normalized" if native_recipe["dataset"] == "thermal" else "native_signed"
    core_config = readout_core_config(
        native_recipe,
        receiver_reduction=receiver_reduction,
        receiver_access=receiver_access,
        geometry_mode=geometry_mode,
    )
    source_sha = file_sha256(path)
    recipe = {
        "schema": READOUT_RECIPE_SCHEMA,
        "family": READOUT_FAMILY,
        "model_family": READOUT_FAMILY,
        "workflow": READOUT_WORKFLOW,
        "protocol": READOUT_PROTOCOL,
        "phase": phase,
        "dataset": native_recipe["dataset"],
        "representation": native_recipe["representation"],
        "architecture": "IAG-C",
        "variant": READOUT_VARIANT,
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": source_sha,
        "source_recipe_sha256": canonical_sha256(source_recipe),
        "source_epoch": expected_source_epoch,
        "source_provider_identity_sha256": canonical_sha256(provider_identity),
        "native_recipe": native_recipe,
        "native_recipe_sha256": canonical_sha256(native_recipe),
        "receiver_reduction": receiver_reduction,
        "receiver_access": receiver_access,
        "geometry_mode": geometry_mode,
        "receiver_readout_init_seed": READOUT_INIT_SEED,
        "core_config": core_config,
        "fit_age_ceiling": READOUT_MAX_FIT_AGE if phase == "frozen_probe" else READOUT_JOINT_FINAL_EPOCH,
        "initial_review_stop": READOUT_HEALTH_REVIEW_AGE if phase == "frozen_probe" else READOUT_JOINT_FINAL_EPOCH,
        "sampler_epoch_mapping": {
            "schema": "fit_age_to_absolute_sampling_epoch_v1",
            "fit_epoch_start": 1 if phase == "frozen_probe" else expected_source_epoch + 1,
            "sampler_epoch_start": (READOUT_SAMPLER_OFFSET + 1
                                    if phase == "frozen_probe" else expected_source_epoch + 1),
            "sampler_epoch_offset": READOUT_SAMPLER_OFFSET if phase == "frozen_probe" else 0,
        },
        "frozen_parameter_policy": ({
            "schema": "round6_frozen_readout_allowlist_v1",
            "trainable_prefixes": ["receiver_readout.", "field_head."],
            "thermal_additional_prefixes": ["response_phi."],
            "non_allowlisted_parameters": "requires_grad_false_and_bitwise_fixed",
            "all_buffers": "bitwise_fixed",
            "optimizer": "fresh_adamw_state",
        } if phase == "frozen_probe" else None),
        "joint_gate_binding": copy.deepcopy(dict(joint_gate_binding)) if joint_gate_binding is not None else None,
    }
    return validate_hyperedge_readout_recipe(recipe)


def validate_hyperedge_readout_recipe(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an exact, non-extensible frozen Round6 H readout contract."""
    if not isinstance(value, Mapping):
        raise TypeError("Round6 hyperedge-readout recipe must be a mapping.")
    recipe = copy.deepcopy(dict(value))
    if set(recipe) != _RECIPE_FIELDS:
        missing, extra = sorted(_RECIPE_FIELDS - set(recipe)), sorted(set(recipe) - _RECIPE_FIELDS)
        raise ValueError(f"Round6 recipe fields differ from schema; missing={missing}, extra={extra}.")
    exact = {
        "schema": READOUT_RECIPE_SCHEMA,
        "family": READOUT_FAMILY,
        "model_family": READOUT_FAMILY,
        "workflow": READOUT_WORKFLOW,
        "protocol": READOUT_PROTOCOL,
        "architecture": "IAG-C",
        "variant": READOUT_VARIANT,
        "receiver_readout_init_seed": READOUT_INIT_SEED,
    }
    for name, expected in exact.items():
        if recipe.get(name) != expected:
            raise ValueError(f"Round6 recipe requires {name}={expected!r}.")
    if recipe.get("dataset") not in ("thermal", "wind"):
        raise ValueError("Round6 dataset must be Thermal or Wind.")
    phase = recipe.get("phase")
    if phase not in ("frozen_probe", "joint_confirmation"):
        raise ValueError("Round6 phase must be frozen_probe or joint_confirmation.")
    expected_source_epoch = (
        READOUT_SOURCE_EPOCH if phase == "frozen_probe" else READOUT_EXTENSION_SOURCE_EPOCH
    )
    if recipe.get("source_epoch") != expected_source_epoch:
        raise ValueError(f"Round6 {phase} source epoch must be {expected_source_epoch}.")
    expected_ceiling = READOUT_MAX_FIT_AGE if phase == "frozen_probe" else READOUT_JOINT_FINAL_EPOCH
    expected_review_stop = READOUT_HEALTH_REVIEW_AGE if phase == "frozen_probe" else READOUT_JOINT_FINAL_EPOCH
    if (recipe.get("fit_age_ceiling") != expected_ceiling
            or recipe.get("initial_review_stop") != expected_review_stop):
        raise ValueError("Round6 fit ceiling or initial review stop differs from its sealed phase contract.")
    expected_representation = "iag_c"
    if recipe.get("representation") != expected_representation:
        raise ValueError("Round6 preserves the IAG-C native representation.")
    _canonical_checkpoint_path(recipe.get("source_checkpoint"))
    if not _is_sha256(recipe.get("source_checkpoint_sha256")):
        raise ValueError("Round6 recipe must pin the source checkpoint SHA-256.")
    if not _is_sha256(recipe.get("source_recipe_sha256")):
        raise ValueError("Round6 recipe must pin the Round5 H recipe SHA-256.")
    if not _is_sha256(recipe.get("source_provider_identity_sha256")):
        raise ValueError("Round6 recipe must pin the source native provider identity.")
    if recipe.get("receiver_reduction") not in READOUT_REDUCTIONS:
        raise ValueError("Round6 receiver reduction is unsupported.")
    if recipe.get("receiver_access") not in READOUT_ACCESS_MODES:
        raise ValueError("Round6 receiver access is unsupported.")
    if (recipe["receiver_access"] == "full_bank_diagnostic"
            and recipe["receiver_reduction"] != "edge_first"):
        raise ValueError("Full-bank diagnostic access requires edge_first reduction.")
    expected_geometry = "scene_frame_normalized" if recipe["dataset"] == "thermal" else "native_signed"
    if recipe.get("geometry_mode") != expected_geometry:
        raise ValueError(f"Round6 {recipe['dataset']} geometry mode must be {expected_geometry!r}.")
    native_recipe = validate_inherited_attention_recipe(recipe.get("native_recipe", {}))
    if (native_recipe.get("dataset") != recipe["dataset"]
            or native_recipe.get("architecture") != "IAG-C"
            or native_recipe.get("representation") != "iag_c"
            or native_recipe.get("normalization") != "soft"
            or native_recipe.get("trajectory") != "fresh_soft"
            or native_recipe.get("total_epochs") != READOUT_SOURCE_EPOCH
            or native_recipe.get("approved_stop_after") != READOUT_SOURCE_EPOCH
            or native_recipe.get("model", {}).get("schema") != MOMENT_CORE_SCHEMA
            or native_recipe.get("model", {}).get("moment_variant") != READOUT_VARIANT
            or native_recipe.get("model", {}).get("query_executor") != "indexed_projected"):
        raise ValueError("Round6 native recipe differs from the pinned soft H e2500 source run.")
    if canonical_sha256(native_recipe) != recipe.get("native_recipe_sha256"):
        raise ValueError("Round6 native recipe digest is inconsistent.")
    expected_core = readout_core_config(
        native_recipe,
        receiver_reduction=recipe["receiver_reduction"],
        receiver_access=recipe["receiver_access"],
        geometry_mode=expected_geometry,
        init_seed=READOUT_INIT_SEED,
    )
    if recipe.get("core_config") != expected_core:
        raise ValueError("Round6 receiver core config differs from the exact native H dimensions or readout mode.")
    expected_mapping = {
        "schema": "fit_age_to_absolute_sampling_epoch_v1",
        "fit_epoch_start": 1 if phase == "frozen_probe" else READOUT_EXTENSION_SOURCE_EPOCH + 1,
        "sampler_epoch_start": (READOUT_SAMPLER_OFFSET + 1 if phase == "frozen_probe"
                                else READOUT_EXTENSION_SOURCE_EPOCH + 1),
        "sampler_epoch_offset": READOUT_SAMPLER_OFFSET if phase == "frozen_probe" else 0,
    }
    if recipe.get("sampler_epoch_mapping") != expected_mapping:
        raise ValueError("Round6 fit age differs from the sealed absolute H-parent sampler mapping.")
    if phase == "frozen_probe":
        if recipe.get("joint_gate_binding") is not None:
            raise ValueError("Frozen probes cannot carry joint-confirmation gate bindings.")
    else:
        _validate_joint_gate_binding(recipe.get("joint_gate_binding"), dataset=recipe["dataset"])
    expected_policy = ({
        "schema": "round6_frozen_readout_allowlist_v1",
        "trainable_prefixes": ["receiver_readout.", "field_head."],
        "thermal_additional_prefixes": ["response_phi."],
        "non_allowlisted_parameters": "requires_grad_false_and_bitwise_fixed",
        "all_buffers": "bitwise_fixed",
        "optimizer": "fresh_adamw_state",
    } if phase == "frozen_probe" else None)
    if recipe.get("frozen_parameter_policy") != expected_policy:
        raise ValueError("Round6 frozen fit must retain its exact trainable-parameter and buffer policy.")
    if not _is_sha256(recipe["source_checkpoint_sha256"]):
        raise ValueError("Round6 source checkpoint digest is malformed.")
    return recipe


def readout_attachment_declaration(recipe: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_hyperedge_readout_recipe(recipe)
    return {
        "schema": READOUT_ATTACHMENT_SCHEMA,
        "family": READOUT_FAMILY,
        "workflow": READOUT_WORKFLOW,
        "protocol": READOUT_PROTOCOL,
        "receiver_reduction": value["receiver_reduction"],
        "receiver_access": value["receiver_access"],
        "geometry_mode": value["geometry_mode"],
        "source_checkpoint": value["source_checkpoint"],
        "source_checkpoint_sha256": value["source_checkpoint_sha256"],
        "source_recipe_sha256": value["source_recipe_sha256"],
        "source_epoch": value["source_epoch"],
    }


__all__ = [
    "READOUT_ACCESS_MODES",
    "READOUT_ATTACHMENT_SCHEMA",
    "READOUT_CORE_SCHEMA",
    "READOUT_EXTENSION_SOURCE_EPOCH",
    "READOUT_FAMILY",
    "READOUT_MAX_FIT_AGE",
    "READOUT_PRIMARY_FIT_AGE",
    "READOUT_PROTOCOL",
    "READOUT_RECIPE_SCHEMA",
    "READOUT_SOURCE_EPOCH",
    "READOUT_WORKFLOW",
    "frozen_extension_resume_amendment",
    "make_hyperedge_readout_recipe",
    "readout_attachment_declaration",
    "readout_core_config",
    "validate_hyperedge_readout_recipe",
    "verify_frozen_extension_decision",
    "verify_joint_gate_binding",
]
