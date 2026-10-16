"""Exact Round4 soft-e500 to trained-sparse inherited-attention child guard."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .inherited_attention_protocol import (
    IAG_ARCHITECTURE_REPRESENTATION,
    IAG_C_ARCHITECTURE,
    IAG_PROTOCOL,
    IAG_Q_ARCHITECTURE,
    IAG_Q_REPRESENTATION_REVIEW_DECISION,
    IAG_REPRESENTATION_REVIEW_SCHEMA,
)

IAG_SPARSE_CHILD_SCHEMA = "inherited_attention_sparse_child_v1"
IAG_NORMALIZATION_CORRECTION_SCHEMA = "inherited_attention_normalization_correction_v1"
IAG_NORMALIZATION_CORRECTION_BINDING_SCHEMA = "inherited_attention_normalization_correction_binding_v1"
IAG_NORMALIZATION_CORRECTION_SCOPE = "entmax15_fp64_prefix_boundary_correctness_v1"
IAG_NORMALIZATION_CORRECTION_THERMAL_SCOPE = (
    "entmax15_fp64_prefix_boundary_with_c_compatibility_v1"
)
IAG_NORMALIZATION_CORRECTION_TARGET_SHA256 = "e2cc51145388c90463a346b745f8007c290c447116c4ebbf18fa233ce6127223"
IAG_NORMALIZATION_CORRECTION_PATH = "src/honf_forward_core/inherited_attention/normalization.py"
IAG_NORMALIZATION_CORRECTION_Q_PATHS = frozenset({
    IAG_NORMALIZATION_CORRECTION_PATH,
    "src/honf_runtime/inherited_attention_child.py",
    "tools/inherited_attention_train.py",
})
IAG_NORMALIZATION_CORRECTION_THERMAL_PATHS = frozenset({
    "src/honf_forward_core/inherited_attention/__init__.py",
    "src/honf_forward_core/inherited_attention/model.py",
    IAG_NORMALIZATION_CORRECTION_PATH,
    "src/honf_forward_core/inherited_attention/query_reader.py",
    "src/honf_runtime/inherited_attention_child.py",
    "src/honf_runtime/inherited_attention_protocol.py",
    "tools/inherited_attention_train.py",
})
IAG_NORMALIZATION_CORRECTION_ADDED_THERMAL_PATH = "src/honf_forward_core/inherited_attention/query_reader.py"
IAG_NORMALIZATION_CORRECTION_ALLOWED_PATHS = (
    IAG_NORMALIZATION_CORRECTION_Q_PATHS | IAG_NORMALIZATION_CORRECTION_THERMAL_PATHS
)
IAG_NORMALIZATION_CORRECTION_FIELDS = frozenset({
    "schema",
    "scope",
    "dataset",
    "architecture",
    "representation",
    "source_checkpoint",
    "source_checkpoint_sha256",
    "source_epoch",
    "source_recipe_sha256",
    "target_recipe_sha256",
    "source_loaded_source_sha256",
    "target_loaded_source_sha256",
    "source_files",
    "target_files",
    "file_changes",
    "reason",
})
IAG_NORMALIZATION_CORRECTION_THERMAL_FIELDS = (
    IAG_NORMALIZATION_CORRECTION_FIELDS | {"thermal_runtime_equivalence"}
)
IAG_NORMALIZATION_CORRECTION_BINDING_FIELDS = frozenset({
    "schema",
    "path",
    "receipt_file_sha256",
    "receipt_sha256",
    "receipt_json",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _source_files(identity: Mapping[str, Any], *, label: str) -> dict[str, str]:
    loaded = identity.get("loaded_source")
    if not isinstance(loaded, Mapping) or set(loaded) != {"files", "sha256"}:
        raise ValueError(f"Normalization correction requires a complete {label} loaded-source inventory.")
    files = loaded.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError(f"Normalization correction requires nonempty {label} loaded-source files.")
    copied: dict[str, str] = {}
    for name, digest in files.items():
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or Path(name).as_posix() != name or ".." in Path(name).parts
                or not _is_sha256(digest)):
            raise ValueError(f"Normalization correction has a malformed {label} loaded-source entry.")
        copied[name] = digest
    copied = dict(sorted(copied.items()))
    if loaded.get("sha256") != _json_sha256(copied):
        raise ValueError(f"Normalization correction {label} loaded-source digest is not canonical.")
    return copied


def _validate_thermal_runtime_equivalence(receipt: Mapping[str, Any]) -> None:
    evidence_binding = receipt.get("thermal_runtime_equivalence")
    if not isinstance(evidence_binding, Mapping) or set(evidence_binding) != {"path", "sha256"}:
        raise ValueError("Thermal IAG-C correction requires its exact runtime-equivalence evidence binding.")
    path_text = evidence_binding.get("path")
    digest = evidence_binding.get("sha256")
    if (not isinstance(path_text, str) or not Path(path_text).is_absolute()
            or str(Path(path_text).resolve()) != path_text or not _is_sha256(digest)):
        raise ValueError("Thermal IAG-C runtime-equivalence evidence path or SHA is noncanonical.")
    evidence_path = Path(path_text).resolve(strict=True)
    if _sha256(evidence_path) != digest:
        raise ValueError("Thermal IAG-C runtime-equivalence evidence differs from its bound SHA256.")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Thermal IAG-C runtime-equivalence evidence must be UTF-8 JSON.") from exc
    expected_keys = {
        "schema", "checkpoint_sha256", "source_loaded_source_sha256",
        "target_loaded_source_sha256", "case_keys", "rows", "passed",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_keys:
        raise ValueError("Thermal IAG-C runtime-equivalence evidence fields differ from the fixed schema.")
    if (evidence.get("schema") != "round4_thermal_soft_runtime_equivalence_v1"
            or evidence.get("checkpoint_sha256") != receipt.get("source_checkpoint_sha256")
            or evidence.get("source_loaded_source_sha256") != receipt.get("source_loaded_source_sha256")
            or evidence.get("target_loaded_source_sha256") != receipt.get("target_loaded_source_sha256")
            or evidence.get("case_keys") != ["0006", "0097", "0342", "0631"]
            or evidence.get("passed") is not True):
        raise ValueError("Thermal IAG-C runtime-equivalence evidence does not match this parent and runtime transition.")
    rows = evidence.get("rows")
    row_keys = {
        "case_key", "module_count", "source_input_sha256", "target_input_sha256",
        "source_output_sha256", "target_output_sha256", "source_named_gradient_sha256",
        "target_named_gradient_sha256",
    }
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("Thermal IAG-C runtime-equivalence evidence must contain the four fixed TRAIN rows.")
    for row, case_key, module_count in zip(rows, ("0006", "0097", "0342", "0631"), (1, 4, 7, 12)):
        if (not isinstance(row, Mapping) or set(row) != row_keys
                or row.get("case_key") != case_key or row.get("module_count") != module_count):
            raise ValueError("Thermal IAG-C runtime-equivalence rows differ from the frozen TRAIN panel.")
        for kind in ("input", "output", "named_gradient"):
            source_hash = row.get(f"source_{kind}_sha256")
            target_hash = row.get(f"target_{kind}_sha256")
            if not _is_sha256(source_hash) or not _is_sha256(target_hash) or source_hash != target_hash:
                raise ValueError("Thermal IAG-C soft runtime equivalence did not match bitwise on the fixed TRAIN panel.")


def _validate_normalization_correction_receipt(
    receipt: Mapping[str, Any],
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> dict[str, str]:
    """Validate the one fixed entmax numerical correction against both identities."""
    if receipt.get("schema") != IAG_NORMALIZATION_CORRECTION_SCHEMA:
        raise ValueError("Unsupported normalization-correction receipt schema.")
    source_recipe = source.get("recipe")
    target_recipe = target.get("recipe")
    if not isinstance(source_recipe, Mapping) or not isinstance(target_recipe, Mapping):
        raise TypeError("Normalization correction requires sealed source and target recipes.")
    architecture = target_recipe.get("architecture")
    representation = IAG_ARCHITECTURE_REPRESENTATION.get(architecture)
    dataset = target_recipe.get("dataset")
    if (representation is None or source_recipe.get("architecture", IAG_C_ARCHITECTURE) != architecture
            or source_recipe.get("representation", representation) != representation
            or target_recipe.get("representation", representation) != representation
            or (architecture, dataset) not in ((IAG_C_ARCHITECTURE, "thermal"), (IAG_Q_ARCHITECTURE, "wind"))
            or source_recipe.get("dataset") != dataset):
        raise ValueError("Normalization correction must preserve the parent dataset and IAG architecture/representation.")
    if architecture == IAG_Q_ARCHITECTURE:
        expected_scope = IAG_NORMALIZATION_CORRECTION_SCOPE
        expected_fields = IAG_NORMALIZATION_CORRECTION_FIELDS
        allowed_paths = IAG_NORMALIZATION_CORRECTION_Q_PATHS
    else:
        expected_scope = IAG_NORMALIZATION_CORRECTION_THERMAL_SCOPE
        expected_fields = IAG_NORMALIZATION_CORRECTION_THERMAL_FIELDS
        allowed_paths = IAG_NORMALIZATION_CORRECTION_THERMAL_PATHS
    if set(receipt) != expected_fields:
        raise ValueError("Normalization-correction receipt fields differ from the fixed architecture scope.")
    if (receipt.get("scope") != expected_scope
            or type(receipt.get("source_epoch")) is not int
            or receipt.get("source_epoch") != 500):
        raise ValueError("Unsupported normalization-correction scope or source epoch.")
    if (receipt.get("dataset") != dataset or receipt.get("architecture") != architecture
            or receipt.get("representation") != representation):
        raise ValueError("Normalization-correction receipt targets a different dataset or IAG representation.")
    if (source_recipe.get("protocol") != IAG_PROTOCOL
            or target_recipe.get("protocol") != IAG_PROTOCOL
            or source_recipe.get("normalization") != "soft"
            or source_recipe.get("trajectory") != "fresh_soft"
            or target_recipe.get("normalization") != "measure_entmax15"
            or target_recipe.get("trajectory") != "e500_sparse_child"):
        raise ValueError("Normalization correction is only valid for a soft e500 to sparse development child.")
    source_recipe_sha = source.get("recipe_sha256")
    target_recipe_sha = target.get("recipe_sha256")
    if (not _is_sha256(source_recipe_sha) or not _is_sha256(target_recipe_sha)
            or receipt.get("source_recipe_sha256") != source_recipe_sha
            or receipt.get("target_recipe_sha256") != target_recipe_sha):
        raise ValueError("Normalization-correction receipt does not bind the exact parent and child recipe hashes.")
    checkpoint = receipt.get("source_checkpoint")
    if (not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute()
            or str(Path(checkpoint).resolve()) != checkpoint
            or not _is_sha256(receipt.get("source_checkpoint_sha256"))):
        raise ValueError("Normalization-correction receipt has a noncanonical parent checkpoint binding.")
    if source_path is not None:
        resolved_source = source_path.expanduser().resolve(strict=True)
        if (str(resolved_source) != checkpoint
                or _sha256(resolved_source) != receipt.get("source_checkpoint_sha256")):
            raise ValueError("Normalization-correction receipt does not match the exact e500 parent checkpoint.")

    source_files = receipt.get("source_files")
    target_files = receipt.get("target_files")
    changes = receipt.get("file_changes")
    if not isinstance(source_files, Mapping) or not isinstance(target_files, Mapping) or not isinstance(changes, Mapping):
        raise TypeError("Normalization-correction receipt must include exact source and target file maps.")
    normalized_source = dict(sorted(source_files.items()))
    normalized_target = dict(sorted(target_files.items()))
    actual_source = _source_files(source, label="parent")
    actual_target = _source_files(target, label="child")
    if normalized_source != actual_source or normalized_target != actual_target:
        raise ValueError("Normalization-correction receipt file maps do not match the sealed loaded-source identities.")
    source_digest = source.get("loaded_source", {}).get("sha256")
    target_digest = target.get("loaded_source", {}).get("sha256")
    if (receipt.get("source_loaded_source_sha256") != source_digest
            or receipt.get("target_loaded_source_sha256") != target_digest):
        raise ValueError("Normalization-correction receipt does not bind the exact loaded-source aggregate hashes.")
    source_names = set(normalized_source)
    target_names = set(normalized_target)
    added = target_names - source_names
    removed = source_names - target_names
    if architecture == IAG_Q_ARCHITECTURE and (added or removed):
        raise ValueError("IAG-Q normalization correction cannot add or remove loaded source files.")
    if architecture == IAG_C_ARCHITECTURE and (
        added != {IAG_NORMALIZATION_CORRECTION_ADDED_THERMAL_PATH} or removed
    ):
        raise ValueError("Thermal IAG-C correction permits only the reviewed query_reader.py source addition.")
    actual_changes = {
        name: {
            "source_sha256": normalized_source.get(name),
            "target_sha256": normalized_target.get(name),
        }
        for name in source_names | target_names
        if normalized_source.get(name) != normalized_target.get(name)
    }
    if dict(sorted(changes.items())) != dict(sorted(actual_changes.items())):
        raise ValueError("Normalization-correction file delta differs from the exact old/new loaded-source maps.")
    if set(actual_changes) != allowed_paths:
        raise ValueError("Normalization correction file delta differs from its exact fixed scope allowlist.")
    if IAG_NORMALIZATION_CORRECTION_PATH not in actual_changes:
        raise ValueError("Normalization correction must include the measure-entmax normalization implementation.")
    if actual_changes[IAG_NORMALIZATION_CORRECTION_PATH]["target_sha256"] != IAG_NORMALIZATION_CORRECTION_TARGET_SHA256:
        raise ValueError("Normalization correction target does not match the reviewed FP64-prefix implementation.")
    if architecture == IAG_C_ARCHITECTURE:
        _validate_thermal_runtime_equivalence(receipt)
    reason = receipt.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Normalization-correction receipt requires a concise recorded reason.")
    return actual_changes


def bind_inherited_attention_normalization_correction(
    receipt_path: Path,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    source_path: Path,
) -> dict[str, Any]:
    """Read and validate the explicit numerical-correction receipt for attachment."""
    path = receipt_path.expanduser().resolve(strict=True)
    raw = path.read_bytes()
    try:
        raw_text = raw.decode("utf-8")
        receipt = json.loads(raw_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Normalization-correction receipt must be valid UTF-8 JSON.") from exc
    if not isinstance(receipt, Mapping):
        raise TypeError("Normalization-correction receipt must be a JSON object.")
    _validate_normalization_correction_receipt(receipt, source, target, source_path=source_path)
    return {
        "schema": IAG_NORMALIZATION_CORRECTION_BINDING_SCHEMA,
        "path": str(path),
        "receipt_file_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_sha256": _json_sha256(receipt),
        "receipt_json": raw_text,
    }


def validate_inherited_attention_normalization_correction_resume(
    binding: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    """Check a stored correction binding against the exact current child identity."""
    if set(binding) != IAG_NORMALIZATION_CORRECTION_BINDING_FIELDS:
        raise ValueError("Stored normalization-correction binding fields are malformed.")
    path = binding.get("path")
    raw_text = binding.get("receipt_json")
    if (binding.get("schema") != IAG_NORMALIZATION_CORRECTION_BINDING_SCHEMA
            or not isinstance(path, str) or not Path(path).is_absolute()
            or str(Path(path).resolve()) != path or not isinstance(raw_text, str)
            or not _is_sha256(binding.get("receipt_file_sha256"))
            or not _is_sha256(binding.get("receipt_sha256"))):
        raise ValueError("Stored normalization-correction binding is not canonical.")
    raw = raw_text.encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != binding["receipt_file_sha256"]:
        raise ValueError("Stored normalization-correction receipt bytes fail their sealed hash.")
    try:
        receipt = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Stored normalization-correction receipt is invalid JSON.") from exc
    if (not isinstance(receipt, Mapping)
            or _json_sha256(receipt) != binding["receipt_sha256"]):
        raise ValueError("Stored normalization-correction receipt fails its canonical hash.")
    source_recipe = {
        "dataset": receipt.get("dataset"),
        "architecture": receipt.get("architecture"),
        "representation": receipt.get("representation"),
        "protocol": IAG_PROTOCOL,
        "normalization": "soft",
        "trajectory": "fresh_soft",
    }
    target_recipe = {
        **source_recipe,
        "normalization": "measure_entmax15",
        "trajectory": "e500_sparse_child",
    }
    source_identity = {
        "recipe": source_recipe,
        "recipe_sha256": receipt.get("source_recipe_sha256"),
        "loaded_source": {
            "files": receipt.get("source_files"),
            "sha256": receipt.get("source_loaded_source_sha256"),
        },
    }
    target_receipt_identity = {
        "recipe": target_recipe,
        "recipe_sha256": receipt.get("target_recipe_sha256"),
        "loaded_source": {
            "files": receipt.get("target_files"),
            "sha256": receipt.get("target_loaded_source_sha256"),
        },
    }
    _validate_normalization_correction_receipt(receipt, source_identity, target_receipt_identity)
    target_recipe = identity.get("recipe")
    if not isinstance(target_recipe, Mapping):
        raise TypeError("Corrected resume requires its sealed target recipe.")
    if (receipt.get("target_recipe_sha256") != identity.get("recipe_sha256")
            or receipt.get("dataset") != target_recipe.get("dataset")
            or receipt.get("architecture") != target_recipe.get("architecture")
            or receipt.get("representation") != target_recipe.get("representation")):
        raise ValueError("Stored normalization correction targets a different resume recipe.")
    actual_target = _source_files(identity, label="resume")
    if (dict(sorted(receipt.get("target_files", {}).items())) != actual_target
            or receipt.get("target_loaded_source_sha256") != identity.get("loaded_source", {}).get("sha256")):
        raise ValueError("Corrected resume loaded-source inventory differs from its bound target runtime.")


def _validated_normalization_correction(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    binding: Any,
    *,
    source_path: Path | None = None,
) -> dict[str, str]:
    if not isinstance(binding, Mapping) or set(binding) != IAG_NORMALIZATION_CORRECTION_BINDING_FIELDS:
        raise ValueError("Sparse-child normalization correction is missing its sealed receipt binding.")
    path = binding.get("path")
    raw_text = binding.get("receipt_json")
    if (binding.get("schema") != IAG_NORMALIZATION_CORRECTION_BINDING_SCHEMA
            or not isinstance(path, str) or not Path(path).is_absolute()
            or str(Path(path).resolve()) != path or not isinstance(raw_text, str)
            or not _is_sha256(binding.get("receipt_file_sha256"))
            or not _is_sha256(binding.get("receipt_sha256"))):
        raise ValueError("Sparse-child normalization-correction binding is noncanonical.")
    raw = raw_text.encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != binding["receipt_file_sha256"]:
        raise ValueError("Sparse-child normalization-correction receipt bytes fail their sealed hash.")
    try:
        receipt = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Sparse-child normalization-correction receipt is invalid JSON.") from exc
    if not isinstance(receipt, Mapping) or _json_sha256(receipt) != binding["receipt_sha256"]:
        raise ValueError("Sparse-child normalization-correction receipt fails its canonical hash.")
    return _validate_normalization_correction_receipt(receipt, source, target, source_path=source_path)


def _mask_verified_source_delta(identity: Any, changed_paths: Mapping[str, Any]) -> Any:
    result = copy.deepcopy(identity)
    loaded = result.get("loaded_source")
    files = loaded["files"]
    for path in changed_paths:
        files[path] = "<receipt-approved-source-correction>"
    loaded["sha256"] = "<receipt-approved-source-correction>"
    result.pop("normalization_correction", None)
    return result


def _soft_identity(value: Any, *, allow_c_scene_change: bool) -> Any:
    """Erase only the sealed normalization/trajectory identity difference."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key == "sparse_transition":
                continue
            if key == "recipe_sha256":
                result[key] = "<normalization-child-recipe>"
            elif key == "normalization":
                result[key] = "soft"
            elif key in ("trajectory",):
                result[key] = "fresh_soft"
            elif key == "absolute_epoch_transition":
                result[key] = None
            elif key == "scene_executor" and allow_c_scene_change:
                # The only executor change allowed at attachment is the
                # mathematically identical dense -> selected contraction.
                # _require_same_soft_identity validates direction and values
                # before this comparison-only normalization is applied.
                result[key] = "dense"
            else:
                result[key] = _soft_identity(item, allow_c_scene_change=allow_c_scene_change)
        return result
    if isinstance(value, (tuple, list)):
        return [_soft_identity(item, allow_c_scene_change=allow_c_scene_change) for item in value]
    return copy.deepcopy(value)


def _identity_values(value: Any, key_name: str) -> list[Any]:
    if isinstance(value, Mapping):
        found = [item for key, item in value.items() if key == key_name]
        for item in value.values():
            found.extend(_identity_values(item, key_name))
        return found
    if isinstance(value, (tuple, list)):
        found = []
        for item in value:
            found.extend(_identity_values(item, key_name))
        return found
    return []


def _require_same_soft_identity(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> dict[str, str]:
    source_recipe = source.get("recipe")
    target_recipe = target.get("recipe")
    if not isinstance(source_recipe, Mapping) or not isinstance(target_recipe, Mapping):
        raise TypeError("Sparse-child source and target identities must embed their recipes.")
    architecture = target_recipe.get("architecture", IAG_C_ARCHITECTURE)
    representation = IAG_ARCHITECTURE_REPRESENTATION.get(architecture)
    source_architecture = source_recipe.get("architecture", IAG_C_ARCHITECTURE)
    if representation is None or source_architecture != architecture:
        raise ValueError("Sparse-child parent and child must use the same IAG architecture.")
    if (source_recipe.get("representation", representation) != representation
            or target_recipe.get("representation", representation) != representation):
        raise ValueError("Sparse-child parent and child representation does not match its architecture.")
    source_scene = _identity_values(source, "scene_executor")
    target_scene = _identity_values(target, "scene_executor")
    allowed_scene = ("dense", "selected") if architecture == IAG_C_ARCHITECTURE else ("cached_tokens",)
    if not source_scene or not target_scene or any(value not in allowed_scene for value in source_scene + target_scene):
        raise ValueError(f"{architecture} sparse-child scene executor identity is missing or unsupported.")
    if len(set(source_scene)) != 1 or len(set(target_scene)) != 1:
        raise ValueError("IAG-C sparse-child identity has conflicting scene executor declarations.")
    source_scene_executor = source_scene[0]
    target_scene_executor = target_scene[0]
    if architecture == IAG_C_ARCHITECTURE and source_scene_executor == "selected" and target_scene_executor == "dense":
        raise ValueError("IAG-C e500 attachment cannot replace selected preparation with the dense executor.")
    source_query = _identity_values(source, "query_executor")
    target_query = _identity_values(target, "query_executor")
    allowed_query = ("indexed",) if architecture == IAG_C_ARCHITECTURE else ("cell_selected", "dense_reference")
    if not source_query or not target_query or len(set(source_query)) != 1 or len(set(target_query)) != 1:
        raise ValueError(f"{architecture} sparse-child query executor identity is missing or conflicting.")
    if source_query[0] not in allowed_query or target_query[0] not in allowed_query:
        if architecture == IAG_C_ARCHITECTURE:
            raise ValueError("IAG-C e500 attachment preserves the indexed query executor exactly.")
        raise ValueError(f"{architecture} sparse-child query executor identity is unsupported.")
    if architecture == IAG_Q_ARCHITECTURE and source_query != target_query:
        raise ValueError("IAG-Q e500 attachment preserves the query executor exactly.")
    if "normalization_correction" in source:
        raise ValueError("A sparse correction cannot attach from an already corrected sparse parent.")
    correction_binding = target.get("normalization_correction")
    changed_paths = None
    if correction_binding is not None:
        changed_paths = _validated_normalization_correction(
            source, target, correction_binding, source_path=source_path,
        )
    left = _soft_identity(source, allow_c_scene_change=architecture == IAG_C_ARCHITECTURE)
    right = _soft_identity(target, allow_c_scene_change=architecture == IAG_C_ARCHITECTURE)
    if changed_paths is not None:
        left = _mask_verified_source_delta(left, changed_paths)
        right = _mask_verified_source_delta(right, changed_paths)
    if left != right:
        raise ValueError(f"{architecture} sparse child changes a field beyond normalization and its absolute transition.")
    return {
        "source": str(source_scene_executor),
        "target": str(target_scene_executor),
        "same_probability_function": True,
    }


def _require_state_layout(model: torch.nn.Module, source_state: Any) -> None:
    current = model.state_dict()
    if not isinstance(source_state, Mapping) or set(current) != set(source_state):
        raise ValueError("Inherited-attention e500 parent and sparse child must have exactly identical state-dict keys.")
    for name, tensor in current.items():
        parent = source_state[name]
        if not torch.is_tensor(parent) or parent.shape != tensor.shape or parent.dtype != tensor.dtype:
            raise ValueError(f"Inherited-attention e500 state tensor {name!r} differs in shape or dtype.")
    current_parameters = dict(model.named_parameters())
    parent_parameters = {
        name for name in current_parameters if name in source_state
    }
    if parent_parameters != set(current_parameters):
        raise ValueError("Sparse child must preserve every parameter name from its soft e500 parent.")


def _optimizer_groups_from_identity(identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract = identity.get("optimizer_schedule_contract")
    if not isinstance(contract, list):
        raise TypeError("Inherited-attention parent identity has no sealed optimizer schedule contract.")
    return [
        {"name": str(item["name"]), "parameter_names": list(item["parameter_names"])}
        for item in contract
    ]


def prepare_inherited_attention_sparse_child(
    model: torch.nn.Module,
    source_payload: Mapping[str, Any],
    sealed_identity: Mapping[str, Any],
    declaration: Mapping[str, Any],
    *,
    source_path: Path,
    case_keys: Sequence[Any],
    stop_after: int,
) -> dict[str, Any]:
    """Validate one literal same-architecture soft e500 parent before loading."""
    base_required = {
        "schema",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_recipe_sha256",
        "target_recipe_sha256",
        "source_epoch",
    }
    if not isinstance(declaration, Mapping) or not base_required.issubset(declaration):
        raise ValueError("Sparse-child declaration is missing fields from the frozen e500 attachment contract.")
    source_path = source_path.expanduser().resolve(strict=True)
    digest = _sha256(source_path)
    if declaration.get("schema") != IAG_SPARSE_CHILD_SCHEMA:
        raise ValueError("Unsupported inherited-attention sparse-child declaration schema.")
    if declaration.get("source_checkpoint") != str(source_path):
        raise ValueError("Sparse-child declaration must name the exact canonical e500 checkpoint path.")
    if declaration.get("source_checkpoint_sha256") != digest:
        raise ValueError("e500 checkpoint checksum differs from the sparse-child declaration.")
    if type(declaration.get("source_epoch")) is not int or declaration["source_epoch"] != 500:
        raise ValueError("Sparse child attaches only to absolute epoch500.")
    if not 501 <= int(stop_after) <= 2500:
        raise ValueError("Sparse-child continuation must stop within absolute epochs501..2500.")
    if source_payload.get("workflow") != "inherited_attention_joint_v1":
        raise ValueError("Sparse child cannot attach from a historical or other-family checkpoint.")
    parent_identity = source_payload.get("experiment_identity")
    if not isinstance(parent_identity, Mapping):
        raise TypeError("IAG-C e500 parent checkpoint has no sealed experiment identity.")
    parent_recipe = parent_identity.get("recipe")
    child_recipe = sealed_identity.get("recipe")
    if not isinstance(parent_recipe, Mapping) or not isinstance(child_recipe, Mapping):
        raise TypeError("Sparse-child parent and child identities must both embed their recipes.")
    architecture = child_recipe.get("architecture", IAG_C_ARCHITECTURE)
    representation = IAG_ARCHITECTURE_REPRESENTATION.get(architecture)
    if representation is None:
        raise ValueError("Sparse child has an unsupported architecture.")
    expected_declaration = base_required | (
        {"representation_review_sha256"} if architecture == IAG_Q_ARCHITECTURE else set()
    )
    if set(declaration) != expected_declaration:
        raise ValueError("Sparse-child declaration fields differ from its architecture-specific contract.")
    if source_payload.get("epoch") != 500 or source_payload.get("arm") != architecture:
        raise ValueError(f"Sparse child parent checkpoint must be the literal {architecture} epoch500 state.")
    if (parent_recipe.get("architecture", IAG_C_ARCHITECTURE) != architecture
            or parent_recipe.get("representation", representation) != representation
            or child_recipe.get("representation", representation) != representation):
        raise ValueError("Sparse child parent and target architecture/representation must match exactly.")
    if architecture == IAG_Q_ARCHITECTURE:
        if parent_recipe.get("dataset") != "wind" or child_recipe.get("dataset") != "wind":
            raise ValueError("IAG-Q sparse children are restricted to their own Wind parent trajectory.")
        parent_review = parent_identity.get("representation_review")
        child_review = sealed_identity.get("representation_review")
        if not isinstance(parent_review, Mapping) or not isinstance(child_review, Mapping):
            raise ValueError("IAG-Q parent and child must retain their representation-review binding.")
        review_digest = parent_review.get("sha256")
        review_path = parent_review.get("path")
        fresh_run_dir = parent_review.get("fresh_run_dir")
        if (set(parent_review) != {"schema", "path", "sha256", "dataset", "decision", "fresh_run_dir"}
                or parent_review.get("schema") != IAG_REPRESENTATION_REVIEW_SCHEMA
                or parent_review.get("dataset") != "wind"
                or parent_review.get("decision") != IAG_Q_REPRESENTATION_REVIEW_DECISION
                or not isinstance(review_path, str) or not Path(review_path).is_absolute()
                or str(Path(review_path).resolve()) != review_path
                or not isinstance(fresh_run_dir, str) or not Path(fresh_run_dir).is_absolute()
                or str(Path(fresh_run_dir).resolve()) != fresh_run_dir
                or not isinstance(review_digest, str) or len(review_digest) != 64
                or any(character not in "0123456789abcdef" for character in review_digest)
                or parent_review != child_review
                or declaration.get("representation_review_sha256") != review_digest):
            raise ValueError("IAG-Q sparse child must preserve and bind the exact review-receipt identity.")
    if (parent_recipe.get("protocol") != "inherited_attention_development_v1"
            or parent_recipe.get("normalization") != "soft"
            or parent_recipe.get("trajectory") != "fresh_soft"):
        raise ValueError(f"Sparse {architecture} parent must be its own fresh soft segmented trajectory.")
    if (child_recipe.get("protocol") != "inherited_attention_development_v1"
            or child_recipe.get("normalization") != "measure_entmax15"
            or child_recipe.get("trajectory") != "e500_sparse_child"):
        raise ValueError(f"Sparse {architecture} child must use its sealed segmented measure-entmax trajectory.")
    if declaration.get("source_recipe_sha256") != parent_identity.get("recipe_sha256"):
        raise ValueError("Sparse-child declaration does not bind the parent's exact soft recipe digest.")
    if declaration.get("target_recipe_sha256") != sealed_identity.get("recipe_sha256"):
        raise ValueError("Sparse-child declaration does not bind the child's exact sparse recipe digest.")
    executor_transition = _require_same_soft_identity(
        parent_identity, sealed_identity, source_path=source_path,
    )
    _require_state_layout(model, source_payload.get("model_state_dict"))
    source_optimizer = source_payload.get("optimizer_state_by_name")
    if not isinstance(source_optimizer, Mapping) or not isinstance(source_optimizer.get("state_by_name"), Mapping):
        raise TypeError("Sparse child requires named AdamW moments from its e500 parent.")
    parent_identity_groups = _optimizer_groups_from_identity(parent_identity)
    child_identity_groups = _optimizer_groups_from_identity(sealed_identity)
    if parent_identity_groups != child_identity_groups:
        raise ValueError("Sparse-child optimizer group names or parameter membership changed.")
    if source_optimizer.get("groups") != parent_identity_groups:
        raise ValueError("Checkpoint AdamW group membership differs from its own sealed identity.")
    expected_names = {name for name, _ in model.named_parameters()}
    state_by_name = source_optimizer["state_by_name"]
    if set(state_by_name) != expected_names:
        raise ValueError("Sparse child must inherit one AdamW state for every original trainable parameter.")
    if source_payload.get("provider_training_state", {}) not in ({}, None):
        raise ValueError("Sparse child does not accept undeclared provider-local training state.")
    history = source_payload.get("history")
    if not isinstance(history, list) or not history or int(history[-1].get("epoch", -1)) != 500:
        raise ValueError("Sparse-child parent history must end at the literal epoch500 checkpoint.")
    if not case_keys:
        raise ValueError("Sparse child cannot attach without its fixed TRAIN membership.")
    result = {
        "schema": IAG_SPARSE_CHILD_SCHEMA,
        "trajectory": "soft_e1_500_then_absolute_sparse_transition_e501_800",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": digest,
        "source_recipe_sha256": declaration["source_recipe_sha256"],
        "target_recipe_sha256": declaration["target_recipe_sha256"],
        "source_epoch": 500,
        "next_epoch": 501,
        "stop_after": int(stop_after),
        "train_case_count": len(case_keys),
        "exact_parameter_layout_verified": True,
        "exact_optimizer_group_contract_verified": True,
        "exact_optimizer_moments_pending_load_verification": True,
        "scene_executor_transition": executor_transition,
        "provider_state_contract": "empty; native sampler age is carried by absolute epoch and restored RNG",
        "source_architecture": architecture,
        "source_representation": representation,
        "target_architecture": architecture,
        "target_representation": representation,
    }
    if architecture == IAG_Q_ARCHITECTURE:
        result["representation_review_sha256"] = declaration["representation_review_sha256"]
    correction_binding = sealed_identity.get("normalization_correction")
    if correction_binding is not None:
        correction_receipt = json.loads(correction_binding["receipt_json"])
        result["normalization_correction"] = {
            "scope": correction_receipt["scope"],
            "receipt_sha256": correction_binding["receipt_sha256"],
            "target_loaded_source_sha256": correction_receipt["target_loaded_source_sha256"],
        }
    return result


def verify_inherited_attention_sparse_child_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source_optimizer_payload: Mapping[str, Any],
) -> None:
    """Verify all moments and parameter memberships survived named restore."""
    from .unified_training import named_optimizer_state

    restored = named_optimizer_state(model, optimizer)
    expected = source_optimizer_payload
    if restored.get("groups") != expected.get("groups"):
        raise ValueError("Sparse child restored optimizer groups differ from its exact e500 parent.")
    left = restored.get("state_by_name", {})
    right = expected.get("state_by_name", {})
    if set(left) != set(right):
        raise ValueError("Sparse child restored AdamW parameter states differ from its exact e500 parent.")
    for name in left:
        if set(left[name]) != set(right[name]):
            raise ValueError(f"e500 optimizer state fields changed for {name!r}.")
        for key in left[name]:
            actual, saved = left[name][key], right[name][key]
            if torch.is_tensor(actual) or torch.is_tensor(saved):
                if not (torch.is_tensor(actual) and torch.is_tensor(saved) and torch.equal(actual, saved)):
                    raise ValueError(f"e500 optimizer tensor changed for {name}.{key}.")
            elif actual != saved:
                raise ValueError(f"e500 optimizer scalar changed for {name}.{key}.")


def verify_inherited_attention_sparse_child_continuation(
    contract: Mapping[str, Any],
    *,
    completed_epoch: int,
    stop_after: int,
) -> None:
    if contract.get("schema") != IAG_SPARSE_CHILD_SCHEMA or contract.get("source_epoch") != 500:
        raise ValueError("IAG-C sparse resume is missing its exact e500 child branch contract.")
    if contract.get("trajectory") != "soft_e1_500_then_absolute_sparse_transition_e501_800":
        raise ValueError("Sparse resume has a different normalization age contract.")
    if (contract.get("target_architecture") == IAG_Q_ARCHITECTURE
            and (contract.get("source_architecture") != IAG_Q_ARCHITECTURE
                or contract.get("source_representation") != "iag_q"
                or contract.get("target_representation") != "iag_q"
                or not isinstance(contract.get("representation_review_sha256"), str))):
        raise ValueError("IAG-Q sparse resume lacks its same-architecture review-bound branch identity.")
    correction = contract.get("normalization_correction")
    if correction is not None and (
        not isinstance(correction, Mapping)
        or set(correction) != {"scope", "receipt_sha256", "target_loaded_source_sha256"}
        or correction.get("scope") != {
            IAG_C_ARCHITECTURE: IAG_NORMALIZATION_CORRECTION_THERMAL_SCOPE,
            IAG_Q_ARCHITECTURE: IAG_NORMALIZATION_CORRECTION_SCOPE,
        }.get(contract.get("target_architecture"))
        or not _is_sha256(correction.get("receipt_sha256"))
        or not _is_sha256(correction.get("target_loaded_source_sha256"))
    ):
        raise ValueError("Sparse resume has a malformed entmax numerical-correction binding.")
    if type(completed_epoch) is not int or not 501 <= completed_epoch < stop_after <= 2500:
        raise ValueError("IAG-C sparse same-run resume must advance inside the absolute 501..2500 horizon.")


__all__ = [
    "IAG_NORMALIZATION_CORRECTION_ALLOWED_PATHS",
    "IAG_NORMALIZATION_CORRECTION_BINDING_SCHEMA",
    "IAG_NORMALIZATION_CORRECTION_PATH",
    "IAG_NORMALIZATION_CORRECTION_SCHEMA",
    "IAG_NORMALIZATION_CORRECTION_SCOPE",
    "IAG_NORMALIZATION_CORRECTION_TARGET_SHA256",
    "IAG_SPARSE_CHILD_SCHEMA",
    "bind_inherited_attention_normalization_correction",
    "prepare_inherited_attention_sparse_child",
    "validate_inherited_attention_normalization_correction_resume",
    "verify_inherited_attention_sparse_child_continuation",
    "verify_inherited_attention_sparse_child_optimizer",
]
