"""Explicit frozen-checkpoint conversion for the Run-1507 static replay."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_runtime.compat import strip_module_prefix

SOURCE_ARCHITECTURE = "task_trained_functional_coalescence_honf"
TARGET_ARCHITECTURE = "source_conditioned_pairwise_honf"

# These are the only serialized state entries removed by the exact frozen
# all-closed specialization. The query Fourier module contains no parameters
# or persistent buffers, so it contributes no state-dict key to this list.
REMOVED_STATE_KEY_SUFFIXES = (
    "router.query_projection.0.weight",
    "router.query_projection.0.bias",
    "router.query_projection.2.weight",
    "router.query_projection.2.bias",
    "router.query_group_projection.weight",
    "functional_detail_controller.node_membership",
    "functional_detail_controller.child_masks",
    "functional_detail_controller.child_refs",
    "functional_detail_controller.ancestor_mask",
    "functional_detail_controller.haar_basis",
    "functional_detail_controller.network.0.weight",
    "functional_detail_controller.network.0.bias",
    "functional_detail_controller.network.2.weight",
    "functional_detail_controller.network.2.bias",
)


def expected_removed_state_keys(*, backend_prefix: str = "core.backend.") -> frozenset[str]:
    """Return the exact Run-1507 keys allowed to disappear in conversion."""

    return frozenset(f"{backend_prefix}{suffix}" for suffix in REMOVED_STATE_KEY_SUFFIXES)


def convert_task_trained_state_dict(
    source_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
    *,
    backend_prefix: str = "core.backend.",
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Map every shared source tensor and require the exact removed-key set.

    ``target_state`` must come from a fully materialized target model. This
    deliberately rejects blanket partial loading: the target may omit only
    the registered query-access and functional-detail controller keys above.
    """

    source = strip_module_prefix(dict(source_state))
    target = dict(target_state)
    source_keys = set(source)
    target_keys = set(target)
    expected_removed = set(expected_removed_state_keys(backend_prefix=backend_prefix))
    removed = source_keys - target_keys
    missing_target = target_keys - source_keys
    if removed != expected_removed:
        raise ValueError(
            "Checkpoint conversion removed-state inventory differs from the explicit contract: "
            f"expected={sorted(expected_removed)}, actual={sorted(removed)}."
        )
    if missing_target:
        raise ValueError(
            "Checkpoint conversion would leave target state uninitialized: "
            f"{sorted(missing_target)}."
        )

    converted: dict[str, Any] = {}
    for key in sorted(target_keys):
        source_value = source[key]
        target_value = target[key]
        if isinstance(target_value, (UninitializedParameter, UninitializedBuffer)):
            raise ValueError(
                f"Target state {key!r} is lazy; run a real materializing forward before conversion."
            )
        if not torch.is_tensor(source_value) or not torch.is_tensor(target_value):
            raise TypeError(f"State entry {key!r} is not a tensor in both models.")
        try:
            source_shape = tuple(source_value.shape)
            target_shape = tuple(target_value.shape)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"State entry {key!r} has an unmaterialized shape.") from error
        if source_shape != target_shape:
            raise ValueError(
                f"State entry {key!r} changed shape during conversion: "
                f"source={source_shape}, target={target_shape}."
            )
        converted[key] = source_value

    inventory = {
        "retained": sorted(target_keys),
        "removed": sorted(removed),
    }
    return converted, inventory


def convert_task_trained_checkpoint(
    checkpoint: Mapping[str, Any],
    target_model: torch.nn.Module,
    *,
    backend_prefix: str = "core.backend.",
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Load a Run-1507 checkpoint into a materialized static model and return it.

    The converted artifact is for frozen replay. Its optimizer and scaler
    state are cleared so it cannot be mistaken for a resumable training
    checkpoint after removing the query/detail modules.
    """

    validate_checkpoint_identity(
        checkpoint,
        case_id="ThermalChannel",
        model_family="honf_forward",
        workflow="forward",
    )
    if checkpoint.get("checkpoint_schema_version") != 1:
        raise ValueError("Run-1507 conversion requires checkpoint_schema_version=1.")
    required_identity = {
        "case_id": "ThermalChannel",
        "model_family": "honf_forward",
        "workflow": "forward",
    }
    missing_identity = sorted(
        key for key, value in required_identity.items() if checkpoint.get(key) != value
    )
    if missing_identity:
        raise ValueError(
            "Run-1507 conversion requires complete checkpoint identity fields; "
            f"missing or mismatched={missing_identity}."
        )

    model_config = getattr(target_model, "config", None)
    core_config = getattr(model_config, "core_honf", None)
    target_architecture = getattr(core_config, "forward_architecture", None)
    if target_architecture != TARGET_ARCHITECTURE:
        raise ValueError(
            f"Target model must use {TARGET_ARCHITECTURE!r}; got {target_architecture!r}."
        )
    source_config = dict(checkpoint.get("model_config") or {})
    source_core = dict(source_config.get("core_honf") or {})
    source_architecture = source_core.get("forward_architecture")
    if source_architecture != SOURCE_ARCHITECTURE:
        raise ValueError(
            f"Source checkpoint must use {SOURCE_ARCHITECTURE!r}; got {source_architecture!r}."
        )
    source_state = checkpoint.get("model_state_dict")
    if not isinstance(source_state, Mapping):
        raise ValueError("Source checkpoint is missing model_state_dict.")

    converted_state, inventory = convert_task_trained_state_dict(
        source_state,
        target_model.state_dict(),
        backend_prefix=backend_prefix,
    )
    target_model.load_state_dict(converted_state, strict=True)

    converted_checkpoint = dict(checkpoint)
    converted_checkpoint["model_state_dict"] = converted_state
    converted_checkpoint["model_config"] = model_config.to_dict()
    converted_checkpoint["optimizer_state_dict"] = None
    converted_checkpoint["optimizer_group_inventory"] = None
    converted_checkpoint["scaler_state_dict"] = None
    selection_state = getattr(target_model, "selection_state", None)
    converted_checkpoint["selection_state"] = (
        dict(selection_state()) if callable(selection_state) else None
    )
    converted_checkpoint["architecture_conversion"] = {
        "source_architecture": SOURCE_ARCHITECTURE,
        "target_architecture": TARGET_ARCHITECTURE,
        "kind": "frozen_all_closed_specialization",
        "removed_state_keys": inventory["removed"],
        "retained_state_key_count": len(inventory["retained"]),
        "optimizer_state_cleared": True,
    }
    return converted_checkpoint, inventory


__all__ = [
    "REMOVED_STATE_KEY_SUFFIXES",
    "SOURCE_ARCHITECTURE",
    "TARGET_ARCHITECTURE",
    "convert_task_trained_checkpoint",
    "convert_task_trained_state_dict",
    "expected_removed_state_keys",
]
