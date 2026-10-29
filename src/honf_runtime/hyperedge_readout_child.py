"""Exact Round6 parent attachment and named-optimizer-state guards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .hyperedge_readout_protocol import (
    READOUT_ATTACHMENT_SCHEMA,
    READOUT_SOURCE_EPOCH,
    READOUT_VARIANT,
    canonical_sha256,
    file_sha256,
    readout_attachment_declaration,
    validate_hyperedge_readout_recipe,
)
from .moment_hyperedge_child import _tree_sha256
from .moment_hyperedge_protocol import (
    MOMENT_FAMILY,
    MOMENT_PROTOCOL,
    MOMENT_WORKFLOW,
    validate_moment_hyperedge_recipe,
)


def _check_source(
    source_payload: Mapping[str, Any], recipe: Mapping[str, Any], source_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]]]:
    value = validate_hyperedge_readout_recipe(recipe)
    path = source_path.expanduser().resolve(strict=True)
    if str(path) != value["source_checkpoint"] or file_sha256(path) != value["source_checkpoint_sha256"]:
        raise ValueError("Round6 H parent path or digest differs from the pinned recipe.")
    source_epoch = value["source_epoch"]
    identity = source_payload.get("experiment_identity")
    if (not isinstance(identity, Mapping)
            or source_payload.get("workflow") != MOMENT_WORKFLOW
            or identity.get("family") != MOMENT_FAMILY
            or identity.get("model_family") != MOMENT_FAMILY
            or identity.get("workflow") != MOMENT_WORKFLOW
            or identity.get("protocol") != MOMENT_PROTOCOL
            or source_payload.get("arm") != "IAG-C"
            or source_payload.get("epoch") != source_epoch):
        raise ValueError("Round6 source checkpoint is not the exact declared H IAG-C parent age.")
    source_recipe = validate_moment_hyperedge_recipe(identity.get("recipe", {}))
    provider_identity = identity.get("provider_identity")
    if (not isinstance(provider_identity, Mapping)
            or source_recipe.get("variant") != READOUT_VARIANT
            or identity.get("recipe_sha256") != value["source_recipe_sha256"]
            or canonical_sha256(provider_identity) != value["source_provider_identity_sha256"]):
        raise ValueError("Round6 source recipe or native provider binding differs from the sealed recipe.")
    if source_payload.get("model_state_dict") is None:
        raise ValueError("Round6 H parent has no model weights.")
    rows = source_payload.get("history")
    if (not isinstance(rows, list) or len(rows) != source_epoch
            or [row.get("epoch") for row in rows if isinstance(row, Mapping)]
            != list(range(1, source_epoch + 1))):
        raise ValueError("Round6 source H checkpoint must retain its complete absolute-age history.")
    if value["phase"] == "joint_confirmation":
        optimizer = source_payload.get("optimizer_state_by_name")
        states = optimizer.get("state_by_name") if isinstance(optimizer, Mapping) else None
        groups = optimizer.get("groups") if isinstance(optimizer, Mapping) else None
        if not isinstance(states, Mapping) or not states or not isinstance(groups, list):
            raise ValueError("Joint confirmation requires the source H named optimizer moments.")
        if not isinstance(source_payload.get("rng_state"), Mapping):
            raise ValueError("Joint confirmation requires the source H RNG state.")
    return identity, provider_identity, rows


def copy_round6_common_state(
    model: torch.nn.Module,
    source_state: Mapping[str, Any],
    *,
    dataset: str,
) -> dict[str, Any]:
    """Copy the complete common H state, preserving only declared reader birth tensors."""
    target_state = model.state_dict()
    if not isinstance(source_state, Mapping):
        raise TypeError("Round6 parent state must be a name-to-tensor mapping.")
    missing = sorted(set(source_state) - set(target_state))
    if missing:
        raise ValueError(f"Round6 model dropped common H state names: {missing}.")
    if dataset == "thermal":
        birth_prefix = "core.receiver_readout."
    elif dataset == "wind":
        birth_prefix = "receiver_readout."
    else:
        raise ValueError("Round6 common-state attachment supports Thermal or Wind.")
    born = sorted(set(target_state) - set(source_state))
    if not born or any(not name.startswith(birth_prefix) for name in born):
        raise ValueError("Round6 attachment introduced state outside its receiver_readout module.")
    named_parameters = dict(model.named_parameters())
    if set(born) != {name for name in born if name in named_parameters}:
        raise ValueError("Round6 receiver birth must contain parameters only; buffers remain parent-owned.")
    for name, source in source_state.items():
        target = target_state[name]
        if (not torch.is_tensor(source) or not torch.is_tensor(target)
                or source.shape != target.shape or source.dtype != target.dtype):
            raise ValueError(f"Round6 common H tensor layout changed for {name!r}.")
        with torch.no_grad():
            target.copy_(source.to(device=target.device))
        if not torch.equal(target.detach().cpu(), source.detach().cpu()):
            raise ValueError(f"Round6 common parent state did not attach exactly for {name!r}.")
    if any(not torch.isfinite(named_parameters[name]).all().item() for name in born):
        raise ValueError("Round6 readout birth contains nonfinite tensors.")
    zero_outputs = [name for name in born if name.endswith("output_projection.weight")]
    if len(zero_outputs) != 1 or not torch.count_nonzero(named_parameters[zero_outputs[0]]).item() == 0:
        raise ValueError("Round6 readout output projection must be exactly zero at attachment.")
    return {
        "common_state_names": len(source_state),
        "common_state_sha256": _tree_sha256({name: target_state[name] for name in sorted(source_state)}),
        "born_parameter_names": born,
        "born_parameter_sha256": _tree_sha256({name: named_parameters[name] for name in born}),
        "new_optimizer_state_count_at_birth": 0,
    }


def prepare_hyperedge_readout_child(
    model: torch.nn.Module,
    source_payload: Mapping[str, Any],
    sealed_identity: Mapping[str, Any],
    declaration: Mapping[str, Any],
    *,
    source_path: Path,
    case_keys: Sequence[Any],
    stop_after: int,
) -> dict[str, Any]:
    """Validate a Round6 branch and verify the exact H parameter/state birth."""
    if not isinstance(declaration, Mapping):
        raise TypeError("Round6 child declaration must be a mapping.")
    recipe = validate_hyperedge_readout_recipe(sealed_identity.get("recipe", {}))
    expected = readout_attachment_declaration(recipe)
    if dict(declaration) != expected:
        raise ValueError("Round6 attachment declaration differs from its sealed recipe.")
    if recipe["phase"] != "joint_confirmation":
        raise ValueError("Only the explicitly gated joint-confirmation phase branches through TrainingEngine.")
    if stop_after != recipe["fit_age_ceiling"] or stop_after != READOUT_SOURCE_EPOCH:
        raise ValueError("Joint confirmation must continue from H e1000 through the absolute e2500 stop.")
    parent_identity, parent_provider, rows = _check_source(source_payload, recipe, source_path)
    if not case_keys:
        raise ValueError("Round6 joint branch requires nonempty selected TRAIN case membership.")
    provider_identity = sealed_identity.get("provider_identity")
    if not isinstance(provider_identity, Mapping):
        raise TypeError("Round6 branch identity omits its native provider binding.")
    from .hyperedge_readout_factory import validate_readout_provider_pair

    provider_binding = validate_readout_provider_pair(
        parent_provider, provider_identity, dataset=recipe["dataset"], core_config=recipe["core_config"],
    )
    state = source_payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("Round6 joint source has no tensor state dictionary.")
    attachment = copy_round6_common_state(model, state, dataset=recipe["dataset"])
    optimizer = source_payload.get("optimizer_state_by_name")
    source_optimizer = optimizer.get("state_by_name") if isinstance(optimizer, Mapping) else None
    if not isinstance(source_optimizer, Mapping):
        raise TypeError("Round6 joint source optimizer state is malformed.")
    common_parameter_names = set(dict(model.named_parameters())) - set(attachment["born_parameter_names"])
    if set(source_optimizer) != common_parameter_names:
        raise ValueError("Round6 joint continuation source moments do not exactly cover common H parameters.")
    return {
        "schema": READOUT_ATTACHMENT_SCHEMA,
        "phase": "joint_confirmation",
        "source_checkpoint": str(source_path.resolve()),
        "source_checkpoint_sha256": file_sha256(source_path),
        "source_epoch": int(source_payload["epoch"]),
        "source_recipe_sha256": recipe["source_recipe_sha256"],
        "parent_identity_sha256": canonical_sha256(parent_identity),
        "native_provider_binding": provider_binding,
        "common_state_attachment": attachment,
        "inherited_optimizer_parameter_names": sorted(source_optimizer),
        "born_optimizer_parameter_names": attachment["born_parameter_names"],
        "inherited_history_rows": len(rows),
        "continuation_semantics": "absolute e1001 through e2500; inherit only common named H AdamW moments",
    }


def verify_joint_optimizer_birth(
    model: torch.nn.Module,
    optimizer_payload: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> None:
    """Require parent moments exactly for common names and none for new reader tensors."""
    states = optimizer_payload.get("state_by_name")
    if not isinstance(states, Mapping):
        raise TypeError("Round6 joint optimizer payload has no named state mapping.")
    born = set(amendment.get("born_optimizer_parameter_names", ()))
    expected = set(amendment.get("inherited_optimizer_parameter_names", ()))
    if set(states) != expected or set(states) & born:
        raise ValueError("Round6 joint optimizer inherited extra or omitted H moments at attachment.")
    names = set(dict(model.named_parameters()))
    if expected | born != names or expected & born:
        raise ValueError("Round6 joint optimizer parameter partition differs from the exact H/readout birth.")
    for name in expected:
        state = states[name]
        if not isinstance(state, Mapping) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise ValueError(f"Round6 inherited optimizer moments are incomplete for {name!r}.")


__all__ = ["copy_round6_common_state", "prepare_hyperedge_readout_child", "verify_joint_optimizer_birth"]
