"""Forward optimizer construction and resume-safe group inventories."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict

import torch

from channelthermal.model import ChannelThermalHONFModel


def _optimizer_group_record(
    name: str,
    learning_rate: float,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> Dict[str, Any]:
    """Build a deterministic, human-auditable optimizer-group inventory."""

    ordered_names = sorted(parameter_name for parameter_name, _ in named_parameters)
    scalar_count = 0
    scalar_count_complete = True
    for _, parameter in named_parameters:
        try:
            scalar_count += int(parameter.numel())
        except ValueError:
            scalar_count_complete = False
    return {
        "name": name,
        "learning_rate": float(learning_rate),
        "parameter_tensor_count": len(named_parameters),
        "trainable_scalar_count": scalar_count if scalar_count_complete else None,
        "scalar_count_complete": scalar_count_complete,
        "parameter_names": ordered_names,
        "ordered_names_sha256": hashlib.sha256("\n".join(ordered_names).encode("utf-8")).hexdigest(),
    }


def _optimizer_inventory_digest(groups: list[Dict[str, Any]]) -> str:
    structure = [
        {
            "name": group["name"],
            "learning_rate": group["learning_rate"],
            "parameter_names": group["parameter_names"],
        }
        for group in groups
    ]
    return hashlib.sha256(repr(structure).encode("utf-8")).hexdigest()


def build_forward_optimizer(
    model: ChannelThermalHONFModel,
    training_config: Dict[str, Any],
) -> tuple[torch.optim.AdamW, Dict[str, Any]]:
    """Build the legacy one-group AdamW or the optional organizer split."""

    learning_rate = float(training_config.get("learning_rate", 2.0e-4))
    weight_decay = float(training_config.get("weight_decay", 1.0e-5))
    organizer_learning_rate = training_config.get("organizer_learning_rate")
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if organizer_learning_rate is None:
        # This is intentionally the literal historical construction path.
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        groups = [_optimizer_group_record("all", learning_rate, trainable)]
        mode = "single"
    else:
        organizer_lr = float(organizer_learning_rate)
        if organizer_lr <= 0.0:
            raise ValueError("organizer_learning_rate must be null or positive.")
        organizer_parameters = [
            item for item in trainable if item[0].startswith("core.organizer.")
        ]
        prediction_parameters = [
            item for item in trainable if not item[0].startswith("core.organizer.")
        ]
        if not organizer_parameters:
            raise ValueError(
                "A split optimizer requires trainable core.organizer.* parameters; "
                "use organizer_learning_rate=null for this model."
            )
        if not prediction_parameters:
            raise ValueError("A split optimizer requires non-organizer trainable parameters.")
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [parameter for _, parameter in organizer_parameters],
                    "lr": organizer_lr,
                },
                {
                    "params": [parameter for _, parameter in prediction_parameters],
                    "lr": learning_rate,
                },
            ],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        groups = [
            _optimizer_group_record("organizer", organizer_lr, organizer_parameters),
            _optimizer_group_record("prediction", learning_rate, prediction_parameters),
        ]
        mode = "split"
    inventory = {
        "mode": mode,
        "weight_decay": weight_decay,
        "groups": groups,
        "group_structure_sha256": _optimizer_inventory_digest(groups),
    }
    return optimizer, inventory


def refresh_optimizer_group_inventory(
    model: ChannelThermalHONFModel,
    inventory: Dict[str, Any],
) -> Dict[str, Any]:
    """Refresh scalar counts after lazy parameters have seen a real batch."""

    named = dict(model.named_parameters())
    refreshed = copy.deepcopy(inventory)
    for group in refreshed["groups"]:
        group_parameters = [(name, named[name]) for name in group["parameter_names"]]
        group.update(
            _optimizer_group_record(
                group["name"],
                group["learning_rate"],
                group_parameters,
            )
        )
    refreshed["group_structure_sha256"] = _optimizer_inventory_digest(refreshed["groups"])
    return refreshed


def _print_optimizer_group_inventory(inventory: Dict[str, Any]) -> None:
    """Print the compact launch inventory; full sorted names remain on disk."""

    for group in inventory["groups"]:
        print(
            "[optimizer] "
            f"group={group['name']} lr={group['learning_rate']:.6g} "
            f"tensors={group['parameter_tensor_count']} scalars={group['trainable_scalar_count']} "
            f"names_sha256={group['ordered_names_sha256']}"
        )


def _validate_optimizer_resume_compatibility(
    checkpoint: Dict[str, Any],
    current_inventory: Dict[str, Any],
) -> None:
    """Reject one-group/split or split-structure drift before optimizer restore."""

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not optimizer_state:
        return
    saved_group_count = len(optimizer_state.get("param_groups") or [])
    current_group_count = len(current_inventory["groups"])
    if saved_group_count != current_group_count:
        raise ValueError(
            "Resume optimizer group structure does not match this launch "
            f"({saved_group_count} saved versus {current_group_count} current groups). "
            "Use --initialize-checkpoint to start a new run across optimizer modes."
        )
    saved_inventory = checkpoint.get("optimizer_group_inventory")
    if current_group_count > 1:
        if not isinstance(saved_inventory, dict):
            raise ValueError(
                "Split-optimizer resume checkpoint lacks optimizer group provenance. "
                "Use --initialize-checkpoint instead."
            )
        if saved_inventory.get("group_structure_sha256") != current_inventory.get(
            "group_structure_sha256"
        ):
            raise ValueError(
                "Resume split-optimizer membership or learning rates do not match this launch. "
                "Use --initialize-checkpoint instead."
            )

