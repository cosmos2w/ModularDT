"""Strict recipe contract for the separately authorized formal A2-D fits.

This contract is intentionally narrower than the shared content-hypergraph
recipe validator: it binds only the two dataset-specific formal profiles and
does not change development or the H6 ``full_train1000`` protocol.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


FORMAL_PROTOCOL = "formal_d_full_train_v1"
FORMAL_SCHEMA = "content_hypergraph_recipe_v2"

_COMMON_MODEL = {
    "context_dim": 128,
    "num_heads": 4,
    "context_depth": 2,
    "num_hyperedges": 6,
    "local_hidden_dim": 128,
    "dropout": 0.0,
}
_THERMAL_MODEL = {
    "module_input_dim": 8,
    "environment_input_dim": 8,
    "coordinate_dim": 2,
    "query_input_dim": 0,
    "operating_input_dim": 14,
    "output_dim": 3,
    **_COMMON_MODEL,
    "thermal_response_rank": 4,
    "init_seed": 0,
}
_WIND_MODEL = {
    **_COMMON_MODEL,
    "thermal_response_rank": 0,
}
_CONTRACTS = {
    "thermal": {
        "seed": 0,
        "sampling_seed": 0,
        "primary_queries": 1024,
        "effective_batch_size": 48,
        "microbatch_size": 24,
        "total_epochs": 5000,
        "field_metric": "field_score",
        "monitor_every": 10,
        "latest_every": 10,
        "checkpoint_epochs": [10, 50, 100, 250, 500, 1000, 2500, 5000],
        "model": _THERMAL_MODEL,
    },
    "wind": {
        "seed": 0,
        "sampling_seed": 42,
        "primary_queries": 8192,
        "effective_batch_size": 16,
        "microbatch_size": 8,
        "total_epochs": 2500,
        "field_metric": "val_volume_mse",
        "monitor_every": 5,
        "latest_every": 5,
        "checkpoint_epochs": [100, 500, 1000, 1500, 2000, 2500],
        "model": _WIND_MODEL,
    },
}
_OPTIMIZER_REQUIRED = {
    "name",
    "weight_decay",
    "bias_and_norm_weight_decay",
}
_OPTIMIZER_OPTIONAL = {"lr", "learning_rate", "betas", "eps"}


def _require_int(recipe: Mapping[str, Any], name: str, expected: int) -> None:
    value = recipe.get(name)
    if type(value) is not int or value != expected:
        raise ValueError(f"Formal D recipe requires {name}={expected}.")


def _require_float(mapping: Mapping[str, Any], name: str, expected: float) -> None:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Formal D optimizer requires {name}={expected}.")
    if not math.isfinite(float(value)) or float(value) != expected:
        raise ValueError(f"Formal D optimizer requires {name}={expected}.")


def validate_formal_d_recipe(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return one exact Thermal or Wind formal A2-D recipe.

    The two formal profiles use different data/query/batch contracts. Recipe
    objects may include dataset-owned path and binding fields, while their
    shared model, optimizer, and execution contracts remain exact.
    """

    if not isinstance(value, Mapping):
        raise TypeError("Formal D recipe must be a mapping.")
    recipe = dict(value)
    if recipe.get("schema") != FORMAL_SCHEMA:
        raise ValueError(f"Formal D requires schema={FORMAL_SCHEMA}.")
    if recipe.get("protocol") != FORMAL_PROTOCOL:
        raise ValueError(f"Formal D requires protocol={FORMAL_PROTOCOL}.")
    if recipe.get("architecture") != "A2-D":
        raise ValueError("Formal full-TRAIN protocol is limited to A2-D.")
    dataset = recipe.get("dataset")
    if not isinstance(dataset, str) or dataset not in _CONTRACTS:
        raise ValueError("Formal D dataset must be thermal or wind.")
    contract = _CONTRACTS[dataset]

    for name in ("seed", "sampling_seed", "primary_queries", "effective_batch_size",
                 "microbatch_size", "total_epochs"):
        _require_int(recipe, name, contract[name])
    _require_int(recipe, "approved_stop_after", contract["total_epochs"])
    if recipe.get("precision") != "fp32":
        raise ValueError("Formal D requires precision=fp32.")
    if recipe.get("edge_access") != "full":
        raise ValueError("Formal D requires full edge access.")
    if type(recipe.get("optimizer_bias_norm_exclusion")) is not bool or recipe[
        "optimizer_bias_norm_exclusion"
    ] is not False:
        raise ValueError("Formal D requires optimizer_bias_norm_exclusion=false.")

    model = recipe.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Formal D model contract must be a mapping.")
    expected_model = contract["model"]
    if dict(model) != expected_model:
        raise ValueError("Formal D model fields must match the sealed dataset-specific A2-D dimensions.")

    optimizer = recipe.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError("Formal D optimizer contract must be a mapping.")
    optimizer = dict(optimizer)
    keys = set(optimizer)
    if not _OPTIMIZER_REQUIRED <= keys or keys - (_OPTIMIZER_REQUIRED | _OPTIMIZER_OPTIONAL):
        raise ValueError("Formal D optimizer fields must be AdamW with declared weight-decay settings.")
    if optimizer.get("name") != "AdamW":
        raise ValueError("Formal D requires AdamW.")
    _require_float(optimizer, "weight_decay", 1.0e-5)
    _require_float(optimizer, "bias_and_norm_weight_decay", 1.0e-5)
    if "lr" in optimizer and "learning_rate" in optimizer:
        if isinstance(optimizer["lr"], bool) or not isinstance(optimizer["lr"], (int, float)):
            raise ValueError("Formal D optimizer lr and learning_rate must agree.")
        if isinstance(optimizer["learning_rate"], bool) or not isinstance(
            optimizer["learning_rate"], (int, float)
        ):
            raise ValueError("Formal D optimizer lr and learning_rate must agree.")
        if float(optimizer["lr"]) != float(optimizer["learning_rate"]):
            raise ValueError("Formal D optimizer lr and learning_rate must agree.")
    if "lr" in optimizer:
        _require_float(optimizer, "lr", 3.0e-4)
    if "learning_rate" in optimizer:
        _require_float(optimizer, "learning_rate", 3.0e-4)
    if "betas" in optimizer and optimizer["betas"] != [0.9, 0.999]:
        raise ValueError("Formal D AdamW betas must be [0.9, 0.999].")
    if "eps" in optimizer:
        _require_float(optimizer, "eps", 1.0e-8)

    execution = recipe.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Formal D execution contract must be a mapping.")
    expected_execution = {
        "monitor_every": contract["monitor_every"],
        "latest_every": contract["latest_every"],
        "curve_every": 100,
        "checkpoint_epochs": contract["checkpoint_epochs"],
        "field_metric": contract["field_metric"],
    }
    if dict(execution) != expected_execution:
        raise ValueError("Formal D execution fields differ from the sealed dataset-specific schedule/selector.")
    for name in ("monitor_every", "latest_every", "curve_every"):
        if type(execution[name]) is not int:
            raise ValueError(f"Formal D execution.{name} must be an integer.")
    if not isinstance(execution["checkpoint_epochs"], list) or any(
        type(epoch) is not int for epoch in execution["checkpoint_epochs"]
    ):
        raise ValueError("Formal D checkpoint_epochs must be an integer list.")
    if not isinstance(execution["field_metric"], str):
        raise ValueError("Formal D field_metric must be a string.")
    return recipe


def validate_formal_d_stop(recipe: Mapping[str, Any], stop_after: int) -> None:
    """Validate an execution segment stop within a sealed formal horizon."""

    validated = validate_formal_d_recipe(recipe)
    horizon = _CONTRACTS[validated["dataset"]]["total_epochs"]
    if type(stop_after) is not int or not 1 <= stop_after <= horizon:
        raise ValueError(f"Formal D execution stop must lie in 1..{horizon}.")


__all__ = ["FORMAL_PROTOCOL", "FORMAL_SCHEMA", "validate_formal_d_recipe", "validate_formal_d_stop"]
