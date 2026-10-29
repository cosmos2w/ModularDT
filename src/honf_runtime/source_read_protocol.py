"""Bounded, formal-semantics source-reader study contracts.

These scopes never grant formal-D's long-run authorization or the historical
H6 qualification. Dataset construction remains owned by each native provider.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

SOURCE_READ_DEVELOPMENT = "source_read_development_v1"
SOURCE_READ_FULL = "source_read_full1000"
SOURCE_READ_PROTOCOLS = frozenset((SOURCE_READ_DEVELOPMENT, SOURCE_READ_FULL))
SOURCE_READ_VARIANTS = frozenset(("D0", "D-scale", "S-Utility", "S-Complete"))
SOURCE_READ_NUMERICS = "deterministic_cuda_v1"


def _exact(mapping: Mapping[str, Any], name: str, expected: Any) -> None:
    value = mapping.get(name)
    if isinstance(expected, int) and not isinstance(expected, bool):
        valid = type(value) is int and value == expected
    else:
        valid = type(value) is type(expected) and value == expected
    if not valid:
        raise ValueError(f"Source-read protocol requires {name}={expected!r}.")


def validate_source_read_recipe(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable study semantics without opening data or CUDA.

    Full-data launch qualification is a separate, source/evidence-bound check;
    a valid recipe alone does not authorize a fresh full-data execution.
    """
    if not isinstance(value, Mapping):
        raise TypeError("Source-read recipe must be a mapping.")
    recipe = dict(value)
    _exact(recipe, "schema", "content_hypergraph_recipe_v2")
    _exact(recipe, "architecture", "A2-D")
    if recipe.get("protocol") not in SOURCE_READ_PROTOCOLS:
        raise ValueError("Use a distinct source_read_development_v1 or source_read_full1000 scope.")
    dataset = recipe.get("dataset")
    if dataset not in ("thermal", "wind"):
        raise ValueError("Source-read dataset must be thermal or wind.")
    full = recipe["protocol"] == SOURCE_READ_FULL
    thermal = dataset == "thermal"
    variant = recipe.get("source_reader_variant")
    if variant not in SOURCE_READ_VARIANTS:
        raise ValueError("Unsupported bounded source-reader variant.")
    if full and variant == "D0":
        raise ValueError("Unchanged D0 does not qualify for another full-TRAIN run.")
    settings = {
        "seed": 0, "sampling_seed": 0 if thermal else 42,
        "primary_queries": 1024 if thermal else 8192,
        "effective_batch_size": 48 if thermal else 16,
        "microbatch_size": 24 if thermal else 8,
        "total_epochs": (5000 if thermal else 2500) if full else 2500,
        "approved_stop_after": 1000 if full else 2500,
        "precision": "fp32", "edge_access": "full",
        "optimizer_bias_norm_exclusion": False,
    }
    for name, expected in settings.items():
        _exact(recipe, name, expected)
    optimizer = recipe.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError("Source-read optimizer must be an explicit AdamW mapping.")
    expected_optimizer = {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-5,
                          "bias_and_norm_weight_decay": 1e-5,
                          "betas": [0.9, 0.999], "eps": 1e-8}
    if dict(optimizer) != expected_optimizer or any(
        isinstance(optimizer[name], bool) or not math.isfinite(float(optimizer[name]))
        for name in ("lr", "weight_decay", "bias_and_norm_weight_decay", "eps")
    ):
        raise ValueError("Source-read optimizer must retain constant all-parameter AdamW3e-4/WD1e-5.")
    model = recipe.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Source-read model must be a mapping.")
    fixed = {"context_dim": 128, "num_heads": 4, "context_depth": 2,
             "num_hyperedges": 6, "thermal_response_rank": 4 if thermal else 0,
             "local_hidden_dim": 128, "dropout": 0.0}
    if thermal:
        fixed.update(module_input_dim=8, environment_input_dim=8, coordinate_dim=2,
                     query_input_dim=0, operating_input_dim=14, output_dim=3, init_seed=0)
    for name, expected in fixed.items():
        actual = model.get(name)
        if actual != expected or isinstance(actual, bool):
            raise ValueError(f"Source-read model requires {name}={expected}.")
    optional = {"reader_execution_backend", "source_value_conditioning", "source_read_policy",
                "source_read_executor", "source_read_rank", "source_read_scorer_width", "reader_query_chunk_size"}
    if set(model) - set(fixed) - optional:
        raise ValueError("Source-read model contains undeclared architecture fields.")
    if "source_read_numerics" in recipe:
        if (recipe.get("protocol") not in SOURCE_READ_PROTOCOLS
                or variant not in ("S-Utility", "S-Complete")
                or recipe["source_read_numerics"] != SOURCE_READ_NUMERICS):
            raise ValueError(
                "source_read_numerics is available only to sparse children as deterministic_cuda_v1."
            )
    if model.get("reader_execution_backend", "legacy") not in ("legacy", "reduced_projection"):
        raise ValueError("Unqualified source-reader execution backend.")
    chunk = model.get("reader_query_chunk_size")
    if chunk is not None and (type(chunk) is not int or chunk not in (128, 512, 2048, 8192)):
        raise ValueError("Source-read query tile must be one of the four bounded qualified choices.")
    if not thermal and chunk not in (None, 8192):
        raise ValueError("Wind source-read scopes preserve whole-query execution; omit the tile or replay the inert8192 metadata.")
    conditioning = model.get("source_value_conditioning", "none")
    host = recipe.get("dense_host_variant", "D0")
    expected_conditioning = "source_scale" if (variant == "D-scale" or (
        variant in ("S-Utility", "S-Complete") and host == "D-scale")) else "none"
    if conditioning != expected_conditioning or host not in ("D0", "D-scale"):
        raise ValueError("Source-scale conditioning differs from the declared dense host/variant.")
    if variant in ("D0", "D-scale") and model.get("source_read_policy", "full") != "full":
        raise ValueError("Dense accuracy controls must execute the full source reader.")
    if variant in ("S-Utility", "S-Complete"):
        expected_policy = "utility" if variant == "S-Utility" else "completion"
        if model.get("source_read_policy") != expected_policy:
            raise ValueError("A sparse child must declare its implemented source-read policy.")
        if model.get("source_read_executor") not in ("packed", "packet", "dense_masked"):
            raise ValueError("A sparse child requires its explicit selected-function executor.")
    if model.get("source_read_rank", 16) != 16 or model.get("source_read_scorer_width", 32) != 32:
        raise ValueError("The bounded completion/scorer widths are fixed at16/32.")
    monitor = (10 if thermal else 5) if full else 100
    stop = settings["approved_stop_after"]
    expected_execution = {"monitor_every": monitor, "latest_every": monitor, "curve_every": 10,
                          "checkpoint_epochs": list(range(100, stop + 1, 100)),
                          "field_metric": "field_score" if thermal else "val_volume_mse"}
    if recipe.get("execution") != expected_execution:
        raise ValueError("Source-read cadence/selector/milestones differ from the declared scope.")
    if not thermal:
        for name, expected in {"volume_queries":6144,"hub_band_queries":2048,
                               "validation_volume_queries":8192,"validation_hub_band_queries":2048}.items():
            _exact(recipe, name, expected)
        objective = {"volume_sampling_fraction":0.75,"volume_loss_weight":0.75,
                     "hub_band_loss_weight":0.25,"validation_volume_loss_weight":0.8,
                     "validation_hub_band_loss_weight":0.2,"channel_weights":[1.0,1.0,1.0],
                     "sampler":"native_uniform_with_replacement_volume_and_hub_band_v1"}
        if recipe.get("objective") != objective:
            raise ValueError("Wind source-read study requires formal two-region equal-channel semantics.")
    return recipe


def validate_source_read_stop(recipe: Mapping[str, Any], stop_after: int) -> None:
    validated = validate_source_read_recipe(recipe)
    maximum = 1000 if validated["protocol"] == SOURCE_READ_FULL else 2500
    if type(stop_after) is not int or not 1 <= stop_after <= maximum:
        raise ValueError(f"Source-read execution stop must lie in1..{maximum}; future horizon is not permission.")


def configure_source_read_numerics(recipe: Mapping[str, Any]) -> bool:
    """Apply the one opt-in deterministic CUDA contract for sparse readers.

    Missing metadata deliberately leaves the process settings unchanged. The
    workspace variable is established before any provider/model construction;
    conflicting values fail before the caller can build a model.
    """
    if not isinstance(recipe, Mapping):
        raise TypeError("Source-read numerical recipe must be a mapping.")
    if "source_read_numerics" not in recipe:
        return False
    variant = recipe.get("source_reader_variant")
    if (recipe.get("protocol") not in SOURCE_READ_PROTOCOLS
            or recipe.get("architecture") != "A2-D"
            or variant not in ("S-Utility", "S-Complete")):
        raise ValueError("Deterministic CUDA numerics are limited to source-read A2-D S-Utility/S-Complete.")
    if recipe.get("source_read_numerics") != SOURCE_READ_NUMERICS:
        raise ValueError("Unsupported source_read_numerics declaration.")
    current_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current_workspace not in (None, ":4096:8"):
        raise ValueError("deterministic_cuda_v1 requires CUBLAS_WORKSPACE_CONFIG=:4096:8.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return True


__all__ = ["SOURCE_READ_DEVELOPMENT", "SOURCE_READ_FULL", "SOURCE_READ_PROTOCOLS",
           "SOURCE_READ_VARIANTS", "SOURCE_READ_NUMERICS", "configure_source_read_numerics",
           "validate_source_read_recipe", "validate_source_read_stop"]
