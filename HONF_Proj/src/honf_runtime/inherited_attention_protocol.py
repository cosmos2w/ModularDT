"""Standalone guards for the opt-in inherited-attention family.

Segmented development and fresh full-TRAIN e1000 recipes share the Round3
native sampling/objective contracts but keep their own schemas and checkpoint
lineage. A valid full-TRAIN recipe is not launch authorization: callers must
also validate the selected-candidate qualification and current resource budget.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

IAG_FAMILY = "inherited_attention_v1"
IAG_REPRESENTATION = "iag_c"
IAG_C_ARCHITECTURE = "IAG-C"
IAG_Q_ARCHITECTURE = "IAG-Q"
IAG_ARCHITECTURE_REPRESENTATION = {
    IAG_C_ARCHITECTURE: "iag_c",
    IAG_Q_ARCHITECTURE: "iag_q",
}
IAG_SCHEMA = "inherited_attention_recipe_v1"
IAG_PROTOCOL = "inherited_attention_development_v1"
IAG_FULL_PROTOCOL = "inherited_attention_full_train1000_v1"
IAG_NUMERICAL_POLICY = "fp32_deterministic_cuda_v1"
IAG_FULL_QUALIFICATION_SCHEMA = "inherited_attention_full_train1000_qualification_v1"
IAG_NORMALIZATIONS = frozenset(("soft", "measure_entmax15"))
IAG_TOTAL_EPOCHS = 2500
IAG_FULL_TOTAL_EPOCHS = 1000
IAG_SPARSE_TRANSITION = {
    "soft_parent_epoch": 500,
    "blend_start_epoch": 501,
    "blend_end_epoch": 800,
    "exact_sparse_start_epoch": 801,
    "hard_stop_epoch": 2500,
}
IAG_FULL_SPARSE_TRANSITION = {
    **IAG_SPARSE_TRANSITION,
    "hard_stop_epoch": IAG_FULL_TOTAL_EPOCHS,
}

IAG_FULL_BUDGET_SCHEMA = "inherited_attention_full_train1000_budget_v1"
IAG_REPRESENTATION_REVIEW_SCHEMA = "inherited_attention_representation_review_v1"
IAG_Q_REPRESENTATION_REVIEW_DECISION = "replace_iag_c_with_iag_q"
IAG_AUTHORIZED_PHYSICAL_GPUS = {
    1: "GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e",
    2: "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39",
}

_DATASET_CONTRACTS = {
    "thermal": {
        "dimensions": {
            "module_input_dim": 8,
            "environment_input_dim": 8,
            "coordinate_dim": 2,
            "query_input_dim": 0,
            "operating_input_dim": 14,
            "output_dim": 3,
        },
        "seed": 0,
        "sampling_seed": 0,
        "sampling_version": "case_epoch_v1",
        "primary_queries": 1024,
        "effective_batch_size": 48,
        "microbatch_size": 24,
        "field_metric": "field_score",
        "thermal_response": True,
        "objective_contract": "thermal_native_flow_temperature_flux_train_response_v1",
    },
    "wind": {
        "dimensions": {
            "module_input_dim": 2,
            "environment_input_dim": 7,
            "coordinate_dim": 3,
            "query_input_dim": 7,
            "operating_input_dim": 11,
            "output_dim": 3,
        },
        "seed": 0,
        "sampling_seed": 42,
        "sampling_version": "case_epoch_v1",
        "primary_queries": 8192,
        "volume_queries": 6144,
        "hub_band_queries": 2048,
        "validation_volume_queries": 8192,
        "validation_hub_band_queries": 2048,
        "effective_batch_size": 16,
        "microbatch_size": 8,
        "field_metric": "val_volume_mse",
        "thermal_response": False,
        "objective_contract": "wind_native_volume_hub_band_standardized_v1",
    },
}

_OPTIMIZER = {
    "name": "AdamW",
    "lr": 3.0e-4,
    "weight_decay": 1.0e-5,
    "bias_and_norm_weight_decay": 1.0e-5,
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
}


def _require_exact(mapping: Mapping[str, Any], name: str, expected: Any) -> None:
    value = mapping.get(name)
    if isinstance(expected, bool):
        valid = type(value) is bool and value is expected
    elif type(expected) is int:
        valid = type(value) is int and value == expected
    elif isinstance(expected, float):
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) == expected
        )
    else:
        valid = type(value) is type(expected) and value == expected
    if not valid:
        raise ValueError(f"Inherited-attention protocol requires {name}={expected!r}.")


def configure_inherited_attention_numerics(policy: str = IAG_NUMERICAL_POLICY) -> None:
    """Apply the one Round4 FP32/CUDA numerical policy before device setup.

    The deterministic cuBLAS workspace is fixed before CUDA context creation;
    this helper intentionally leaves SDPA backend availability unchanged.
    """
    if policy != IAG_NUMERICAL_POLICY:
        raise ValueError(f"Unsupported inherited-attention numerical policy {policy!r}.")
    workspace = ":4096:8"
    if torch.cuda.is_initialized():
        if (os.environ.get("CUBLAS_WORKSPACE_CONFIG") != workspace
                or not torch.are_deterministic_algorithms_enabled()
                or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32
                or not torch.backends.cudnn.deterministic
                or torch.backends.cudnn.benchmark):
            raise RuntimeError("Inherited-attention numerics must be configured before CUDA initialization.")
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_inherited_attention_recipe(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one segmented IAG-C/IAG-Q trajectory or guarded full probe.

    Data paths and dataset-bound manifest identities remain provider-owned.
    A sparse recipe only seals the absolute normalization transition; the
    training command must separately require the exact same-architecture soft
    epoch-500 checkpoint declaration before it may branch.
    """

    if not isinstance(value, Mapping):
        raise TypeError("Inherited-attention recipe must be a mapping.")
    recipe = dict(value)
    protocol = recipe.get("protocol")
    if protocol not in (IAG_PROTOCOL, IAG_FULL_PROTOCOL):
        raise ValueError("Inherited-attention protocol must be segmented development or guarded full_train1000.")
    full = protocol == IAG_FULL_PROTOCOL
    horizon = IAG_FULL_TOTAL_EPOCHS if full else IAG_TOTAL_EPOCHS
    architecture = recipe.get("architecture")
    representation = (
        IAG_ARCHITECTURE_REPRESENTATION.get(architecture)
        if isinstance(architecture, str) else None
    )
    if representation is None:
        raise ValueError("Inherited-attention architecture must be IAG-C or IAG-Q.")
    for name, expected in (
        ("schema", IAG_SCHEMA),
        ("family", IAG_FAMILY),
        ("representation", representation),
        ("architecture", architecture),
        ("model_family", IAG_FAMILY),
        ("total_epochs", horizon),
        ("approved_stop_after", horizon),
        ("precision", "fp32"),
        ("edge_access", "full"),
        ("optimizer_bias_norm_exclusion", False),
        ("numerical_policy", IAG_NUMERICAL_POLICY),
    ):
        _require_exact(recipe, name, expected)

    dataset = recipe.get("dataset")
    if not isinstance(dataset, str) or dataset not in _DATASET_CONTRACTS:
        raise ValueError("Inherited-attention dataset must be thermal or wind.")
    if architecture == IAG_Q_ARCHITECTURE and dataset != "wind":
        raise ValueError("IAG-Q is authorized only for the Wind dataset in this Round4 revision.")
    contract = _DATASET_CONTRACTS[dataset]
    for name, expected in contract.items():
        if name in ("objective_contract", "dimensions", "field_metric"):
            continue
        _require_exact(recipe, name, expected)
    model = recipe.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("Inherited-attention model configuration must be a mapping.")
    for name, expected in contract["dimensions"].items():
        _require_exact(model, name, expected)
    if recipe.get("objective_contract") != contract["objective_contract"]:
        raise ValueError("Inherited-attention objective must retain the Round3 formal-semantic dataset law.")
    expected_profile = (
        "thermal_content_hypergraph_original600_train_canonical89_validation_v1"
        if full and dataset == "thermal"
        else "wind_original420_train_fullVALID90_followup1000_v1"
        if full and dataset == "wind"
        else "fixed25_v1"
        if dataset == "thermal"
        else "fixed24_v1"
    )
    _require_exact(recipe, "dataset_profile", expected_profile)

    normalization = recipe.get("normalization")
    if not isinstance(normalization, str) or normalization not in IAG_NORMALIZATIONS:
        raise ValueError("Inherited-attention normalization must be soft or measure_entmax15.")
    if normalization == "measure_entmax15":
        transition = recipe.get("sparse_transition")
        expected_transition = IAG_FULL_SPARSE_TRANSITION if full else IAG_SPARSE_TRANSITION
        if not isinstance(transition, Mapping) or set(transition) != set(expected_transition):
            raise ValueError("Sparse inherited-attention recipes require the fixed absolute-epoch 501–800 transition.")
        for name, expected in expected_transition.items():
            _require_exact(transition, name, expected)
        expected_trajectory = "fresh_full_sparse_schedule" if full else "e500_sparse_child"
        if recipe.get("trajectory") != expected_trajectory:
            raise ValueError("Sparse inherited-attention trajectory must match its segmented child or fresh full-TRAIN schedule.")
    else:
        expected_trajectory = "fresh_full_soft" if full else "fresh_soft"
        if "sparse_transition" in recipe or recipe.get("trajectory") != expected_trajectory:
            raise ValueError("Soft inherited-attention must be a fresh trajectory without a sparse transition.")

    optimizer = recipe.get("optimizer")
    if not isinstance(optimizer, Mapping) or set(optimizer) != set(_OPTIMIZER):
        raise ValueError("Inherited-attention optimizer must use the sealed all-parameter AdamW contract.")
    for name, expected in _OPTIMIZER.items():
        _require_exact(optimizer, name, expected)

    execution = recipe.get("execution")
    if not isinstance(execution, Mapping):
        raise TypeError("Inherited-attention execution contract must be a mapping.")
    expected_execution = {
        "monitor_every": 100,
        "latest_every": 100,
        "curve_every": 10,
        "checkpoint_epochs": list(range(100, horizon + 1, 100)),
        "field_metric": contract["field_metric"],
    }
    if set(execution) != set(expected_execution):
        raise ValueError("Inherited-attention execution cadence and selector are fixed by the Round4 contract.")
    for name, expected in expected_execution.items():
        _require_exact(execution, name, expected)

    for name, expected in (
        ("context_dim", 128),
        ("num_heads", 4),
        ("context_depth", 2),
        ("local_hidden_dim", 128),
        ("dropout", 0.0),
        ("init_seed", 0),
        ("signed_carrier_width", 8),
        ("thermal_response", contract["thermal_response"]),
    ):
        _require_exact(model, name, expected)
    if model.get("normalization") != normalization:
        raise ValueError("Inherited-attention core normalization must match the sealed recipe normalization.")
    if architecture == IAG_C_ARCHITECTURE:
        if model.get("scene_executor") not in ("dense", "selected"):
            raise ValueError("IAG-C scene_executor must be dense or selected.")
        legacy_query_executor = model.get("query_executor") in ("indexed", "dense_reference")
        round5_projected_core = (
            model.get("schema") == "scene_prepared_moment_hyperedge_core_v1"
            and model.get("query_executor") == "indexed_projected"
            and model.get("collect_exact_work_counts") is False
            and model.get("moment_variant") in ("C0", "M", "H")
            and model.get("moment_width") == 8
            and model.get("processor_hidden_dim") == 128
            and model.get("moment_init_seed") == 0
        )
        if not (legacy_query_executor or round5_projected_core):
            raise ValueError("IAG-C query_executor must be indexed or dense_reference.")
    else:
        if model.get("scene_executor") != "cached_tokens":
            raise ValueError("IAG-Q scene_executor must be cached_tokens; it does not prepare P@V.")
        if model.get("query_executor") not in ("cell_selected", "dense_reference"):
            raise ValueError("IAG-Q query_executor must be cell_selected or dense_reference.")
    if any(name in model for name in (
        "num_hyperedges",
        "thermal_response_rank",
        "source_read_policy",
        "source_read_executor",
        "source_read_rank",
        "source_read_scorer_width",
        "source_value_conditioning",
        "reader_execution_backend",
        "reader_query_chunk_size",
    )):
        raise ValueError("Inherited-attention model config cannot carry legacy H6/direct/source-read controls.")

    if "source_reader_variant" in recipe or "dense_host_variant" in recipe:
        raise ValueError("Inherited-attention recipes cannot declare a Round3 source-read arm.")
    return recipe


def validate_inherited_attention_representation_review(
    recipe: Mapping[str, Any], receipt: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the compact prior-evidence binding required for Wind IAG-Q."""
    validated = validate_inherited_attention_recipe(recipe)
    if validated["architecture"] != IAG_Q_ARCHITECTURE:
        if receipt is not None:
            raise ValueError("Representation-review receipts are reserved for IAG-Q.")
        return None
    if receipt is None:
        raise ValueError("Wind IAG-Q requires its measured representation-review receipt.")
    if not isinstance(receipt, Mapping):
        raise TypeError("Wind IAG-Q representation review must be a mapping.")
    for name, expected in (
        ("schema", IAG_REPRESENTATION_REVIEW_SCHEMA),
        ("dataset", "wind"),
        ("decision", IAG_Q_REPRESENTATION_REVIEW_DECISION),
        ("source_architecture", IAG_C_ARCHITECTURE),
        ("target_architecture", IAG_Q_ARCHITECTURE),
        ("review_epoch", 500),
        ("substantive_revisions_used_for_dataset", 1),
        ("maximum_revisions_for_dataset", 1),
    ):
        _require_exact(receipt, name, expected)
    for name in (
        "source_checkpoint", "capacity_diagnosis", "fresh_run_dir", "future_sparse_child_run_dir",
    ):
        value = receipt.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"IAG-Q representation review must bind {name}.")
    for name in ("source_checkpoint_sha256", "capacity_diagnosis_sha256"):
        value = receipt.get(name)
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"IAG-Q representation review must bind a lowercase {name}.")
    if Path(receipt["fresh_run_dir"]).expanduser().resolve() == Path(
        receipt["future_sparse_child_run_dir"]
    ).expanduser().resolve():
        raise ValueError("IAG-Q soft and sparse-child runs must have distinct output directories.")
    return dict(receipt)


def validate_inherited_attention_full_train_authorization(
    recipe: Mapping[str, Any],
    qualification: Mapping[str, Any] | None,
    budget: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require a selected e2500 qualification and current bounded budget.

    The receipt is intentionally compact and caller-owned. This guard checks
    that the selected same-dataset checkpoint passed the native and
    priority/cost gates, then binds one physical GPU UUID and a resource
    allowance. It does not query devices or create authorization itself.
    """
    validated = validate_inherited_attention_recipe(recipe)
    if validated["protocol"] != IAG_FULL_PROTOCOL:
        raise ValueError("Full-TRAIN authorization applies only to the fresh inherited-attention e1000 protocol.")
    if not isinstance(qualification, Mapping) or qualification.get("schema") != IAG_FULL_QUALIFICATION_SCHEMA:
        raise ValueError("A qualified same-dataset e2500 receipt is required before full-TRAIN dispatch.")
    if qualification.get("dataset") != validated["dataset"]:
        raise ValueError("Full-TRAIN qualification receipt belongs to a different dataset.")
    if (qualification.get("architecture") != validated["architecture"]
            or qualification.get("selected_epoch") != IAG_TOTAL_EPOCHS):
        raise ValueError("Full-TRAIN qualification must select this architecture at literal epoch2500.")
    if qualification.get("qualified") is not True:
        raise ValueError("Full-TRAIN candidate qualification did not pass.")
    gates = qualification.get("gates")
    required_gates = {"native_contracts", "priority_accuracy", "complete_cost"}
    if not isinstance(gates, Mapping) or set(gates) != required_gates or any(gates.get(key) is not True for key in required_gates):
        raise ValueError("Full-TRAIN qualification must record passing native, priority-accuracy, and complete-cost gates.")
    checkpoint = qualification.get("selected_checkpoint")
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("path"), str) or not checkpoint["path"]:
        raise ValueError("Full-TRAIN qualification must bind the selected epoch2500 checkpoint path.")
    digest = checkpoint.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Full-TRAIN qualification must bind a lowercase SHA-256 checkpoint digest.")

    if not isinstance(budget, Mapping) or budget.get("schema") != IAG_FULL_BUDGET_SCHEMA:
        raise ValueError("A separately declared current resource-budget receipt is required before full-TRAIN dispatch.")
    if budget.get("dataset") != validated["dataset"] or budget.get("approved") is not True:
        raise ValueError("Full-TRAIN budget approval must bind the selected dataset and approved status.")
    devices = budget.get("physical_gpu_uuids")
    physical_index = budget.get("physical_gpu_index")
    if (type(physical_index) is not int or physical_index not in IAG_AUTHORIZED_PHYSICAL_GPUS
            or not isinstance(devices, list) or len(devices) != 1
            or devices[0] != IAG_AUTHORIZED_PHYSICAL_GPUS[physical_index]):
        raise ValueError("A full-TRAIN probe must bind exactly physical GPU1 or GPU2 by index and UUID.")
    hours = budget.get("max_gpu_process_hours")
    elapsed = budget.get("max_elapsed_hours")
    for name, value in (("max_gpu_process_hours", hours), ("max_elapsed_hours", elapsed)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"Full-TRAIN budget field {name} must be positive and finite.")
    if budget.get("qualification_checkpoint_sha256") != digest:
        raise ValueError("Full-TRAIN budget must bind the exact qualified candidate checkpoint.")
    return dict(qualification), dict(budget)


def validate_inherited_attention_stop(recipe: Mapping[str, Any], stop_after: int) -> None:
    """Bound a development or exact child continuation to absolute epoch2500."""

    validated = validate_inherited_attention_recipe(recipe)
    maximum = IAG_FULL_TOTAL_EPOCHS if validated["protocol"] == IAG_FULL_PROTOCOL else IAG_TOTAL_EPOCHS
    if type(stop_after) is not int or not 1 <= stop_after <= maximum:
        raise ValueError(f"Inherited-attention stop must lie in1..{maximum}; the schedule is not launch authorization.")
    if validated["protocol"] == IAG_FULL_PROTOCOL and stop_after != IAG_FULL_TOTAL_EPOCHS:
        raise ValueError("Full-TRAIN IAG workflow is capped at its exact epoch1000 endpoint.")
    if (validated["protocol"] == IAG_PROTOCOL
            and validated["normalization"] == "measure_entmax15" and stop_after <= IAG_SPARSE_TRANSITION[
        "soft_parent_epoch"
    ]):
        raise ValueError("Sparse inherited-attention child stop cannot precede its absolute epoch500 prefix.")


__all__ = [
    "IAG_ARCHITECTURE_REPRESENTATION",
    "IAG_AUTHORIZED_PHYSICAL_GPUS",
    "IAG_C_ARCHITECTURE",
    "IAG_FAMILY",
    "IAG_FULL_BUDGET_SCHEMA",
    "IAG_FULL_PROTOCOL",
    "IAG_FULL_QUALIFICATION_SCHEMA",
    "IAG_FULL_TOTAL_EPOCHS",
    "IAG_NORMALIZATIONS",
    "IAG_PROTOCOL",
    "IAG_Q_ARCHITECTURE",
    "IAG_Q_REPRESENTATION_REVIEW_DECISION",
    "IAG_REPRESENTATION",
    "IAG_REPRESENTATION_REVIEW_SCHEMA",
    "IAG_SCHEMA",
    "IAG_SPARSE_TRANSITION",
    "IAG_TOTAL_EPOCHS",
    "validate_inherited_attention_full_train_authorization",
    "validate_inherited_attention_recipe",
    "validate_inherited_attention_representation_review",
    "validate_inherited_attention_stop",
]
