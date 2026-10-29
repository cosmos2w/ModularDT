"""State-preserving R13-4 Guide e500 attachment for the R13-5 joint95 reader."""
from __future__ import annotations

from collections.abc import Mapping
import re
from pathlib import Path

import numpy as np
import torch


TRANSFER_SCHEMA = "thermal_r13_4_guide_e500_to_r13_5_joint_member_routing_transfer_v1"
SOURCE_WORKFLOW = "interaction_reused_member_routing_joint_training_v3"
TARGET_WORKFLOW = "interaction_reused_joint_member_routing_joint_training_v1"


def _logical_cuda_index(value):
    match = re.fullmatch(r"cuda:(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def restore_joint_member_routing_transfer_rng_state(state, *, source_device, target_device):
    """Restore a Guide transfer's RNG on the selected target device.

    The engine's ordinary restore remains unchanged for historical exact resumes.
    A new transfer can run with a different CUDA-visible device count, so its
    active source stream is mapped explicitly from the source's declared logical
    device to the selected target logical device. CPU and host RNGs use the
    regular exact restore path.
    """
    from .unified_training import restore_rng_state

    cpu_state = {key: value for key, value in state.items() if key != "torch_cuda"}
    restore_rng_state(cpu_state)

    saved_cuda = list(state.get("torch_cuda", ()))
    target = torch.device(target_device)
    target_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    receipt = {
        "source_saved_cuda_device_count": len(saved_cuda),
        "target_visible_cuda_device_count": target_count,
        "saved_cuda_streams_identical": bool(saved_cuda) and all(
            torch.equal(saved_cuda[0].detach().cpu(), item.detach().cpu())
            for item in saved_cuda[1:]
        ),
        "source_declared_training_device": str(source_device) if source_device is not None else None,
        "source_cuda_state_index": None,
        "target_training_device": str(target),
        "restored_cuda_rng_scope": "selected_training_device_only",
        "cuda_rng_mapping": [],
    }
    if target.type != "cuda" or not saved_cuda or target_count == 0:
        receipt["status"] = "no_cuda_stream_to_restore"
        return receipt

    target_index = torch.cuda.current_device() if target.index is None else target.index
    if not 0 <= target_index < target_count:
        raise ValueError("R13-5 transfer target CUDA device is outside the visible device inventory.")
    source_index = _logical_cuda_index(source_device)
    if source_index is None:
        if not receipt["saved_cuda_streams_identical"]:
            raise ValueError(
                "R13-5 transfer needs the source's declared logical CUDA device when saved streams differ."
            )
        source_index = 0
    if not 0 <= source_index < len(saved_cuda):
        raise ValueError("R13-5 transfer source CUDA device is outside the saved RNG inventory.")

    torch.cuda.set_rng_state(saved_cuda[source_index].detach().cpu(), device=target)
    receipt["source_cuda_state_index"] = source_index
    receipt["cuda_rng_mapping"] = [{
        "source_logical_device": source_index,
        "target_logical_device": target_index,
    }]
    receipt["status"] = "selected_source_stream_restored"
    return receipt


def same_state(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, Mapping):
        return isinstance(right, Mapping) and left.keys() == right.keys() and all(
            same_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            same_state(a, b) for a, b in zip(left, right)
        )
    return left == right


def prepare_joint_member_routing_transfer(model, provider, source, current, declaration, *, source_path):
    """Validate the literal R13-4 parent and preserve compatible full training state."""
    initialization = current.get("initialization", {})
    expected_path = Path(str(initialization.get("source_checkpoint", ""))).expanduser().resolve()
    source_path = Path(source_path).resolve(strict=True)
    if (not isinstance(declaration, Mapping)
            or declaration.get("schema") != TRANSFER_SCHEMA
            or declaration.get("source_epoch") != 500
            or declaration.get("source_checkpoint") != str(source_path)
            or expected_path != source_path
            or source.get("epoch") != 500
            or source.get("arm") != "R13-4-Guide"
            or source.get("workflow") != SOURCE_WORKFLOW
            or current.get("workflow") != TARGET_WORKFLOW
            or current.get("training_membership") != "full_train600"
            or model.architecture not in ("R13-5-M-Joint95", "R13-5-ME-Joint95")):
        raise ValueError("R13-5 transfer requires the literal R13-4 Guide epoch500 and a full-TRAIN600 arm.")
    previous = source["experiment_identity"]
    old_provider = previous["provider_identity"]
    new_provider = current["provider_identity"]
    for name in ("training_case_ids", "validation_case_ids"):
        if old_provider.get(name) != new_provider.get(name):
            raise ValueError(f"R13-5 transfer changed the source {name}.")
    if (not isinstance(old_provider.get("source_binding"), Mapping)
            or not same_state(old_provider["source_binding"], new_provider.get("source_binding"))):
        raise ValueError("R13-5 transfer changed the existing dataset/manifest source binding.")

    old_adapter, new_adapter = old_provider["adapter_state"], new_provider["adapter_state"]
    for name in ("normalization_stats", "heat_scale", "native_grid_nx_ny", "environment_grid_nx_ny"):
        if not same_state(old_adapter[name], new_adapter[name]):
            raise ValueError(f"R13-5 transfer changed TRAIN normalization or Thermal adapter field {name}.")
    old_core, new_core = old_adapter["core"], new_adapter["core"]
    for name in ("base_model_config", "input_dataset", "input_capability", "query_tile_size"):
        if not same_state(old_core[name], new_core[name]):
            raise ValueError(f"R13-5 transfer changed compatible R13-4 core field {name}.")
    if (previous.get("optimizer") != current.get("optimizer")
            or previous.get("optimizer_schedule_contract") != current.get("optimizer_schedule_contract")
            or previous.get("sampling_dataset_id") != current.get("sampling_dataset_id")
            or previous.get("engine_config", {}).get("seed") != current.get("engine_config", {}).get("seed")
            or previous.get("engine_config", {}).get("effective_cases") != current.get("engine_config", {}).get("effective_cases")
            or previous.get("engine_config", {}).get("microbatch_cases") != current.get("engine_config", {}).get("microbatch_cases")
            or previous.get("engine_config", {}).get("sampling_version") != current.get("engine_config", {}).get("sampling_version")):
        raise ValueError("R13-5 transfer changed optimizer, schedule, or native sampling/packing identity.")

    source_model = source["model_state_dict"]
    target_model = model.state_dict()
    if (target_model.keys() != source_model.keys()
            or any(not torch.is_tensor(value) or value.shape != source_model[name].shape
                   for name, value in target_model.items())):
        raise ValueError("R13-5 must preserve every R13-4 named parameter and shape at attachment.")
    source_optimizer = source["optimizer_state_by_name"]
    source_parameter_names = {
        name for group in source_optimizer.get("groups", ()) for name in group.get("parameter_names", ())
    }
    target_parameter_names = set(dict(model.named_parameters()))
    if source_parameter_names != target_parameter_names:
        raise ValueError("R13-5 transfer changed named optimizer membership.")
    return {
        "schema": TRANSFER_SCHEMA,
        "kind": "r13_4_guide_e500_joint95_reader_branch",
        "source_checkpoint": str(source_path),
        "source_epoch": 500,
        "source_arm": source["arm"],
        "source_training_device": (
            (old_provider.get("recipe", {}).get("execution", {}).get("preferred_devices") or [None])[0]
        ),
        "target_arm": model.architecture,
        "inherited_state": (
            "all model parameters, named AdamW moments, RNG, sampler, TRAIN normalization, "
            "and provider numerical clocks"
        ),
        "new_parameter_names": [],
        "changes": (
            "M/ME address law, joint M/E measure and whole-row retrieval, one joint95 member prefix; "
            "general joint field head and paired native physical objective are retained"
        ),
        "health_comparison_origin": "reset for R13-5 at parent epoch500",
        "native_physical_objective_weights": "unchanged; full and joint95 losses are averaged by field role",
    }


__all__ = [
    "TRANSFER_SCHEMA", "prepare_joint_member_routing_transfer",
    "restore_joint_member_routing_transfer_rng_state", "same_state",
]
