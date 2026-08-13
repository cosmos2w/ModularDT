from __future__ import annotations

import pytest
import torch

from honf_runtime.devices import resolve_device
from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_runtime.reproducibility import environment_snapshot, seed_all


def test_seed_all_repeats_cpu_torch_sequence() -> None:
    seed_all(123)
    first = torch.rand(5)
    seed_all(123)
    assert torch.equal(first, torch.rand(5))


def test_device_resolution_and_environment_snapshot() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda:0")
    snapshot = environment_snapshot()
    assert snapshot["python"]
    assert snapshot["torch"] == torch.__version__


def test_checkpoint_identity_accepts_history_and_rejects_mismatch() -> None:
    validate_checkpoint_identity(
        {}, case_id="ThermalChannel", model_family="honf_forward", workflow="forward"
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_checkpoint_identity(
            {
                "checkpoint_schema_version": 1,
                "case_id": "AnotherCase",
                "model_family": "honf_forward",
                "workflow": "forward",
            },
            case_id="ThermalChannel",
            model_family="honf_forward",
            workflow="forward",
        )
    with pytest.raises(ValueError, match="Unsupported"):
        validate_checkpoint_identity(
            {"checkpoint_schema_version": 99},
            case_id="ThermalChannel",
            model_family="honf_forward",
            workflow="forward",
        )
