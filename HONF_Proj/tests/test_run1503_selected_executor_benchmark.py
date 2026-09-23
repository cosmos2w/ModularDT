"""CPU contracts for the Run-1503 adaptive executor benchmark."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools" / "diagnostics"))

from run_run1503_selected_executor_benchmark import (
    EXPECTED_ARCHITECTURE,
    _adaptive_ledger,
    _measure,
    _require_checkpoint_identity,
    build_parser,
)


def _model(architecture: str = EXPECTED_ARCHITECTURE):
    return SimpleNamespace(
        config=SimpleNamespace(
            core_honf=SimpleNamespace(forward_architecture=architecture)
        )
    )


def test_identity_gate_requires_adaptive_architecture_and_exact_gate_epoch() -> None:
    record = _require_checkpoint_identity(_model(), {"epoch": 150}, expected_epoch=150)
    assert record == {
        "architecture": EXPECTED_ARCHITECTURE,
        "epoch": 150,
        "strict_model_state_load": True,
        "expected_gate_epoch": 150,
    }
    with pytest.raises(ValueError, match="exact architecture"):
        _require_checkpoint_identity(_model("sparse_incidence_group_control_honf"), {"epoch": 150})
    with pytest.raises(ValueError, match="requires epoch 500"):
        _require_checkpoint_identity(_model(), {"epoch": 150}, expected_epoch=500)


def test_parser_defaults_to_matched_2048_receiver_chunk(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            f"e150={tmp_path / 'epoch_0150_model.pt'}",
            "--output",
            str(tmp_path / "benchmark.json"),
        ]
    )
    assert args.receiver_chunk_size == 2048
    assert args.case_id == []
    assert args.expected_epoch is None


def test_adaptive_ledger_keeps_fine_coarse_and_source_rows_distinct() -> None:
    output = {
        "pred_field": torch.zeros(1, 5, 2),
        "interaction_aux": {
            "group_control_adaptive_fine_group_rows_forward": torch.tensor([[6.0, 4.0, 9.0]]),
            "group_control_adaptive_fine_group_rows_logical": torch.tensor([[5.0, 3.0, 8.0]]),
            "group_control_adaptive_fine_group_rows_padded": torch.tensor([[1.0, 1.0, 1.0]]),
            "group_control_adaptive_fine_rows_per_query": torch.tensor([[3.0, 2.0, 1.0, 4.0, 2.0]]),
            "group_control_adaptive_fine_rows_logical": torch.tensor(16.0),
            "group_control_adaptive_fine_rows_forward": torch.tensor(19.0),
            "group_control_adaptive_fine_rows_padded": torch.tensor(3.0),
            "group_control_adaptive_fine_work_ratio": torch.tensor(0.95),
            "group_control_adaptive_fine_executor_selected": torch.ones((1, 5)),
            "group_control_adaptive_fine_executor_batch_limit": torch.tensor(128.0),
            "group_control_adaptive_fine_scalar_block_calls": torch.tensor(12.0),
            "group_control_adaptive_fine_batched_block_calls": torch.tensor(1.0),
            "group_control_adaptive_fine_block_call_reduction": torch.tensor(11.0),
            "group_control_adaptive_fine_scalar_gemm_launches": torch.tensor(12.0),
            "group_control_adaptive_fine_batched_gemm_launches": torch.tensor(1.0),
            "group_control_adaptive_fine_gemm_launch_reduction": torch.tensor(11.0),
            "group_control_adaptive_fine_batched_checkpoint_calls": torch.tensor(0.0),
            "group_control_adaptive_batched_fine_group_rows_forward": torch.tensor(
                [[12.0, 12.0, 12.0]]
            ),
            "group_control_adaptive_batched_fine_rows": torch.tensor(36.0),
            "group_control_adaptive_batched_fine_rows_padded": torch.tensor(20.0),
            "group_control_adaptive_batched_fine_rows_recompute": torch.tensor(0.0),
            "group_control_adaptive_coarse_rows": torch.tensor(15.0),
            "group_control_adaptive_full_rectangle_rows": torch.tensor(20.0),
            "group_control_adaptive_environment_source_mass": torch.tensor(
                [[[0.2, 0.0, 0.1], [0.0, 0.3, 0.0], [0.1, 0.0, 0.4], [0.0, 0.2, 0.0]]]
            ),
            "group_control_environment_incidence": torch.tensor(
                [[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]
            ),
            "group_control_environment_measure": torch.tensor([[0.3, 0.3, 0.2, 0.2]]),
            "group_control_environment_unique_pairs": torch.tensor(11.0),
            "group_control_environment_logical_paths": torch.tensor(13.0),
            "group_control_environment_support_rows": torch.tensor(11.0),
        },
    }
    ledger = _adaptive_ledger(output, receiver_chunk_size=2048)
    assert ledger["status"] == "complete"
    assert ledger["query_chunk_count"] == 1
    assert ledger["fine_rows_forward"] == 19.0
    assert ledger["coarse_rows"] == 15.0
    assert ledger["source_rows_per_group"] == [2, 2, 2]
    assert ledger["fine"]["group_rows_forward"] == [6.0, 4.0, 9.0]
    assert ledger["fine"]["rows_forward"] == 19.0
    assert ledger["coarse"]["rows"] == 15.0
    assert ledger["source"]["rows_per_group"] == [2, 2, 2]
    assert ledger["support"]["unique_pairs"] == 11.0
    assert ledger["executor"] == {
        "selected_batched": [1.0, 1.0, 1.0, 1.0, 1.0],
        "batch_limit": 128.0,
        "scalar_block_calls": 12.0,
        "batched_block_calls": 1.0,
        "block_call_reduction": 11.0,
        "scalar_gemm_launches": 12.0,
        "batched_gemm_launches": 1.0,
        "gemm_launch_reduction": 11.0,
        "batched_checkpoint_calls": 0.0,
        "batched_group_rows_forward": [12.0, 12.0, 12.0],
        "batched_rows": 36.0,
        "batched_rows_padded": 20.0,
        "batched_rows_recompute": 0.0,
    }
    assert ledger["maps_probe_timed"] is False


def test_measure_cpu_contract_has_cuda_and_absolute_memory_fields() -> None:
    result = _measure(
        torch,
        torch.device("cpu"),
        lambda: torch.ones(2, 3),
        receiver_chunk_size=2048,
        warmups=1,
        repetitions=2,
        name="full_physical_forward",
    )
    assert result["status"] == "complete"
    assert result["timed_maps"] is False
    assert result["peak_total_allocated_bytes"] is None
    assert result["peak_incremental_allocated_bytes"] is None
    assert all(sample["peak_allocated_bytes"] is None for sample in result["samples"])
