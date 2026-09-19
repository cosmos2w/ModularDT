"""CPU/static contracts for the Run-1406 executor evidence runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools" / "diagnostics"))

from run_run1406_executor_evidence import (
    DEFAULT_CASE_IDS,
    _compare_evidence,
    _is_ledger_key,
    _phase_request,
    build_parser,
    build_plan,
)


def _args(tmp_path: Path, *extra: str):
    checkpoint = tmp_path / "epoch_0050_model.pt"
    checkpoint.write_bytes(b"placeholder")
    return build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "evidence.json"),
            *extra,
        ]
    )


def _raw(label: str, *, scale: float = 1.0) -> dict[str, object]:
    value = torch.tensor([[1.0, 2.0]], dtype=torch.float32) * scale
    physical = {key: value.clone() for key in (
        "pred_field",
        "pred_internal_temperature",
        "pred_interface",
        "pred_port_condition",
        "pred_port_condition_raw",
        "local_port_condition_used",
        "pred_port_global_temperature",
        "pred_port_global_temperature_target",
        "pred_port_global_consistency_mask",
    )}
    return {
        "label": label,
        "physical": {
            case_id: {
                "physical": physical,
                "provisional": {"pred_field": value.clone()},
            }
            for case_id in DEFAULT_CASE_IDS
        },
        "phase_tensors": {
            case_id: {"initial_port_physical_context": value.clone()}
            for case_id in DEFAULT_CASE_IDS
        },
        "phase_summaries": {case_id: {} for case_id in DEFAULT_CASE_IDS},
        "backward": {
            "status": "pass",
            "batch_size": 48,
            "query_count": 1024,
            "module_count": 12,
            "port_condition_mode": "predicted",
            "optimizer_update": False,
            "loss_terms": {"loss": 3.0 * scale},
            "gradient_count": 1,
            "none_gradient_count": 0,
            "none_gradients": [],
            "gradients": {"core.backend.weight": value.clone()},
        },
    }


def test_plan_records_strict_trusted_loading_and_timed_ledger_policy(tmp_path: Path) -> None:
    plan = build_plan(_args(tmp_path))

    assert plan["status"] == "plan_only"
    assert plan["checkpoint_policy"]["strict_load"] is True
    assert "load_trusted_checkpoint" in plan["checkpoint_policy"]["trusted_loader"]
    assert plan["protocol"]["timed_maps"] is False
    assert plan["protocol"]["timed_detailed_ledgers"] is False
    assert plan["protocol"]["timed_profiler"] is False
    assert plan["protocol"]["case_ids"] == list(DEFAULT_CASE_IDS)
    assert plan["protocol"]["backward"]["port_condition_mode"] == "predicted"


def test_plan_rejects_noncontrolled_protocol(tmp_path: Path) -> None:
    args = _args(tmp_path, "--query-count", "1024")
    with pytest.raises(ValueError, match="query_count=8192"):
        build_plan(args)


def test_timed_request_disables_maps_and_ledger_surface() -> None:
    batch = {
        "structure": {"module_centers": torch.zeros(1, 1, 2)},
        "query_xy": torch.zeros(1, 2, 2),
    }
    request = _phase_request(batch, batch["query_xy"], consistency=False)

    assert request["local_port_condition_mode"] == "predicted"
    assert request["return_routing_maps"] is False
    assert request["return_port_global_consistency"] is False
    assert request["return_organizer_passes"] is False


def test_ledger_markers_are_not_treated_as_physical_parity_tensors() -> None:
    assert _is_ledger_key("group_control_module_logical_paths")
    assert _is_ledger_key("group_control_environment_fine_rows_forward")
    assert _is_ledger_key("group_control_module_valid_pair_denominator")
    assert not _is_ledger_key("initial_port_physical_context")


def test_same_checkpoint_output_and_gradient_comparison_is_strict() -> None:
    reference = _raw("reference")
    candidate = _raw("optimized")
    result = _compare_evidence(reference, candidate)

    assert result["status"] == "pass"
    assert result["backward"]["status"] == "pass"
    assert result["contract"]["detailed_ledgers_in_timed_reads"] is False

    candidate = _raw("optimized", scale=1.01)
    failed = _compare_evidence(reference, candidate)
    assert failed["status"] == "fail"
    assert failed["cases"]["0273"]["physical_outputs"]["status"] == "fail"


def test_absent_optional_provisional_maps_do_not_fail_physical_parity() -> None:
    reference = _raw("reference")
    candidate = _raw("optimized")
    for payload in (reference, candidate):
        for case_id in DEFAULT_CASE_IDS:
            payload["physical"][case_id]["provisional"] = {}

    result = _compare_evidence(reference, candidate)

    assert result["status"] == "pass"
    assert result["cases"]["0273"]["provisional_outputs"]["status"] == "pass"
