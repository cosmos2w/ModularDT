"""CPU contracts for the reversible Run-1503 branch interventions."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools" / "diagnostics"))

from run_run1503_interventions import (
    EXPECTED_ARCHITECTURE,
    adaptive_branch_intervention,
    build_parser,
)


class _Backend:
    def __init__(self) -> None:
        self.calls = 0

    def _read_environment(self, *_args, **_kwargs):
        self.calls += 1
        context = torch.full((1, 2, 3), 10.0)
        return context, {
            "group_control_adaptive_coarse_contribution": torch.full((1, 2, 3), 2.0),
            "group_control_adaptive_fine_contribution": torch.full((1, 2, 3), 3.0),
        }


def _model() -> SimpleNamespace:
    backend = _Backend()
    return SimpleNamespace(
        config=SimpleNamespace(
            core_honf=SimpleNamespace(forward_architecture=EXPECTED_ARCHITECTURE)
        ),
        core=SimpleNamespace(backend=backend, _interface_read_role="p2_field"),
    )


def test_coarse_and_fine_suppression_subtract_only_requested_branch() -> None:
    model = _model()
    original = model.core.backend._read_environment
    with adaptive_branch_intervention(model, "coarse_suppressed"):
        coarse, _ = model.core.backend._read_environment(None)
    assert torch.all(coarse == 8.0)
    assert model.core.backend._read_environment == original
    with adaptive_branch_intervention(model, "fine_suppressed"):
        fine, _ = model.core.backend._read_environment(None)
    assert torch.all(fine == 7.0)
    assert model.core.backend._read_environment == original


def test_intervention_is_scoped_to_p2_role_and_restores_on_missing_branch() -> None:
    model = _model()
    model.core._interface_read_role = "p1_refinement"
    with adaptive_branch_intervention(model, "coarse_suppressed"):
        unchanged, _ = model.core.backend._read_environment(None)
    assert torch.all(unchanged == 10.0)

    model.core._interface_read_role = "p2_field"
    original = model.core.backend._read_environment

    def missing(*_args, **_kwargs):
        return torch.ones(1, 2, 3), {}

    model.core.backend._read_environment = missing
    with pytest.raises(RuntimeError, match="required"), adaptive_branch_intervention(
        model, "fine_suppressed"
    ):
        model.core.backend._read_environment(None)
    assert model.core.backend._read_environment == missing
    model.core.backend._read_environment = original


def test_parser_keeps_interventions_opt_in_and_exposes_gate_epoch(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            f"e500={tmp_path / 'epoch_0500_model.pt'}",
            "--output",
            str(tmp_path / "interventions.json"),
            "--expected-epoch",
            "500",
            "--mode",
            "fine_suppressed",
        ]
    )
    assert args.mode == ["fine_suppressed"]
    assert args.expected_epoch == 500
    assert args.case_id == []
