"""Deterministic correction-disabled generation and early resource errors."""

from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch

from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner
from channelthermal.inverse.evaluation.candidate_evaluator import ThermalChannelCandidateEvaluator
from channelthermal.inverse.verifier import FrozenThermalChannelVerifier
from tests.inverse_test_utils import request_batch


def _designer() -> HierarchicalInverseDesigner:
    return HierarchicalInverseDesigner.from_config(
        {
            "num_edges": 3, "max_modules": 5, "request_hidden_dim": 16,
            "plan_hidden_dim": 24, "plan_layers": 1, "plan_sampling_steps": 2,
            "layout_hidden_dim": 24, "layout_layers": 1, "layout_sampling_steps": 2,
            "corrector_enabled": False, "dropout": 0.0,
        }
    ).eval()


def test_same_seed_is_deterministic_and_correction_can_be_disabled() -> None:
    designer = _designer()
    request = request_batch(1)
    context = torch.zeros(1, 10)
    geometry = torch.zeros(1, 8)
    geometry[:, 0] = 1.0 / 5.0
    geometry[:, 1] = 4.0 / 5.0
    first = designer.sample_candidates(
        request=request, context=context, geometry_constraints=geometry,
        num_plans=2, layouts_per_plan=2, correct_once=False, seed=99,
    )
    second = designer.sample_candidates(
        request=request, context=context, geometry_constraints=geometry,
        num_plans=2, layouts_per_plan=2, correct_once=False, seed=99,
    )
    torch.testing.assert_close(first.plans, second.plans)
    torch.testing.assert_close(first.layouts, second.layouts)
    assert first.corrected is None
    json.dumps(first.to_dict())


def test_missing_forward_checkpoint_fails_before_model_loading(tmp_path: Path) -> None:
    missing = tmp_path / "best_predicted_model.pt"
    with pytest.raises(FileNotFoundError, match="Frozen forward checkpoint not found"):
        FrozenThermalChannelVerifier(missing)


def test_public_verify_dispatches_to_object_adapter() -> None:
    class Adapter:
        def verify(self, value: int, *, scale: int) -> int:
            return value * scale

    designer = _designer().attach_verifier(Adapter())
    assert designer.verify(4, scale=3) == 12


def test_public_verify_rejects_invalid_adapter() -> None:
    designer = _designer().attach_verifier(object())
    with pytest.raises(TypeError, match="callable or expose verify"):
        designer.verify()


def test_tensor_mode_correction_requires_case_runtime_and_exact_verification() -> None:
    designer = HierarchicalInverseDesigner.from_config(
        {
            "num_edges": 3, "max_modules": 5, "request_hidden_dim": 16,
            "plan_hidden_dim": 24, "plan_layers": 1, "plan_sampling_steps": 2,
            "layout_hidden_dim": 24, "layout_layers": 1, "layout_sampling_steps": 2,
            "corrector_enabled": True, "corrector_hidden_dim": 24,
            "corrector_blocks": 1, "dropout": 0.0,
        }
    ).eval()
    geometry = torch.zeros(1, 8)
    geometry[:, 1] = 1.0
    with pytest.raises(RuntimeError, match="before exact verification"):
        designer.sample_candidates(
            request=request_batch(1), context=torch.zeros(1, 10),
            geometry_constraints=geometry, correct_once=True,
        )


def test_case_runtime_rejects_invalid_sampling_counts_before_forward_calls() -> None:
    evaluator = object.__new__(ThermalChannelCandidateEvaluator)
    with pytest.raises(ValueError, match="num_plans must be positive"):
        evaluator.sample_candidates(
            request=None, context=None, num_plans=0, layouts_per_plan=4, top_k=1
        )
    with pytest.raises(ValueError, match="layouts_per_plan must be positive"):
        evaluator.sample_candidates(
            request=None, context=None, num_plans=2, layouts_per_plan=0, top_k=1
        )
    with pytest.raises(ValueError, match="top_k must be positive"):
        evaluator.sample_candidates(
            request=None, context=None, num_plans=2, layouts_per_plan=2, top_k=5
        )
