"""Real-checkpoint integration checks for the frozen inverse verifier.

Physical design ``D`` and context ``c`` come from one mapped case. Request
``R`` is represented by the exact functional evaluation. Compact plan ``G`` is
not sampled here; the test checks deterministic realized plan ``G_hat`` from
autonomous predicted ports and the final organizer.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from channelthermal.inverse.verifier import FrozenThermalChannelVerifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORWARD_CHECKPOINT = (
    PROJECT_ROOT
    / "Trained_Results/ThermalChannel/Inverse_Resources"
    / "best_predicted_model_snapshot.pt"
)


@pytest.mark.skipif(not FORWARD_CHECKPOINT.is_file(), reason="real self-contained forward checkpoint unavailable")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_frozen_verifier_is_finite_autonomous_and_deterministic() -> None:
    """Replay one case twice without teacher ports on visible physical GPU 1."""

    expected_physical_gpu = os.environ.get("HONF_TEST_PHYSICAL_GPU")
    if expected_physical_gpu is not None:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == expected_physical_gpu
    verifier = FrozenThermalChannelVerifier(FORWARD_CHECKPOINT, device="cuda:0", query_batch_size=32768)
    first = verifier.verify_case(case_index=0, return_outputs=("predicted_ports", "environment"))
    second = verifier.verify_design(
        first.design,
        first.context,
        return_outputs=("predicted_ports",),
    )

    assert not verifier.model.training
    assert not any(parameter.requires_grad for parameter in verifier.model.parameters())
    assert first.checkpoint_provenance["local_port_condition_mode"] == "predicted"
    assert first.checkpoint_provenance["organizer_source"] == "final_after_local_response_fusion"
    assert first.compact_plan.raw.shape == second.compact_plan.raw.shape
    np.testing.assert_allclose(first.compact_plan.raw, second.compact_plan.raw, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(first.outputs["predicted_ports"], second.outputs["predicted_ports"], rtol=0.0, atol=0.0)
    assert np.isfinite(first.outputs["predicted_ports"]).all()
    assert np.isfinite(first.outputs["environment"]["pred_field_grid"]).all()
    assert all(value.valid and np.isfinite(value.value) for value in first.functionals.values())
