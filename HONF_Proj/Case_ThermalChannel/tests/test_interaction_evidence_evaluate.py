"""Scientific metric contracts for response and finite-pool decisions."""

import numpy as np
import pytest
from channelthermal.interaction_evidence.evaluate import (
    compare_candidate_ranking,
    compare_response_block,
)


def test_unresolved_response_and_excluded_receiver_do_not_claim_sign_evidence():
    result = compare_response_block(
        np.array([[0.01], [0.02], [100.0]]),
        np.array([[-0.01], [0.04], [-100.0]]),
        output_role="final_port_temperature",
        units="temperature",
        observed_mask=np.array([True, True, False]),
        noise_floor=0.03,
        weights=np.array([1.0, 3.0, 100.0]),
    )
    assert result.observed_count == 2
    assert result.resolved_sign_count == 0
    assert result.resolved_sign_accuracy is None
    assert result.weighted_mae == pytest.approx(0.02)


def test_finite_pool_regret_is_not_global_optimum_regret():
    result = compare_candidate_ranking(
        np.array([4.0, 2.0, 3.0]),
        np.array([1.0, 3.0, 2.0]),
        baseline_reference_objective=2.5,
        baseline_predicted_objective=3.5,
        improvement_floor=0.1,
    )
    assert result.selected_index == 1
    assert result.reference_best_index == 0
    assert result.finite_pool_regret == pytest.approx(2.0)
    assert result.resolved_improvement_count == 3
    assert result.improvement_sign_accuracy == pytest.approx(1.0 / 3.0)
