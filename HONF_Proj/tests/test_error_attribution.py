from __future__ import annotations

import numpy as np
import pytest

from honf_forward_core.evaluation import weighted_squared_error_attribution


def test_weighted_squared_error_attribution_closes_with_shared_mask_and_query_weights() -> None:
    baseline_error = np.array([[1.0, -2.0], [3.0, 4.0], [np.nan, np.nan]])
    adaptation_delta = np.array([[-0.5, 1.0], [1.0, -2.0], [np.nan, np.nan]])
    access_delta = np.array([[0.25, -0.5], [-2.0, 1.0], [np.nan, np.nan]])
    mask = np.array([True, True, False])
    weights = np.array([1.0, 3.0, 5.0])

    result = weighted_squared_error_attribution(
        baseline_error,
        adaptation_delta,
        access_delta,
        weights=weights,
        mask=mask,
    )

    normalized = np.array([0.25, 0.75])[:, None]
    e_base = baseline_error[:2]
    e_adapt = e_base + adaptation_delta[:2]
    e_access = e_adapt + access_delta[:2]
    assert result.selected_entry_count == 4
    assert result.normalized_weight_sum == pytest.approx(1.0)
    assert result.baseline_squared_error == pytest.approx(float(np.sum(normalized * e_base**2) / 2))
    assert result.adapted_full_squared_error == pytest.approx(float(np.sum(normalized * e_adapt**2) / 2))
    assert result.accessed_squared_error == pytest.approx(float(np.sum(normalized * e_access**2) / 2))
    assert result.adaptation_delta == pytest.approx(
        result.adaptation_cross_term + result.adaptation_squared_term
    )
    assert result.access_delta == pytest.approx(result.access_cross_term + result.access_squared_term)
    assert result.adaptation_closure_error == pytest.approx(0.0, abs=1e-14)
    assert result.access_closure_error == pytest.approx(0.0, abs=1e-14)
    assert result.total_closure_error == pytest.approx(0.0, abs=1e-14)


def test_negative_cross_terms_are_reported_separately_from_squared_perturbation() -> None:
    result = weighted_squared_error_attribution(
        np.array([1.0, -2.0]),
        np.array([-1.0, 2.0]),
        np.zeros(2),
    )

    assert result.adaptation_cross_term == pytest.approx(-5.0)
    assert result.adaptation_squared_term == pytest.approx(2.5)
    assert result.adaptation_delta == pytest.approx(-2.5)
    assert result.adapted_full_squared_error == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("arrays", "kwargs", "message"),
    [
        (([1.0], [1.0, 2.0], [0.0]), {}, "aligned"),
        (([1.0], [np.nan], [0.0]), {}, "finite"),
        (([1.0], [0.0], [0.0]), {"weights": np.array([-1.0])}, "nonnegative"),
        (([1.0], [0.0], [0.0]), {"mask": np.array([False])}, "no entries"),
    ],
)
def test_invalid_or_empty_attribution_inputs_are_rejected(
    arrays: tuple[object, object, object], kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        weighted_squared_error_attribution(*arrays, **kwargs)
