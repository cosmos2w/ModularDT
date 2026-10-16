from __future__ import annotations

import numpy as np
import pytest
from windfarm.training.native_role_evaluation import physical_role_metrics, summarize_physical_role_metrics


def test_physical_roles_preserve_vector_norm_tails_and_qualified_reference():
    from source_read_wind_evaluate import (
        physical_role_metrics as reference_metrics,
    )
    from source_read_wind_evaluate import (
        summarize_physical_role_metrics as reference_summary,
    )

    roles = np.repeat(np.arange(5), [1640, 1640, 1640, 1640, 1632])
    target = np.zeros((8192, 3), dtype=np.float32)
    prediction = np.broadcast_to(np.asarray([1.0, 2.0, 2.0], dtype=np.float32), target.shape)
    records = []
    for row, multiplier in enumerate((1.0, 2.0, 4.0)):
        metadata = {"row_index": row, "layout_index": row // 3, "n_turbines": 8}
        result = physical_role_metrics(metadata, prediction * multiplier, target, roles)
        assert result == reference_metrics(metadata, prediction * multiplier, target, roles)
        assert result["roles"]["near_turbine"]["vector_rmse_mps"] == 3.0 * multiplier
        assert result["roles"]["background"]["query_count"] == 1632
        records.append(result)
    summary = summarize_physical_role_metrics(records)
    assert summary == reference_summary(records)
    assert len(summary["by_layout_index"]) == 1
    assert summary["overall"]["roles"]["near_turbine"]["vector_rmse_mps"]["mean"] == 7.0
    assert summary["overall"]["roles"]["near_turbine"]["vector_rmse_mps"]["p90"] == 10.8


def test_physical_role_metrics_reject_missing_roles_and_nonfinite_values():
    target = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="no queries"):
        physical_role_metrics({}, target, target, np.zeros(10, dtype=np.int64))
    prediction = target.copy()
    prediction[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        physical_role_metrics({}, prediction, target, np.repeat(np.arange(5), 2))
