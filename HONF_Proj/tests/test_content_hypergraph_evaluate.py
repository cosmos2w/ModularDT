from __future__ import annotations

import pytest

from tools.content_hypergraph_evaluate import label_monitor_metrics


def test_monitor_labels_use_observed_role_queries_without_mutating_native_metrics():
    original = {
        "field_score": 0.25,
        "query_count_per_row": 8192,
        "endpoint_rows": [4, 8],
        "endpoint_selection": "all fixed development rows",
        "rows": [
            {"roles": {"volume": {"query_count": 820}, "other": {"query_count": 3276}}},
            {"roles": {"volume": {"query_count": 820}, "other": {"query_count": 3276}}},
        ],
    }
    result = label_monitor_metrics(original)
    assert result["query_count_per_row"] == 4096
    assert result["monitor_rows"] == [4, 8]
    assert result["field_score"] == original["field_score"]
    assert "endpoint_selection" not in result and "endpoint_rows" not in result
    assert original["query_count_per_row"] == 8192


def test_monitor_rejects_inconsistent_observed_query_counts():
    with pytest.raises(ValueError, match="one observed query count"):
        label_monitor_metrics({
            "query_count_per_row": 8192,
            "rows": [{"roles": {"volume": {"query_count": 4096}}},
                     {"roles": {"volume": {"query_count": 8192}}}],
        })
