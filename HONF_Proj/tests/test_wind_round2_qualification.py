"""Focused Wind guards for the Round2 same-panel native field gate."""

import pytest

from honf_runtime.content_hypergraph_qualification import (
    FIRST_DIRECT,
    WIND_DEV_PANEL_SHA256,
    WIND_ROLE_QUERY_COUNTS,
    field_decision,
    native_record,
)


ROLES = tuple(WIND_ROLE_QUERY_COUNTS)
PRIORITY = ("near_turbine", "downstream_envelope")
AXES = ("Ux", "Uy", "Uz")


def wind_record(architecture, *, component=0.85, vector=0.9, score=0.08,
                checkpoint=None, panel=WIND_DEV_PANEL_SHA256):
    rows = []
    for index in range(24):
        rows.append({
            "row_index": index,
            "roles": {
                role: {
                    "component_rmse_mps": [component, component, component],
                    "vector_rmse_mps": vector if role in PRIORITY else 0.8,
                    "query_count": WIND_ROLE_QUERY_COUNTS[role],
                }
                for role in ROLES
            },
        })
    return {
        "architecture": architecture,
        "epoch": 2500,
        "checkpoint_sha256": checkpoint or f"checkpoint-{architecture}",
        "field_score": score,
        "panel_sha256": panel,
        "rows": rows,
    }


def wind_decision(candidate=None, direct=None, baseline=None):
    candidate = candidate or wind_record("A2-H6", component=0.85, vector=0.9, score=0.08)
    direct = direct or wind_record("A2-D", component=0.9, vector=1.0, score=0.09)
    baseline = baseline or wind_record("A-D", component=1.0, vector=1.0, score=0.1,
                                      checkpoint=FIRST_DIRECT["wind"])
    floors = {"field_score": 1e-8}
    floors.update({role: 1e-8 for role in ROLES})
    floors.update({f"p90/{role}": 1e-8 for role in PRIORITY})
    floors.update({f"{role}/{axis}": 1e-8 for role in PRIORITY for axis in AXES})
    return field_decision("wind", candidate, direct, baseline, floors=floors)


def test_wind_field_gate_uses_six_component_ratios_and_both_priority_p90s():
    result = wind_decision()
    assert result["qualified"]
    ratios = result["ratios"]
    assert all(f"{role}/{axis}" in ratios for role in PRIORITY for axis in AXES)
    assert ratios["six_component_geometric_mean"] == pytest.approx(0.85)
    assert result["checks"]["near_turbine_vs_matched"]
    assert result["checks"]["downstream_envelope_vs_matched"]
    assert result["checks"]["p90/near_turbine"]
    assert result["checks"]["p90/downstream_envelope"]


def test_wind_priority_mean_and_p90_gates_are_independent():
    candidate = wind_record("A2-H6", component=0.85, vector=1.16, score=0.08)
    result = wind_decision(candidate=candidate)
    assert not result["checks"]["near_turbine_vs_matched"]
    assert result["checks"]["p90/near_turbine"]

    candidate = wind_record("A2-H6", component=0.85, vector=0.9, score=0.08)
    for row in candidate["rows"][:6]:
        row["roles"]["near_turbine"]["vector_rmse_mps"] = 1.8
    for row in candidate["rows"][6:]:
        row["roles"]["near_turbine"]["vector_rmse_mps"] = 0.8
    result = wind_decision(candidate=candidate)
    assert result["checks"]["near_turbine_vs_matched"]
    assert result["checks"]["p90/near_turbine"] is False


@pytest.mark.parametrize("arm", ["candidate", "matched", "baseline"])
@pytest.mark.parametrize("panel", [None, "0" * 64, "not-a-digest"])
def test_wind_requires_the_sealed_panel_for_each_coherent_arm(arm, panel):
    candidate = wind_record("A2-H6")
    direct = wind_record("A2-D", component=0.9, vector=1.0, score=0.09)
    baseline = wind_record("A-D", component=1.0, vector=1.0, score=0.1,
                          checkpoint=FIRST_DIRECT["wind"])
    {"candidate": candidate, "matched": direct, "baseline": baseline}[arm]["panel_sha256"] = panel
    with pytest.raises(ValueError, match="sealed same-panel"):
        wind_decision(candidate=candidate, direct=direct, baseline=baseline)


@pytest.mark.parametrize("arm", ["candidate", "matched", "baseline"])
def test_wind_requires_exact_per_role_quotas_even_when_total_stays_q8192(arm):
    candidate = wind_record("A2-H6")
    direct = wind_record("A2-D", component=0.9, vector=1.0, score=0.09)
    baseline = wind_record("A-D", component=1.0, vector=1.0, score=0.1,
                          checkpoint=FIRST_DIRECT["wind"])
    record = {"candidate": candidate, "matched": direct, "baseline": baseline}[arm]
    row = record["rows"][0]
    row["roles"]["volume"]["query_count"] += 1
    row["roles"]["background"]["query_count"] -= 1
    assert sum(row["roles"][role]["query_count"] for role in ROLES) == 8192
    with pytest.raises(ValueError, match="exact per-role native query quotas"):
        wind_decision(candidate=candidate, direct=direct, baseline=baseline)


@pytest.mark.parametrize("values", [[0.85, 0.85], [0.85] * 4, [0.85, float("nan"), 0.85],
                                      [0.85, float("inf"), 0.85], [0.85, -0.01, 0.85]])
def test_wind_component_vectors_must_be_three_finite_nonnegative_values(values):
    candidate = wind_record("A2-H6")
    candidate["rows"][0]["roles"]["near_turbine"]["component_rmse_mps"] = values
    with pytest.raises(ValueError, match="component RMSE vectors"):
        wind_decision(candidate=candidate)


def test_wind_keeps_one_coherent_first_round_direct_checkpoint_not_channel_oracle():
    oracle = wind_record("A-D", checkpoint="assembled-channel-oracle")
    with pytest.raises(ValueError, match="coherent"):
        wind_decision(baseline=oracle)


def test_wind_native_board_extraction_is_bound_to_one_explicit_model_label():
    board = {
        "endpoint_models": {
            "candidate": {
                "checkpoint": {"sha256": "candidate-sha", "completed_epoch": 2500,
                               "architecture": "A2-H6"},
                "row_metrics": [{"row_index": 0}],
                "field_score_mean": 0.08,
                "endpoint_panel_sha256": WIND_DEV_PANEL_SHA256,
            }
        }
    }
    record = native_record(board, "wind", label="candidate")
    assert record["panel_sha256"] == WIND_DEV_PANEL_SHA256
    assert record["rows"] == [{"row_index": 0}]
    with pytest.raises(KeyError):
        native_record(board, "wind", label="other")
