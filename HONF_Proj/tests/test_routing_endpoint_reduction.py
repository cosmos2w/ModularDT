"""Prevent incomplete populations and wrong-budget checkpoints in the report."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "diagnostics"))
import reduce_routing_endpoint as reduction


def rows():
    metric_keys = [reduction._raw_metric_key(base) for base in
                   reduction.CORE_BASES + reduction.CHANNEL_BASES + reduction.ENGINEERING_KPI_BASES]
    return [
        {"case_id": f"{index:04d}", **{key: "1.0" for key in metric_keys},
         **{key: "fixture" for key in reduction.STRATA},
         "field_u_fluid_norm_target_sse": "10", "field_u_fluid_norm_num_values": "100"}
        for index in range(90)
    ]


@pytest.mark.parametrize("defect", ["duplicate", "missing_metric", "missing_count", "blank_target", "changed_stratum"])
def test_endpoint_rejects_incomplete_or_unmatched_cases(defect):
    reference = rows()
    candidate = copy.deepcopy(reference)
    reduction.validate_population(candidate, reference)
    if defect == "duplicate":
        candidate[1]["case_id"] = candidate[0]["case_id"]
    elif defect == "missing_metric":
        del candidate[0]["global_field_fluid_norm_l2"]
    elif defect == "missing_count":
        del candidate[0]["field_u_fluid_norm_num_values"]
    elif defect == "blank_target":
        candidate[0]["field_u_fluid_norm_target_sse"] = ""
    else:
        candidate[0][reduction.STRATA[0]] = "other"
    with pytest.raises(ValueError):
        reduction.validate_population(candidate, reference)


@pytest.mark.parametrize("policy,epoch,filename", [
    ("exact500", 100, "epoch_0100_model.pt"),
    ("saved_best", 501, "best_by_field_mse_model.pt"),
    ("saved_best", 450, "best_by_temperature_mse_model.pt"),
])
def test_endpoint_rejects_wrong_budget_or_selection_policy(tmp_path, monkeypatch, policy, epoch, filename):
    candidate = rows()
    for row in candidate:
        row.update(model_label=f"{policy}:Run2000", checkpoint=f"/fixture/Run_2000_fixture/{filename}")
    if policy != "exact500":
        candidate += [dict(row, model_label="exact500:Run2000", checkpoint="/fixture/Run_2000_fixture/epoch_0500_model.pt")
                      for row in rows()]
    source = tmp_path / "metrics.csv"
    reduction.write_csv(source, candidate)
    monkeypatch.setattr(reduction, "_load_trusted_checkpoint", lambda path: {"epoch": 500 if path.name == "epoch_0500_model.pt" else epoch})
    with pytest.raises(ValueError):
        reduction.reduce_endpoint(source, tmp_path / "output")
