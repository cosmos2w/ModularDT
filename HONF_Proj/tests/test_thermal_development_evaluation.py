"""Quarter-cohort CLI/loader wiring without native checkpoints or inference."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location("thermal_campaign_benchmark", TOOLS / "thermal_campaign_benchmark.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)

import thermal_campaign_evaluate as evaluation
import thermal_campaign_responses as responses
import thermal_development as development


def manifest():
    return {"manifest_sha256": "a" * 64, "partitions": {
        "train": {"case_ids": ["0001", "0304"]},
        "test": {"case_ids": [f"{600+i:04}" for i in range(22)]}}}


@pytest.mark.parametrize("stage", [100, 500, 1000])
def test_all22_development_metrics_but_only_panel_graphs_and_arrays(stage):
    args = SimpleNamespace(stage=stage, interventions=["normal"], panel_config=Path("historical.json"),
        panel_size=4, capture_phase_graphs=True, fixed_summary_train_cases=600)
    evaluation.configure_evaluation_scope(args, manifest())
    assert evaluation.evaluation_indices(list(range(22)), args) == list(range(22))
    assert args.panel_config is None and args.fixed_summary_train_cases == 10
    assert sum(evaluation.save_case_arrays(args, i, {1, 4, 8, 20}) for i in range(22)) == 4
    assert sum(evaluation.capture_case_phases(args, i, {1, 4, 8, 20}) for i in range(22)) == 4


def test_explicit_panel_quick_diagnostic_and_historical_archives_preserved(monkeypatch):
    panel = Path("explicit.json")
    args = SimpleNamespace(stage=500, interventions=["normal"], panel_config=panel, panel_config_explicit=True,
        fixed_summary_train_cases=3, fixed_summary_train_cases_explicit=True, panel_size=4,
        quick_diagnostic=True, capture_phase_graphs=True)
    evaluation.configure_evaluation_scope(args, manifest())
    monkeypatch.setattr(evaluation, "fixed_screen_indices", lambda dataset, path, count: [1, 2, 3, 4])
    assert evaluation.evaluation_indices(list(range(22)), args) == [1, 2, 3, 4]
    assert args.panel_config is panel and args.fixed_summary_train_cases == 3
    evaluation.configure_evaluation_scope(args, None)
    assert evaluation.save_case_arrays(args, 10, {1})
    assert evaluation.capture_case_phases(args, 10, {1})


@pytest.mark.parametrize("scope", ["auto", "formal-full"])
def test_native_loader_shares_saved_normalizer_and_exact_raw_membership(monkeypatch, scope):
    from channelthermal.data import datasets
    from channelthermal.evaluation import loading

    from honf_runtime import compat

    selected = manifest() if scope == "auto" else None
    checkpoint = {"train_config": {"dataset": {"packed_h5_path": "unused.h5", "normalize_inputs": True,
        "normalize_targets": True}}, "global_normalization_stats": {"steady_field_mean": [1., 2.]}}
    calls = []
    dummy = SimpleNamespace(eval=lambda: None, requires_grad_=lambda value: None)
    monkeypatch.setattr(loading, "load_model", lambda *args: (dummy, checkpoint))
    monkeypatch.setattr(compat, "resolve_demo_path", lambda path: Path(path))
    monkeypatch.setattr(development, "resolve_evaluation_manifest", lambda *args, **kwargs: selected)

    def dataset(path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(selected_case_ids=kwargs.get("case_ids", ("all",)))

    monkeypatch.setattr(datasets, "GlobalChannelThermalDataset", dataset)
    result = benchmark.load_native("sentinel.pt", None, torch.device("cpu"), dataset_scope=scope)
    assert result[0] is dummy and len(calls) == 2
    assert calls[0]["normalizer"] is calls[1]["normalizer"]
    np.testing.assert_array_equal(calls[0]["normalizer"].stats["steady_field_mean"], [1., 2.])
    assert calls[0]["normalize_targets"] and "normalize_targets" not in calls[1]
    if selected is not None:
        assert calls[0]["case_ids"] == calls[1]["case_ids"] == tuple(selected["partitions"]["test"]["case_ids"])
    else:
        assert "case_ids" not in calls[0] and "case_ids" not in calls[1]


def test_development_missing_saved_stats_fails_before_dataset_construction(monkeypatch):
    from channelthermal.data import datasets
    from channelthermal.evaluation import loading

    from honf_runtime import compat

    dummy = SimpleNamespace(eval=lambda: None, requires_grad_=lambda value: None)
    monkeypatch.setattr(loading, "load_model", lambda *args: (dummy, {"train_config": {"dataset": {"packed_h5_path": "unused"}}}))
    monkeypatch.setattr(compat, "resolve_demo_path", Path)
    monkeypatch.setattr(development, "resolve_evaluation_manifest", lambda *args, **kwargs: manifest())
    monkeypatch.setattr(datasets, "GlobalChannelThermalDataset", lambda *args, **kwargs: pytest.fail("Dataset constructed"))
    with pytest.raises(ValueError, match="saved selected-training"):
        benchmark.load_native("sentinel.pt", None, torch.device("cpu"))


def test_response_defaults_use_only_checkpoint_anchors_and_validate_actual_family_membership(tmp_path):
    checkpoint = {"train_config": {"training": {"campaign": {"response_stencils": ["/data/train_0304_responses.npz"]}}}}
    assert responses.response_paths(checkpoint, manifest(), None, tmp_path / "unused") == [Path("/data/train_0304_responses.npz")]
    with pytest.raises(ValueError, match="no silent full-atlas"):
        responses.response_paths({}, manifest(), None, tmp_path / "unused")
    with pytest.raises(ValueError, match="at most four"):
        responses.response_paths({}, manifest(), [f"/data/{i}.npz" for i in range(5)], None)
    stencil = SimpleNamespace(split=SimpleNamespace(value="train"))
    responses.validate_response_membership(stencil, {"anchor_id": "0304"}, manifest())
    for metadata in ({"anchor_id": "0318"}, {}):
        with pytest.raises(ValueError, match="outside selected"):
            responses.validate_response_membership(stencil, metadata, manifest())
    with pytest.raises(ValueError, match="outside selected"):
        responses.validate_response_membership(SimpleNamespace(split=SimpleNamespace(value="calibration")),
            {"anchor_id": "0304"}, manifest())
    responses.validate_response_membership(stencil, {"anchor_id": "0318"}, None)


def test_cli_exposes_explicit_full_override_without_reading_panels():
    args = responses.parse_args(["--checkpoint", "sentinel", "--dataset", "unused", "--output-dir", "unused",
        "--evaluation-scope", "formal-full"])
    assert args.evaluation_scope == "formal-full" and args.stencils is None
