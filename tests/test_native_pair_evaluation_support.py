"""The reported change must use the same physical support in both scenes."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("missing", ["material", "pressure", "region"])
def test_pair_evaluation_rejects_missing_prediction_support(tmp_path, monkeypatch, missing):
    tool = Path(__file__).resolve().parents[1] / "tools/thermal_native_pair_evaluate.py"
    spec = importlib.util.spec_from_file_location("thermal_native_pair_evaluate", tool)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    path = tmp_path / "native.h5"
    config = {"domain": {"lx": 2, "ly": 1, "nx": 2, "ny": 1},
              "layout": {"num_modules": 1}, "flow": {"re": 1}}
    with h5py.File(path, "w") as h5:
        for cid in ("001", "002"):
            h5.create_group(f"cases/{cid}").create_dataset("case_config_json", data=json.dumps(config))
    masks = {"A": np.array([[True, False]]), "B": np.array([[True, False]])}
    monkeypatch.setattr(evaluator, "native_pair_support", lambda *args: masks)
    monkeypatch.setattr(evaluator, "native_pair_references", lambda *args: torch.zeros((1, 2, 4)))
    monkeypatch.setattr(evaluator, "_endpoint_functionals", lambda *args: torch.zeros((1, 2)))
    monkeypatch.setattr(evaluator, "describe_native_scene", lambda *args: SimpleNamespace(
        descriptor=np.zeros((1, 3)), compatibility_key='same prescribed controls'))
    monkeypatch.setattr(evaluator, "native_input_distance", lambda *args: 1.)

    def fake_panel(model, provider, *, case_observer, **kwargs):
        for cid in ("001", "002"):
            bad = cid == "001"
            valid = (torch.tensor([[False, True]]) if bad and missing in ("pressure", "region")
                     else torch.ones((1, 2), dtype=torch.bool))
            material = torch.tensor([[[not (bad and missing == "material")]]])
            prediction = SimpleNamespace(native_main={"pred_field": torch.zeros((1, 2, 5))},
                native_prepared=SimpleNamespace(stencils={"fluid": SimpleNamespace(valid=valid),
                    "material": SimpleNamespace(valid=material)}, source_present=torch.ones((1, 1))))
            xy = torch.tensor([[[0. if missing == "pressure" else 1., 0.], [1., 0.]]])
            batch = SimpleNamespace(targets=SimpleNamespace(field_targets=torch.zeros((1, 2, 5)),
                point_weights=torch.ones((1, 2)), material_targets=torch.ones((1, 1, 1))),
                receivers=SimpleNamespace(fluid_xy=xy))
            case_observer(cid, prediction, batch)
        return {}

    monkeypatch.setattr(evaluator, "evaluate_native_panel", fake_panel)
    with pytest.raises(ValueError, match="receiver"):
        evaluator.evaluate_pair_panel(torch.nn.Linear(1, 1), SimpleNamespace(data_path=path), epoch=1,
            split="validation", case_ids=("001", "002"), pairs=(("001", "002"),), scales=[1]*4,
            floors=[1]*4, mode="restricted", export_cases=(), output=tmp_path / "result.json")


def test_change_summary_reports_physical_and_normalized_errors():
    tool = Path(__file__).resolve().parents[1] / "tools/thermal_native_pair_evaluate.py"
    spec = importlib.util.spec_from_file_location("thermal_native_pair_evaluate_summary", tool)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    rows = [{"pair": ["one", "two"], "change_error": [2., -4., 0., 8.]},
            {"pair": ["three", "four"], "change_error": [0., 4., 6., 0.]}]
    result = evaluator.summarize_changes(rows, [2., 4., 3., 8.])
    assert result["normalized_change_rmse"] == pytest.approx(np.sqrt(8 / 8))
    assert result["observables"]["J"]["mae"] == pytest.approx(1.)
    assert result["observables"]["P"]["rmse"] == pytest.approx(4.)
    assert result["observables"]["A"]["rmse"] == pytest.approx(np.sqrt(18))
    assert result["contributing_case_count"] == 4


def test_pair_evaluation_rejects_duplicate_pairs():
    tool = Path(__file__).resolve().parents[1] / "tools/thermal_native_pair_evaluate.py"
    spec = importlib.util.spec_from_file_location("thermal_native_pair_evaluate_duplicates", tool)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    with pytest.raises(ValueError, match="duplicate|unique|repeated"):
        evaluator.evaluate_pair_panel(None, None, epoch=1, split="validation", case_ids=("001", "002"),
            pairs=(("001", "002"), ("001", "002")), scales=[1]*4, floors=[1]*4,
            mode="restricted", export_cases=(), output=Path("unused"))
