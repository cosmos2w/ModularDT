"""Explicit inverse panel identity, selected membership and resume order."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "tools/thermal_campaign_heat_inference.py"
spec = importlib.util.spec_from_file_location("heat_panel_tool", path)
heat_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heat_tool)


def test_explicit_panel_replays_requested_order_and_avoids_default_selection(monkeypatch):
    dataset = SimpleNamespace(selected_case_ids=["0687", "0277", "0288", "0291", "0294"])
    monkeypatch.setattr(heat_tool, "screen_indices", lambda *_: (_ for _ in ()).throw(RuntimeError("default selection was used")))
    indices, requested = heat_tool.explicit_case_panel(dataset, 4, ("0277", "0291", "0294", "0687"))
    assert indices == [1, 3, 4, 0]
    assert requested == ("0277", "0291", "0294", "0687")
    heat_tool.validate_case_panel_resume({"requested_case_ids": list(requested)}, requested)
    with pytest.raises(ValueError, match="resume changed"):
        heat_tool.validate_case_panel_resume({"requested_case_ids": list(requested[::-1])}, requested)
    with pytest.raises(ValueError, match="resume changed"):
        heat_tool.validate_case_panel_resume({}, requested)


@pytest.mark.parametrize("requested", [("0277", "0277"), ("0277",), ("0277", "excluded")])
def test_explicit_panel_rejects_duplicates_wrong_budget_and_unselected_ids(requested):
    dataset = SimpleNamespace(selected_case_ids=["0277", "0291"])
    with pytest.raises(ValueError):
        heat_tool.explicit_case_panel(dataset, 2, requested)


def test_legacy_default_selection_and_resume_remain_available(monkeypatch):
    monkeypatch.setattr(heat_tool, "screen_indices", lambda _dataset, _count: [3, 1])
    assert heat_tool.explicit_case_panel(SimpleNamespace(), 2) == ([3, 1], None)
    heat_tool.validate_case_panel_resume({}, None)
