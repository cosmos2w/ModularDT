"""Matched ordering and actual native input preparation for complete calls."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from thermal_response_refinement_cost import alternating_orders, authorized_device, benchmark, prepare_inputs


def test_alternating_repetitions_preserve_all_arms_and_change_position():
    names = ("G-fast", "H-add", "H-joint")
    orders = alternating_orders(names)
    assert len(orders) == 5
    assert all(set(order) == set(names) for order in orders)
    assert len({order.index("G-fast") for order in orders}) == 3
    assert orders[0] == list(names)
    with pytest.raises(ValueError, match="Unique"):
        alternating_orders(("a", "a"))


def test_cost_horizon_is_fixed_before_any_model_load(tmp_path):
    with pytest.raises(ValueError, match="exactly five"):
        benchmark({}, tmp_path, device="cpu", repeats=4)


def test_cost_physical_device_requires_explicit_new_authorization(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    with pytest.raises(ValueError, match="physical GPU"):
        authorized_device("cuda:0")
    assert authorized_device("cuda:0", (0, 2)) == torch.device("cuda:0")
    assert authorized_device("cuda:1", (0, 2)) == torch.device("cuda:1")
    with pytest.raises(ValueError, match="explicitly authorized"):
        authorized_device("cuda:0", (2,))


def test_real_saved_parent_prepares_complete_native_conditions_without_model_calls():
    from honf_runtime.compat import load_trusted_checkpoint

    summary = Path("/data/wanglz/ModularDT/thermal_development/lean_interaction_20261005/evaluation/g-fast_absolute1000_additional500/summary.json")
    if not summary.exists():
        pytest.skip("Existing G-fast metadata is unavailable")
    checkpoint = Path(json.loads(summary.read_text())["checkpoint"])
    saved = load_trusted_checkpoint(checkpoint, map_location="cpu")
    inputs, panels, manifest, _ = prepare_inputs(saved, torch.device("cpu"))
    assert {name: (arguments["query_xy"].shape[0], arguments["query_xy"].shape[1])
            for name, arguments in inputs.items()} == {
                "B8Q1024_low": (8, 1024), "B8Q1024_high": (8, 1024),
                "B1Q8192_low": (1, 8192), "B1Q8192_high": (1, 8192)}
    for name, arguments in inputs.items():
        assert arguments["local_port_condition_mode"] == "predicted"
        assert arguments["mixed_teacher_ratio"] == 0
        assert arguments["return_predicted_port_outputs"]
        assert arguments["local_module_params"] is not None
        assert set(panels[name]["case_ids"]).issubset(manifest["partitions"]["train"]["case_ids"])
    assert panels["B8Q1024_low"]["module_counts"] == [1] * 8
    assert panels["B8Q1024_high"]["module_counts"] == [12] * 8
