from __future__ import annotations

from types import SimpleNamespace

import pytest
from channelthermal.training.unified_task import ThermalRefinementTask


class _Core:
    def __init__(self):
        self.policy = None

    def set_execution(self, **policy):
        self.policy = policy


@pytest.mark.parametrize(
    ("adapter_mode", "core_mode"),
    (
        ("warmup", "all_fine"),
        ("full_detail", "all_fine"),
        ("adaptive_detail", "adaptive"),
        ("all_fine", "all_fine"),
        ("all_base", "all_base"),
        ("nearest", "nearest"),
        ("upstream", "upstream"),
        ("shuffle", "shuffle"),
    ),
)
def test_native_adapter_preserves_refinement_evaluation_modes(adapter_mode, core_mode):
    core = _Core()
    model = SimpleNamespace(core=core)
    ThermalRefinementTask._set_execution(
        None, model, adapter_mode, "hard", 0.25, training=False, threshold=0.6
    )
    assert core.policy == {
        "mode": core_mode,
        "phase": "hard",
        "threshold": 0.6,
        "temperature": 0.25,
        "training_signal": False,
    }


def test_parent_native_adapter_all_fine_needs_no_refinement_policy():
    model = SimpleNamespace(core=object())
    ThermalRefinementTask._set_execution(None, model, "full_detail", "hard", 1.0, training=False)


def test_route_control_rejects_unknown_mode():
    model = SimpleNamespace(core=_Core())
    with pytest.raises(ValueError, match="Unknown Thermal refinement evaluation mode"):
        ThermalRefinementTask._set_execution(None, model, "route_search", "hard", 1.0, training=False)


def test_unrefined_parent_rejects_subset_route_controls():
    model = SimpleNamespace(core=object())
    with pytest.raises(TypeError, match="require the opt-in refined source-response core"):
        ThermalRefinementTask._set_execution(None, model, "nearest", "hard", 1.0, training=False)
