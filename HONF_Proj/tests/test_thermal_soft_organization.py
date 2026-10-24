"""Bank calibration support, zero VJPs, full-gradient cap and fixed input panel."""
from types import SimpleNamespace

import pytest
import torch

from honf_runtime.thermal_soft_organization import (
    TERMS, SoftOrganizationRuntime, gradient_balance, select_calibration_panel,
    validate_softonly_config,
)
from honf_runtime import research_training as research
from tests.test_research_training_runner import PRESETS


def test_panel_uses_padded_presence_mask_and_covers_every_stratum_order_independently():
    cases = [{"case_id": f"{count:02}_{i:02}", "structure": {
        # Match the packed dataset: centers and presence use fixed max_modules
        # width, while module_present carries the actual physical count.
        "module_centers": torch.zeros(12, 2),
        "module_present": torch.tensor([1.] * count + [0.] * (12 - count)),
    }} for count in range(1, 13) for i in range(20)]
    panel = select_calibration_panel(cases)
    assert len(panel) == len(set(panel)) == 96
    assert panel == select_calibration_panel(list(reversed(cases)))
    assert {p.split("_")[0] for p in panel} == {f"{i:02}" for i in range(1, 13)}
    assert all(sum(p.startswith(f"{i:02}_") for p in panel) == 8 for i in range(1, 13))


def test_gradient_balance_uses_same_declared_set_and_zero_fills():
    left = {"a": torch.tensor([3., 4.]), "b": torch.tensor([12.])}
    right = {"b": torch.tensor([-5.])}
    row = gradient_balance(left, right, ["a", "b", "c"])
    assert row["native_norm"] == 13 and row["term_norm"] == 5
    assert row["signed_dot"] == -60 and row["opposed_sign_elements"] == 1
    assert row["term_missing_names_zero_filled"] == ["a", "c"]
    assert row["native_missing_names_zero_filled"] == ["c"]


def test_combined_structural_cap_uses_native_full_model_gradient():
    policy_m = torch.nn.Parameter(torch.tensor([1.]))
    policy_e = torch.nn.Parameter(torch.tensor([1.]))
    shared = torch.nn.Parameter(torch.tensor([1.]))
    policy_m.grad = torch.tensor([0.])
    policy_e.grad = torch.tensor([0.])
    shared.grad = torch.tensor([10.])
    names = {"core.compact_retrieval.compatibility.M.weight": policy_m,
             "core.compact_retrieval.compatibility.E.weight": policy_e,
             "core.shared.weight": shared}
    runtime = object.__new__(SoftOrganizationRuntime)
    runtime._named = lambda: names
    runtime.state = {"coefficients": {name: 1. for name in TERMS}}
    runtime.calibrating = False
    objective = SimpleNamespace(
        additive_denominators={name: 2 for name in TERMS},
        additive_gradients={name: {policy_m: torch.tensor([200.])} for name in TERMS},
        additive_values={name: [torch.tensor(2.)] for name in TERMS}, losses={}, work={})
    runtime.apply_additives(objective)
    assert policy_m.grad.item() == pytest.approx(1.)
    assert shared.grad.item() == 10.
    assert runtime.last_record["structural_full_norm"] == pytest.approx(1.)
    assert runtime.last_record["structural_cap_scale"] < 1
    assert runtime.last_record["terms"]["compact_tie_M"]["parameter_names"] == [next(iter(names))]


def test_softonly_config_rejects_a_cut_freeze_or_validation_tuning():
    config, _ = research.load_research_config(PRESETS / "thermal_compact_softonly_full1000_scratch_v2.json")
    validate_softonly_config(config)
    for key, value in (("compact_arm", "merge"), ("freeze_policy", True),
                       ("calibration_epochs", [1, 100, 200, 300]),
                       ("geometry_train_families", ["0304"])):
        changed = {**config, "round12": {**config["round12"], key: value}}
        with pytest.raises(ValueError):
            research._validate_research_config(changed)


def test_in_process_state_rollback_preserves_failed_attempt_wall_time(monkeypatch):
    runtime = object.__new__(SoftOrganizationRuntime)
    runtime.state = {"schema": "thermal_soft_organization_state_v2"}
    runtime.panel = ("a",)
    runtime.geometry = SimpleNamespace(metadata={"reference": "fixed"})
    runtime.elapsed_base = 100.
    runtime.started = 20.
    monkeypatch.setattr("honf_runtime.thermal_soft_organization.time.monotonic", lambda: 35.)
    runtime.load_state_dict({"schema": runtime.state["schema"], "panel_case_ids": ["a"],
                             "geometry_identity": runtime.geometry.metadata,
                             "training_elapsed_seconds": 100.})
    assert runtime.elapsed_base == 115. and runtime.started == 35.
