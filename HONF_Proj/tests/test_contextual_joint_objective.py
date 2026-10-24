"""CPU contracts for contextual organizer VJPs, calibration, and safety caps."""
from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from channelthermal.training.contextual_joint_objective import ContextualJointObjectiveRuntime
from honf_forward_core.contextual_organization.forest import ContextualFrontierPolicy
from honf_runtime.unified_training import capture_rng_state


class _ToyCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_policy = torch.nn.Parameter(torch.tensor(0.5))
        self.environment_policy = torch.nn.Parameter(torch.tensor(-0.25))
        self.shared = torch.nn.Parameter(torch.tensor(1.25))
        self.native_only = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("organization_epoch", torch.tensor(0, dtype=torch.long))
        self.register_buffer("organization_update", torch.tensor(0, dtype=torch.long))
        self._organization_epoch = 0
        self._organization_update = 0

    def organizer_parameters(self, bank):
        if bank == "module":
            return {"frontier_policies.module.score": self.module_policy, "shared": self.shared}
        if bank == "environment":
            return {"frontier_policies.environment.score": self.environment_policy, "shared": self.shared}
        raise ValueError(bank)

    def set_organization_clock(self, epoch, update):
        self._organization_epoch = int(epoch)
        self._organization_update = int(update)
        self.organization_epoch.fill_(int(epoch))
        self.organization_update.fill_(int(update))


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _ToyCore()


class _Provider:
    def __init__(self, model):
        self.model = model
        self.counter = 7
        self.training_samples = 11
        self.validation_samples = 13
        self._segment_started = 17.0
        self.training_cases = tuple({
            "case_id": f"case-{index}",
            "structure": {"module_present": np.asarray([1.0] * (1 + index % 2) + [0.0] * (4 - (1 + index % 2)))},
        } for index in range(4))

    def training_state_dict(self):
        return {"counter": self.counter}

    def load_training_state_dict(self, state):
        self.counter = int(state["counter"])


def _frontier(core, *, count, log_scale=0.0, bank="module"):
    policy = core.module_policy if bank == "module" else core.environment_policy
    return SimpleNamespace(
        expected_K=count * policy.square(),
        log_prob=policy * log_scale,
        reached_internal=(0,) if log_scale else (),
    )


def _predictions(core, module_counts, *, log_scale=0.0, environment_counts=None):
    environment_counts = environment_counts or [0] * len(module_counts)
    module_banks = tuple(SimpleNamespace(
        present_count=count,
        frontier=_frontier(core, count=count, log_scale=log_scale, bank="module") if count else None,
    ) for count in module_counts)
    environment_banks = tuple(SimpleNamespace(
        present_count=count,
        frontier=_frontier(core, count=count, log_scale=log_scale, bank="environment") if count else None,
    ) for count in environment_counts)
    prepared = SimpleNamespace(module_banks=module_banks, environment_banks=environment_banks)
    return SimpleNamespace(native_prepared=SimpleNamespace(organization=prepared))


def _gradient_norm(values):
    return torch.sqrt(sum(value.detach().double().square().sum() for value in values.values())).item()


def _assert_nested_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_collect_normalizes_each_scene_and_zero_fills_absent_bank_vjps():
    model = _ToyModel()
    runtime = ContextualJointObjectiveRuntime(model)
    predictions = _predictions(model.core, [2, 4], log_scale=0.0, environment_counts=[0, 0])
    losses = torch.tensor([1.0, 2.0]) * model.core.shared

    assert runtime.collect(predictions, losses, 21, case_ids=("a", "b")) is None
    summary = runtime.last_collection

    assert summary["banks"]["module"]["eligible_cases"] == 2
    assert summary["banks"]["module"]["normalized_expected_k_sum"] == pytest.approx(0.5)
    assert summary["banks"]["environment"]["eligible_cases"] == 0
    raw = runtime._accumulators["capacity_module"]
    # Each E[K]/N is w^2, so the sum of two case VJPs is 4w.
    assert raw["gradients"]["core.module_policy"].item() == pytest.approx(2.0)
    assert "core.shared" not in raw["gradients"]  # exact absent VJP; inventory later zero-fills it
    assert "capacity_environment" not in runtime._accumulators
    assert runtime._current_case_coverage == {"module": 2, "environment": 0}
    runtime.state["coefficients"]["capacity_module"] = 1.0
    model.core.native_only.grad = torch.tensor(1.0)
    record = runtime.before_step(model, None, {}, {}, epoch=21, update_index=0)
    assert record["terms"]["capacity_module"]["term_missing_names_zero_filled"] == ["core.shared"]


def test_reinforce_uses_only_detached_native_loss_and_updates_per_case_ema_afterward():
    model = _ToyModel()
    runtime = ContextualJointObjectiveRuntime(model, baseline_decay=0.5)
    prediction = _predictions(model.core, [3], log_scale=4.0)
    native_loss = torch.tensor([3.0]) * model.core.shared

    runtime.collect(prediction, native_loss, 21, case_ids=("case-a",))

    reinforce = runtime._accumulators["reinforce_module"]["gradients"]
    assert reinforce["core.module_policy"].item() == pytest.approx(15.0)
    assert "core.shared" not in reinforce
    assert runtime._baseline == {}  # the action used the previous-update baseline
    assert runtime._pending_baseline == {"case-a": [3.75]}

    runtime.state["coefficients"] = {"reinforce_module": 1.0}
    model.core.native_only.grad = torch.tensor(1.0)
    runtime.before_step(model, None, {}, {}, epoch=21, update_index=0)
    assert runtime._baseline["case-a"] == pytest.approx(3.75)


def test_loading_provider_snapshot_discards_partial_oom_update_before_retry():
    model = _ToyModel()
    runtime = ContextualJointObjectiveRuntime(model)
    snapshot = runtime.state_dict()

    failed_prediction = _predictions(model.core, [3], log_scale=4.0)
    runtime.collect(
        failed_prediction,
        torch.tensor([2.0]) * model.core.shared,
        21,
        case_ids=("case-a",),
    )
    assert runtime._accumulators
    assert runtime._pending_baseline == {"case-a": [2.5]}

    # The maintained engine uses provider restoration to roll back a partially
    # collected macro update after OOM. A persistent state snapshot has no
    # partial VJPs, so loading it must clear every transient update buffer.
    runtime.load_state_dict(snapshot)
    assert runtime._accumulators == {}
    assert runtime._pending_baseline == {}
    assert runtime._native_calibration_gradients == {}
    assert runtime._native_calibration_cases == 0
    assert runtime._current_case_coverage == {"module": 0, "environment": 0}
    assert runtime._current_vjp_coverage == {}
    assert runtime.last_collection is None

    retry_prediction = _predictions(model.core, [3], log_scale=4.0)
    runtime.collect(
        retry_prediction,
        torch.tensor([3.0]) * model.core.shared,
        21,
        case_ids=("case-a",),
    )

    fresh_model = _ToyModel()
    fresh_runtime = ContextualJointObjectiveRuntime(fresh_model)
    fresh_runtime.load_state_dict(snapshot)
    fresh_runtime.collect(
        _predictions(fresh_model.core, [3], log_scale=4.0),
        torch.tensor([3.0]) * fresh_model.core.shared,
        21,
        case_ids=("case-a",),
    )
    _assert_nested_equal(runtime._accumulators, fresh_runtime._accumulators)
    _assert_nested_equal(runtime._pending_baseline, fresh_runtime._pending_baseline)
    _assert_nested_equal(runtime._current_case_coverage, fresh_runtime._current_case_coverage)
    _assert_nested_equal(runtime._current_vjp_coverage, fresh_runtime._current_vjp_coverage)


def test_full_native_gradient_cap_applies_after_combining_banks_and_raw_terms():
    model = _ToyModel()
    runtime = ContextualJointObjectiveRuntime(model, full_gradient_cap=0.1)
    runtime.state["coefficients"] = {"probe_module": 1.0, "probe_environment": 1.0}
    runtime.add_raw_gradients("probe_module", "module", {
        "frontier_policies.module.score": torch.tensor(100.0), "shared": torch.tensor(100.0),
    })
    runtime.add_raw_gradients("probe_environment", "environment", {
        "frontier_policies.environment.score": torch.tensor(100.0), "shared": torch.tensor(100.0),
    })
    model.core.native_only.grad = torch.tensor(10.0)

    record = runtime.before_step(model, None, {}, {}, epoch=30, update_index=2)

    added = {
        "module": model.core.module_policy.grad,
        "environment": model.core.environment_policy.grad,
        "shared": model.core.shared.grad,
    }
    assert _gradient_norm(added) <= 1.0 + 1e-6
    assert record["native_full_gradient_norm"] == pytest.approx(10.0)
    assert record["organizer_cap_scale"] < 1.0
    assert record["organizer_weighted_gradient_norm"] == pytest.approx(1.0)
    assert record["terms"]["probe_module"]["term_missing_names_zero_filled"] == []


def test_zero_native_reference_never_creates_a_positive_cap_or_amplification():
    model = _ToyModel()
    runtime = ContextualJointObjectiveRuntime(model)
    runtime.state["coefficients"] = {"probe_module": 1.0}
    runtime.add_raw_gradients("probe_module", "module", {"core.module_policy": torch.tensor(12.0)})

    record = runtime.before_step(model, None, {}, {}, epoch=30, update_index=2)

    assert record["native_full_gradient_norm"] == 0.0
    assert record["organizer_cap"] == 0.0
    assert record["organizer_cap_scale"] == 0.0
    assert model.core.module_policy.grad is None


def test_frontier_ramp_scales_organizer_before_full_native_cap_at_epoch_21_and_200():
    multipliers = {
        21: ContextualFrontierPolicy.structural_multiplier_for_epoch(21),
        200: ContextualFrontierPolicy.structural_multiplier_for_epoch(200),
    }
    assert multipliers[21] == pytest.approx(
        0.1 + 0.45 * (1.0 - np.cos(np.pi * 20 / 199.0))
    )
    assert multipliers[200] == pytest.approx(1.0)

    for epoch, multiplier in multipliers.items():
        model = _ToyModel()
        model.core.frontier_policies = {
            "module": SimpleNamespace(
                structural_multiplier_for_epoch=ContextualFrontierPolicy.structural_multiplier_for_epoch
            ),
            "environment": SimpleNamespace(
                structural_multiplier_for_epoch=ContextualFrontierPolicy.structural_multiplier_for_epoch
            ),
        }
        runtime = ContextualJointObjectiveRuntime(model)
        runtime.state["coefficients"] = {"probe_module": 1.0}
        runtime.add_raw_gradients("probe_module", "module", {
            "frontier_policies.module.score": torch.tensor(2.0),
            "shared": torch.tensor(1.0),
        })
        model.core.native_only.grad = torch.tensor(100.0)
        work = {}

        record = runtime.before_step(model, None, {}, work, epoch=epoch, update_index=0)

        assert record["structural_multiplier"] == pytest.approx(multiplier)
        assert record["organizer_cap_scale"] == pytest.approx(1.0)
        assert record["organizer_weighted_gradient_norm"] == pytest.approx(np.sqrt(5.0) * multiplier)
        assert model.core.module_policy.grad.item() == pytest.approx(2.0 * multiplier)
        assert model.core.shared.grad.item() == pytest.approx(multiplier)
        assert work["contextual_structural_multiplier"] == pytest.approx(multiplier)


def test_inactive_frontier_flags_disable_capacity_and_reinforce_during_warmup():
    model = _ToyModel()
    runtime = ContextualJointObjectiveRuntime(model)
    prediction = _predictions(model.core, [3], log_scale=4.0)
    frontier = prediction.native_prepared.organization.module_banks[0].frontier
    frontier.capacity_active = False
    frontier.score_function_active = False

    runtime.collect(prediction, torch.tensor([3.0]) * model.core.shared, 20, case_ids=("case-a",))

    assert runtime._accumulators == {}
    assert runtime._pending_baseline == {"case-a": [3.75]}
    assert runtime.last_collection["banks"]["module"]["capacity_active_cases"] == 0
    assert runtime.last_collection["banks"]["module"]["score_function_active_cases"] == 0


def test_calibration_is_no_step_and_exactly_restores_rng_model_provider_clock_and_optimizer():
    random.seed(5)
    np.random.seed(6)
    torch.manual_seed(7)
    model = _ToyModel()
    provider = _Provider(model)
    runtime = ContextualJointObjectiveRuntime(model, provider, calibration_max_cases=4)
    runtime.calibration_extra = [{"prior": "preserve"}]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.core.set_organization_clock(8, 3)
    model.train(True)
    model.core.training = False  # calibration must preserve per-module modes
    model.core.module_policy.grad = torch.tensor(0.75)
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    provider_before = provider.training_state_dict()
    counters_before = (provider.training_samples, provider.validation_samples)
    segment_started_before = provider._segment_started
    rng_before = capture_rng_state()
    runtime_before = runtime.state_dict()

    def runner(active_provider, panel, epoch, objective):
        active_provider.counter = 99
        active_provider.training_samples += 5
        active_provider.validation_samples += 3
        model.core.set_organization_clock(900, 12)
        _ = torch.rand(1), np.random.rand(), random.random()
        objective.calibration_extra.append({"probe_index": 0, "bank": "module"})
        rows = []
        for index, case_id in enumerate(panel):
            count = 1 + index % 2
            native = (model.core.shared * (index + 1.0) - 1.0).square().reshape(1)
            prediction = _predictions(model.core, [count], log_scale=2.0)
            rows.append((prediction, native, (case_id,)))
        return rows

    report = runtime.calibrate(provider, 1, optimizer=optimizer, panel_runner=runner)

    assert report["optimizer_steps"] == 0
    assert len(report["case_ids"]) == 4
    assert report["terms"]["capacity_module"]["coefficient"] > 0.0
    assert report["terms"]["reinforce_module"]["coefficient"] > 0.0
    assert report["terms"]["capacity_module"]["parameter_names"] == [
        "core.module_policy", "core.shared",
    ]
    assert "same_sign_elements" in report["terms"]["reinforce_module"]
    assert report["terms"]["reinforce_module"]["samples"] == len(report["case_ids"])
    assert report["functional_probe_exposure"] == [{"probe_index": 0, "bank": "module"}]
    assert report["calibration_coefficients_are_pre_ramp"] is True
    _assert_nested_equal(model.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    assert provider.training_state_dict() == provider_before
    assert (provider.training_samples, provider.validation_samples) == counters_before
    assert provider._segment_started == segment_started_before
    assert model.training is True and model.core.training is False
    assert model.core._organization_epoch == 8 and model.core._organization_update == 3
    torch.testing.assert_close(model.core.module_policy.grad, torch.tensor(0.75), rtol=0, atol=0)
    _assert_nested_equal(capture_rng_state(), rng_before)
    # Persistent calibration is the only intended runtime-state change.
    assert runtime.state["calibrated_epochs"] == [1]
    assert runtime.state["calibrations"][-1]["optimizer_steps"] == 0
    assert runtime.calibration_extra == [{"prior": "preserve"}]
    assert runtime_before["calibrated_epochs"] == []


def test_calibration_uses_zero_coefficient_when_native_reference_norm_is_zero():
    model = _ToyModel()
    provider = _Provider(model)
    runtime = ContextualJointObjectiveRuntime(model, provider)

    def runner(active_provider, panel, epoch, objective):
        for case_id in panel:
            prediction = _predictions(model.core, [2], log_scale=1.0)
            objective.collect(prediction, torch.tensor([2.0]), epoch, case_ids=(case_id,))

    report = runtime.calibrate(provider, 1, panel_runner=runner)

    assert report["terms"]["capacity_module"]["native_norm"] == 0.0
    assert report["terms"]["capacity_module"]["term_norm"] > 0.0
    assert report["terms"]["capacity_module"]["coefficient"] == 0.0
