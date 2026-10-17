"""Calibration is a read-only probe of scientific model/provider/RNG state."""
from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from honf_forward_core.joint_native.model import JointNativeCore
from honf_runtime.joint_native_calibration import calibrate_joint_native_task
from tests.test_content_hypergraph_v2_core import _core
from tests.test_native_budgeted_protocol import _Provider, _engine


@pytest.mark.parametrize("mode,arm", [("joint", "H9-ME"), ("direct", "S9-ME")])
@pytest.mark.parametrize("fail_preparation", [False, True])
def test_calibration_restores_exposure_and_state_even_before_forward(monkeypatch, mode, arm, fail_preparation):
    provider = _Provider()
    provider.training_samples = 17
    provider.validation_samples = 19
    provider.physical_training_samples = 23
    provider.physical_training_queries = 29
    model = JointNativeCore.from_a2d(_core("A2-D"), routing_mode=mode).train()
    model.environment_reader.eval()  # Preserve heterogeneous submodule modes.
    if mode == "direct":
        model.router.route = model.router.route  # Preserve a caller-owned instance override exactly.
    had_instance_route = "route" in vars(model.router)
    original_instance_route = vars(model.router).get("route")
    with torch.no_grad():
        model.environment_conditioning.weight.fill_(.02)
    engine = _engine()
    engine.epoch_case_order = lambda provider, epoch: (["fixture-case"], "fixture-dataset")
    engine.epoch_update_slices = lambda count: (slice(0, count),)

    def prepare(*args, **kwargs):
        provider.training_samples += 1
        provider.validation_samples += 2
        provider.physical_training_samples += 3
        provider.physical_training_queries += 4
        provider.new_training_queries = 7
        random.random()
        np.random.rand()
        torch.rand(1)
        if fail_preparation:
            raise RuntimeError("injected batch preparation failure")
        return [provider.batch]

    engine.prepare_update_microbatches = prepare
    task = {"model": model, "provider": provider, "engine": engine, "identity": {"recipe": {
        "arm": arm, "dataset": "thermal", "source_epoch": 3000,
        "source_checkpoint_sha256": "a" * 64, "microbatch_cases": 1,
    }}}
    original = copy.deepcopy(model.state_dict())
    modes = {name: module.training for name, module in model.named_modules()}
    clocks = (model._joint_epoch, model._joint_update_index, model._joint_macro_clock)
    counters = {name: value for name, value in vars(provider).items() if isinstance(value, (int, float))}
    hook_counts = {name: len(module._forward_hooks) for name, module in model.named_modules()}
    rng_python, rng_numpy, rng_torch = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()

    # CPU calibration must not initialize or read a CUDA RNG/context.
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("CPU calibration initialized CUDA RNGs"))
    if fail_preparation:
        with pytest.raises(RuntimeError, match="injected batch preparation"):
            calibrate_joint_native_task(task, max_batches=1)
    else:
        receipt = calibrate_joint_native_task(task, max_batches=1)
        assert receipt["optimizer_updates"] == 0
        assert receipt["provider_numeric_counters_restored"] is True

    for name, value in original.items():
        actual = model.state_dict()[name]
        assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual
    assert modes == {name: module.training for name, module in model.named_modules()}
    assert clocks == (model._joint_epoch, model._joint_update_index, model._joint_macro_clock)
    assert counters == {name: value for name, value in vars(provider).items() if isinstance(value, (int, float))}
    assert hook_counts == {name: len(module._forward_hooks) for name, module in model.named_modules()}
    assert ("route" in vars(model.router)) == had_instance_route
    assert vars(model.router).get("route") is original_instance_route
    assert random.getstate() == rng_python
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == rng_numpy[0] and np.array_equal(actual_numpy[1], rng_numpy[1])
    assert actual_numpy[2:] == rng_numpy[2:]
    assert torch.equal(torch.get_rng_state(), rng_torch)
