"""Real optimizer/checkpoint continuation at a requested epoch boundary."""

from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from channelthermal.training.checkpoints import _restore_rng_state, save_checkpoint
from channelthermal.training.stop_request import acknowledge_stop, stop_requested

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.run_layout import RunLayout


class TinyTrainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
        self.register_buffer("fixed_normalizer", torch.tensor([2., 3.]))
        empty = torch.empty(0)
        self.local_coupling = SimpleNamespace(
            **{name: empty for name in (
                "local_module_params_mean", "local_module_params_std", "local_port_tokens_mean",
                "local_port_tokens_std", "local_internal_temperature_mean", "local_internal_temperature_std",
                "local_interface_targets_mean", "local_interface_targets_std")},
            local_surrogate=None, local_surrogate_checkpoint_path=None, local_surrogate_frozen=True,
            local_surrogate_normalize_inputs=False, local_surrogate_normalize_targets=False,
        )
        self.input_adapter = SimpleNamespace(feature_names=["x"], global_context_names=[],
                                             content_context_names=[], fixed_heat_scale=1.)
        self.campaign_training_state = {"sampler_epoch": 0, "completed_updates": 0}
        self.age = 0

    def selection_state(self):
        return {"epoch": self.age, "total_epochs": 1000}

    def epoch(self, optimizer):
        optimizer.zero_grad(set_to_none=True)
        # Actual two-microbatch accumulation, stochastic dropout and all saved
        # random streams make a lost/advanced resume state observable.
        for _ in range(2):
            x = torch.randn(2, 3) + float(np.random.uniform()) + random.random()
            target = torch.randn(2, 1)
            loss = (self.net(x) - target).square().mean() / 2
            loss.backward()
        optimizer.step()
        self.age += 1
        self.campaign_training_state.update(sampler_epoch=self.age, completed_updates=self.age)


def seed():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)


def checkpoint_kwargs(model, optimizer):
    config = SimpleNamespace(to_dict=lambda: {"channelthermal": {"internal_prediction_mode": "global_head"}},
                             channelthermal=SimpleNamespace(local_surrogate_checkpoint_path=None))
    dataset = SimpleNamespace(channel_order=["temperature"], field_dim=1,
        interface_condition_feature_names=[], interface_target_names=[], max_num_modules=1,
        selected_module_counts=[1], normalizer=SimpleNamespace(stats={"field_std": np.array([3.], np.float32)}))
    return {"model": model, "model_config": config, "train_config": {"dataset": {}},
            "dataset": dataset, "epoch": model.age, "best_metric": 1., "optimizer": optimizer}


def assert_nested_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_requested_stop_resume_next_optimizer_update_equals_uninterrupted(tmp_path):
    seed()
    uninterrupted = TinyTrainer()
    optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=3e-4, weight_decay=1e-5)
    uninterrupted.epoch(optimizer)
    (tmp_path / "stop_requested").touch()
    assert stop_requested(tmp_path)
    save_checkpoint(tmp_path / "latest_model.pt", **checkpoint_kwargs(uninterrupted, optimizer))
    acknowledge_stop(tmp_path, epoch=1)
    assert not stop_requested(tmp_path)
    assert not (tmp_path / "latest_model.pt.tmp").exists()
    saved = load_trusted_checkpoint(tmp_path / "latest_model.pt", map_location="cpu")
    uninterrupted.epoch(optimizer)
    expected_parameters = copy.deepcopy(uninterrupted.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    resumed = TinyTrainer()
    resumed.load_state_dict(saved["model_state_dict"])
    restored_optimizer = torch.optim.AdamW(resumed.parameters(), lr=3e-4, weight_decay=1e-5)
    restored_optimizer.load_state_dict(saved["optimizer_state_dict"])
    resumed.age = saved["epoch"]
    resumed.campaign_training_state = copy.deepcopy(saved["campaign_training_state"])
    _restore_rng_state(saved)
    resumed.epoch(restored_optimizer)
    assert_nested_equal(resumed.state_dict(), expected_parameters)
    assert_nested_equal(restored_optimizer.state_dict(), expected_optimizer)
    assert resumed.campaign_training_state == uninterrupted.campaign_training_state
    np.testing.assert_array_equal(saved["global_normalization_stats"]["field_std"], [3.])
    assert saved["selection_state"] == {"epoch": 1, "total_epochs": 1000}


def test_atomic_save_failure_preserves_previous_latest_and_unacknowledged_request(tmp_path, monkeypatch):
    seed()
    model = TinyTrainer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.epoch(optimizer)
    path = tmp_path / "latest_model.pt"
    save_checkpoint(path, **checkpoint_kwargs(model, optimizer))
    previous = path.read_bytes()
    (tmp_path / "stop_requested").touch()

    def fail_after_partial_write(payload, temporary):
        temporary.write_bytes(b"interrupted temporary write")
        raise OSError("fixture write failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="write failure"):
        save_checkpoint(path, **checkpoint_kwargs(model, optimizer))
    assert path.read_bytes() == previous
    assert stop_requested(tmp_path)


def test_stop_acknowledgement_records_resumable_status(tmp_path):
    import json

    (tmp_path / "run_manifest.json").write_text(json.dumps({"status": "running"}))
    (tmp_path / "stop_requested").touch()
    acknowledge_stop(tmp_path, epoch=7)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "stopped_resumable"
    assert manifest["last_completed_epoch"] == 7
    assert manifest["resume_checkpoint"] == str(RunLayout(tmp_path).path("latest_model.pt"))
