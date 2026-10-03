"""Native coverage, stage continuation and matched initializer boundaries."""

from __future__ import annotations

import copy

import pytest
import torch
from channelthermal.data.collation import ModuleCountBucketBatchSampler
from channelthermal.training.campaign import (
    CampaignMicrobatchLoader,
    copy_matched_physical_initial_state,
    record_structural_calibration,
    structural_calibration_candidate,
    validate_campaign,
    validate_campaign_resume,
)
from channelthermal.training.native_denominators import native_denominators_enabled
from torch import nn


def _config():
    return {"training": {"seed": 0, "epochs": 100, "learning_rate": .0003,
                         "max_train_batches_per_epoch": None,
                         "campaign": {"arm": "H-tree", "schedule_total_epochs": 5000,
                                      "require_full_epoch": True, "matched_fresh_initialization": True}},
            "dataset": {"split": "train", "batch_size": 48}, "loss": {"value": 1.0}}


def test_campaign_rejects_partial_epochs_and_schedule_reset():
    config = _config()
    assert validate_campaign(config)["schedule_total_epochs"] == 5000
    with pytest.raises(ValueError, match="complete native"):
        validate_campaign(config, max_train_batches=2)
    changed = copy.deepcopy(config)
    changed["training"]["epochs"] = 5001
    with pytest.raises(ValueError, match="horizon"):
        validate_campaign(changed)


def test_stage_resume_preserves_optimizer_schedule_and_task():
    config = _config()
    checkpoint = {"train_config": copy.deepcopy(config), "optimizer_state_dict": {"state": 1}, "rng_state": {"torch": 1}}
    config["training"]["epochs"] = 500
    validate_campaign_resume(checkpoint, config)
    changed = copy.deepcopy(config)
    changed["training"]["campaign"]["schedule_total_epochs"] = 500
    with pytest.raises(ValueError, match="schedule/lineage"):
        validate_campaign_resume(checkpoint, changed)
    changed = copy.deepcopy(config)
    changed["loss"]["value"] = 2
    with pytest.raises(ValueError, match="loss"):
        validate_campaign_resume(checkpoint, changed)
    checkpoint["optimizer_state_dict"] = None
    with pytest.raises(ValueError, match="optimizer and RNG"):
        validate_campaign_resume(checkpoint, config)


def test_amp_resume_rejects_missing_scaler_state():
    config = _config()
    config["training"]["amp"] = True
    checkpoint = {"train_config": copy.deepcopy(config), "optimizer_state_dict": {"state": 1}, "rng_state": {"torch": 1}}
    with pytest.raises(ValueError, match="gradient-scaler"):
        validate_campaign_resume(checkpoint, config)
    checkpoint["scaler_state_dict"] = {"scale": 1024.}
    validate_campaign_resume(checkpoint, config)


def test_structural_calibration_spans_distinct_training_strata_and_records_norms():
    training_counts = list(range(1, 11)) * 60
    state = {}
    assert structural_calibration_candidate(state, [2, 3], training_counts) is None
    for count in (1, 3, 5, 7, 9):
        stratum = structural_calibration_candidate(state, [count] * 8, training_counts)
        assert stratum is not None
        record_structural_calibration(state, stratum, [count] * 8, task_norm=2., cost_norm=100.,
                                      max_weight=.001, epoch=26, native_batch=count)
        assert structural_calibration_candidate(state, [count] * 8, training_counts) is None
    assert state["structural_calibration_complete"]
    assert len(state["structural_scale_samples"]) == 5
    assert state["structural_scale"] == pytest.approx(.0004)
    assert [sample["observed_module_counts"] for sample in state["structural_calibration_samples"]] == [[1], [3], [5], [7], [9]]
    assert all(sample["task_organizer_gradient_norm"] == 2. for sample in state["structural_calibration_samples"])
    # An absent task signal never turns a tiny structural gradient into pressure.
    zero = {}
    stratum = structural_calibration_candidate(zero, [3], [3])
    record_structural_calibration(zero, stratum, [3], task_norm=0., cost_norm=1e-30,
                                  max_weight=.001, epoch=26, native_batch=1)
    assert zero["structural_scale"] == 0.
    assert zero["structural_calibration_complete"]


def test_native_bucket_epoch_and_query_stream_resume_by_absolute_epoch():
    sampler = ModuleCountBucketBatchSampler([i % 10 + 1 for i in range(600)], batch_size=48, seed=0)
    sampler.set_epoch(100)
    first = list(sampler)
    assert len(first) == 13
    assert sorted(i for batch in first for i in batch) == list(range(600))
    resumed = ModuleCountBucketBatchSampler(sampler.module_counts, batch_size=48, seed=0)
    resumed.set_epoch(101)
    sampler.set_epoch(101)
    assert list(resumed) == list(sampler)


class _Loader:
    dataset = tuple(range(7))
    def __len__(self):
        return 2
    def __iter__(self):
        for start, stop in ((0, 5), (5, 7)):
            values = torch.arange(start, stop, dtype=torch.float32).reshape(-1, 1)
            yield {"field_targets": values, "inputs": values, "case_id": [str(i) for i in range(start, stop)]}


def test_microbatches_preserve_actual_case_weight_and_effective_steps():
    native = nn.Linear(1, 1)
    micro = copy.deepcopy(native)
    native_optimizer, micro_optimizer = torch.optim.AdamW(native.parameters()), torch.optim.AdamW(micro.parameters())
    for batch in _Loader():
        native_optimizer.zero_grad(set_to_none=True)
        native(batch["inputs"]).square().mean().backward()
        native_optimizer.step()
    seen = []
    steps = 0
    for batch in CampaignMicrobatchLoader(_Loader(), 2):
        if batch["_optimizer_start"]:
            micro_optimizer.zero_grad(set_to_none=True)
        (micro(batch["inputs"]).square().mean() * batch["_accumulation_weight"]).backward()
        seen.extend(batch["case_id"])
        if batch["_optimizer_boundary"]:
            micro_optimizer.step()
            steps += 1
    assert seen == [str(i) for i in range(7)]
    assert steps == 2
    for actual, expected in zip(micro.parameters(), native.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_matched_initializer_rejects_physical_shape_drift():
    canonical, target = nn.Linear(2, 3), nn.Linear(2, 3)
    inventory = copy_matched_physical_initial_state(target, canonical)
    assert len(inventory["loaded"]) == 2
    assert torch.equal(target.weight, canonical.weight)
    with pytest.raises(ValueError, match="unmatched tensors"):
        copy_matched_physical_initial_state(nn.Linear(3, 3), canonical)


def test_native_denominator_policy_has_one_explicit_epoch100_migration():
    config = _config()
    saved = {"train_config": copy.deepcopy(config), "epoch": 100,
             "optimizer_state_dict": {"state": 1}, "rng_state": {"torch": 1}}
    config["training"]["epochs"] = 500
    config["training"]["campaign"].update(physical_loss_policy_version=2, native_loss_denominators_start_epoch=101)
    amendment = validate_campaign_resume(saved, config)
    assert amendment["activation_epoch"] == 101
    explicit_default = copy.deepcopy(saved)
    explicit_default["train_config"]["training"]["campaign"]["physical_loss_policy_version"] = 1
    assert validate_campaign_resume(explicit_default, config) == amendment
    assert not native_denominators_enabled(config["training"]["campaign"], 100)
    assert native_denominators_enabled(config["training"]["campaign"], 101)
    for epoch in (25, 99, 101, 500):
        changed = copy.deepcopy(saved)
        changed["epoch"] = epoch
        with pytest.raises(ValueError, match="schedule/lineage"):
            validate_campaign_resume(changed, config)
    for key, value in (("physical_loss_policy_version", 3), ("native_loss_denominators_start_epoch", 100), ("arm", "another")):
        changed = copy.deepcopy(config)
        changed["training"]["campaign"][key] = value
        with pytest.raises(ValueError, match="schedule/lineage"):
            validate_campaign_resume(saved, changed)
    later = {**saved, "epoch": 500, "train_config": copy.deepcopy(config)}
    config["training"]["epochs"] = 1000
    assert validate_campaign_resume(later, config) is None
