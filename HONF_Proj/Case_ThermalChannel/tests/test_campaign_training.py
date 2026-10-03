"""Native coverage, stage continuation and matched initializer boundaries."""

from __future__ import annotations

import copy

import pytest
import torch
from channelthermal.data.collation import ModuleCountBucketBatchSampler
from channelthermal.training.campaign import (
    CampaignMicrobatchLoader,
    copy_matched_physical_initial_state,
    validate_campaign,
    validate_campaign_resume,
)
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
