from __future__ import annotations

import pytest
import torch

from channelthermal.data.collation import ChannelThermalBatchCollator, ModuleCountBucketBatchSampler


def _sample(present: list[float], offset: float) -> dict:
    width = len(present)
    module_values = torch.arange(width, dtype=torch.float32) + offset
    affinity = module_values[:, None] * 10.0 + module_values[None, :]
    return {
        "structure": {
            "module_centers": torch.stack([module_values, module_values + 0.5], dim=-1),
            "heat_powers": module_values,
            "module_present": torch.tensor(present),
        },
        "interface_target": module_values[:, None, None],
        "structure_targets": {
            "env_module_influence_target": module_values[None, :],
            "module_affinity_target": affinity,
        },
        "case_id": f"case-{offset}",
    }


def test_dynamic_collator_compacts_and_pads_to_batch_active_maximum() -> None:
    batch = ChannelThermalBatchCollator()(
        [
            _sample([1.0, 0.0, 1.0, 0.0, 0.0], 0.0),
            _sample([0.0, 1.0, 1.0, 0.0, 1.0], 10.0),
        ]
    )

    assert batch["structure"]["module_present"].shape == (2, 3)
    assert torch.equal(batch["module_count"], torch.tensor([2, 3]))
    assert torch.equal(batch["structure"]["heat_powers"][0], torch.tensor([0.0, 2.0, 1.0]))
    assert torch.equal(batch["structure"]["heat_powers"][1], torch.tensor([11.0, 12.0, 14.0]))
    expected_affinity = torch.tensor([[0.0, 2.0, 1.0], [20.0, 22.0, 21.0], [10.0, 12.0, 11.0]])
    assert torch.equal(batch["structure_targets"]["module_affinity_target"][0], expected_affinity)


def test_dynamic_collator_enforces_optional_memory_safeguard() -> None:
    collator = ChannelThermalBatchCollator(max_modules_per_batch=2)

    with pytest.raises(ValueError, match="max_modules_per_batch=2"):
        collator([_sample([1.0, 1.0, 1.0], 0.0)])


def test_module_count_bucketing_is_epoch_deterministic() -> None:
    sampler = ModuleCountBucketBatchSampler(
        [1, 8, 2, 7, 3, 6, 4, 5],
        batch_size=2,
        bucket_size_multiplier=4,
        seed=9,
    )
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    second = list(sampler)

    assert first == second
    assert sorted(index for batch in first for index in batch) == list(range(8))
    assert all(
        max(sampler.module_counts[index] for index in batch)
        - min(sampler.module_counts[index] for index in batch)
        <= 1
        for batch in first
    )
