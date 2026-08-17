"""Dynamic module-axis collation for ChannelThermal forward batches."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Sampler, default_collate


def _gather_module_axis(value: torch.Tensor, order: torch.Tensor, axis: int) -> torch.Tensor:
    """Apply one per-case module permutation to a batched tensor axis."""

    view_shape = [order.shape[0]] + [1] * (value.ndim - 1)
    view_shape[axis] = order.shape[1]
    index_shape = list(value.shape)
    index_shape[axis] = order.shape[1]
    index = order.view(view_shape).expand(index_shape)
    return torch.gather(value, axis, index)


def _slice_axis(value: torch.Tensor, axis: int, width: int) -> torch.Tensor:
    slices = [slice(None)] * value.ndim
    slices[axis] = slice(0, width)
    return value[tuple(slices)]


@dataclass(frozen=True)
class ChannelThermalBatchCollator:
    """Compact active modules and pad only to the largest count in this batch."""

    dynamic_module_padding: bool = True
    max_modules_per_batch: int | None = None

    def __post_init__(self) -> None:
        if self.max_modules_per_batch is not None and int(self.max_modules_per_batch) <= 0:
            raise ValueError("max_modules_per_batch must be positive when configured.")

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        batch = default_collate(samples)
        present = batch["structure"]["module_present"]
        if present.ndim != 2:
            raise ValueError(f"module_present must collate to [B,M], got {tuple(present.shape)}.")
        active = present > 0.5
        active_counts = active.sum(dim=1)
        required = max(int(active_counts.max().item()), 1)
        runtime_width = required if self.dynamic_module_padding else int(present.shape[1])
        if self.max_modules_per_batch is not None and runtime_width > int(self.max_modules_per_batch):
            raise ValueError(
                f"Batch requires M={runtime_width}, exceeding the collation safeguard "
                f"max_modules_per_batch={self.max_modules_per_batch}."
            )
        if not self.dynamic_module_padding:
            return batch

        batch_size, source_width = present.shape
        slots = torch.arange(source_width, device=present.device).expand(batch_size, -1)
        sort_key = torch.where(active, slots, slots + source_width)
        order = torch.argsort(sort_key, dim=1)

        structure = batch["structure"]
        for key in ("module_centers", "heat_powers", "module_present"):
            structure[key] = _slice_axis(_gather_module_axis(structure[key], order, 1), 1, required)

        for key in (
            "module_internal_temperature_points",
            "interface_condition",
            "interface_condition_valid_mask",
            "interface_target",
            "teacher_port_tokens",
            "local_module_params",
        ):
            if key in batch:
                batch[key] = _slice_axis(_gather_module_axis(batch[key], order, 1), 1, required)

        targets = batch.get("structure_targets")
        if isinstance(targets, dict):
            for key in ("env_module_influence_target", "env_module_target_mask"):
                if key in targets:
                    targets[key] = _slice_axis(_gather_module_axis(targets[key], order, 2), 2, required)
            for key in ("module_affinity_target", "module_affinity_target_mask"):
                if key in targets:
                    value = _gather_module_axis(targets[key], order, 1)
                    value = _gather_module_axis(value, order, 2)
                    targets[key] = _slice_axis(_slice_axis(value, 1, required), 2, required)

        batch["module_count"] = active_counts
        return batch


class ModuleCountBucketBatchSampler(Sampler[list[int]]):
    """Randomize cases while forming local batches with similar module counts."""

    def __init__(
        self,
        module_counts: Sequence[int],
        *,
        batch_size: int,
        bucket_size_multiplier: int = 8,
        seed: int = 0,
    ) -> None:
        self.module_counts = tuple(int(value) for value in module_counts)
        self.batch_size = int(batch_size)
        self.bucket_size = self.batch_size * int(bucket_size_multiplier)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size <= 0 or self.bucket_size <= 0:
            raise ValueError("batch_size and bucket_size_multiplier must be positive.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.module_counts)))
        rng.shuffle(indices)
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=self.module_counts.__getitem__)
            batches.extend(
                bucket[offset : offset + self.batch_size]
                for offset in range(0, len(bucket), self.batch_size)
            )
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return int(math.ceil(len(self.module_counts) / max(self.batch_size, 1)))


__all__ = ["ChannelThermalBatchCollator", "ModuleCountBucketBatchSampler"]
