"""Frozen-model geometry reliance and quadrature diagnostics for the study."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import torch

from honf_forward_core.config import BatchData


def discrepancy(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | None]:
    difference = (candidate.double() - reference.double()).reshape(-1)
    denominator = float(torch.linalg.vector_norm(reference.double()))
    return {
        "max_absolute": float(difference.abs().max()),
        "rms": float(difference.square().mean().sqrt()),
        "relative_l2": float(torch.linalg.vector_norm(difference)) / denominator if denominator else None,
    }


def _read(model: Any, batch: BatchData, *, chunk: int = 128) -> torch.Tensor:
    prepared = model.prepare_case(batch)
    return model.decode(prepared, batch.query_xy, batch.query_features,
                        receiver_chunk_size=chunk)["pred_field"]


def _pad(batch: BatchData, extra: int) -> BatchData:
    updates = {}
    for name in ("module_centers", "module_features", "module_present"):
        value = getattr(batch, name)
        shape = list(value.shape)
        shape[1] = extra
        updates[name] = torch.cat((value, value.new_zeros(shape)), dim=1)
    return replace(batch, **updates)


def _join(first: BatchData, second: BatchData) -> BatchData:
    width = max(first.module_present.shape[1], second.module_present.shape[1])
    first = _pad(first, width - first.module_present.shape[1])
    second = _pad(second, width - second.module_present.shape[1])
    updates = {}
    for item in fields(BatchData):
        a, b = getattr(first, item.name), getattr(second, item.name)
        if torch.is_tensor(a) and torch.is_tensor(b):
            updates[item.name] = torch.cat((a, b), dim=0)
    updates["metadata"] = {}
    return replace(first, **updates)


def _refined_environment(batch: BatchData, scale: torch.Tensor) -> BatchData:
    lower = batch.global_context[:, -6:-3] * scale
    extent = batch.global_context[:, -3:] * scale
    axes = [(torch.arange(n, device=scale.device, dtype=scale.dtype) + .5) / n for n in (32, 8, 4)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, -1, 3)
    coords = lower[:, None] + extent[:, None] * grid
    features = torch.cat(((coords - lower[:, None]) / scale,
                          (lower[:, None] + extent[:, None] - coords) / scale,
                          coords[..., 2:3] / scale[2]), dim=-1)
    weights = batch.env_weights.sum(dim=1, keepdim=True).expand(-1, 1024) / 1024
    return replace(batch, env_coords=coords, env_features=features, env_weights=weights)


@torch.no_grad()
def geometry_diagnostics(model: Any, batches: list[BatchData]) -> list[dict[str, Any]]:
    """Compare fixed receivers under representation changes, without retraining.

    Inputs are three single-case validation batches with equal query counts.
    Domain corruption and route muting are fitted-model reliance tests only;
    their altered inputs do not define new solved physical cases.
    """
    if len(batches) != 3 or any(batch.module_present.shape[0] != 1 for batch in batches):
        raise ValueError("Geometry diagnostics require three single-case validation batches.")
    model.eval()
    records = []
    for index, batch in enumerate(batches):
        reference = _read(model, batch)
        record: dict[str, Any] = {"case_index": index, "query_count": batch.query_xy.shape[1]}
        record["same_chunk_repeat"] = discrepancy(reference, _read(model, batch))
        record["chunk_1024"] = discrepancy(reference, _read(model, batch, chunk=1024))
        permutation = torch.arange(batch.module_present.shape[1] - 1, -1, -1, device=batch.module_present.device)
        permuted = replace(batch, **{name: getattr(batch, name)[:, permutation]
                                   for name in ("module_centers", "module_present", "module_features")})
        record["module_permutation"] = discrepancy(reference, _read(model, permuted))
        record["additional_padding"] = discrepancy(reference, _read(model, _pad(batch, 7)))
        together = _join(batch, batches[(index + 1) % len(batches)])
        record["different_domain_batch"] = discrepancy(reference, _read(model, together)[:1])
        reverse = torch.arange(batch.query_xy.shape[1] - 1, -1, -1, device=batch.query_xy.device)
        reversed_batch = replace(batch, query_xy=batch.query_xy[:, reverse], query_features=batch.query_features[:, reverse])
        record["query_permutation"] = discrepancy(reference, _read(model, reversed_batch)[:, reverse])
        duplicate = replace(batch,
                            env_coords=batch.env_coords.repeat_interleave(2, dim=1),
                            env_features=batch.env_features.repeat_interleave(2, dim=1),
                            env_weights=batch.env_weights.repeat_interleave(2, dim=1) / 2)
        record["split_weight_duplication"] = discrepancy(reference, _read(model, duplicate))
        scale = batch.query_xy.new_tensor(model.config.coordinate_scale)
        record["environment_1024"] = discrepancy(reference, _read(model, _refined_environment(batch, scale)))
        corrupted_context = batch.global_context.clone()
        corrupted_context[:, -6:] = 0
        corrupted = replace(batch, global_context=corrupted_context,
                            query_features=torch.zeros_like(batch.query_features),
                            env_features=torch.zeros_like(batch.env_features))
        corrupted_prediction = _read(model, corrupted)
        record["zeroed_support_features"] = discrepancy(reference, corrupted_prediction)

        if model.architecture == "legacy_honf":
            route = model.core.decoder.pairwise_kernel
            hook = route.register_forward_hook(lambda _module, _args, output: (torch.zeros_like(output[0]), output[1], output[2]))
            route_name = "classic query-module pair context only; hyperedge and near routes retained"
        else:
            route = model.core.backend.query_module_output
            hook = route.register_forward_hook(lambda _module, _args, output: torch.zeros_like(output))
            route_name = "dense direct query-module context only; environment, coarse and local routes retained"
        try:
            muted = _read(model, batch)
        finally:
            hook.remove()
        record["muted_route"] = {"definition": route_name, **discrepancy(reference, muted)}
        if batch.target_field is not None:
            target = batch.target_field
            record["standardized_reference_mse"] = float((reference - target).square().mean())
            record["standardized_muted_route_mse"] = float((muted - target).square().mean())
            record["standardized_corrupted_support_mse"] = float((corrupted_prediction - target).square().mean())
        record["environment_volume_D3"] = float(batch.env_weights.sum())
        record["environment_cell_volume_D3"] = float(batch.env_weights.mean())
        records.append(record)
    return records
