"""Audited parameter initialization for a three-term full-access refit.

The source-conditioned checkpoint already uses the three-term field head, so
its common and fine tensors can initialize the new operator. Its organizer and
control tensors are deliberately discarded. Dense checkpoints use a different
field head: only their encoder and fine-reader tensors are copied. Neither
transition promises equal predictions before refitting.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

_SOURCES = {"source_conditioned_pairwise_honf", "dense_pairwise_field"}
_DENSE_PREFIXES = (
    "global_encoder.",
    "module_feature_encoder.",
    "module_position_encoder.",
    "env_encoder.",
    "backend.",
)


def warm_start_three_term_full_access(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    source_architecture: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build a strict-loadable target state plus a complete transfer inventory.

    Both models must first be materialized with their native case adapters.
    The returned mapping is passed to ``load_state_dict(..., strict=True)``.
    A missing or shape-mismatched eligible tensor is an error, never a partial
    load. ``initialized`` names retain the caller's freshly initialized value.
    """

    if source_architecture not in _SOURCES:
        raise ValueError(f"unsupported source architecture: {source_architecture!r}")
    if not source_state or not target_state:
        raise ValueError("source and target states must be materialized and nonempty")
    result: dict[str, torch.Tensor] = {}
    copied: list[str] = []
    initialized: list[str] = []
    for name, target in target_state.items():
        if isinstance(target, (UninitializedParameter, UninitializedBuffer)):
            raise TypeError(f"target tensor is unmaterialized: {name}")
        eligible = (
            source_architecture == "source_conditioned_pairwise_honf"
            or name.startswith(_DENSE_PREFIXES)
        )
        if not eligible:
            result[name] = target.detach().clone()
            initialized.append(name)
            continue
        source = source_state.get(name)
        if source is None or isinstance(source, (UninitializedParameter, UninitializedBuffer)):
            raise ValueError(f"missing or unmaterialized source tensor: {name}")
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(
                f"source tensor {name} has shape/dtype {tuple(source.shape)}/{source.dtype}, "
                f"expected {tuple(target.shape)}/{target.dtype}"
            )
        result[name] = source.detach().clone()
        copied.append(name)
    inventory = {
        "source_architecture": source_architecture,
        "copied": tuple(copied),
        "initialized": tuple(initialized),
        "discarded_source": tuple(sorted(set(source_state) - set(copied))),
        "prediction_identity_claim": False,
    }
    return result, inventory


__all__ = ["warm_start_three_term_full_access"]
