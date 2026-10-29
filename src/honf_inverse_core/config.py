"""Strict root validation for inverse `R,c -> G -> D -> G_hat` workflows.

Modeling objects retain their typed runtime contracts; this helper only keeps
inverse dataset/training/evaluation configuration from silently accepting
misspelled or forward-only keys. It never widens the forward config loader.
"""

from __future__ import annotations

from typing import Any, Mapping, Set


def validate_config_keys(
    config: Mapping[str, Any],
    *,
    allowed: Set[str],
    required: Set[str],
    label: str,
) -> dict[str, Any]:
    unknown = sorted(set(config) - set(allowed))
    missing = sorted(set(required) - set(config))
    if unknown:
        raise ValueError(f"Unknown {label} config keys: {unknown}")
    if missing:
        raise ValueError(f"Missing {label} config keys: {missing}")
    return dict(config)


__all__ = ["validate_config_keys"]
