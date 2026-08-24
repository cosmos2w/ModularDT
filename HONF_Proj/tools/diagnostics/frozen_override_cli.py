"""Shared CLI support for prediction-preserving frozen arithmetic screens."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Iterable

from channelthermal.workflows.evaluate_forward import apply_frozen_forward_overrides


LOCALITY_MODES = {
    "none",
    "compact_kernel",
    "bounded_gaussian",
    "gaussian_bounded",
    "inherit_environment",
}


def add_frozen_override_arguments(parser: argparse.ArgumentParser) -> None:
    """Add label-scoped Stage-6 frozen-screen arguments to ``parser``."""

    parser.add_argument(
        "--mechanism-residual-scale",
        action="append",
        default=[],
        metavar="LABEL=FLOAT",
        help="Evaluation-only descriptor-first content-residual scale.",
    )
    parser.add_argument(
        "--query-locality-mode-override",
        action="append",
        default=[],
        metavar="LABEL=MODE",
        help="Evaluation-only query-locality mode.",
    )
    parser.add_argument(
        "--query-locality-strength-override",
        action="append",
        default=[],
        metavar="LABEL=FLOAT",
        help="Evaluation-only independent query-locality strength.",
    )


def _labeled_specs(
    values: Iterable[str],
    *,
    converter: Callable[[str], Any],
    description: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected LABEL={description}, got {item!r}")
        label, raw_value = item.split("=", 1)
        cleaned = label.strip()
        if not cleaned:
            raise ValueError(f"Frozen override label is empty in {item!r}")
        if cleaned in result:
            raise ValueError(f"Duplicate frozen override for label {cleaned!r}")
        result[cleaned] = converter(raw_value.strip())
    return result


def resolve_frozen_overrides(args: argparse.Namespace, labels: Iterable[str]) -> None:
    """Parse overrides, validate label coverage, and attach maps to ``args``."""

    args.mechanism_scale_overrides = _labeled_specs(
        args.mechanism_residual_scale,
        converter=float,
        description="FLOAT",
    )
    args.query_locality_mode_overrides = _labeled_specs(
        args.query_locality_mode_override,
        converter=str,
        description="MODE",
    )
    args.query_locality_strength_overrides = _labeled_specs(
        args.query_locality_strength_override,
        converter=float,
        description="FLOAT",
    )
    known = set(labels)
    referenced = (
        set(args.mechanism_scale_overrides)
        | set(args.query_locality_mode_overrides)
        | set(args.query_locality_strength_overrides)
    )
    unknown = sorted(referenced - known)
    if unknown:
        raise ValueError(f"Frozen overrides reference unknown checkpoint labels: {unknown}")
    bad_modes = sorted(
        {
            value
            for value in args.query_locality_mode_overrides.values()
            if value not in LOCALITY_MODES
        }
    )
    if bad_modes:
        raise ValueError(f"Unsupported query-locality modes: {bad_modes}")


def apply_label_frozen_overrides(
    model: Any,
    label: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Apply the resolved override values for one checkpoint label."""

    return apply_frozen_forward_overrides(
        model,
        mechanism_latent_residual_scale=args.mechanism_scale_overrides.get(label),
        query_locality_mode=args.query_locality_mode_overrides.get(label),
        query_locality_strength=args.query_locality_strength_overrides.get(label),
    )
