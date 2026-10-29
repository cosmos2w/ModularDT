"""ThermalChannel operating-context contract for inverse design.

Physical design ``D`` is generated separately. Context ``c`` is the ten-value
operating/material/domain vector defined here. Request ``R`` conditions the
hierarchy, compact plan ``G`` is sampled from ``(R,c)``, and realized plan
``G_hat`` is verified under the same physical context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from honf_inverse_core.contracts import NamedContext


CONTEXT_SCHEMA_NAME = "thermalchannel_inverse_context"
CONTEXT_SCHEMA_VERSION = 1
CONTEXT_FEATURE_NAMES = (
    "re",
    "u_in",
    "nu",
    "solid_alpha",
    "fluid_alpha",
    "solid_k",
    "fluid_k",
    "module_radius",
    "domain_length_x",
    "domain_length_y",
)
_ROOT_KEYS = {"schema_name", "schema_version", *CONTEXT_FEATURE_NAMES}


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Context {name} must be finite.")
    return number


def parse_context(payload: Mapping[str, Any]) -> NamedContext:
    """Parse and strictly validate one context-schema-v1 mapping."""

    unknown = sorted(set(payload) - _ROOT_KEYS)
    missing = sorted(_ROOT_KEYS - set(payload))
    if unknown:
        raise ValueError(f"Unknown ThermalChannel context keys: {unknown}")
    if missing:
        raise ValueError(f"ThermalChannel context is missing keys: {missing}")
    if payload["schema_name"] != CONTEXT_SCHEMA_NAME or int(payload["schema_version"]) != 1:
        raise ValueError(
            f"Expected {CONTEXT_SCHEMA_NAME!r} schema_version=1; got "
            f"{payload['schema_name']!r} v{payload['schema_version']!r}."
        )
    values = np.asarray([_finite(payload[name], name=name) for name in CONTEXT_FEATURE_NAMES], dtype=np.float32)
    named = {name: float(values[index]) for index, name in enumerate(CONTEXT_FEATURE_NAMES)}
    if named["re"] < 0.0 or named["u_in"] < 0.0:
        raise ValueError("Context re and u_in must be nonnegative.")
    for name in (
        "nu",
        "solid_alpha",
        "fluid_alpha",
        "solid_k",
        "fluid_k",
        "module_radius",
        "domain_length_x",
        "domain_length_y",
    ):
        if named[name] <= 0.0:
            raise ValueError(f"Context {name} must be positive.")
    if 2.0 * named["module_radius"] >= named["domain_length_y"]:
        raise ValueError("Context module diameter must be smaller than the channel height.")
    return NamedContext(CONTEXT_FEATURE_NAMES, values, CONTEXT_SCHEMA_NAME, CONTEXT_SCHEMA_VERSION)


def load_context(source: Mapping[str, Any] | str | Path) -> NamedContext:
    """Load a context from a mapping or JSON path."""

    if isinstance(source, Mapping):
        return parse_context(source)
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Context JSON not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Context JSON root must be an object: {path}")
    return parse_context(payload)


def context_from_forward_structure(structure: Mapping[str, Any]) -> NamedContext:
    """Build ``c`` from the maintained raw forward structure mapping."""

    material = np.asarray(structure["material_params"], dtype=np.float64).reshape(-1)
    if material.shape[0] < 6:
        raise ValueError("Forward material_params must contain six ordered values.")
    scalar = lambda key: float(np.asarray(structure[key]).reshape(-1)[0])
    payload = {
        "schema_name": CONTEXT_SCHEMA_NAME,
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "re": scalar("re"),
        "u_in": scalar("u_in"),
        "nu": float(material[0]),
        "solid_alpha": float(material[1]),
        "fluid_alpha": float(material[2]),
        "solid_k": float(material[3]),
        "fluid_k": float(material[4]),
        "module_radius": float(material[5]),
        "domain_length_x": scalar("domain_length_x"),
        "domain_length_y": scalar("domain_length_y"),
    }
    return parse_context(payload)


def forward_material_params(context: NamedContext) -> np.ndarray:
    """Return the exact six-column forward material order from ``c``."""

    values = context.as_mapping()
    return np.asarray(
        [
            values["nu"],
            values["solid_alpha"],
            values["fluid_alpha"],
            values["solid_k"],
            values["fluid_k"],
            values["module_radius"],
        ],
        dtype=np.float32,
    )


__all__ = [
    "CONTEXT_FEATURE_NAMES",
    "CONTEXT_SCHEMA_NAME",
    "CONTEXT_SCHEMA_VERSION",
    "context_from_forward_structure",
    "forward_material_params",
    "load_context",
    "parse_context",
]
