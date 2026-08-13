"""Dynamic loading of case plugins without hard-coded case branches."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from .case_protocol import CasePlugin


@dataclass(frozen=True)
class ModelFamilySpec:
    """Runtime availability and workflow surface of a reusable model family."""

    family_id: str
    available: bool
    workflows: tuple[str, ...]
    description: str


MODEL_FAMILIES = {
    "honf_forward": ModelFamilySpec(
        family_id="honf_forward",
        available=True,
        workflows=("forward", "local_module", "compare"),
        description="Hypergraph Operator Neural Field forward model",
    ),
    "honf_inverse": ModelFamilySpec(
        family_id="honf_inverse",
        available=False,
        workflows=("inverse",),
        description="Reserved inverse-design namespace; no implementation in version 0.1",
    ),
}


def require_model_family(family_id: str, workflow: str) -> ModelFamilySpec:
    """Reject unknown, unavailable, or workflow-incompatible model families."""

    try:
        spec = MODEL_FAMILIES[str(family_id)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model_family={family_id!r}; registered={sorted(MODEL_FAMILIES)}. "
            "Install/register a concrete baseline implementation before selecting it."
        ) from exc
    if not spec.available:
        raise NotImplementedError(spec.description)
    if workflow not in spec.workflows:
        raise ValueError(f"model_family={family_id!r} does not support workflow={workflow!r}.")
    return spec


def load_object(dotted_path: str) -> Any:
    """Load ``module:attribute`` and return the referenced object."""

    module_name, separator, attribute_name = str(dotted_path).partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(f"Expected a plugin factory in 'module:attribute' form; got {dotted_path!r}.")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute_name)
    except AttributeError as exc:
        raise ImportError(f"{module_name!r} has no attribute {attribute_name!r}.") from exc


def load_case_plugin(factory_path: str) -> CasePlugin:
    """Instantiate and structurally validate a configured case plugin."""

    factory = load_object(factory_path)
    plugin = factory()
    required = ("case_id", "display_name", "version", "validate_config", "inspect_launch", "train", "evaluate")
    missing = [name for name in required if not hasattr(plugin, name)]
    if missing:
        raise TypeError(f"Case plugin {factory_path!r} is missing required members: {missing}")
    return plugin
