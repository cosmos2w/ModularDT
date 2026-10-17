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
    "inherited_attention_v1": ModelFamilySpec(
        family_id="inherited_attention_v1",
        available=True,
        workflows=("inherited_attention_train", "inherited_attention_evaluate"),
        description="Dataset-native IAG-C inherited-gathering operator with geometry-derived receiver covers",
    ),
    "moment_hyperedges_round5": ModelFamilySpec(
        family_id="moment_hyperedges_round5",
        available=True,
        workflows=("moment_hyperedges_train", "moment_hyperedges_evaluate"),
        description="Round5 scene-prepared signed-moment hyperedge continuation on native IAG-C providers",
    ),
    "hyperedge_readout_round6": ModelFamilySpec(
        family_id="hyperedge_readout_round6",
        available=True,
        workflows=("hyperedge_readout_train", "hyperedge_readout_evaluate"),
        description="Round6 matched receiver-conditioned readouts over completed H hyperedge states",
    ),
    "native_retrieval_v1": ModelFamilySpec(
        family_id="native_retrieval_v1",
        available=True,
        workflows=("native_retrieval_train", "native_retrieval_evaluate"),
        description="Round7 matched A2-D continuations with typed native-token retrieval and strict no-E primary access",
    ),
    "native_budgeted_dispatch_v2": ModelFamilySpec(
        family_id="native_budgeted_dispatch_v2",
        available=True,
        workflows=("native_budgeted_dispatch_train", "native_budgeted_dispatch_evaluate"),
        description="Round8 matched D0 continuations with learned budgeted native hyperedge dispatch and strict no-E primary access",
    ),
    "honf_content_hypergraph": ModelFamilySpec(
        family_id="honf_content_hypergraph",
        available=True,
        workflows=("forward", "compare"),
        description="Typed attention with content-bearing hypergraph decoding and native adapters",
    ),
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
