"""Sealed Wind typed-centroid R1 recipe and provider bindings."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from windfarm.training.content_hypergraph_task import (
    _validate_recipe_v2,
    _validate_wind_v2_core_binding,
)

ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = ROOT / "src/config_core/forward/content_hypergraph"
MODE = "typed_w_centroids_v1"


def _recipe(name: str) -> dict:
    return json.loads((RECIPE_DIR / name).read_text())


def _core_config(recipe: dict) -> dict:
    architecture = recipe["architecture"]
    return {
        "schema": "content_hypergraph_core_v2",
        "module_input_dim": 2,
        "environment_input_dim": 7,
        "coordinate_dim": 3,
        "query_input_dim": 7,
        "operating_input_dim": 11,
        "output_dim": 3,
        "architecture": architecture,
        "context_dim": 128,
        "num_heads": 4,
        "context_depth": 2,
        "num_hyperedges": 6 if architecture == "A2-H6" else 0,
        "thermal_response_rank": 0,
        "local_hidden_dim": 128,
        "dropout": 0.0,
        "init_seed": 42,
        **({"query_edge_geometry_mode": MODE} if recipe["model"].get("query_edge_geometry_mode") else {}),
    }


@pytest.mark.parametrize(
    ("default_name", "r1_name"),
    [
        ("wind_a2-h6_development.json", "wind_a2-h6_r1_centroids_development.json"),
        ("wind_a2-h6_full1000.json", "wind_a2-h6_r1_centroids_full1000.json"),
    ],
)
def test_wind_r1_recipe_changes_only_h6_query_geometry_mode(default_name: str, r1_name: str) -> None:
    default = _recipe(default_name)
    candidate = _recipe(r1_name)
    stripped = copy.deepcopy(candidate)

    assert stripped["model"].pop("query_edge_geometry_mode") == MODE
    assert stripped == default
    validated, architecture, microbatch = _validate_recipe_v2(candidate)
    assert validated["model"]["query_edge_geometry_mode"] == MODE
    assert architecture == "A2-H6"
    assert microbatch == 12


@pytest.mark.parametrize("architecture", ["A2-D", "A2-H6"])
def test_wind_core_binding_accepts_only_recipe_matched_mode(architecture: str) -> None:
    name = "wind_a2-d_development.json" if architecture == "A2-D" else "wind_a2-h6_r1_centroids_development.json"
    recipe = _recipe(name)
    config = _core_config(recipe)

    _validate_wind_v2_core_binding(recipe, config)

    changed_width = dict(config, context_dim=64)
    with pytest.raises(ValueError, match="dimensions"):
        _validate_wind_v2_core_binding(recipe, changed_width)
    if architecture == "A2-D":
        with pytest.raises(ValueError, match="unsupported fields"):
            _validate_wind_v2_core_binding(
                recipe, {**config, "query_edge_geometry_mode": MODE},
            )
    else:
        with pytest.raises(ValueError, match="missing or unsupported fields"):
            _validate_wind_v2_core_binding(recipe, {key: value for key, value in config.items()
                                                       if key != "query_edge_geometry_mode"})


@pytest.mark.parametrize("mutation", ["unknown_model_flag", "loss_override", "wrong_centroid_mode", "direct_mode"])
def test_wind_r1_provider_rejects_unapproved_recipe_changes(mutation: str) -> None:
    recipe = _recipe("wind_a2-h6_r1_centroids_development.json")
    if mutation == "unknown_model_flag":
        recipe["model"]["reader_width"] = 64
    elif mutation == "loss_override":
        recipe["native_loss_override"] = {"velocity": 0.5}
    elif mutation == "wrong_centroid_mode":
        recipe["model"]["query_edge_geometry_mode"] = "typed_w_centroids_v2"
    else:
        recipe["architecture"] = "A2-D"
    with pytest.raises((ValueError, TypeError)):
        _validate_recipe_v2(recipe)

