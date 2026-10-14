"""Execution limits of the new family do not alter historical recipe guards."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools/content_hypergraph_train.py"
_SPEC = importlib.util.spec_from_file_location("content_hypergraph_train", _PATH)
launcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(launcher)


def recipe(protocol="development"):
    return {"schema": launcher.SCHEMA, "dataset": "thermal", "architecture": "A-H6", "seed": 0,
            "primary_queries": 1024, "effective_batch_size": 48, "microbatch_size": 8,
            "protocol": protocol, "total_epochs": 5000 if protocol == "full_train1000" else 2500,
            "approved_stop_after": 1000}


def test_full_schedule_never_authorizes_5000_execution():
    value = launcher.validate_recipe(recipe("full_train1000"))
    launcher.validate_stop(value, 1000)
    with pytest.raises(ValueError, match="1..1000"):
        launcher.validate_stop(value, 5000)


@pytest.mark.parametrize("name,value", [("seed", 42), ("primary_queries", 4096),
                                         ("edge_access", "top3"), ("precision", "bf16")])
def test_primary_scientific_identity_rejects_silent_recipe_changes(name, value):
    data = recipe()
    data[name] = value
    with pytest.raises(ValueError):
        launcher.validate_recipe(data)


def test_full_followup_requires_organized_model_and_explicit_stop():
    data = recipe("full_train1000")
    data["architecture"] = "A-D"
    with pytest.raises(ValueError, match="organized"):
        launcher.validate_recipe(data)
    data["architecture"] = "A-H6"
    del data["approved_stop_after"]
    with pytest.raises(ValueError, match="1000 stop"):
        launcher.validate_recipe(data)
