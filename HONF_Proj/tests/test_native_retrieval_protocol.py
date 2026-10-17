from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from honf_runtime import native_retrieval_protocol as protocol
from honf_runtime.native_retrieval_child import verify_native_retrieval_resume

ROOT = Path(__file__).resolve().parents[1]


def _recipe(monkeypatch, tmp_path, *, dataset="thermal", treatment="G-native-M", backend="selected", tau_q=0.20):
    base = json.loads((ROOT / "src/config_core/forward/content_hypergraph" /
                       f"{dataset}_a2-d_source_read_development_d0_v1.json").read_text())
    source = tmp_path / "d0_e500.pt"
    source.touch()
    identity = {
        "recipe": base,
        "recipe_sha256": protocol.canonical_sha256(base),
        "provider_identity": {"dataset": dataset, "subset_manifest_sha256": "a" * 64},
    }
    payload = {"epoch": 500}
    monkeypatch.setattr(
        protocol, "_source_contract",
        lambda _path: (copy.deepcopy(payload), copy.deepcopy(identity), "b" * 64),
    )
    return protocol.make_native_retrieval_recipe(
        source, treatment=treatment, native_backend=backend, tau_q=tau_q,
    )


@pytest.mark.parametrize("dataset", ["thermal", "wind"])
def test_recipe_binds_native_no_e_treatment_to_exact_d0_child(monkeypatch, tmp_path, dataset):
    recipe = _recipe(monkeypatch, tmp_path, dataset=dataset)
    assert recipe["source_epoch"] == 500
    assert recipe["treatment"] == "G-native-M"
    assert recipe["native_backend"] == "selected"
    assert recipe["environment_value_reader"] is False
    assert recipe["permission_curriculum"]["exploration_replaces_sparse_epoch"] is True
    assert protocol.validate_native_retrieval_recipe(recipe) == recipe
    child = protocol.native_retrieval_child_declaration(recipe)
    assert child["absolute_epoch_policy"] == "inherited_500_plus_2000_new_epochs"
    assert child["optimizer_policy"] == "exact_named_adamw_state_and_schedule_continuation"


def test_d_control_is_the_unchanged_full_direct_arm(monkeypatch, tmp_path):
    recipe = _recipe(monkeypatch, tmp_path, treatment="D-control", backend="full_direct", tau_q=0.0)
    assert recipe["native_tau_mm"] == recipe["native_tau_me"] == recipe["native_tau_q"] == 0.0
    with pytest.raises(ValueError, match="original full"):
        _recipe(monkeypatch, tmp_path, treatment="D-control", backend="selected", tau_q=0.20)


@pytest.mark.parametrize("field,value", [
    ("environment_value_reader", True),
    ("native_backend", "unregistered"),
    ("architecture", "A2-H6"),
])
def test_strict_recipe_rejects_scope_expansion(monkeypatch, tmp_path, field, value):
    recipe = _recipe(monkeypatch, tmp_path)
    recipe[field] = value
    with pytest.raises(ValueError):
        protocol.validate_native_retrieval_recipe(recipe)


def test_exact_resume_requires_absolute_age_buffer_and_same_child_contract(monkeypatch, tmp_path):
    recipe = _recipe(monkeypatch, tmp_path)
    declaration = protocol.native_retrieval_child_declaration(recipe)
    epoch_key = "core.native_training_epoch" if recipe["dataset"] == "thermal" else "native_training_epoch"
    payload = {
        "epoch": 501,
        "branch_parent": {"native_retrieval_child": declaration},
        "model_state_dict": {epoch_key: 501},
    }
    verify_native_retrieval_resume(payload, recipe=recipe, stop_after=600)
    payload["model_state_dict"][epoch_key] = 500
    with pytest.raises(ValueError, match="stale"):
        verify_native_retrieval_resume(payload, recipe=recipe, stop_after=600)
    payload["model_state_dict"][epoch_key] = 501
    payload["branch_parent"]["native_retrieval_child"]["source_epoch"] = 499
    with pytest.raises(ValueError, match="declaration"):
        verify_native_retrieval_resume(payload, recipe=recipe, stop_after=600)
