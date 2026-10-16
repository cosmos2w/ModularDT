from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
for directory in (
    PROJECT / "src",
    PROJECT / "Case_ThermalChannel/src",
    PROJECT / "Case_WindFarm/src",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from honf_forward_core.inherited_attention.model import InheritedAttentionCoreV1
from honf_forward_core.inherited_attention.moment_model import MomentHyperedgeCoreV1
from honf_runtime.inherited_attention_protocol import validate_inherited_attention_recipe
from honf_runtime.moment_hyperedge_child import _tree_sha256
from honf_runtime.moment_hyperedge_factory import validate_moment_provider_pair
from honf_runtime.moment_hyperedge_protocol import (
    MOMENT_CORE_SCHEMA,
    MOMENT_FINAL_EPOCH,
    MOMENT_FIRST_SELECTION_EPOCH,
    MOMENT_SELECTION_INTERVAL,
    canonical_sha256,
    make_moment_hyperedge_recipe,
    moment_selection_eligible,
    native_recipe_for_moment_hyperedges,
    validate_moment_hyperedge_recipe,
)
from honf_runtime.unified_training import _reported_best_field_score

RECIPE_DIR = PROJECT / "src/config_core/forward/inherited_attention"


def _parent_recipe(dataset: str) -> dict:
    name = f"{dataset}_iag_c_soft_development_v1.json"
    return validate_inherited_attention_recipe(json.loads((RECIPE_DIR / name).read_text(encoding="utf-8")))


def _recipe(tmp_path: Path, dataset: str, variant: str) -> dict:
    source = tmp_path / f"{dataset}_parent_e500.pt"
    source.write_bytes(f"pinned-{dataset}-parent".encode())
    return make_moment_hyperedge_recipe(_parent_recipe(dataset), source, variant)


@pytest.mark.parametrize("dataset", ["thermal", "wind"])
@pytest.mark.parametrize("variant", ["C0", "M", "H"])
def test_round5_recipe_is_exact_native_iag_child(tmp_path: Path, dataset: str, variant: str):
    recipe = validate_moment_hyperedge_recipe(_recipe(tmp_path, dataset, variant))
    native = native_recipe_for_moment_hyperedges(recipe)
    parent = _parent_recipe(dataset)
    assert {key: value for key, value in native.items() if key != "model"} == {
        key: value for key, value in parent.items() if key != "model"
    }
    assert native["model"]["schema"] == MOMENT_CORE_SCHEMA
    assert native["model"]["query_executor"] == "indexed_projected"
    assert recipe["source_recipe_sha256"] == canonical_sha256(parent)


@pytest.mark.parametrize("dataset", ["thermal", "wind"])
def test_old_iag_schema_does_not_gain_projected_executor(tmp_path: Path, dataset: str):
    recipe = _parent_recipe(dataset)
    recipe["model"]["query_executor"] = "indexed_projected"
    with pytest.raises(ValueError, match="query_executor"):
        validate_inherited_attention_recipe(recipe)


@pytest.mark.parametrize("dataset,expected_born_count", [("thermal", 121696), ("wind", 122880)])
def test_round5_model_birth_layout_and_m_h_initialization(
    tmp_path: Path, dataset: str, expected_born_count: int,
):
    parent = _parent_recipe(dataset)
    old_config = {"schema": "inherited_attention_core_v1", "architecture": "IAG-C", **parent["model"]}
    old_model = InheritedAttentionCoreV1.from_config(old_config)
    old_parameters = dict(old_model.named_parameters())
    m_recipe = _recipe(tmp_path, dataset, "M")
    h_recipe = _recipe(tmp_path, dataset, "H")
    c0_recipe = _recipe(tmp_path, dataset, "C0")
    model_config = lambda recipe: {
        "schema": MOMENT_CORE_SCHEMA,
        "architecture": "IAG-C",
        **recipe["model"],
    }
    c0 = MomentHyperedgeCoreV1.from_config(model_config(c0_recipe))
    m = MomentHyperedgeCoreV1.from_config(model_config(m_recipe))
    h = MomentHyperedgeCoreV1.from_config(model_config(h_recipe))

    assert dict(c0.named_parameters()).keys() == old_parameters.keys()
    assert {name: tuple(value.shape) for name, value in c0.named_parameters()} == {
        name: tuple(value.shape) for name, value in old_parameters.items()
    }
    m_parameters = dict(m.named_parameters())
    h_parameters = dict(h.named_parameters())
    born = sorted(set(m_parameters) - set(old_parameters))
    assert born == sorted(set(h_parameters) - set(old_parameters))
    assert sum(m_parameters[name].numel() for name in born) == expected_born_count
    for name in born:
        assert torch.equal(m_parameters[name], h_parameters[name]), name
        assert m_parameters[name].requires_grad


@pytest.mark.parametrize("dataset", ["thermal", "wind"])
def test_provider_identity_allows_only_enumerated_round5_core_fields(
    tmp_path: Path, dataset: str,
):
    parent_recipe = _parent_recipe(dataset)
    recipe = _recipe(tmp_path, dataset, "M")
    native_child = native_recipe_for_moment_hyperedges(recipe)
    old_config = {"schema": "inherited_attention_core_v1", "architecture": "IAG-C", **parent_recipe["model"]}
    new_config = {"schema": MOMENT_CORE_SCHEMA, "architecture": "IAG-C", **recipe["model"]}
    parent_provider = {
        "recipe": parent_recipe,
        "dataset_membership_sha256": "a" * 64,
        "normalizer_sha256": "b" * 64,
    }
    child_provider = copy.deepcopy(parent_provider)
    child_provider["recipe"] = native_child
    if dataset == "thermal":
        parent_provider.update(model_config={"core": old_config}, adapter_state={"core": old_config})
        child_provider.update(model_config={"core": new_config}, adapter_state={"core": new_config})
    else:
        parent_provider.update(model_config=old_config, adapter_state={"normalizer": {"sha256": "c" * 64}})
        child_provider.update(model_config=new_config, adapter_state={"normalizer": {"sha256": "c" * 64}})

    binding = validate_moment_provider_pair(
        parent_provider, child_provider, dataset=dataset, recipe=recipe,
    )
    assert binding["schema"] == "moment_hyperedge_native_provider_binding_v1"
    changed = copy.deepcopy(child_provider)
    changed["normalizer_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="provider identity changed"):
        validate_moment_provider_pair(parent_provider, changed, dataset=dataset, recipe=recipe)


def test_selection_ages_are_only_saved_common_ages_600_through_2500():
    selected = [epoch for epoch in range(1, MOMENT_FINAL_EPOCH + 1) if moment_selection_eligible(epoch)]
    expected = list(range(MOMENT_FIRST_SELECTION_EPOCH, MOMENT_FINAL_EPOCH + 1, MOMENT_SELECTION_INTERVAL))
    assert selected == expected
    assert not moment_selection_eligible(500)
    assert not moment_selection_eligible(501)
    assert not moment_selection_eligible(550)
    assert not moment_selection_eligible(2501)


def test_round5_preselection_summary_is_json_safe_without_changing_old_behavior():
    score, status = _reported_best_field_score(float("inf"), moment_hyperedge=True)
    assert score is None
    assert status == "not_yet_eligible"
    json.dumps({"best_field_score": score, "status": status}, allow_nan=False)
    old_score, old_status = _reported_best_field_score(float("inf"), moment_hyperedge=False)
    assert old_score == float("inf")
    assert old_status is None


def test_tree_state_digest_handles_scalar_optimizer_steps():
    before = _tree_sha256({"step": torch.tensor(500.0), "exp_avg": torch.ones(2)})
    after = _tree_sha256({"step": torch.tensor(501.0), "exp_avg": torch.ones(2)})
    assert before != after
