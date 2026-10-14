"""Literal-age/native improvement decisions and pre-allocation execution limits."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from honf_runtime.content_hypergraph_qualification import FIRST_DIRECT, field_decision, recipe_architecture

_PATH = Path(__file__).resolve().parents[1] / "tools/content_hypergraph_train.py"
_SPEC = importlib.util.spec_from_file_location("round2_launcher", _PATH)
launcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(launcher)


def recipe(dataset="thermal", full=False):
    thermal = dataset == "thermal"
    value = {"schema": "content_hypergraph_recipe_v2", "dataset": dataset, "architecture": "A2-H6",
             "protocol": "full_train1000" if full else "development", "seed": 0 if thermal else 42,
             "primary_queries": 1024 if thermal else 4096, "effective_batch_size": 48 if thermal else 24,
             "microbatch_size": 24 if thermal else 12, "total_epochs": 5000 if full else 2500,
             "precision": "fp32", "edge_access": "full", "optimizer_bias_norm_exclusion": True,
             "model": {"context_dim": 128, "num_heads": 4, "context_depth": 2, "num_hyperedges": 6,
                       "thermal_response_rank": 4 if thermal else 0, "local_hidden_dim": 128, "dropout": 0.0}}
    if full:
        value["approved_stop_after"] = 1000
    return value


def thermal_record(architecture, fluid, score):
    return {"architecture": architecture, "epoch": 2500,
            "checkpoint_sha256": FIRST_DIRECT["thermal"] if architecture == "A-D" else architecture,
            "field_score": score,
            "rows": [{"case_id": str(index), "module_count": 3,
                      "roles": {role: {"rmse": fluid if role == "temperature_fluid" else 1.0}
                                for role in ("u", "v", "p", "omega", "temperature_fluid",
                                             "temperature_surface", "temperature_material")},
                      "q_proxy_rmse": 1.0} for index in range(22)]}


def fields(candidate=None, direct=None, baseline=None):
    return field_decision("thermal", candidate or thermal_record("A2-H6", 2.5, .08),
                          direct or thermal_record("A2-D", 2.6, .09),
                          baseline or thermal_record("A-D", 3.0, .10),
                          floors={name: 1e-8 for name in ("field_score", "u", "v", "p", "omega",
                                  "temperature_fluid", "temperature_surface", "temperature_material", "q_proxy")})


def test_field_decision_uses_coherent_literal_age_priority_and_nonpriority_roles():
    assert fields()["qualified"]
    assert not fields(candidate=thermal_record("A2-H6", 2.71, .08))["qualified"]
    candidate = thermal_record("A2-H6", 2.5, .08)
    for row in candidate["rows"]:
        row["roles"]["v"]["rmse"] = 1.251
    result = fields(candidate=candidate)
    assert not result["qualified"]
    assert result["checks"]["nonpriority/v"] is False


def test_field_decision_keeps_casewise_p90_and_panel_membership():
    candidate = thermal_record("A2-H6", 1.0, .08)
    for row in candidate["rows"][-4:]:
        row["roles"]["temperature_fluid"]["rmse"] = 4.0
    result = fields(candidate=candidate)
    assert result["checks"]["priority_ten_percent_gain"]
    assert not result["checks"]["p90/temperature_fluid"]
    assert "not a pointwise" in result["summaries"]["organized"]["p90_definition"]
    candidate["rows"][0]["case_id"] = "different"
    with pytest.raises(ValueError, match="memberships"):
        fields(candidate=candidate)


@pytest.mark.parametrize("mutation", ["epoch", "checkpoint"])
def test_field_decision_rejects_selected_age_and_incoherent_baseline(mutation):
    baseline = thermal_record("A-D", 3.0, .1)
    baseline["epoch" if mutation == "epoch" else "checkpoint_sha256"] = 2100 if mutation == "epoch" else "oracle"
    with pytest.raises(ValueError, match="literal2500|coherent"):
        fields(baseline=baseline)


@pytest.mark.parametrize("dataset", ["thermal", "wind"])
def test_v2_full5000_is_rejected_before_any_model_or_dataset_allocation(tmp_path, monkeypatch, dataset):
    value = recipe(dataset, full=True)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(launcher, "build", lambda *a, **k: pytest.fail("model/data allocation before stop check"))
    with pytest.raises(ValueError, match="1..1000"):
        launcher.main(["--recipe", str(path), "--stop-after", "5000", "--dry-run"])
    assert not (tmp_path / "run").exists()


def test_v2_recipe_versions_and_shared_capacity_are_explicit():
    value = recipe()
    assert launcher.validate_recipe(value) == value
    for key, bad in (("architecture", "A-H6"), ("optimizer_bias_norm_exclusion", False)):
        changed = {**value, key: bad}
        with pytest.raises(ValueError):
            launcher.validate_recipe(changed)
    changed = copy.deepcopy(value)
    changed["model"]["local_hidden_dim"] = 64
    with pytest.raises(ValueError, match="local_hidden_dim"):
        launcher.validate_recipe(changed)
    assert recipe_architecture(value) == recipe_architecture(recipe(full=True))


@pytest.mark.parametrize("dataset", ["thermal", "wind"])
def test_typed_centroid_mode_is_explicit_h6_only_and_architecture_bound(dataset):
    baseline = recipe(dataset)
    typed_development = copy.deepcopy(baseline)
    typed_development["model"]["query_edge_geometry_mode"] = "typed_w_centroids_v1"
    typed_full = recipe(dataset, full=True)
    typed_full["model"]["query_edge_geometry_mode"] = "typed_w_centroids_v1"

    assert launcher.validate_recipe(typed_development) == typed_development
    assert launcher.validate_recipe(typed_full) == typed_full
    assert recipe_architecture(typed_development) == recipe_architecture(typed_full)
    assert recipe_architecture(typed_development) != recipe_architecture(baseline)

    direct = copy.deepcopy(typed_development)
    direct["architecture"] = "A2-D"
    with pytest.raises(ValueError, match="only for A2-H6"):
        launcher.validate_recipe(direct)
    for mode in (None, "unknown", "typed_w_centroids_v2"):
        invalid = copy.deepcopy(typed_development)
        invalid["model"]["query_edge_geometry_mode"] = mode
        with pytest.raises(ValueError, match="query_edge_geometry_mode"):
            launcher.validate_recipe(invalid)


def test_thermal_r1_recipes_change_only_the_explicit_query_geometry_mode():
    root = Path(__file__).resolve().parents[1] / "src/config_core/forward/content_hypergraph"
    pairs = (
        ("thermal_a2-h6_development.json", "thermal_a2-h6_r1_centroids_development.json"),
        ("thermal_a2-h6_full1000.json", "thermal_a2-h6_r1_centroids_full1000.json"),
    )
    recipes = []
    for baseline_name, candidate_name in pairs:
        baseline = json.loads((root / baseline_name).read_text())
        candidate = json.loads((root / candidate_name).read_text())
        candidate_without_mode = copy.deepcopy(candidate)
        assert candidate_without_mode["model"].pop("query_edge_geometry_mode") == "typed_w_centroids_v1"
        assert candidate_without_mode == baseline
        assert launcher.validate_recipe(candidate) == candidate
        recipes.append(candidate)
    assert recipe_architecture(recipes[0]) == recipe_architecture(recipes[1])
