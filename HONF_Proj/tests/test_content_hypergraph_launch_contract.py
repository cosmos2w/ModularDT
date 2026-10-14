"""Execution limits of the new family do not alter historical recipe guards."""

import importlib.util
import json
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


def wind_full_recipe():
    return {
        **recipe("full_train1000"),
        "dataset": "wind",
        "seed": 42,
        "primary_queries": 4096,
        "effective_batch_size": 24,
    }


def pooling_amendment():
    return {
        "schema": "content_hypergraph_pooling_numerical_repair_v1",
        "approved_stop_after": 1000,
    }


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


def test_fresh_wind_legacy_recipe_rejects_missing_optimizer_declaration_before_build(tmp_path, monkeypatch):
    data = {**recipe(), "dataset": "wind", "seed": 42, "primary_queries": 4096,
            "effective_batch_size": 24}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(launcher, "build", lambda *a, **k: pytest.fail("rejected recipe constructed data/model"))
    with pytest.raises(ValueError, match="explicitly enable optimizer_bias_norm_exclusion"):
        launcher.main(["--recipe", str(path), "--device", "cpu", "--dry-run"])


@pytest.mark.parametrize(
    ("data", "extra_args", "message"),
    [
        (recipe(), ["--resume", "latest_model.pt", "--stop-after", "100"], "limited to Wind A-H6 full_train1000"),
        (wind_full_recipe(), ["--resume", "latest_model.pt", "--stop-after", "1001"], "1..1000"),
        (wind_full_recipe(), ["--stop-after", "1000"], "requires --resume"),
        (wind_full_recipe(), ["--resume", "latest_model.pt", "--stop-after", "1000", "--dry-run"],
         "only valid for an exact scientific resume"),
    ],
)
def test_pooling_amendment_rejects_wrong_scope_stop_resume_and_dry_run_before_build(
    tmp_path, monkeypatch, data, extra_args, message
):
    recipe_path = tmp_path / "recipe.json"
    amendment_path = tmp_path / "pooling_amendment.json"
    recipe_path.write_text(json.dumps(data))
    amendment_path.write_text(json.dumps(pooling_amendment()))
    monkeypatch.setattr(launcher, "build", lambda *a, **k: pytest.fail("rejected amendment constructed data/model"))
    argv = ["--recipe", str(recipe_path), "--device", "cpu", "--numerical-pooling-amendment",
            str(amendment_path), *extra_args]
    if "--dry-run" not in extra_args:
        argv.extend(["--output", str(tmp_path / "run"), "--yes"])
    with pytest.raises(ValueError, match=message):
        launcher.main(argv)


def test_pooling_amendment_is_loaded_and_passed_only_to_qualified_resume(tmp_path, monkeypatch, capsys):
    recipe_path = tmp_path / "recipe.json"
    amendment_path = tmp_path / "pooling_amendment.json"
    recipe_path.write_text(json.dumps(wind_full_recipe()))
    amendment = pooling_amendment()
    amendment_path.write_text(json.dumps(amendment))
    captured = {}

    class Model:
        def parameters(self):
            return []

    class Provider:
        def identity_payload(self):
            return {"schema": "test"}

    class Engine:
        config = type("Config", (), {})()

        def fit(self, model, provider, output, **kwargs):
            captured["kwargs"] = kwargs
            return {"status": "accepted"}

    monkeypatch.setattr(launcher, "validate_qualification", lambda *_: {"qualified": True})
    monkeypatch.setattr(launcher, "build", lambda *a, **k: (Model(), Provider()))
    monkeypatch.setattr(launcher, "engine_for", lambda *a, **k: Engine())
    monkeypatch.setattr(launcher, "loaded_source_binding", lambda: {"sha256": "test"})
    result = launcher.main([
        "--recipe", str(recipe_path),
        "--device", "cpu",
        "--output", str(tmp_path / "run"),
        "--stop-after", "1000",
        "--resume", str(tmp_path / "latest_model.pt"),
        "--qualification", str(tmp_path / "qualification.json"),
        "--numerical-pooling-amendment", str(amendment_path),
        "--yes",
    ])
    assert result == 0
    assert captured["kwargs"]["numerical_pooling_amendment"] == amendment
    assert captured["kwargs"]["resume_checkpoint"] == tmp_path / "latest_model.pt"
    assert "optimizer_bias_norm_exclusion" not in json.loads(recipe_path.read_text())
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"status": "accepted"}


def test_pooling_amendment_rejects_unknown_schema_before_build(tmp_path, monkeypatch):
    recipe_path = tmp_path / "recipe.json"
    amendment_path = tmp_path / "pooling_amendment.json"
    recipe_path.write_text(json.dumps(wind_full_recipe()))
    amendment_path.write_text(json.dumps({"schema": "wrong"}))
    monkeypatch.setattr(launcher, "build", lambda *a, **k: pytest.fail("invalid amendment constructed data/model"))
    with pytest.raises(ValueError, match="Unsupported numerical pooling amendment schema"):
        launcher.main([
            "--recipe", str(recipe_path), "--device", "cpu", "--output", str(tmp_path / "run"),
            "--stop-after", "1000", "--resume", str(tmp_path / "latest_model.pt"),
            "--qualification", str(tmp_path / "qualification.json"),
            "--numerical-pooling-amendment", str(amendment_path), "--yes",
        ])
