"""Source-read scopes do not inherit historical full-D/H6 permissions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from honf_runtime.source_read_protocol import (
    SOURCE_READ_FULL, validate_source_read_recipe, validate_source_read_stop,
)

ROOT = Path(__file__).resolve().parents[1]


def recipe(dataset="wind", scope="development", variant="d0"):
    path = ROOT / "src/config_core/forward/content_hypergraph" / (
        f"{dataset}_a2-d_source_read_{scope}_{variant}_v1.json")
    return json.loads(path.read_text())


@pytest.mark.parametrize("dataset", ["thermal", "wind"])
def test_source_read_stops_and_formal_semantics(dataset):
    from content_hypergraph_train import engine_for, validate_recipe, validate_stop
    value = recipe(dataset)
    assert validate_recipe(value) == value
    engine = engine_for(value, "cpu")
    assert engine.config.seed == (0 if dataset == "thermal" else 42)
    assert engine.config.monitor_every == 100
    assert engine.config.total_epochs == 2500
    assert engine.config.curve_every == 10
    assert engine.selection.field_metric == ("field_score" if dataset == "thermal" else "val_volume_mse")
    validate_stop(value, 2500)
    full = recipe(dataset, "full1000", "d_scale")
    assert full["protocol"] == SOURCE_READ_FULL
    validate_source_read_stop(full, 1000)
    for stop in (1001, 2500, 5000, True):
        with pytest.raises(ValueError, match="execution stop"):
            validate_stop(full, stop)


@pytest.mark.parametrize("key,value", [
    ("seed", 42), ("primary_queries", 4096), ("effective_batch_size", 24),
    ("optimizer_bias_norm_exclusion", True), ("approved_stop_after", 5000),
])
def test_old_wind_development_settings_do_not_enter_new_scope(key, value):
    changed = recipe()
    changed[key] = value
    with pytest.raises(ValueError):
        validate_source_read_recipe(changed)


def test_scale_conditioning_and_sparse_declarations_are_not_silent_aliases():
    changed = recipe()
    changed["source_reader_variant"] = "D-scale"
    with pytest.raises(ValueError, match="conditioning"):
        validate_source_read_recipe(changed)
    changed["source_reader_variant"] = "S-Utility"
    with pytest.raises(ValueError, match="policy"):
        validate_source_read_recipe(changed)
    changed = recipe("wind", "full1000", "d_scale")
    changed["source_reader_variant"] = "D0"
    with pytest.raises(ValueError, match="Unchanged D0"):
        validate_source_read_recipe(changed)


@pytest.mark.parametrize("dataset", ["thermal", "wind"])
def test_full1001_rejects_before_constructor_or_run_allocation(dataset, monkeypatch, tmp_path):
    import content_hypergraph_train as cli
    value = recipe(dataset, "full1000", "d_scale")
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(value))
    calls = []
    monkeypatch.setattr(cli, "build", lambda *a, **k: calls.append(1))
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="execution stop"):
        cli.main(["--recipe", str(path), "--stop-after", "1001", "--output", str(output), "--dry-run"])
    assert calls == [] and not output.exists()
