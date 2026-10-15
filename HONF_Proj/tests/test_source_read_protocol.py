"""Source-read scopes do not inherit historical full-D/H6 permissions."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import pytest

from honf_runtime.source_read_protocol import (
    SOURCE_READ_FULL, SOURCE_READ_NUMERICS, configure_source_read_numerics,
    validate_source_read_recipe, validate_source_read_stop,
)

ROOT = Path(__file__).resolve().parents[1]


def recipe(dataset="wind", scope="development", variant="d0"):
    path = ROOT / "src/config_core/forward/content_hypergraph" / (
        f"{dataset}_a2-d_source_read_{scope}_{variant}_v1.json")
    return json.loads(path.read_text())


def sparse_recipe(dataset="wind", variant="S-Utility"):
    value = recipe(dataset)
    value["source_reader_variant"] = variant
    value["dense_host_variant"] = "D0"
    value["model"].update(
        source_read_policy="utility" if variant == "S-Utility" else "completion",
        source_read_executor="packed",
        source_read_rank=16,
        source_read_scorer_width=32,
    )
    return value


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


def test_wind_frozen_whole_query_execution_replays_only_its_inert8192_field():
    value = recipe("wind")
    value["model"]["reader_query_chunk_size"] = 8192
    assert validate_source_read_recipe(value) == value
    value["model"]["reader_query_chunk_size"] = 512
    with pytest.raises(ValueError, match="whole-query"):
        validate_source_read_recipe(value)


@pytest.mark.parametrize("variant", ["S-Utility", "S-Complete"])
def test_deterministic_cuda_numerics_is_a_top_level_sparse_recipe_option(variant):
    value = sparse_recipe(variant=variant)
    value["source_read_numerics"] = SOURCE_READ_NUMERICS
    assert validate_source_read_recipe(value) == value

    changed = sparse_recipe(variant=variant)
    assert canonical_recipe_sha256(changed) != canonical_recipe_sha256(value)


def canonical_recipe_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@pytest.mark.parametrize("variant", ["D0", "D-scale"])
def test_deterministic_cuda_numerics_is_rejected_for_dense_variants(variant):
    value = recipe()
    value["source_reader_variant"] = variant
    value["source_read_numerics"] = SOURCE_READ_NUMERICS
    with pytest.raises(ValueError, match="only to sparse children"):
        validate_source_read_recipe(value)


@pytest.mark.parametrize("setting", [None, "deterministic_cuda_v2", True])
def test_deterministic_cuda_numerics_rejects_undeclared_values(setting):
    value = sparse_recipe()
    value["source_read_numerics"] = setting
    with pytest.raises(ValueError, match="only to sparse children"):
        validate_source_read_recipe(value)


def test_configure_deterministic_cuda_numerics_is_opt_in_and_restores_cleanly(monkeypatch):
    import torch

    value = sparse_recipe()
    prior = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        assert not configure_source_read_numerics(value)
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        assert torch.are_deterministic_algorithms_enabled() == prior["deterministic"]
        assert torch.backends.cuda.matmul.allow_tf32 == prior["matmul_tf32"]
        assert torch.backends.cudnn.allow_tf32 == prior["cudnn_tf32"]
        assert torch.backends.cudnn.deterministic == prior["cudnn_deterministic"]
        assert torch.backends.cudnn.benchmark == prior["cudnn_benchmark"]

        value["source_read_numerics"] = SOURCE_READ_NUMERICS
        assert configure_source_read_numerics(value)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
    finally:
        torch.use_deterministic_algorithms(prior["deterministic"], warn_only=prior["warn_only"])
        torch.backends.cuda.matmul.allow_tf32 = prior["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = prior["cudnn_tf32"]
        torch.backends.cudnn.deterministic = prior["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = prior["cudnn_benchmark"]


def test_configure_deterministic_cuda_numerics_rejects_conflicting_workspace_before_backend_change(monkeypatch):
    import torch

    value = sparse_recipe()
    value["source_read_numerics"] = SOURCE_READ_NUMERICS
    previous = torch.are_deterministic_algorithms_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_source_read_numerics(value)
    assert torch.are_deterministic_algorithms_enabled() == previous


@pytest.mark.parametrize("field,value", [
    ("protocol", "historical_formal_a2d"),
    ("architecture", "A2-H6"),
])
def test_misattached_deterministic_numerics_reject_before_mutating_runtime(field, value, monkeypatch):
    import torch

    recipe_value = sparse_recipe()
    recipe_value["source_read_numerics"] = SOURCE_READ_NUMERICS
    recipe_value[field] = value
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with pytest.raises(ValueError, match="limited to source-read A2-D"):
        configure_source_read_numerics(recipe_value)

    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    ) == previous


def test_absent_numerics_on_historical_recipe_remains_untouched(monkeypatch):
    import torch

    historical_recipe = {"protocol": "formal_a2d", "architecture": "A2-H6"}
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    assert not configure_source_read_numerics(historical_recipe)

    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    ) == previous


def test_training_and_standalone_loaders_apply_numerics_before_model_construction(monkeypatch):
    import content_hypergraph_train as cli
    import honf_runtime.content_hypergraph as loader
    import channelthermal.training.content_hypergraph_task_v2 as thermal_task

    value = sparse_recipe("thermal")
    value["source_read_numerics"] = SOURCE_READ_NUMERICS
    events = []
    monkeypatch.setattr(cli, "configure_source_read_numerics", lambda recipe: events.append("train_numerics"))
    monkeypatch.setattr(
        thermal_task, "build_thermal_source_read_task_v1",
        lambda recipe, device: (events.append("train_model") or object(), object()),
    )
    cli.build(value, device="cpu")
    assert events == ["train_numerics", "train_model"]

    class _Model:
        def get_config(self):
            return {}

        def load_state_dict(self, state, strict=True):
            assert state == {} and strict

        def eval(self):
            return self

    from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2

    recipe_hash = cli.canonical_sha256(value)
    payload = {
        "workflow": "content_hypergraph_joint_v2",
        "arm": "A2-D",
        "model_state_dict": {},
        "experiment_identity": {
            "model_family": "honf_content_hypergraph",
            "recipe": value,
            "recipe_sha256": recipe_hash,
            "provider_identity": {"model_config": {}, "adapter_state": {}},
        },
    }
    events.clear()
    monkeypatch.setattr(loader, "load_trusted_checkpoint", lambda *args, **kwargs: payload)
    monkeypatch.setattr(loader, "require_model_family", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "configure_source_read_numerics", lambda recipe: events.append("load_numerics"))
    monkeypatch.setattr(
        ThermalContentHypergraphAdapterV2,
        "from_state_payload",
        classmethod(lambda cls, state, device: (events.append("load_model") or _Model())),
    )
    loader.load_content_hypergraph_checkpoint("unused.pt", device="cpu")
    assert events == ["load_numerics", "load_model"]


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
