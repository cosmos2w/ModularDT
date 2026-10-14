"""The formal A2-D contract is distinct from development and H6 follow-up."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from honf_runtime.content_hypergraph_formal import (
    FORMAL_PROTOCOL,
    validate_formal_d_recipe,
    validate_formal_d_stop,
)

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT / "tools"))
import content_hypergraph_train as train_cli


def recipe(dataset: str) -> dict:
    thermal = dataset == "thermal"
    model = {
        "context_dim": 128,
        "num_heads": 4,
        "context_depth": 2,
        "num_hyperedges": 6,
        "thermal_response_rank": 4 if thermal else 0,
        "local_hidden_dim": 128,
        "dropout": 0.0,
    }
    if thermal:
        model = {
            "module_input_dim": 8,
            "environment_input_dim": 8,
            "coordinate_dim": 2,
            "query_input_dim": 0,
            "operating_input_dim": 14,
            "output_dim": 3,
            **model,
            "init_seed": 0,
        }
    return {
        "schema": "content_hypergraph_recipe_v2",
        "dataset": dataset,
        "architecture": "A2-D",
        "protocol": FORMAL_PROTOCOL,
        "seed": 0,
        "sampling_seed": 0 if thermal else 42,
        "primary_queries": 1024 if thermal else 8192,
        "microbatch_size": 24 if thermal else 8,
        "effective_batch_size": 48 if thermal else 16,
        "total_epochs": 5000 if thermal else 2500,
        "approved_stop_after": 5000 if thermal else 2500,
        "precision": "fp32",
        "edge_access": "full",
        "optimizer_bias_norm_exclusion": False,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": 1e-5,
            "bias_and_norm_weight_decay": 1e-5,
        },
        "execution": {
            "monitor_every": 10 if thermal else 5,
            "latest_every": 10 if thermal else 5,
            "curve_every": 10,
            "checkpoint_epochs": (
                [10, 50, 100, 250, 500, 1000, 2500, 5000]
                if thermal else [100, 500, 1000, 1500, 2000, 2500]
            ),
            "field_metric": "field_score" if thermal else "val_volume_mse",
        },
        "model": model,
    }


@pytest.mark.parametrize("dataset", ["thermal", "wind"])
def test_formal_recipe_is_accepted_and_returned_unchanged(dataset: str) -> None:
    value = recipe(dataset)
    assert validate_formal_d_recipe(value) == value


@pytest.mark.parametrize("dataset", ["thermal", "wind"])
def test_formal_recipe_keeps_new_curve_default_and_accepts_legacy_100_for_replay(dataset: str) -> None:
    value = recipe(dataset)
    assert value["execution"]["curve_every"] == 10
    assert validate_formal_d_recipe(value) == value
    legacy = copy.deepcopy(value)
    legacy["execution"]["curve_every"] = 100
    assert validate_formal_d_recipe(legacy) == legacy
    validate_formal_d_stop(legacy, 1)
    for unsupported in (1, 5, 20, True):
        changed = copy.deepcopy(value)
        changed["execution"]["curve_every"] = unsupported
        with pytest.raises(ValueError, match="curve_every"):
            validate_formal_d_recipe(changed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema="content_hypergraph_recipe_v1"), "schema"),
        (lambda value: value.update(protocol="development"), "protocol"),
        (lambda value: value.update(architecture="A2-H6"), "A2-D"),
        (lambda value: value.update(total_epochs=1000), "total_epochs"),
        (lambda value: value.update(approved_stop_after=1000), "approved_stop_after"),
        (lambda value: value.update(seed=42), "seed"),
        (lambda value: value.update(sampling_seed=7), "sampling_seed"),
        (lambda value: value.update(precision="bf16"), "precision"),
        (lambda value: value.update(edge_access="top3"), "edge access"),
        (lambda value: value.update(optimizer_bias_norm_exclusion=True), "exclusion=false"),
        (lambda value: value["optimizer"].update(weight_decay=0.0001), "weight_decay"),
        (lambda value: value["optimizer"].update(bias_and_norm_weight_decay=0.0), "bias_and_norm"),
        (lambda value: value["optimizer"].update(lr=1e-4), "lr"),
        (lambda value: value["model"].update(context_dim=64), "model fields"),
        (lambda value: value["model"].update(query_edge_geometry_mode="typed_w_centroids_v1"), "model fields"),
        (lambda value: value["execution"].update(monitor_every=100), "execution fields"),
        (lambda value: value["execution"].update(field_metric="field_score"), "execution fields"),
    ],
)
def test_formal_recipe_rejects_changed_contract(mutation, message: str) -> None:
    value = recipe("wind")
    mutation(value)
    with pytest.raises(ValueError, match=message):
        validate_formal_d_recipe(value)


@pytest.mark.parametrize(
    ("dataset", "key", "value"),
    [
        ("thermal", "microbatch_size", 8),
        ("thermal", "effective_batch_size", 24),
        ("wind", "primary_queries", 4096),
        ("wind", "microbatch_size", 12),
        ("wind", "effective_batch_size", 24),
    ],
)
def test_formal_recipe_rejects_development_or_quarter_batch_settings(
    dataset: str, key: str, value: int,
) -> None:
    changed = recipe(dataset)
    changed[key] = value
    with pytest.raises(ValueError, match=key):
        validate_formal_d_recipe(changed)


@pytest.mark.parametrize("dataset,horizon", [("thermal", 5000), ("wind", 2500)])
def test_formal_stop_accepts_segments_inside_sealed_horizon(dataset: str, horizon: int) -> None:
    value = recipe(dataset)
    validate_formal_d_stop(value, 1)
    validate_formal_d_stop(value, horizon)
    for bad in (0, horizon + 1, True, float(horizon)):
        with pytest.raises(ValueError, match="execution stop"):
            validate_formal_d_stop(value, bad)


def test_formal_stop_revalidates_recipe_before_accepting_segment() -> None:
    value = copy.deepcopy(recipe("thermal"))
    value["approved_stop_after"] = 2500
    with pytest.raises(ValueError, match="approved_stop_after"):
        validate_formal_d_stop(value, 1000)


@pytest.mark.parametrize("lr_key", ["lr", "learning_rate"])
def test_formal_recipe_accepts_only_the_declared_constant_lr_alias(lr_key: str) -> None:
    value = recipe("wind")
    value["optimizer"][lr_key] = 3e-4
    assert validate_formal_d_recipe(value)["optimizer"][lr_key] == 3e-4


def test_formal_recipe_rejects_conflicting_lr_aliases() -> None:
    value = recipe("wind")
    value["optimizer"].update(lr=3e-4, learning_rate=1e-4)
    with pytest.raises(ValueError, match="must agree"):
        validate_formal_d_recipe(value)


def _write_recipe(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("bad_stop", [5001, 2501])
def test_cli_rejects_stop_beyond_formal_horizon_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_stop: int,
) -> None:
    value = recipe("thermal" if bad_stop == 5001 else "wind")
    recipe_path = _write_recipe(tmp_path / "formal.json", value)
    monkeypatch.setattr(train_cli, "build", lambda *args, **kwargs: pytest.fail("build must not run"))
    with pytest.raises(ValueError, match="execution stop"):
        train_cli.main(["--recipe", str(recipe_path), "--dry-run", "--stop-after", str(bad_stop)])


def test_cli_rejects_formal_h6_and_h6_qualification_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_cli, "build", lambda *args, **kwargs: pytest.fail("build must not run"))
    h6 = recipe("thermal")
    h6["architecture"] = "A2-H6"
    h6_path = _write_recipe(tmp_path / "h6.json", h6)
    with pytest.raises(ValueError, match="A2-D"):
        train_cli.main(["--recipe", str(h6_path), "--dry-run", "--qualification", str(tmp_path / "qualification.json")])

    formal_path = _write_recipe(tmp_path / "formal.json", recipe("thermal"))
    with pytest.raises(ValueError, match="separate from H6 qualification"):
        train_cli.main([
            "--recipe", str(formal_path), "--dry-run", "--qualification",
            str(tmp_path / "qualification.json"),
        ])


def test_cli_rejects_nonempty_fresh_formal_output_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_path = _write_recipe(tmp_path / "formal.json", recipe("thermal"))
    output = tmp_path / "occupied-run"
    output.mkdir()
    (output / "keep.txt").write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(train_cli, "build", lambda *args, **kwargs: pytest.fail("build must not run"))
    with pytest.raises(ValueError, match="empty new run directory"):
        train_cli.main([
            "--recipe", str(recipe_path), "--output", str(output), "--yes",
            "--stop-after", "5000",
        ])
    assert (output / "keep.txt").read_text(encoding="utf-8") == "fixture"


@pytest.mark.parametrize("dataset,horizon", [("thermal", 5000), ("wind", 2500)])
def test_formal_engine_keeps_full_schedule_horizon_before_final_stop(dataset: str, horizon: int) -> None:
    value = validate_formal_d_recipe(recipe(dataset))
    engine = train_cli.engine_for(value, "cpu")
    assert engine.config.total_epochs == horizon
    assert engine.config.effective_cases == value["effective_batch_size"]
    assert engine.config.microbatch_cases == value["microbatch_size"]
