"""Fail-closed CLI routing for audited source-read sparse children."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from honf_runtime.unified_training import EngineConfig, SelectionPolicy

_PROJECT = Path(__file__).resolve().parents[1]
_CLI_PATH = _PROJECT / "tools/content_hypergraph_train.py"
_SPEC = importlib.util.spec_from_file_location("source_read_cli_test", _CLI_PATH)
cli = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(cli)

_RECIPES = _PROJECT / "src/config_core/forward/content_hypergraph"


def sparse_recipe(*, full: bool = False, variant: str = "S-Utility") -> dict:
    name = (
        "thermal_a2-d_source_read_full1000_d_scale_v1.json" if full else
        "thermal_a2-d_source_read_development_d0_v1.json"
    )
    recipe = json.loads((_RECIPES / name).read_text())
    recipe["source_reader_variant"] = variant
    if full or "dense_host_variant" in recipe:
        recipe["dense_host_variant"] = "D-scale" if full else "D0"
    recipe["model"].update(
        source_read_policy="utility" if variant == "S-Utility" else "completion",
        source_read_executor="packed",
        source_read_rank=16,
        source_read_scorer_width=32,
    )
    return cli.validate_recipe(recipe)


def test_source_read_child_cli_options_bind_the_exact_parent_and_callback(tmp_path, monkeypatch):
    recipe = sparse_recipe()
    parent = tmp_path / "dense_e500.pt"
    parent.write_bytes(b"fixture checkpoint")
    declaration = {
        "source_checkpoint": str(parent.resolve()),
        "source_reader_variant": "S-Utility",
    }
    import honf_runtime.source_read_auxiliary as auxiliary

    calls = {}

    def fit(model, provider, optimizer, **kwargs):
        calls.update(model=model, provider=provider, optimizer=optimizer, **kwargs)
        return {"schema": "fixture"}

    monkeypatch.setattr(auxiliary, "fit_source_read_child_auxiliary", fit)
    options = cli.source_read_child_fit_options(recipe, declaration, parent)
    assert options["branch_from_checkpoint"] == parent.resolve()
    assert options["source_read_child"] is declaration
    receipt = options["source_read_child_auxiliary_fit"](
        "model", "provider", "optimizer", max_updates=32, source_epoch=500,
    )
    assert receipt == {"schema": "fixture"}
    assert calls["variant"] == "S-Utility"
    assert calls["source_epoch"] == 500


def test_sparse_modules_are_imported_before_child_source_identity_is_sealed():
    cli._load_source_read_child_modules()
    files = cli.loaded_source_binding()["files"]
    for module in (
        "query_chunks_v1.py", "source_read_decoder_v1.py",
        "source_read_sparse.py", "source_read_response.py",
    ):
        path = f"src/honf_forward_core/content_hypergraph/{module}"
        assert path in files


def test_sparse_exact_resume_imports_child_runtime_before_identity_binding(tmp_path):
    recipe = sparse_recipe()
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    assert cli.source_read_child_fit_options(
        recipe, None, None, resume_checkpoint=checkpoint,
    ) == {}
    files = cli.loaded_source_binding()["files"]
    assert "src/honf_forward_core/content_hypergraph/source_read_decoder_v1.py" in files
    assert "src/honf_forward_core/content_hypergraph/source_read_sparse.py" in files
    assert "src/honf_forward_core/content_hypergraph/source_read_response.py" in files


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"declaration": None, "parent": Path("parent.pt"), "resume": None}, "requires --source-read-child"),
        ({"declaration": None, "parent": None, "resume": None}, "audited dense e500"),
    ],
)
def test_source_read_sparse_cli_rejects_unbound_fresh_or_parent_path(kwargs, match):
    recipe = sparse_recipe()
    with pytest.raises(ValueError, match=match):
        cli.source_read_child_fit_options(
            recipe,
            kwargs["declaration"],
            kwargs["parent"],
            resume_checkpoint=kwargs["resume"],
        )


def test_source_read_child_cli_rejects_parent_or_variant_mismatch(tmp_path):
    recipe = sparse_recipe()
    parent = tmp_path / "dense_e500.pt"
    parent.write_bytes(b"checkpoint")
    declaration = {"source_checkpoint": str(parent.resolve()), "source_reader_variant": "S-Complete"}
    with pytest.raises(ValueError, match="variants differ"):
        cli.source_read_child_fit_options(recipe, declaration, parent)
    declaration["source_reader_variant"] = "S-Utility"
    other_parent = tmp_path / "other_dense_e500.pt"
    other_parent.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="exact canonical path"):
        cli.source_read_child_fit_options(recipe, declaration, other_parent)
    with pytest.raises(ValueError, match="no resume"):
        cli.source_read_child_fit_options(
            recipe, declaration, parent, resume_checkpoint=parent,
        )


def test_source_read_full_sparse_is_closed_before_build_or_allocation(tmp_path, monkeypatch):
    recipe = sparse_recipe(full=True)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    monkeypatch.setattr(cli, "build", lambda *args, **kwargs: pytest.fail("full sparse reached build"))
    output = tmp_path / "sparse_full"
    with pytest.raises(ValueError, match="Full sparse source-read training is not wired"):
        cli.main(["--recipe", str(path), "--dry-run", "--output", str(output)])
    assert not output.exists()


def test_source_read_sparse_development_cannot_start_fresh_without_dense_parent(tmp_path, monkeypatch):
    recipe = sparse_recipe()
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    output = tmp_path / "sparse_child"
    monkeypatch.setattr(cli, "build", lambda *args, **kwargs: pytest.fail("unbound sparse run reached build"))
    with pytest.raises(ValueError, match="audited dense e500"):
        cli.main(["--recipe", str(path), "--dry-run", "--output", str(output)])
    assert not output.exists()


def test_source_read_child_preflight_checks_checkpoint_without_allocating_output(tmp_path, monkeypatch):
    parent = tmp_path / "dense_e500.pt"
    torch.save({"epoch": 500}, parent)
    recipe = sparse_recipe()
    declaration = {
        "source_checkpoint_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "source_reader_variant": "S-Utility",
        "auxiliary_fit_mode": "calibrated",
        "auxiliary_update_limit": 32,
    }

    def prepare(*args, **kwargs):
        assert kwargs["source_path"] == parent.resolve()
        return {
            "exact_parent_state_copied": True,
            "source_changes": {"src/example.py": {"parent_sha256": "a", "child_sha256": "b"}},
        }

    import honf_runtime.source_read_child as child

    monkeypatch.setattr(child, "prepare_source_read_child", prepare)

    class Provider:
        def identity_payload(self):
            return {"dataset": "fixture", "training_case_ids": ["train-a"]}

        def optimizer_groups(self, model, architecture, stage):
            assert architecture == "A2-D" and stage == "joint"
            return ()

        def epoch_cases(self, epoch, seed):
            assert epoch == 501 and seed == 0
            return ("train-a",)

    output = tmp_path / "new_child_run"
    identity = {"loaded_source": {"files": {}, "sha256": "ignored"}}
    result = cli.preflight_source_read_child(
        object(), Provider(),
        SimpleNamespace(
            config=EngineConfig(seed=0, microbatch_cases=24, effective_cases=48, total_epochs=2500),
            selection=SelectionPolicy(field_metric="field_score"),
        ),
        identity=identity,
        recipe=recipe,
        declaration=declaration,
        parent_checkpoint=parent,
        output=output,
        stop_after=600,
    )
    assert result["status"] == "preflight_passed"
    assert result["source_epoch"] == 500
    assert result["exact_parent_state_copied"] is True
    assert not output.exists()
