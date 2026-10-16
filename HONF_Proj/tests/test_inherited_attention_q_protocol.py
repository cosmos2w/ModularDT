from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "tools"):
    sys.path.insert(0, str(directory))

import inherited_attention_train

from honf_runtime.inherited_attention_child import (
    IAG_SPARSE_CHILD_SCHEMA,
    prepare_inherited_attention_sparse_child,
)
from honf_runtime.inherited_attention_protocol import (
    IAG_Q_REPRESENTATION_REVIEW_DECISION,
    IAG_REPRESENTATION_REVIEW_SCHEMA,
    validate_inherited_attention_recipe,
    validate_inherited_attention_representation_review,
)

RECIPE_DIR = PROJECT / "src/config_core/forward/inherited_attention"


def _recipe(name: str) -> dict[str, Any]:
    return json.loads((RECIPE_DIR / name).read_text(encoding="utf-8"))


def _review_receipt(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source-e500.pt"
    source.write_bytes(b"source checkpoint evidence")
    diagnosis = tmp_path / "capacity-diagnosis.json"
    diagnosis.write_text("{\"diagnosis\":\"fixed\"}\n", encoding="utf-8")
    return {
        "schema": IAG_REPRESENTATION_REVIEW_SCHEMA,
        "dataset": "wind",
        "decision": IAG_Q_REPRESENTATION_REVIEW_DECISION,
        "source_architecture": "IAG-C",
        "target_architecture": "IAG-Q",
        "review_epoch": 500,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "capacity_diagnosis": str(diagnosis),
        "capacity_diagnosis_sha256": hashlib.sha256(diagnosis.read_bytes()).hexdigest(),
        "substantive_revisions_used_for_dataset": 1,
        "maximum_revisions_for_dataset": 1,
        "fresh_run_dir": str(tmp_path / "wind_iag_q_soft_dev"),
        "future_sparse_child_run_dir": str(tmp_path / "wind_iag_q_entmax15_dev"),
    }


def _review_identity(receipt: dict[str, Any], receipt_path: Path) -> dict[str, Any]:
    return {
        "schema": receipt["schema"],
        "path": str(receipt_path.resolve()),
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "dataset": receipt["dataset"],
        "decision": receipt["decision"],
        "fresh_run_dir": str(Path(receipt["fresh_run_dir"]).resolve()),
    }


@pytest.mark.parametrize(
    "name",
    (
        "wind_iag_q_soft_development_v1.json",
        "wind_iag_q_measure_entmax15_development_v1.json",
        "wind_iag_q_soft_full_train1000_v1.json",
        "wind_iag_q_measure_entmax15_full_train1000_v1.json",
    ),
)
def test_wind_q_recipes_keep_the_new_representation_and_executor_law(name: str):
    recipe = validate_inherited_attention_recipe(_recipe(name))
    assert recipe["dataset"] == "wind"
    assert recipe["architecture"] == "IAG-Q"
    assert recipe["representation"] == "iag_q"
    assert recipe["model"]["scene_executor"] == "cached_tokens"
    assert recipe["model"]["query_executor"] == "cell_selected"
    assert recipe["model"]["normalization"] == recipe["normalization"]


def test_wind_q_recipe_rejects_c_executor_or_non_wind_scope():
    recipe = _recipe("wind_iag_q_soft_development_v1.json")
    recipe["model"]["query_executor"] = "indexed"
    with pytest.raises(ValueError, match="IAG-Q query_executor"):
        validate_inherited_attention_recipe(recipe)
    thermal = _recipe("thermal_iag_c_soft_development_v1.json")
    thermal["architecture"] = "IAG-Q"
    thermal["representation"] = "iag_q"
    thermal["model"]["scene_executor"] = "cached_tokens"
    thermal["model"]["query_executor"] = "cell_selected"
    with pytest.raises(ValueError, match="authorized only for the Wind"):
        validate_inherited_attention_recipe(thermal)


def test_q_review_receipt_is_validated_and_output_paths_are_bound(tmp_path: Path):
    recipe = validate_inherited_attention_recipe(_recipe("wind_iag_q_soft_development_v1.json"))
    receipt = _review_receipt(tmp_path)
    receipt_path = tmp_path / "review.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    parsed = inherited_attention_train._read_json_object(receipt_path, "receipt")
    validated = validate_inherited_attention_representation_review(recipe, parsed)
    assert validated == receipt

    verified, identity = inherited_attention_train._representation_review_binding(recipe, receipt_path)
    assert verified == receipt
    assert identity == _review_identity(receipt, receipt_path)
    inherited_attention_train._validate_review_output_binding(
        recipe, receipt, Path(receipt["fresh_run_dir"]),
    )
    with pytest.raises(ValueError, match="must match the review-bound fresh_run_dir"):
        inherited_attention_train._validate_review_output_binding(
            recipe, receipt, tmp_path / "wrong-output",
        )
    sparse_recipe = validate_inherited_attention_recipe(
        _recipe("wind_iag_q_measure_entmax15_development_v1.json")
    )
    inherited_attention_train._validate_review_output_binding(
        sparse_recipe, receipt, Path(receipt["future_sparse_child_run_dir"]),
    )
    with pytest.raises(ValueError, match="must match the review-bound future_sparse_child_run_dir"):
        inherited_attention_train._validate_review_output_binding(
            sparse_recipe, receipt, tmp_path / "wrong-child-output",
        )

    receipt["capacity_diagnosis_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="capacity_diagnosis is missing or differs"):
        inherited_attention_train._representation_review_binding(recipe, receipt_path)


def test_q_cli_requires_review_before_building_native_data(monkeypatch: pytest.MonkeyPatch):
    builds: list[bool] = []
    monkeypatch.setattr(inherited_attention_train, "build", lambda *args, **kwargs: builds.append(True))
    with pytest.raises(ValueError, match="requires --representation-review"):
        inherited_attention_train.main([
            "--recipe", str(RECIPE_DIR / "wind_iag_q_soft_development_v1.json"),
            "--dry-run",
        ])
    assert builds == []


def _q_child_fixture(tmp_path: Path):
    source_recipe = validate_inherited_attention_recipe(_recipe("wind_iag_q_soft_development_v1.json"))
    target_recipe = validate_inherited_attention_recipe(
        _recipe("wind_iag_q_measure_entmax15_development_v1.json")
    )
    receipt = _review_receipt(tmp_path)
    receipt_path = tmp_path / "review.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    review_identity = _review_identity(receipt, receipt_path)
    model = nn.Linear(1, 1)
    source_provider = {
        "model_config": {
            "architecture": "IAG-Q",
            "representation": "iag_q",
            "normalization": "soft",
            "scene_executor": "cached_tokens",
            "query_executor": "cell_selected",
        },
        "training": {"normalization": "soft", "absolute_epoch_transition": None},
    }
    target_provider = copy.deepcopy(source_provider)
    target_provider["model_config"]["normalization"] = "measure_entmax15"
    target_provider["training"]["normalization"] = "measure_entmax15"
    target_provider["training"]["absolute_epoch_transition"] = target_recipe["sparse_transition"]
    source_id = {
        "workflow": "inherited_attention_joint_v1",
        "model_family": "inherited_attention_v1",
        "recipe": source_recipe,
        "recipe_sha256": inherited_attention_train.canonical_sha256(source_recipe),
        "representation_review": review_identity,
        "absolute_epoch_transition": None,
        "provider_identity": source_provider,
        "loaded_source": {"sha256": "d" * 64},
        "optimizer_schedule_contract": [{"name": "all", "parameter_names": ["weight", "bias"]}],
        "engine_config": {"total_epochs": 2500},
        "selection_policy": {"field_metric": "val_volume_mse"},
    }
    target_id = copy.deepcopy(source_id)
    target_id.update(
        recipe=target_recipe,
        recipe_sha256=inherited_attention_train.canonical_sha256(target_recipe),
        absolute_epoch_transition=target_recipe["sparse_transition"],
        provider_identity=target_provider,
    )
    source_path = tmp_path / "wind-iag-q-e500.pt"
    source_path.write_bytes(b"parent checkpoint identity")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    state = copy.deepcopy(model.state_dict())
    source_payload = {
        "workflow": "inherited_attention_joint_v1",
        "arm": "IAG-Q",
        "epoch": 500,
        "experiment_identity": source_id,
        "model_state_dict": state,
        "optimizer_state_by_name": {
            "groups": [{"name": "all", "parameter_names": ["weight", "bias"]}],
            "state_by_name": {
                name: {
                    "step": torch.tensor(500.0),
                    "exp_avg": torch.zeros_like(value),
                    "exp_avg_sq": torch.zeros_like(value),
                }
                for name, value in state.items()
            },
        },
        "provider_training_state": {},
        "history": [{"epoch": 500}],
    }
    declaration = {
        "schema": IAG_SPARSE_CHILD_SCHEMA,
        "source_checkpoint": str(source_path.resolve()),
        "source_checkpoint_sha256": digest,
        "source_recipe_sha256": source_id["recipe_sha256"],
        "target_recipe_sha256": target_id["recipe_sha256"],
        "source_epoch": 500,
        "representation_review_sha256": review_identity["sha256"],
    }
    return model, source_payload, target_id, declaration, source_path, source_recipe, target_recipe


def test_q_sparse_child_preserves_its_own_e500_state_and_review_binding(tmp_path: Path):
    model, payload, target, declaration, path, _, _ = _q_child_fixture(tmp_path)
    result = prepare_inherited_attention_sparse_child(
        model, payload, target, declaration,
        source_path=path, case_keys=("wind-0001",), stop_after=2500,
    )
    assert result["source_architecture"] == "IAG-Q"
    assert result["target_architecture"] == "IAG-Q"
    assert result["source_representation"] == "iag_q"
    assert result["target_representation"] == "iag_q"
    assert result["representation_review_sha256"] == declaration["representation_review_sha256"]
    assert result["scene_executor_transition"] == {
        "source": "cached_tokens",
        "target": "cached_tokens",
        "same_probability_function": True,
    }


def test_q_cli_child_declaration_binds_review_sha_and_source_recipe(tmp_path: Path, monkeypatch):
    _, payload, _, declaration, path, _, sparse_recipe = _q_child_fixture(tmp_path)
    source_identity = payload["experiment_identity"]
    review_identity = source_identity["representation_review"]
    validated_sparse_recipe = validate_inherited_attention_recipe(sparse_recipe)
    result = inherited_attention_train._validate_sparse_declaration(
        validated_sparse_recipe, declaration, path, None, review_identity,
    )
    assert result["branch_from_checkpoint"] == path.resolve()
    assert result["declaration"]["representation_review_sha256"] == review_identity["sha256"]

    mismatched = {**declaration, "representation_review_sha256": "0" * 64}
    with pytest.raises(ValueError, match="must bind the supplied representation-review"):
        inherited_attention_train._validate_sparse_declaration(
            validated_sparse_recipe, mismatched, path, None, review_identity,
        )

    from honf_runtime import compat

    monkeypatch.setattr(compat, "load_trusted_checkpoint", lambda *args, **kwargs: payload)
    loaded = inherited_attention_train._load_parent_payload(
        path, declaration, validated_sparse_recipe, review_identity,
    )
    assert loaded is payload


def test_q_sparse_child_rejects_c_parent_and_review_or_executor_drift(tmp_path: Path):
    model, payload, target, declaration, path, _, _ = _q_child_fixture(tmp_path)
    c_recipe = _recipe("wind_iag_c_soft_development_v1.json")
    payload["arm"] = "IAG-C"
    payload["experiment_identity"]["recipe"] = c_recipe
    with pytest.raises(ValueError, match="literal IAG-Q epoch500 state"):
        prepare_inherited_attention_sparse_child(
            model, payload, target, declaration,
            source_path=path, case_keys=("wind-0001",), stop_after=2500,
        )

    model, payload, target, declaration, path, _, _ = _q_child_fixture(tmp_path / "second")
    target["representation_review"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="preserve and bind the exact review"):
        prepare_inherited_attention_sparse_child(
            model, payload, target, declaration,
            source_path=path, case_keys=("wind-0001",), stop_after=2500,
        )

    model, payload, target, declaration, path, _, _ = _q_child_fixture(tmp_path / "third")
    target["recipe"]["model"]["query_executor"] = "dense_reference"
    target["provider_identity"]["model_config"]["query_executor"] = "dense_reference"
    with pytest.raises(ValueError, match="preserves the query executor exactly"):
        prepare_inherited_attention_sparse_child(
            model, payload, target, declaration,
            source_path=path, case_keys=("wind-0001",), stop_after=2500,
        )
