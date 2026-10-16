from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
for directory in (
    PROJECT / "src",
    PROJECT / "tools",
    PROJECT / "Case_ThermalChannel/src",
    PROJECT / "Case_WindFarm/src",
):
    sys.path.insert(0, str(directory))

import inherited_attention_train

from honf_runtime import content_hypergraph as checkpoint_loader
from honf_runtime.inherited_attention_child import (
    IAG_SPARSE_CHILD_SCHEMA,
    prepare_inherited_attention_sparse_child,
)
from honf_runtime.inherited_attention_protocol import (
    IAG_AUTHORIZED_PHYSICAL_GPUS,
    IAG_FULL_BUDGET_SCHEMA,
    IAG_FULL_QUALIFICATION_SCHEMA,
    IAG_NUMERICAL_POLICY,
    configure_inherited_attention_numerics,
    validate_inherited_attention_full_train_authorization,
    validate_inherited_attention_recipe,
)

RECIPE_DIR = PROJECT / "src/config_core/forward/inherited_attention"


def _recipe(name: str) -> dict:
    return json.loads((RECIPE_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", sorted(RECIPE_DIR.glob("*.json")), ids=lambda path: path.name)
def test_all_opt_in_iag_recipes_resolve_without_legacy_controls(path: Path):
    recipe = validate_inherited_attention_recipe(json.loads(path.read_text(encoding="utf-8")))
    assert recipe["family"] == "inherited_attention_v1"
    assert recipe["architecture"] == "IAG-C"
    assert recipe["model"]["signed_carrier_width"] == 8
    assert recipe["model"]["query_executor"] == "indexed"
    assert recipe["numerical_policy"] == IAG_NUMERICAL_POLICY
    assert "thermal_response_rank" not in recipe["model"]
    assert "num_hyperedges" not in recipe["model"]


def test_iag_numerical_policy_is_fixed_and_does_not_change_sdpa_availability(monkeypatch: pytest.MonkeyPatch):
    old_workspace = __import__("os").environ.get("CUBLAS_WORKSPACE_CONFIG")
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    sdpa_state = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    try:
        configure_inherited_attention_numerics(IAG_NUMERICAL_POLICY)
        assert __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert sdpa_state == (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        if old_workspace is None:
            __import__("os").environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] = old_workspace


def test_full_train_guard_requires_qualified_checkpoint_and_index_uuid_budget():
    recipe = validate_inherited_attention_recipe(_recipe("thermal_iag_c_soft_full_train1000_v1.json"))
    digest = "a" * 64
    qualification = {
        "schema": IAG_FULL_QUALIFICATION_SCHEMA,
        "dataset": "thermal",
        "architecture": "IAG-C",
        "selected_epoch": 2500,
        "qualified": True,
        "gates": {"native_contracts": True, "priority_accuracy": True, "complete_cost": True},
        "selected_checkpoint": {"path": "/tmp/frozen-iag-e2500.pt", "sha256": digest},
    }
    physical_index = 1
    budget = {
        "schema": IAG_FULL_BUDGET_SCHEMA,
        "dataset": "thermal",
        "approved": True,
        "physical_gpu_index": physical_index,
        "physical_gpu_uuids": [IAG_AUTHORIZED_PHYSICAL_GPUS[physical_index]],
        "max_gpu_process_hours": 2.0,
        "max_elapsed_hours": 3.0,
        "qualification_checkpoint_sha256": digest,
    }
    accepted, accepted_budget = validate_inherited_attention_full_train_authorization(
        recipe, qualification, budget,
    )
    assert accepted["selected_checkpoint"]["sha256"] == digest
    assert accepted_budget["physical_gpu_index"] == physical_index
    with pytest.raises(ValueError, match="qualified same-dataset"):
        validate_inherited_attention_full_train_authorization(recipe, None, budget)
    with pytest.raises(ValueError, match="physical GPU1 or GPU2"):
        validate_inherited_attention_full_train_authorization(
            recipe, qualification, {**budget, "physical_gpu_index": 0},
        )
    with pytest.raises(ValueError, match="physical GPU1 or GPU2"):
        validate_inherited_attention_full_train_authorization(
            recipe, qualification,
            {**budget, "physical_gpu_uuids": [IAG_AUTHORIZED_PHYSICAL_GPUS[2]]},
        )


def test_full_train_cli_receipt_gate_precedes_native_provider_build(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(inherited_attention_train, "build", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="both a qualified e2500 receipt"):
        inherited_attention_train.main([
            "--recipe", str(RECIPE_DIR / "thermal_iag_c_soft_full_train1000_v1.json"),
            "--stop-after", "1000",
            "--dry-run",
        ])
    assert calls == []


def _child_identity(*, scene_executor: str, normalization: str, sparse: bool) -> dict:
    recipe = {
        "protocol": "inherited_attention_development_v1",
        "normalization": normalization,
        "trajectory": "e500_sparse_child" if sparse else "fresh_soft",
        "sparse_transition": {"blend_start_epoch": 501} if sparse else None,
        "model": {
            "scene_executor": scene_executor,
            "query_executor": "indexed",
            "normalization": normalization,
        },
    }
    if not sparse:
        recipe.pop("sparse_transition")
    digest = "c" * 64 if sparse else "b" * 64
    return {
        "workflow": "inherited_attention_joint_v1",
        "model_family": "inherited_attention_v1",
        "recipe": recipe,
        "recipe_sha256": digest,
        "absolute_epoch_transition": {"blend_start_epoch": 501} if sparse else None,
        "optimizer_schedule_contract": [{"name": "all", "parameter_names": ["weight"]}],
        "provider_identity": {
            "model_config": {
                "scene_executor": scene_executor,
                "query_executor": "indexed",
                "normalization": normalization,
            },
            "training": {
                "normalization": normalization,
                "absolute_epoch_transition": {"blend_start_epoch": 501} if sparse else None,
            },
        },
        "loaded_source": {"sha256": "d" * 64},
    }


def _child_fixture(tmp_path: Path, *, source_scene: str = "dense", target_scene: str = "selected",
                   query_executor: str = "indexed", extra_target_field: bool = False):
    model = nn.Linear(1, 1, bias=False)
    source_identity = _child_identity(scene_executor=source_scene, normalization="soft", sparse=False)
    target_identity = _child_identity(scene_executor=target_scene, normalization="measure_entmax15", sparse=True)
    target_identity["recipe"]["model"]["query_executor"] = query_executor
    target_identity["provider_identity"]["model_config"]["query_executor"] = query_executor
    if extra_target_field:
        target_identity["new_control"] = "changed"
    path = tmp_path / "parent-e500.pt"
    path.write_bytes(b"synthetic sealed parent checkpoint bytes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "workflow": "inherited_attention_joint_v1",
        "arm": "IAG-C",
        "epoch": 500,
        "experiment_identity": source_identity,
        "model_state_dict": model.state_dict(),
        "optimizer_state_by_name": {
            "groups": [{"name": "all", "parameter_names": ["weight"]}],
            "state_by_name": {"weight": {"step": torch.tensor(5), "exp_avg": torch.zeros_like(model.weight),
                                           "exp_avg_sq": torch.zeros_like(model.weight)}},
        },
        "provider_training_state": {},
        "history": [{"epoch": 500}],
    }
    declaration = {
        "schema": IAG_SPARSE_CHILD_SCHEMA,
        "source_checkpoint": str(path.resolve()),
        "source_checkpoint_sha256": digest,
        "source_recipe_sha256": source_identity["recipe_sha256"],
        "target_recipe_sha256": target_identity["recipe_sha256"],
        "source_epoch": 500,
    }
    return model, source_identity, target_identity, payload, declaration, path


def test_sparse_child_allows_only_dense_to_selected_same_probability_function(tmp_path: Path):
    model, _, target, payload, declaration, path = _child_fixture(tmp_path)
    result = prepare_inherited_attention_sparse_child(
        model, payload, target, declaration, source_path=path, case_keys=("case-1",), stop_after=2500,
    )
    assert result["scene_executor_transition"] == {
        "source": "dense", "target": "selected", "same_probability_function": True,
    }


def test_engine_sparse_child_preflight_reaches_parent_load_without_auxiliary_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from honf_runtime import compat

    model, _, target, _, declaration, path = _child_fixture(tmp_path)
    provider = SimpleNamespace(identity_payload=lambda: target["provider_identity"])
    recipe = _recipe("thermal_iag_c_measure_entmax15_development_v1.json")
    engine = inherited_attention_train.engine_for(recipe, "cpu")

    def stop_at_parent_load(source, *, map_location):
        assert Path(source) == path
        assert map_location == "cpu"
        raise RuntimeError("reached exact IAG parent loading")

    monkeypatch.setattr(compat, "load_trusted_checkpoint", stop_at_parent_load)
    with pytest.raises(RuntimeError, match="reached exact IAG parent loading"):
        engine.fit(
            model, provider, tmp_path / "child", identity=target, arm="IAG-C",
            stop_after=2500, branch_from_checkpoint=path,
            inherited_attention_sparse_child=declaration,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_scene": "selected", "target_scene": "dense"}, "cannot replace selected preparation"),
        ({"query_executor": "dense_reference"}, "preserves the indexed query executor"),
        ({"extra_target_field": True}, "changes a field beyond normalization"),
    ],
)
def test_sparse_child_rejects_executor_or_identity_drift(tmp_path: Path, kwargs: dict, message: str):
    model, _, target, payload, declaration, path = _child_fixture(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=message):
        prepare_inherited_attention_sparse_child(
            model, payload, target, declaration, source_path=path, case_keys=("case-1",), stop_after=2500,
        )


def test_dataset_free_sparse_checkpoint_load_requires_recipe_bound_core_law():
    recipe = validate_inherited_attention_recipe(
        _recipe("thermal_iag_c_measure_entmax15_development_v1.json")
    )
    recipe_digest = inherited_attention_train.canonical_sha256(recipe)
    expected_core = {"schema": "inherited_attention_core_v1", "architecture": "IAG-C", **recipe["model"]}
    provider = {
        "model_config": expected_core,
        "adapter_state": {"core": {**expected_core, "normalization": "soft"}},
    }
    payload = {
        "workflow": "inherited_attention_joint_v1",
        "arm": "IAG-C",
        "epoch": 900,
        "experiment_identity": {
            "workflow": "inherited_attention_joint_v1",
            "model_family": "inherited_attention_v1",
            "recipe": recipe,
            "recipe_sha256": recipe_digest,
            "provider_identity": provider,
        },
        "model_state_dict": {},
    }
    with pytest.raises(ValueError, match="core law/executor differs"):
        checkpoint_loader._load_inherited_attention_checkpoint(payload, device="cpu")


def test_dataset_free_sparse_checkpoint_requires_e500_attachment_receipt():
    recipe = validate_inherited_attention_recipe(
        _recipe("thermal_iag_c_measure_entmax15_development_v1.json")
    )
    recipe_digest = inherited_attention_train.canonical_sha256(recipe)
    expected_core = {"schema": "inherited_attention_core_v1", "architecture": "IAG-C", **recipe["model"]}
    provider = {"model_config": expected_core, "adapter_state": {"core": expected_core}}
    payload = {
        "workflow": "inherited_attention_joint_v1",
        "arm": "IAG-C",
        "epoch": 900,
        "experiment_identity": {
            "workflow": "inherited_attention_joint_v1",
            "model_family": "inherited_attention_v1",
            "recipe": recipe,
            "recipe_sha256": recipe_digest,
            "provider_identity": provider,
        },
        "model_state_dict": {},
    }
    with pytest.raises(ValueError, match="exact soft e500 branch contract"):
        checkpoint_loader._load_inherited_attention_checkpoint(payload, device="cpu")
