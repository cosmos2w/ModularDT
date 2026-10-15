"""CPU contracts for exact dense-e500 -> source-read child attachment."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import honf_runtime.unified_training as runtime
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.run_layout import resolve_checkpoint
from honf_runtime.source_read_child import (
    AUXILIARY_FIT_SCHEMA,
    SCHEMA,
    prepare_source_read_child,
)
from honf_runtime.unified_training import (
    EngineConfig,
    LossTerm,
    OptimizerGroupSpec,
    ScheduleSpec,
    TaskBatch,
    TrainingEngine,
    _engine_config_payload,
    _optimizer_spec_payload,
    _sampling_dataset_id,
    capture_rng_state,
    named_optimizer_state,
)


def _recipe(variant: str, *, host: str = "D0") -> dict:
    recipe = {
        "schema": "content_hypergraph_recipe_v2",
        "dataset": "thermal",
        "model_family": "honf_content_hypergraph",
        "adapter_schema": "thermal_content_hypergraph_adapter_v2",
        "architecture": "A2-D",
        "protocol": "source_read_development_v1",
        "source_reader_variant": variant,
        "seed": 0,
        "sampling_seed": 0,
        "primary_queries": 1024,
        "microbatch_size": 24,
        "effective_batch_size": 48,
        "total_epochs": 2500,
        "approved_stop_after": 2500,
        "precision": "fp32",
        "edge_access": "full",
        "optimizer_bias_norm_exclusion": False,
        "optimizer": {
            "name": "AdamW", "lr": 3.0e-4, "weight_decay": 1.0e-5,
            "bias_and_norm_weight_decay": 1.0e-5, "betas": [0.9, 0.999], "eps": 1.0e-8,
        },
        "execution": {
            "monitor_every": 100, "latest_every": 100, "curve_every": 10,
            "checkpoint_epochs": list(range(100, 2501, 100)), "field_metric": "field_score",
        },
        "model": {
            "module_input_dim": 8, "environment_input_dim": 8, "coordinate_dim": 2,
            "query_input_dim": 0, "operating_input_dim": 14, "output_dim": 3,
            "context_dim": 128, "num_heads": 4, "context_depth": 2, "num_hyperedges": 6,
            "thermal_response_rank": 4, "local_hidden_dim": 128, "dropout": 0.0, "init_seed": 0,
            "reader_execution_backend": "legacy",
            "source_value_conditioning": "none" if host == "D0" else "source_scale",
        },
        "manifest": "fixed25_v1",
    }
    if variant.startswith("S-"):
        recipe["dense_host_variant"] = host
        recipe["model"].update({
            "source_read_policy": "utility" if variant == "S-Utility" else "completion",
            "source_read_executor": "packed",
            "source_read_rank": 16,
            "source_read_scorer_width": 32,
        })
    return recipe


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _loaded_source(value: str) -> dict:
    files = {"src/honf_forward_core/content_hypergraph/readers_v2.py": value * 64}
    return {"files": files, "sha256": _sha(files)}


def _provider_identity(variant: str, *, host: str = "D0", changed_objective: bool = False) -> dict:
    recipe = _recipe(variant, host=host)
    core = {
        "schema": "content_hypergraph_core_v2",
        "architecture": "A2-D",
        **recipe["model"],
    }
    return {
        "schema": "thermal_content_hypergraph_provider_v2",
        "dataset": "ThermalChannel",
        "task": "ThermalChannel",
        "architecture": "A2-D",
        "protocol": "source_read_development_v1",
        "dataset_profile": "fixed25_v1_TRAIN150_DEV22",
        "source_reader_variant": variant,
        "training_case_ids": list(range(150)),
        "training_case_ids_sha256": "1" * 64,
        "validation_case_ids": list(range(200, 222)),
        "subset_manifest_sha256": "2" * 64,
        "normalization_stats_sha256": "3" * 64,
        "model_config": {"schema": "thermal_content_hypergraph_adapter_v2", "core": copy.deepcopy(core)},
        "adapter_state": {"core": copy.deepcopy(core), "normalization_stats_sha256": "3" * 64},
        "source_read_contract": {
            "protocol": "source_read_development_v1", "dataset_profile": "fixed25_v1_TRAIN150_DEV22",
            "source_reader_variant": variant, "reader_execution_backend": "legacy",
            "source_value_conditioning": recipe["model"]["source_value_conditioning"],
            "normalization_scope": "fixed25 selected TRAIN150 only", "future_horizon_epochs": 2500,
            "approved_stop_after": 2500, "execution_cadence": recipe["execution"],
        },
        "source_binding": {
            "protocol": "source_read_development_v1", "cohort_fingerprint": "fixed25_v1",
            "normalizer_fingerprint": "train150-only", "source_reader_variant": variant,
        },
        "objective": {
            "name": "equal_case_native_flow_temperature_and_flux_v1",
            "response_weight": 0.25 if changed_objective else 0.2,
            "target_partition_access": "TRAIN optimization, exposed DEV metrics only",
        },
        "training": {
            "initialization_seed": 0, "sampling_version": "case_epoch_v1",
            "effective_batch_size": 48, "microbatch_size": 24, "primary_queries_per_case": 1024,
            "total_epochs": 2500, "optimizer": "AdamW", "source_reader_variant": variant,
        },
        "recipe": recipe,
    }


def _external_identity(provider_identity: dict, loaded_source: dict) -> dict:
    recipe = provider_identity["recipe"]
    return {
        "workflow": "content_hypergraph_joint_v1",
        "model_family": "honf_content_hypergraph",
        "recipe": copy.deepcopy(recipe),
        "recipe_sha256": _sha(recipe),
        "design_sha256": "4" * 64,
        "loaded_source": copy.deepcopy(loaded_source),
        "source_read_plan_sha256": "5" * 64,
        "source_read_authorization": {
            "protocol": "source_read_development_v1", "variant": recipe["source_reader_variant"],
            "hard_stop": 2500, "full_data_launch_requires_measured_qualification": True,
        },
        "data_locks": {"WindTEST": "locked", "new_solver_attempts": 0},
    }


def _source_patch(tmp_path: Path) -> tuple[Path, str]:
    relative = "src/honf_forward_core/content_hypergraph/readers_v2.py"
    path = tmp_path / "source_read.patch"
    path.write_text(
        f"diff --git a/{relative} b/{relative}\n"
        f"index {'a' * 40}..{'b' * 40} 100644\n"
        f"--- a/{relative}\n+++ b/{relative}\n@@ -1 +1 @@\n-dense\n+sparse controller\n",
        encoding="utf-8",
    )
    return path.resolve(), hashlib.sha256(path.read_bytes()).hexdigest()


def _declaration(source_path: Path, source_sha: str, added_names, tmp_path: Path, *, mode="calibrated"):
    patch_path, patch_sha = _source_patch(tmp_path)
    return {
        "schema": SCHEMA,
        "source_checkpoint": str(source_path.resolve()),
        "source_checkpoint_sha256": source_sha,
        "source_epoch": 500,
        "source_arm": "A2-D",
        "source_reader_variant": "S-Utility",
        "dense_host_variant": "D0",
        "added_parameter_names": sorted(added_names),
        "review_epochs": list(range(600, 2501, 100)),
        "final_epoch": 2500,
        "auxiliary_fit_mode": mode,
        "auxiliary_update_limit": 2 if mode == "calibrated" else 0,
        "reason": "Attach the Round3 learned source-read policy to its exact dense e500 prefix.",
        "allowed_loaded_source_changes": {
            "src/honf_forward_core/content_hypergraph/readers_v2.py": {
                "parent_sha256": "a" * 64, "child_sha256": "b" * 64,
            },
        },
        "parent_loaded_source_sha256": _loaded_source("a")["sha256"],
        "child_loaded_source_sha256": _loaded_source("b")["sha256"],
        "source_patch_artifact": str(patch_path),
        "source_patch_sha256": patch_sha,
    }


class _TinySourceRead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.register_buffer("train_context_scale", torch.tensor(1.0))

    def forward(self, value):
        return self.linear(value) * self.train_context_scale


class _TinyA2D(nn.Module):
    def __init__(self, child: bool = False, *, extra_parameter: bool = False,
                 replacement_shape: bool = False, remove_bias: bool = False,
                 extra_buffer: bool = False):
        super().__init__()
        self.core = nn.Module()
        self.core.physical = nn.Linear(1, 2 if replacement_shape else 1, bias=not remove_bias)
        if child:
            self.core.direct_reader = nn.Module()
            self.core.direct_reader.source_read = _TinySourceRead()
        if extra_parameter:
            self.unrelated_addition = nn.Parameter(torch.tensor(0.1))
        if extra_buffer:
            self.register_buffer("unrelated_buffer", torch.tensor(1.0))

    def forward(self, value):
        result = self.core.physical(value)
        if hasattr(self.core, "direct_reader"):
            result = result + self.core.direct_reader.source_read(value)
        if hasattr(self, "unrelated_addition"):
            result = result + self.unrelated_addition
        return result


class _SourceReadProvider:
    def __init__(self, model: nn.Module, variant: str, *, capture_epoch: int | None = None,
                 changed_objective: bool = False, dense_host: str = "D0"):
        self.model = model
        self.variant = variant
        self.capture_epoch = capture_epoch
        self.changed_objective = changed_objective
        self.dense_host = dense_host
        self.first_epoch_rng = None
        self.first_epoch_weights = None

    def identity_payload(self):
        return _provider_identity(self.variant, host=self.dense_host,
                                  changed_objective=self.changed_objective)

    def training_state_dict(self):
        return {"calibration": "train_only_fixed", "normalizer": "fixed25_train150"}

    def load_training_state_dict(self, value):
        if value != self.training_state_dict():
            raise ValueError("provider state differs")

    def epoch_cases(self, epoch, seed):
        assert seed == 0
        return tuple(range(150))

    def make_batch(self, case_keys, key):
        if (self.capture_epoch is not None and getattr(key, "epoch", None) == self.capture_epoch
                and self.first_epoch_weights is None):
            self.first_epoch_rng = capture_rng_state()
            self.first_epoch_weights = copy.deepcopy(self.model.state_dict())
        x = torch.tensor(case_keys, dtype=torch.float32).reshape(-1, 1) / 150.0
        y = 2 * x + 1
        return TaskBatch(x, None, y, case_keys=tuple(case_keys))

    def loss_denominators(self, batches, phase, arm):
        del phase, arm
        return {"native": sum(batch.targets.numel() for batch in batches)}

    def make_scene(self, value):
        return value

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch, temperature):
        del receivers, execution_mode, phase, temperature
        return model(scene), None

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        del phase, auxiliary_state
        return {"native": LossTerm((predictions - targets).square().sum(), targets.numel())}

    def validation_batches(self):
        for keys in ((0, 1, 2), (3, 4)):
            yield self.make_batch(keys, None)

    def validation_metrics(self, predictions, targets, auxiliary_state):
        del auxiliary_state
        return {"squared_error": float((predictions - targets).square().sum()), "count": targets.numel()}

    def reduce_native_metrics(self, records):
        total = sum(row["squared_error"] for row in records)
        count = sum(row["count"] for row in records)
        return {"field_score": total / count}

    def work_counts(self, batch, predictions, auxiliary_state):
        del batch, predictions, auxiliary_state
        return {}

    def optimizer_groups(self, model, arm, stage):
        del arm, stage
        names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        schedule = ScheduleSpec(
            peak_lr=3.0e-4, warmup_start_lr=3.0e-4, warmup_epochs=0,
            hold_through_epoch=2499, total_epochs=2500, final_lr=3.0e-4,
        )
        return (OptimizerGroupSpec("source_read_joint", names, schedule, weight_decay=1.0e-5),)

def _config():
    return EngineConfig(
        seed=0, microbatch_cases=24, effective_cases=48, total_epochs=2500,
        warmup_epochs=0, open_through_epoch=0, soft_through_epoch=0,
        monitor_every=100, monitor_epochs=(), checkpoint_epochs=tuple(range(100, 2501, 100)),
        latest_every=100, curve_every=10, progress_enabled=False,
        sampling_version="case_epoch_v1", training_mode="joint",
    )


def _sealed_identity(provider: _SourceReadProvider, *, loaded_source: dict):
    extra = _external_identity(provider.identity_payload(), loaded_source)
    config = _config()
    specs = provider.optimizer_groups(provider.model, "A2-D", "joint")
    return {
        **extra,
        "provider_identity": provider.identity_payload(),
        "engine_config": _engine_config_payload(config),
        "selection_policy": asdict(TrainingEngine(_config(), device="cpu").selection),
        "optimizer_schedule_contract": [_optimizer_spec_payload(spec) for spec in specs],
        "sampling_dataset_id": _sampling_dataset_id(provider.identity_payload()),
    }


@pytest.fixture(scope="module")
def dense_parent(tmp_path_factory):
    root = tmp_path_factory.mktemp("source-read-dense-e500")
    torch.manual_seed(0)
    source_model = _TinyA2D()
    provider = _SourceReadProvider(source_model, "D0")
    identity = _external_identity(provider.identity_payload(), _loaded_source("a"))
    engine = TrainingEngine(_config(), device="cpu")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
        patch.setattr(runtime, "print", lambda *args, **kwargs: None, raising=False)
        engine.fit(source_model, provider, root / "dense", identity=identity, arm="A2-D", stop_after=500)
    path = resolve_checkpoint(root / "dense", "latest")
    payload = load_trusted_checkpoint(path, map_location="cpu")
    return root, path, payload


def _child_inputs(root: Path, path: Path, payload: dict, *, extra_parameter: bool = False,
                  changed_objective: bool = False):
    torch.manual_seed(888)
    model = _TinyA2D(child=True, extra_parameter=extra_parameter)
    provider = _SourceReadProvider(
        model, "S-Utility", capture_epoch=501,
        changed_objective=changed_objective,
    )
    loaded = _loaded_source("b")
    identity = _external_identity(provider.identity_payload(), loaded)
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    child_names = sorted(set(dict(model.named_parameters())) - set(payload["model_state_dict"]))
    declaration = _declaration(path, source_sha, child_names, root)
    return model, provider, identity, declaration


def _auxiliary_fit(model, provider, optimizer, *, added_parameter_names, train_case_keys,
                   max_updates, parent_sha256, source_epoch):
    assert source_epoch == 500
    assert max_updates == 2
    assert tuple(train_case_keys) == tuple(range(150))
    # Exercise all RNG streams; the wrapper must restore the exact parent stream.
    random.random()
    np.random.rand()
    torch.rand(())
    module = model.core.direct_reader.source_read
    x = torch.tensor(train_case_keys[:4], dtype=torch.float32).reshape(-1, 1) / 150.0
    target = torch.ones_like(x)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (module(x) - target).square().mean().backward()
        optimizer.step()
    module.train_context_scale.add_(0.25)
    steps = {
        name: int(optimizer.state[dict(model.named_parameters())[name]]["step"].item())
        for name in added_parameter_names
    }
    return {
        "schema": AUXILIARY_FIT_SCHEMA,
        "partition": "TRAIN",
        "source_checkpoint_sha256": parent_sha256,
        "source_epoch": source_epoch,
        "train_case_keys": list(train_case_keys),
        "train_case_count": len(train_case_keys),
        "auxiliary_fit_case_keys": list(train_case_keys[:4]),
        "auxiliary_fit_case_count": 4,
        "auxiliary_fit_case_share": 4 / len(train_case_keys),
        "teacher_rows": 8,
        "optimizer_updates": 2,
        "updated_parameter_names": sorted(added_parameter_names),
        "changed_buffer_names": ["core.direct_reader.source_read.train_context_scale"],
        "optimizer_steps_by_parameter": steps,
    }


def test_source_read_child_preserves_dense_e500_state_moments_rng_history_and_resume(
    dense_parent, monkeypatch,
):
    root, parent_path, parent = dense_parent
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    parent_state = {name: value.clone() for name, value in parent["model_state_dict"].items()}
    parent_rng = copy.deepcopy(parent["rng_state"])
    parent_moments = copy.deepcopy(parent["optimizer_state_by_name"]["state_by_name"])
    model, provider, identity, declaration = _child_inputs(root, parent_path, parent)
    engine = TrainingEngine(_config(), device="cpu")
    make_optimizer = engine._make_optimizer
    attached_optimizer = {}

    def capture_optimizer(*args, **kwargs):
        optimizer, specs = make_optimizer(*args, **kwargs)
        attached_optimizer.update(named_optimizer_state(args[0], optimizer))
        return optimizer, specs

    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "print", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(engine, "_make_optimizer", capture_optimizer)
    child_dir = root / "child-resumed"
    engine.fit(
        model, provider, child_dir, identity=identity, arm="A2-D", stop_after=600,
        branch_from_checkpoint=parent_path, source_read_child=declaration,
        source_read_child_auxiliary_fit=_auxiliary_fit,
    )
    for name, state in parent_moments.items():
        for key, expected in state.items():
            torch.testing.assert_close(attached_optimizer["state_by_name"][name][key], expected, rtol=0, atol=0)
    child_latest = resolve_checkpoint(child_dir, "latest")
    child = load_trusted_checkpoint(child_latest, map_location="cpu")
    assert provider.first_epoch_weights is not None
    for name, expected in parent_state.items():
        torch.testing.assert_close(provider.first_epoch_weights[name], expected, rtol=0, atol=0)
    assert provider.first_epoch_rng["python"] == parent_rng["python"]
    assert provider.first_epoch_rng["numpy"][0] == parent_rng["numpy"][0]
    np.testing.assert_array_equal(provider.first_epoch_rng["numpy"][1], parent_rng["numpy"][1])
    assert provider.first_epoch_rng["numpy"][2:] == parent_rng["numpy"][2:]
    torch.testing.assert_close(provider.first_epoch_rng["torch_cpu"], parent_rng["torch_cpu"], rtol=0, atol=0)
    assert child["epoch"] == 600
    assert [row["epoch"] for row in child["history"]] == list(range(1, 601))
    assert child["history"][:500] == parent["history"]
    assert child["branch_parent"]["inherited_case_visits"] == sum(row["case_visits"] for row in parent["history"])
    assert child["branch_parent"]["inherited_optimizer_updates"] == sum(
        row["optimizer_updates"] for row in parent["history"]
    )
    receipt = child["branch_parent"]["source_read_child"]["auxiliary_fit_receipt"]
    assert receipt["mode"] == "calibrated" and receipt["teacher_rows"] == 8
    assert receipt["train_case_keys"] == list(range(150))
    assert receipt["train_case_count"] == 150
    assert receipt["auxiliary_fit_case_count"] == 4
    assert receipt["auxiliary_fit_case_share"] == 4 / 150
    assert receipt["optimizer_updates"] == 2
    assert receipt["optimizer_state_policy"].startswith("joint AdamW")
    assert receipt["changed_buffer_names"] == ["core.direct_reader.source_read.train_context_scale"]
    changed_buffer = receipt["controller_buffer_changes"][receipt["changed_buffer_names"][0]]
    assert changed_buffer["before_sha256"] != changed_buffer["after_sha256"]
    assert all("auxiliary" not in row for row in child["history"])
    added_names = declaration["added_parameter_names"]
    final_named = child["optimizer_state_by_name"]["state_by_name"]
    assert all(int(final_named[name]["step"]) == 402 for name in added_names)
    assert all(int(final_named[name]["step"]) == 2400 for name in parent["optimizer_state_by_name"]["state_by_name"])
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == parent_sha

    uninterrupted_model, uninterrupted_provider, uninterrupted_identity, uninterrupted_declaration = _child_inputs(
        root, parent_path, parent,
    )
    uninterrupted_declaration["source_patch_artifact"] = declaration["source_patch_artifact"]
    uninterrupted_declaration["source_patch_sha256"] = declaration["source_patch_sha256"]
    uninterrupted_dir = root / "child-uninterrupted"
    TrainingEngine(_config(), device="cpu").fit(
        uninterrupted_model, uninterrupted_provider, uninterrupted_dir, identity=uninterrupted_identity,
        arm="A2-D", stop_after=700, branch_from_checkpoint=parent_path,
        source_read_child=uninterrupted_declaration, source_read_child_auxiliary_fit=_auxiliary_fit,
    )
    resumed_model, resumed_provider, resumed_identity, _ = _child_inputs(root, parent_path, parent)
    TrainingEngine(_config(), device="cpu").fit(
        resumed_model, resumed_provider, child_dir, identity=resumed_identity, arm="A2-D",
        stop_after=700, resume_checkpoint=child_latest,
    )
    resumed = load_trusted_checkpoint(child_latest, map_location="cpu")
    uninterrupted = load_trusted_checkpoint(resolve_checkpoint(uninterrupted_dir, "latest"), map_location="cpu")
    for name, value in uninterrupted["model_state_dict"].items():
        torch.testing.assert_close(value, resumed["model_state_dict"][name], rtol=0, atol=0)
    for name, state in uninterrupted["optimizer_state_by_name"]["state_by_name"].items():
        for key, value in state.items():
            other = resumed["optimizer_state_by_name"]["state_by_name"][name][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                assert value == other
    def without_wall_times(rows):
        return [
            {key: value for key, value in row.items()
             if key not in {"train_seconds", "validation_seconds"}}
            for row in rows
        ]

    assert without_wall_times(resumed["history"]) == without_wall_times(uninterrupted["history"])
    assert resumed["epoch"] == 700
    assert len(resumed["history"]) == 700


@pytest.mark.parametrize("failure", [
    "wrong_parent_age", "wrong_protocol", "identity", "unrelated_parameter",
    "parameter_shape_replacement", "parameter_removal", "unrelated_buffer",
])
def test_source_read_child_rejects_wrong_parent_protocol_identity_and_added_names(
    dense_parent, tmp_path, failure,
):
    root, path, payload = dense_parent
    model, provider, identity, declaration = _child_inputs(
        root, path, payload, extra_parameter=failure == "unrelated_parameter",
    )
    if failure in ("parameter_shape_replacement", "parameter_removal", "unrelated_buffer"):
        torch.manual_seed(888)
        model = _TinyA2D(
            child=True, replacement_shape=failure == "parameter_shape_replacement",
            remove_bias=failure == "parameter_removal",
            extra_buffer=failure == "unrelated_buffer",
        )
        provider = _SourceReadProvider(model, "S-Utility", capture_epoch=501)
        identity = _external_identity(provider.identity_payload(), _loaded_source("b"))
        child_names = sorted(set(dict(model.named_parameters())) - set(payload["model_state_dict"]))
        declaration = _declaration(
            path, hashlib.sha256(path.read_bytes()).hexdigest(), child_names, root,
        )
    if failure == "wrong_parent_age":
        payload = copy.deepcopy(payload)
        payload["epoch"] = 499
    elif failure == "identity":
        provider = _SourceReadProvider(model, "S-Utility", changed_objective=True)
        identity = _external_identity(provider.identity_payload(), _loaded_source("b"))
    elif failure == "unrelated_parameter":
        declaration["added_parameter_names"] = sorted(
            set(dict(model.named_parameters())) - set(payload["model_state_dict"])
        )
    sealed = _sealed_identity(provider, loaded_source=_loaded_source("b"))
    if failure == "wrong_protocol":
        sealed["recipe"]["protocol"] = "formal_d_full_train_v1"
        sealed["provider_identity"]["recipe"]["protocol"] = "formal_d_full_train_v1"
        sealed["recipe_sha256"] = _sha(sealed["recipe"])
    with pytest.raises((ValueError, TypeError)):
        prepare_source_read_child(
            model, payload, sealed,
            declaration, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_path=path, case_keys=tuple(range(150)), stop_after=600,
        )


def test_source_read_child_rejects_unreported_controller_buffer_mutation(dense_parent, monkeypatch):
    root, path, payload = dense_parent
    model, provider, identity, declaration = _child_inputs(root, path, payload)

    def falsified_buffer_receipt(*args, **kwargs):
        receipt = _auxiliary_fit(*args, **kwargs)
        receipt["changed_buffer_names"] = []
        return receipt

    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "print", lambda *args, **kwargs: None, raising=False)
    with pytest.raises(ValueError, match="TRAIN-only boundary"):
        TrainingEngine(_config(), device="cpu").fit(
            model, provider, root / "child-unreported-buffer", identity=identity,
            arm="A2-D", stop_after=600, branch_from_checkpoint=path,
            source_read_child=declaration, source_read_child_auxiliary_fit=falsified_buffer_receipt,
        )


def test_source_read_child_rejects_reusing_dense_parent_output_directory(dense_parent):
    root, path, payload = dense_parent
    model, provider, identity, declaration = _child_inputs(root, path, payload)
    with pytest.raises(ValueError, match="fresh run directory"):
        TrainingEngine(_config(), device="cpu").fit(
            model, provider, root / "dense", identity=identity, arm="A2-D", stop_after=600,
            branch_from_checkpoint=path, source_read_child=declaration,
        )
