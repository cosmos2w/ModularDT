from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]

for directory in (PROJECT / "src", PROJECT / "tools"):
    sys.path.insert(0, str(directory))

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.inherited_attention_child import IAG_SPARSE_CHILD_SCHEMA
from honf_runtime.run_layout import resolve_checkpoint
from honf_runtime.unified_training import (
    EngineConfig,
    LossTerm,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    SelectionPolicy,
    TaskBatch,
    TrainingEngine,
    _engine_config_payload,
    _optimizer_spec_payload,
    _sampling_dataset_id,
    capture_rng_state,
)


class _LifecycleModel(nn.Linear):
    def __init__(self) -> None:
        super().__init__(1, 1)
        self.register_buffer("normalization_epoch", torch.tensor(0, dtype=torch.int64))

    def set_inherited_attention_training_epoch(self, epoch: int) -> None:
        if epoch:
            self.normalization_epoch.fill_(epoch)


class _LifecycleProvider:
    def __init__(self, provider_identity: dict[str, Any]) -> None:
        self._provider_identity = provider_identity
        self.training_draws: list[tuple[int, float, float, float]] = []
        self.prediction_epochs: list[int] = []

    def identity_payload(self) -> dict[str, Any]:
        return copy.deepcopy(self._provider_identity)

    def epoch_cases(self, epoch: int, seed: int) -> tuple[str, ...]:
        del epoch, seed
        return ("case-0",)

    def make_batch(self, case_keys: tuple[str, ...], key: SamplingKey) -> TaskBatch:
        sample_hash = hashlib.sha256(
            json.dumps([key.seed_for("lifecycle-query"), list(case_keys)], separators=(",", ":")).encode()
        ).hexdigest()
        return TaskBatch(
            scene_inputs=torch.ones((len(case_keys), 1), dtype=torch.float32),
            receivers=None,
            targets=torch.full((len(case_keys), 1), 1.7, dtype=torch.float32),
            auxiliary={"query_sampling_sha256": sample_hash},
            case_keys=case_keys,
        )

    def loss_denominators(self, batches, phase, arm):
        del phase, arm
        return {"native": float(sum(batch.targets.numel() for batch in batches))}

    def make_scene(self, scene_inputs):
        return scene_inputs

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch, temperature):
        del receivers, execution_mode, phase, temperature
        model.set_inherited_attention_training_epoch(int(epoch))
        self.prediction_epochs.append(int(model.normalization_epoch))
        prediction = model(scene)
        if model.training:
            py_draw = random.random()
            np_draw = float(np.random.random())
            torch_draw = float(torch.rand(()))
            self.training_draws.append((epoch, py_draw, np_draw, torch_draw))
            prediction = prediction + (py_draw + np_draw + torch_draw) * 0.01
        return prediction, None

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        del phase, auxiliary_state
        return {"native": LossTerm((predictions - targets).square().sum(), targets.numel())}

    def validation_batches(self):
        yield self.make_batch(("case-0",), SamplingKey(7, 501, 0, 0, "joint", "IAG-C"))

    def validation_metrics(self, predictions, targets, auxiliary_state):
        del auxiliary_state
        return {"squared_error": float((predictions - targets).square().sum()), "count": targets.numel()}

    def reduce_native_metrics(self, records):
        total = sum(row["squared_error"] for row in records)
        count = sum(row["count"] for row in records)
        return {"field_score": total / count}

    def optimizer_groups(self, model, arm, stage):
        del stage
        if arm != "IAG-C":
            raise ValueError("The lifecycle fixture only accepts IAG-C.")
        return (
            OptimizerGroupSpec(
                name="iag_adamw_all_parameters",
                parameter_names=("weight", "bias"),
                schedule=ScheduleSpec(
                    peak_lr=3.0e-4,
                    warmup_start_lr=3.0e-4,
                    warmup_epochs=0,
                    hold_through_epoch=501,
                    total_epochs=502,
                    final_lr=3.0e-4,
                ),
                weight_decay=1.0e-5,
            ),
        )

    def work_counts(self, batch, predictions, auxiliary_state):
        del predictions, auxiliary_state
        return {"query_rows": int(batch.targets.numel())}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _draw_from_rng(state: dict[str, Any]) -> tuple[float, float, float]:
    previous = capture_rng_state()
    try:
        from honf_runtime.unified_training import restore_rng_state

        restore_rng_state(state)
        return random.random(), float(np.random.random()), float(torch.rand(()))
    finally:
        from honf_runtime.unified_training import restore_rng_state

        restore_rng_state(previous)


def _make_recipe(*, sparse: bool) -> dict[str, Any]:
    if sparse:
        transition = {
            "soft_parent_epoch": 500,
            "blend_start_epoch": 501,
            "blend_end_epoch": 800,
            "exact_sparse_start_epoch": 801,
            "hard_stop_epoch": 2500,
        }
        return {
            "protocol": "inherited_attention_development_v1",
            "normalization": "measure_entmax15",
            "trajectory": "e500_sparse_child",
            "sparse_transition": transition,
            "model": {
                "normalization": "measure_entmax15",
                "scene_executor": "selected",
                "query_executor": "indexed",
            },
        }
    return {
        "protocol": "inherited_attention_development_v1",
        "normalization": "soft",
        "trajectory": "fresh_soft",
        "model": {
            "normalization": "soft",
            "scene_executor": "dense",
            "query_executor": "indexed",
        },
    }


def _base_identity(recipe: dict[str, Any], provider_identity: dict[str, Any]) -> dict[str, Any]:
    transition = recipe.get("sparse_transition")
    return {
        "workflow": "inherited_attention_joint_v1",
        "model_family": "inherited_attention_v1",
        "recipe": recipe,
        "recipe_sha256": _canonical_sha256(recipe),
        "loaded_source": {"files": {"toy/lifecycle.py": "a" * 64}, "sha256": "b" * 64},
        "provider_identity": provider_identity,
        "absolute_epoch_transition": transition,
    }


def _provider_identity(*, sparse: bool) -> dict[str, Any]:
    normalization = "measure_entmax15" if sparse else "soft"
    transition = _make_recipe(sparse=True)["sparse_transition"] if sparse else None
    return {
        "dataset": "toy",
        "manifest_fingerprint": "c" * 64,
        "model_config": {
            "normalization": normalization,
            "scene_executor": "selected" if sparse else "dense",
            "query_executor": "indexed",
        },
        "training": {
            "normalization": normalization,
            "absolute_epoch_transition": transition,
        },
    }


def _sealed_identity(base: dict[str, Any], model: nn.Module, provider: _LifecycleProvider,
                     engine: TrainingEngine) -> dict[str, Any]:
    groups = tuple(provider.optimizer_groups(model, "IAG-C", "joint"))
    provider_identity = provider.identity_payload()
    return {
        **base,
        "provider_identity": provider_identity,
        "engine_config": _engine_config_payload(engine.config),
        "selection_policy": asdict(engine.selection),
        "optimizer_schedule_contract": [_optimizer_spec_payload(spec) for spec in groups],
        "sampling_dataset_id": _sampling_dataset_id(provider_identity),
    }


def _synthetic_history() -> list[dict[str, Any]]:
    return [
        {
            "epoch": epoch,
            "case_visits": 1,
            "optimizer_updates": 1,
            "microbatches": 1,
            "partial_update_cases": 0,
        }
        for epoch in range(1, 501)
    ]


def test_sparse_child_attaches_at_501_and_exactly_resumes_at_502(
    tmp_path: Path,
) -> None:
    torch.manual_seed(19)
    random.seed(23)
    np.random.seed(29)
    config = EngineConfig(
        seed=7,
        microbatch_cases=1,
        effective_cases=1,
        total_epochs=502,
        monitor_every=1,
        checkpoint_epochs=(501, 502),
        latest_every=1,
        curve_every=None,
        progress_enabled=False,
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        training_mode="joint",
    )
    engine = TrainingEngine(config, device="cpu", selection=SelectionPolicy(field_metric="field_score"))
    source_model = _LifecycleModel()
    with torch.no_grad():
        source_model.weight.fill_(0.35)
        source_model.bias.fill_(-0.1)
        source_model.normalization_epoch.fill_(500)

    sparse_provider_identity = _provider_identity(sparse=True)
    provider = _LifecycleProvider(sparse_provider_identity)
    sparse_recipe = _make_recipe(sparse=True)
    sparse_base = _base_identity(sparse_recipe, sparse_provider_identity)
    sparse_identity = _sealed_identity(sparse_base, source_model, provider, engine)

    soft_recipe = _make_recipe(sparse=False)
    soft_provider_identity = _provider_identity(sparse=False)
    soft_base = _base_identity(soft_recipe, soft_provider_identity)
    soft_identity = _sealed_identity(soft_base, source_model, _LifecycleProvider(soft_provider_identity), engine)

    optimizer_schedule_contract = sparse_identity["optimizer_schedule_contract"]
    optimizer_groups = [
        {"name": item["name"], "parameter_names": item["parameter_names"]}
        for item in optimizer_schedule_contract
    ]
    optimizer_state = {
        "groups": optimizer_groups,
        "state_by_name": {
            "weight": {
                "step": torch.tensor(500.0),
                "exp_avg": torch.full_like(source_model.weight, 0.02),
                "exp_avg_sq": torch.full_like(source_model.weight, 0.003),
            },
            "bias": {
                "step": torch.tensor(500.0),
                "exp_avg": torch.full_like(source_model.bias, -0.01),
                "exp_avg_sq": torch.full_like(source_model.bias, 0.004),
            },
        },
    }
    torch.manual_seed(37)
    random.seed(41)
    np.random.seed(43)
    parent_rng = capture_rng_state()
    expected_501 = _draw_from_rng(parent_rng)

    parent_dir = tmp_path / "soft-parent"
    parent_dir.mkdir()
    parent_path = parent_dir / "epoch_0500_model.pt"
    source_payload = {
        "checkpoint_schema_version": 1,
        "workflow": "inherited_attention_joint_v1",
        "experiment_identity": soft_identity,
        "arm": "IAG-C",
        "epoch": 500,
        "current_epoch": 500,
        "model_state_dict": copy.deepcopy(source_model.state_dict()),
        "optimizer_state_by_name": optimizer_state,
        "optimizer_group_names": [group["name"] for group in optimizer_groups],
        "optimizer_schedule_contract": optimizer_schedule_contract,
        "rng_state": parent_rng,
        "sampler_state": {"completed_epoch": 500, "next_epoch": 501},
        "history": _synthetic_history(),
        "best_field_score": 0.25,
        "best_response_guarded_score": float("inf"),
        "provider_training_state": {},
    }
    torch.save(source_payload, parent_path)
    source_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    declaration = {
        "schema": IAG_SPARSE_CHILD_SCHEMA,
        "source_checkpoint": str(parent_path.resolve()),
        "source_checkpoint_sha256": source_sha,
        "source_recipe_sha256": soft_identity["recipe_sha256"],
        "target_recipe_sha256": sparse_identity["recipe_sha256"],
        "source_epoch": 500,
    }

    # One actual CPU AdamW update attaches the sparse child at absolute epoch 501.
    model = _LifecycleModel()
    output = tmp_path / "sparse-child"
    result_501 = engine.fit(
        model,
        provider,
        output,
        identity=sparse_base,
        arm="IAG-C",
        stop_after=501,
        branch_from_checkpoint=parent_path,
        inherited_attention_sparse_child=declaration,
    )
    latest_501 = resolve_checkpoint(output, "latest")
    payload_501 = load_trusted_checkpoint(latest_501, map_location="cpu")
    assert result_501["completed_epoch"] == 501
    assert provider.training_draws[0][0] == 501
    assert provider.training_draws[0][1:] == expected_501
    assert int(payload_501["model_state_dict"]["normalization_epoch"]) == 501
    assert [row["epoch"] for row in payload_501["history"][:500]] == list(range(1, 501))
    assert payload_501["history"][-1]["epoch"] == 501
    assert sum(row["case_visits"] for row in payload_501["history"]) == 501
    assert sum(row["optimizer_updates"] for row in payload_501["history"]) == 501
    assert sum(row["optimizer_updates"] for row in payload_501["history"] if row["epoch"] >= 501) == 1
    assert payload_501["branch_parent"]["sha256"] == source_sha
    child_receipt = payload_501["branch_parent"]["inherited_attention_sparse_child"]
    assert child_receipt["source_epoch"] == 500
    assert child_receipt["next_epoch"] == 501
    assert child_receipt["exact_inherited_moments_verified"] is True
    assert child_receipt["new_optimizer_states_at_attachment"] == 0
    state_501 = payload_501["optimizer_state_by_name"]["state_by_name"]
    assert set(state_501) == {"weight", "bias"}
    assert {int(state["step"]) for state in state_501.values()} == {501}
    assert not torch.equal(state_501["weight"]["exp_avg"], optimizer_state["state_by_name"]["weight"]["exp_avg"])

    expected_502 = _draw_from_rng(payload_501["rng_state"])
    # A second actual CPU AdamW update resumes only from the child's current latest state.
    result_502 = engine.fit(
        model,
        provider,
        output,
        identity=sparse_base,
        arm="IAG-C",
        stop_after=502,
        resume_checkpoint=latest_501,
    )
    latest_502 = resolve_checkpoint(output, "latest")
    payload_502 = load_trusted_checkpoint(latest_502, map_location="cpu")
    assert result_502["completed_epoch"] == 502
    assert len(provider.training_draws) == 2
    assert provider.training_draws[1][0] == 502
    assert provider.training_draws[1][1:] == expected_502
    assert int(payload_502["model_state_dict"]["normalization_epoch"]) == 502
    assert [row["epoch"] for row in payload_502["history"][:500]] == list(range(1, 501))
    assert [row["epoch"] for row in payload_502["history"][-2:]] == [501, 502]
    assert sum(row["case_visits"] for row in payload_502["history"]) == 502
    assert sum(row["optimizer_updates"] for row in payload_502["history"]) == 502
    assert sum(row["optimizer_updates"] for row in payload_502["history"] if row["epoch"] >= 501) == 2
    assert payload_502["branch_parent"] == payload_501["branch_parent"]
    state_502 = payload_502["optimizer_state_by_name"]["state_by_name"]
    assert {int(state["step"]) for state in state_502.values()} == {502}
    assert latest_502.name == "latest_model.pt"
    assert resolve_checkpoint(output, "best_field").is_file()
    assert load_trusted_checkpoint(resolve_checkpoint(output, "best_field"), map_location="cpu")["epoch"] in (501, 502)
    assert provider.prediction_epochs.count(501) >= 2
    assert provider.prediction_epochs.count(502) >= 2
