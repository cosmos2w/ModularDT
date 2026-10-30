"""Actual shared-engine clean-stop/resume qualification for R13-8 A and L IDs."""
from __future__ import annotations

import json
import os
import random
import copy
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from channelthermal.training.r13_8_anchored_task import (
    ARM_A,
    ARM_L,
    CORE_SCHEMA,
    PROVIDER_SCHEMA,
    TRAINING_STATE_SCHEMA,
    WORKFLOW,
)
from honf_runtime.run_layout import resolve_checkpoint
from honf_runtime.unified_training import (
    EngineConfig,
    LossTerm,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    TaskBatch,
    TrainingEngine,
)


_EPOCH_HORIZON = 500


class _SyntheticAnchoredModel(nn.Module):
    def __init__(self, architecture: str):
        super().__init__()
        self.architecture = architecture
        self.linear = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class _SyntheticAnchoredProvider:
    """Tiny stochastic provider with explicit checkpointable sampling state."""

    def __init__(self, architecture: str, *, clean_stop_at: int | None = None):
        self.architecture = architecture
        self.clean_stop_at = clean_stop_at
        self.generator = torch.Generator(device="cpu").manual_seed(1308)
        self.batch_calls = 0
        self.completed_epoch_calls = 0
        self.last_sampling_key: tuple[int, int, int, int, str, str] | None = None
        self.recipe = {
            "schema": "synthetic_r13_8_shared_engine_resume_recipe_v1",
            "model": {"schema": CORE_SCHEMA, "architecture": architecture},
            "training": {
                "total_epochs": _EPOCH_HORIZON,
                "optimizer_schedule_horizon": _EPOCH_HORIZON,
                "synthetic_case_count": 2,
                "effective_cases": 2,
                "microbatch_cases": 2,
            },
        }

    def identity_payload(self) -> dict:
        return {
            "schema": PROVIDER_SCHEMA,
            "training_state_schema": TRAINING_STATE_SCHEMA,
            "workflow": WORKFLOW,
            "architecture": self.architecture,
            "member_routing_schema": CORE_SCHEMA,
            "dataset": "synthetic_only_no_native_training_data",
            "training_case_ids": ["synthetic-case-0", "synthetic-case-1"],
            "native_only_training": True,
            "recipe": self.recipe,
        }

    def epoch_cases(self, epoch: int, seed: int):
        del epoch, seed
        return ("synthetic-case-0", "synthetic-case-1")

    def make_batch(self, case_keys, key: SamplingKey):
        self.batch_calls += 1
        self.last_sampling_key = (
            int(key.seed), int(key.epoch), int(key.update_index), int(key.microbatch_index),
            str(key.stage), str(key.arm),
        )
        local = torch.rand((len(case_keys), 1), generator=self.generator)
        # Process RNG is checkpointed separately from provider-local state.
        process_noise = torch.rand((len(case_keys), 1)) * 0.03
        inputs = local + process_noise
        targets = 1.7 * inputs - 0.35
        return TaskBatch(
            scene_inputs=inputs,
            receivers=None,
            targets=targets,
            auxiliary={"workflow": WORKFLOW, "sampling_epoch": int(key.epoch)},
            case_keys=tuple(case_keys),
        )

    def loss_denominators(self, batches, phase, arm):
        del phase, arm
        return {"native": float(sum(batch.targets.numel() for batch in batches))}

    def make_scene(self, scene_inputs):
        return scene_inputs

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch, temperature):
        del receivers, execution_mode, phase, epoch, temperature
        return model(scene), None

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        del phase, auxiliary_state
        return {"native": LossTerm((predictions - targets).square().sum(), targets.numel())}

    def validation_batches(self):
        inputs = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float32)
        targets = 1.7 * inputs - 0.35
        yield TaskBatch(inputs, None, targets, case_keys=("validation-0", "validation-1", "validation-2"))

    def validation_metrics(self, predictions, targets, auxiliary_state):
        del auxiliary_state
        return {"squared_error": float((predictions - targets).square().sum()), "count": targets.numel()}

    def reduce_native_metrics(self, records):
        total = sum(row["squared_error"] for row in records)
        count = sum(row["count"] for row in records)
        return {"field_score": total / count}

    def optimizer_groups(self, model, arm, stage):
        del arm, stage
        return (OptimizerGroupSpec(
            name="synthetic_predictor",
            parameter_names=tuple(name for name, _parameter in model.named_parameters()),
            schedule=ScheduleSpec(
                peak_lr=0.01,
                warmup_start_lr=0.01,
                hold_through_epoch=400,
                total_epochs=_EPOCH_HORIZON,
                final_lr=0.001,
            ),
            weight_decay=0.01,
        ),)

    def work_counts(self, batch, predictions, auxiliary_state):
        del predictions, auxiliary_state
        return {"synthetic_rows": batch.targets.numel()}

    def completed_training_epoch(self, model, engine, epoch: int, output_dir: Path, row):
        del model, engine, row
        self.completed_epoch_calls += 1
        if epoch == self.clean_stop_at:
            request_path = Path(output_dir) / "CLEAN_STOP_REQUEST.json"
            request_path.write_text(json.dumps({
                "request_id": f"r13-8-{self.architecture}-synthetic-clean-stop-e{epoch}",
                "requested_by": "synthetic shared-engine resume test",
            }), encoding="utf-8")

    def training_state_dict(self):
        return {
            "schema": TRAINING_STATE_SCHEMA,
            "generator_state": self.generator.get_state().clone(),
            "batch_calls": self.batch_calls,
            "completed_epoch_calls": self.completed_epoch_calls,
            "last_sampling_key": self.last_sampling_key,
        }

    def load_training_state_dict(self, state):
        if not isinstance(state, Mapping) or state.get("schema") != TRAINING_STATE_SCHEMA:
            raise ValueError("Synthetic R13-8 provider state schema differs.")
        self.generator.set_state(state["generator_state"].cpu())
        self.batch_calls = int(state["batch_calls"])
        self.completed_epoch_calls = int(state["completed_epoch_calls"])
        self.last_sampling_key = state["last_sampling_key"]


def _engine_config() -> EngineConfig:
    return EngineConfig(
        seed=138,
        microbatch_cases=2,
        effective_cases=2,
        total_epochs=_EPOCH_HORIZON,
        warmup_epochs=0,
        monitor_every=_EPOCH_HORIZON,
        monitor_epochs=(100,),
        checkpoint_epochs=(100, 101, _EPOCH_HORIZON),
        latest_every=None,
        curve_every=None,
        progress_enabled=False,
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        training_mode="joint",
    )


def _identity(architecture: str) -> dict:
    recipe = _SyntheticAnchoredProvider(architecture).recipe
    return {
        "workflow": WORKFLOW,
        "architecture": architecture,
        "member_routing_schema": CORE_SCHEMA,
        "fixture_kind": "synthetic_engine_resume_qualification_only",
        "recipe": recipe,
    }


def _start(architecture: str, *, clean_stop_at: int | None = None):
    seed = 138
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return _SyntheticAnchoredModel(architecture), _SyntheticAnchoredProvider(
        architecture, clean_stop_at=clean_stop_at,
    )


def _assert_nested_exact(left, right, path="root"):
    if torch.is_tensor(left):
        assert torch.is_tensor(right), path
        assert torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray), path
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping) and set(left) == set(right), path
        for key in left:
            _assert_nested_exact(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left)) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _assert_nested_exact(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


@pytest.mark.parametrize("architecture", (ARM_A, ARM_L))
def test_r13_8_workflow_clean_stops_at_100_and_resumes_exactly_to_101(tmp_path, architecture):
    temp_root = Path(os.environ.get("TMPDIR", "/tmp")).resolve()
    assert tmp_path.resolve().is_relative_to(temp_root), "fixture outputs must stay under TMPDIR"
    assert "wanglz" in str(temp_root) and "/data2/" in f"{temp_root}/", "TMPDIR must use DATA2"

    identity = _identity(architecture)
    config = _engine_config()
    engine = TrainingEngine(config, device="cpu")
    split_dir = tmp_path / f"split_{architecture}"
    split_model, split_provider = _start(architecture, clean_stop_at=100)
    stopped = engine.fit(
        split_model,
        split_provider,
        split_dir,
        identity=identity,
        arm=architecture,
        stop_after=_EPOCH_HORIZON,
    )
    assert stopped["status"] == "clean_stopped"
    assert stopped["completed_epoch"] == 100
    assert json.loads((split_dir / "logs" / "clean_stop_acknowledged.json").read_text())[
        "epoch"] == 100
    source = torch.load(resolve_checkpoint(split_dir, "latest"), map_location="cpu", weights_only=False)
    assert source["epoch"] == source["sampler_state"]["completed_epoch"] == 100
    assert source["workflow"] == source["experiment_identity"]["workflow"] == WORKFLOW
    assert source["experiment_identity"]["architecture"] == architecture
    assert source["experiment_identity"]["engine_config"]["total_epochs"] == _EPOCH_HORIZON
    assert source["optimizer_schedule_contract"][0]["schedule"]["total_epochs"] == _EPOCH_HORIZON
    assert source["provider_training_state"]["batch_calls"] == 100

    resumed_model, resumed_provider = _start(architecture)
    resumed = engine.fit(
        resumed_model,
        resumed_provider,
        split_dir,
        identity=identity,
        arm=architecture,
        stop_after=101,
        resume_checkpoint=resolve_checkpoint(split_dir, "latest"),
    )
    assert resumed["status"] == "completed"
    assert resumed["completed_epoch"] == 101

    control_dir = tmp_path / f"control_{architecture}"
    control_model, control_provider = _start(architecture)
    control = engine.fit(
        control_model,
        control_provider,
        control_dir,
        identity=identity,
        arm=architecture,
        stop_after=101,
    )
    assert control["completed_epoch"] == 101

    resumed_state = torch.load(resolve_checkpoint(split_dir, "latest"), map_location="cpu", weights_only=False)
    control_state = torch.load(resolve_checkpoint(control_dir, "latest"), map_location="cpu", weights_only=False)
    assert resumed_state["epoch"] == control_state["epoch"] == 101
    assert resumed_state["experiment_identity"] == control_state["experiment_identity"]
    assert resumed_state["experiment_identity"]["engine_config"]["total_epochs"] == _EPOCH_HORIZON
    assert resumed_state["optimizer_schedule_contract"][0]["schedule"]["total_epochs"] == _EPOCH_HORIZON
    assert resumed_state["sampler_state"]["completed_epoch"] == 101
    assert resumed_state["sampler_state"]["next_epoch"] == 102
    assert resumed_state["history"][-1]["epoch"] == control_state["history"][-1]["epoch"] == 101
    assert resumed_state["history"][-1]["train_losses"] == control_state["history"][-1]["train_losses"]

    _assert_nested_exact(resumed_state["model_state_dict"], control_state["model_state_dict"], "model")
    _assert_nested_exact(resumed_state["optimizer_state_by_name"], control_state["optimizer_state_by_name"], "optimizer")
    _assert_nested_exact(resumed_state["rng_state"], control_state["rng_state"], "process_rng")
    _assert_nested_exact(
        resumed_state["provider_training_state"], control_state["provider_training_state"], "provider_state",
    )
    _assert_nested_exact(resumed_state["sampler_state"], control_state["sampler_state"], "sampler_state")
    assert resumed_state["provider_training_state"]["completed_epoch_calls"] == 101
    assert resumed_state["provider_training_state"]["batch_calls"] == 101


@pytest.mark.parametrize("measure", ("inventory_unit_total", "active_domain"))
def test_detail_workflow_engine_state_resumes_exactly(tmp_path, measure):
    """Exercise the shared engine with R14 identity and a synthetic provider."""
    from channelthermal.training.r13_8_anchored_task import (
        ARM_DETAIL, DETAIL_CORE_SCHEMA, DETAIL_PROVIDER_SCHEMA, DETAIL_STATE_SCHEMA, DETAIL_WORKFLOW,
    )

    class DetailProvider(_SyntheticAnchoredProvider):
        def identity_payload(self):
            return {**super().identity_payload(), 'schema': DETAIL_PROVIDER_SCHEMA,
                    'training_state_schema': DETAIL_STATE_SCHEMA, 'workflow': DETAIL_WORKFLOW,
                    'member_routing_schema': DETAIL_CORE_SCHEMA}

        def training_state_dict(self):
            return {**super().training_state_dict(), 'schema': DETAIL_STATE_SCHEMA}

        def load_training_state_dict(self, state):
            if state.get('schema') != DETAIL_STATE_SCHEMA:
                raise ValueError('Synthetic Detail provider state schema differs.')
            super().load_training_state_dict({**state, 'schema': TRAINING_STATE_SCHEMA})

    def start(clean_stop_at=None):
        model, _ = _start(ARM_DETAIL)
        provider = DetailProvider(ARM_DETAIL, clean_stop_at=clean_stop_at)
        provider.recipe['model'] = {'schema': DETAIL_CORE_SCHEMA, 'architecture': ARM_DETAIL,
                                    'access_measure': measure}
        return model, provider

    model, provider = start(clean_stop_at=1)
    identity = {'workflow': DETAIL_WORKFLOW, 'architecture': ARM_DETAIL,
                'member_routing_schema': DETAIL_CORE_SCHEMA, 'recipe': copy.deepcopy(provider.recipe),
                'provider_schema': DETAIL_PROVIDER_SCHEMA, 'fixture_kind': 'synthetic_engine_resume'}
    config = replace(_engine_config(), monitor_epochs=(1,), checkpoint_epochs=(1, 2, _EPOCH_HORIZON))
    engine = TrainingEngine(config, device='cpu')
    split_dir = tmp_path / 'split'
    stopped = engine.fit(model, provider, split_dir, identity=identity, arm=ARM_DETAIL, stop_after=2)
    assert stopped['status'] == 'clean_stopped' and stopped['completed_epoch'] == 1
    model, provider = start()
    resumed = engine.fit(model, provider, split_dir, identity=identity, arm=ARM_DETAIL, stop_after=2,
                         resume_checkpoint=resolve_checkpoint(split_dir, 'latest'))
    assert resumed['completed_epoch'] == 2
    control_model, control_provider = start()
    control_dir = tmp_path / 'control'
    engine.fit(control_model, control_provider, control_dir, identity=identity, arm=ARM_DETAIL, stop_after=2)
    resumed_state = torch.load(resolve_checkpoint(split_dir, 'latest'), map_location='cpu', weights_only=False)
    control_state = torch.load(resolve_checkpoint(control_dir, 'latest'), map_location='cpu', weights_only=False)
    for name in ('model_state_dict', 'optimizer_state_by_name', 'rng_state', 'sampler_state',
                 'provider_training_state', 'optimizer_schedule_contract', 'experiment_identity'):
        _assert_nested_exact(resumed_state[name], control_state[name], name)
    assert resumed_state['epoch'] == control_state['epoch'] == 2


@pytest.mark.parametrize("architecture", (ARM_A, ARM_L))
def test_r13_8_completed_100_horizon_extends_to_500_with_exact_101_recovery(tmp_path, architecture):
    """Exercise the formal endpoint's changing execution horizon, not only a pause."""
    from honf_runtime.unified_training import _resume_identity_amendment

    identity = _identity(architecture)
    initial_config = replace(_engine_config(), total_epochs=100, checkpoint_epochs=(100,))
    extended_config = replace(
        initial_config, total_epochs=500, checkpoint_epochs=(100, 200, 300, 400, 500),
    )
    split_dir = tmp_path / f"extended_{architecture}"
    model, provider = _start(architecture)
    completed = TrainingEngine(initial_config, device="cpu").fit(
        model, provider, split_dir, identity=identity, arm=architecture, stop_after=100,
    )
    assert completed["status"] == "completed" and completed["completed_epoch"] == 100
    source_path = resolve_checkpoint(split_dir, "latest")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    assert source["experiment_identity"]["engine_config"]["total_epochs"] == 100
    assert source["optimizer_schedule_contract"][0]["schedule"]["total_epochs"] == 500

    resumed_model, resumed_provider = _start(architecture)
    resumed = TrainingEngine(extended_config, device="cpu").fit(
        resumed_model, resumed_provider, split_dir, identity=identity, arm=architecture,
        stop_after=101, resume_checkpoint=source_path, allow_horizon_extension=True,
    )
    assert resumed["status"] == "completed" and resumed["completed_epoch"] == 101
    resumed_state = torch.load(resolve_checkpoint(split_dir, "latest"), map_location="cpu", weights_only=False)
    amendment = resumed_state["resume_amendments"][-1]
    assert amendment["kind"] == "explicit_member_routing_horizon_extension"
    assert amendment["previous_horizon"] == 100 and amendment["execution_horizon"] == 500
    assert amendment["optimizer_schedule_unchanged"] is True
    assert resumed_state["experiment_identity"]["engine_config"]["checkpoint_epochs"] == (100, 200, 300, 400, 500)
    _assert_nested_exact(source["optimizer_schedule_contract"], resumed_state["optimizer_schedule_contract"])

    control_dir = tmp_path / f"extended_control_{architecture}"
    control_model, control_provider = _start(architecture)
    control = TrainingEngine(extended_config, device="cpu").fit(
        control_model, control_provider, control_dir, identity=identity, arm=architecture, stop_after=101,
    )
    assert control["completed_epoch"] == 101
    control_state = torch.load(resolve_checkpoint(control_dir, "latest"), map_location="cpu", weights_only=False)
    for name in ("model_state_dict", "optimizer_state_by_name", "rng_state", "sampler_state",
                 "provider_training_state", "experiment_identity"):
        _assert_nested_exact(resumed_state[name], control_state[name], name)
    assert resumed_state["history"][-1]["train_losses"] == control_state["history"][-1]["train_losses"]
    assert resumed_state["history"][-1]["epoch"] == 101

    changed_recipe = copy.deepcopy(resumed_state["experiment_identity"])
    changed_recipe["recipe"]["training"]["optimizer_schedule_horizon"] = 100
    with pytest.raises(ValueError, match="identity"):
        _resume_identity_amendment(source["experiment_identity"], changed_recipe, allow_horizon_extension=True)
