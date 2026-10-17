"""CPU algebraic check of native child checkpoint-boundary continuation.

This fixture exercises the maintained optimizer and checkpoint serializers with
synthetic model data. It is not a D0 fit or evidence about native training.
"""

from __future__ import annotations

import io
import random
from dataclasses import replace

import numpy as np
import torch

from honf_forward_core.native_retrieval.core import NativeRetrievalCore
from honf_runtime.unified_training import (
    EngineConfig,
    LossTerm,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    TaskBatch,
    TrainingEngine,
    _sampling_dataset_id,
    named_optimizer_state,
    restore_rng_state,
)
from tests.test_content_hypergraph_v2_core import _core, _query, _scene


class _OneCaseNativeEngineProvider:
    def __init__(self):
        self.batch = TaskBatch(
            scene_inputs=_scene(1),
            receivers=_query(1, 3),
            targets=torch.zeros((1, 3, 3)),
            case_keys=("fit-case-0001",),
        )
        self.seen_routing_states = []

    def identity_payload(self):
        return {
            "dataset": "native-engine-fit-fixture",
            "training_case_ids": ["fit-case-0001"],
        }

    def epoch_cases(self, _epoch, _seed):
        return ["fit-case-0001"]

    def make_batch(self, case_keys, _key):
        return TaskBatch(
            self.batch.scene_inputs, self.batch.receivers, self.batch.targets,
            case_keys=tuple(case_keys),
        )

    def optimizer_groups(self, model, _arm, _phase):
        return (OptimizerGroupSpec(
            name="synthetic_native_core",
            parameter_names=tuple(name for name, parameter in model.named_parameters()
                                  if parameter.requires_grad),
            schedule=ScheduleSpec(peak_lr=2e-4, hold_through_epoch=1, total_epochs=2),
            weight_decay=1e-5,
        ),)

    def loss_denominators(self, batches, _phase, _arm):
        return {"native_field": float(sum(batch.targets.numel() for batch in batches))}

    def make_scene(self, scene_inputs):
        return scene_inputs

    def predict_native(self, model, scene, receivers, _arm, _phase, epoch, temperature):
        del temperature
        self.seen_routing_states.append(model.native_routing_state())
        return model.decode(model.prepare(scene), receivers), {}

    def loss_terms(self, predictions, targets, _phase, _auxiliary_state):
        error = predictions.nonlinear_field - targets
        return {"native_field": LossTerm(error.square().sum(), targets.numel())}

    def work_counts(self, _batch, _predictions, _auxiliary_state):
        return {"synthetic_native_forward": 1}

    def validation_batches(self):
        yield self.batch

    def validation_metrics(self, predictions, targets, _auxiliary_state):
        error = predictions.nonlinear_field - targets
        return {"field_score": float(error.square().mean().detach())}

    def reduce_native_metrics(self, records):
        return {"field_score": sum(float(row["field_score"]) for row in records) / len(records)}


class _OptimizerBoundaryFixture:
    """Only the identity, case order, and named optimizer boundary are needed."""

    def __init__(self, model):
        self._identity = {
            "dataset": "native-retrieval-resume-fixture",
            "training_case_ids": ["case-a", "case-b", "case-c"],
        }
        self.case_keys = tuple(self._identity["training_case_ids"])
        self.model = model

    def identity_payload(self):
        return self._identity

    def epoch_cases(self, _epoch, _seed):
        return list(self.case_keys)

    def optimizer_groups(self, model, _arm, _phase):
        return (OptimizerGroupSpec(
            name="a2d_native_fixture",
            parameter_names=tuple(name for name, parameter in model.named_parameters()
                                  if parameter.requires_grad),
            schedule=ScheduleSpec(peak_lr=2e-4, hold_through_epoch=1000, total_epochs=2500),
            weight_decay=1e-5,
        ),)


def _synthetic_step(model, optimizer, engine, provider, epoch, dataset_id):
    order, observed_dataset_id = engine.epoch_case_order(provider, epoch)
    assert observed_dataset_id == dataset_id
    case_key = order[0]
    sampling_key = SamplingKey(
        engine.config.seed, engine.sampler_epoch(epoch), 0, 0,
        engine.config.stage_for_epoch(epoch), "A2-D",
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        dataset_id=dataset_id,
    )
    query_rng = sampling_key.numpy_rng(case_key, "synthetic-native-query")
    coordinates = torch.as_tensor(query_rng.uniform(0.05, 0.95, size=(1, 3, 2)), dtype=torch.float32)
    query = replace(_query(1, 3), coordinates=coordinates)
    model.set_native_training_epoch(epoch)
    scene = _scene(1)
    heat = torch.tensor([[1.0, -0.2, 0.5]])
    prepared = model.prepare(scene)
    output = model.decode(prepared, query, heat)
    # The random scalar makes process RNG restoration part of the same update.
    random_scale = torch.rand(())
    loss = (output.nonlinear_field.square().sum()
            + output.thermal_temperature.square().sum()
            + 0.01 * random_scale * output.nonlinear_field.sum())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return case_key, order, coordinates, random_scale.detach().clone(), loss.detach().clone()


def _assert_named_optimizer_equal(left, right):
    assert left["groups"] == right["groups"]
    assert set(left["state_by_name"]) == set(right["state_by_name"])
    for name in left["state_by_name"]:
        a, b = left["state_by_name"][name], right["state_by_name"][name]
        assert set(a) == set(b)
        for key, value in a.items():
            if torch.is_tensor(value):
                assert torch.equal(value, b[key]), (name, key)
            else:
                assert value == b[key], (name, key)


def test_native_checkpoint_restores_named_adamw_age_sampler_rng_and_next_update():
    random.seed(781)
    np.random.seed(781)
    torch.manual_seed(781)
    parent = _core("A2-D").train()
    provider = _OptimizerBoundaryFixture(parent)
    engine = TrainingEngine(
        EngineConfig(
            seed=29, microbatch_cases=1, effective_cases=1,
            total_epochs=2500, warmup_epochs=500, training_mode="joint",
            sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        ),
        device="cpu",
    )

    # Seed a real, named AdamW boundary using one synthetic A2-D model step.
    source_optimizer, source_specs = engine._make_optimizer(
        parent, provider, "A2-D", 500, None,
    )
    scene, query = _scene(1), _query(1, 3)
    source_output = parent.decode(parent.prepare(scene), query, torch.tensor([[1.0, -0.2, 0.5]]))
    source_loss = source_output.nonlinear_field.square().sum() + source_output.thermal_temperature.square().sum()
    source_optimizer.zero_grad(set_to_none=True)
    source_loss.backward()
    source_optimizer.step()
    source_moments = named_optimizer_state(parent, source_optimizer)
    nonzero_moments = [
        value["exp_avg"] for value in source_moments["state_by_name"].values()
        if torch.count_nonzero(value.get("exp_avg", torch.zeros(()))).item()
    ]
    assert nonzero_moments, "fixture must cross a nonzero named AdamW boundary"

    native = NativeRetrievalCore.from_a2d(parent).train()
    age_1007 = engine._set_native_retrieval_epoch(native, 1007)
    assert age_1007["absolute_epoch"] == 1007
    assert age_1007["exploration"] is False
    age_1008 = engine._set_native_retrieval_epoch(native, 1008)
    assert age_1008["absolute_epoch"] == 1008
    assert age_1008["exploration"] is True
    assert age_1008["strength"] == 0.0
    age_1007 = engine._set_native_retrieval_epoch(native, 1007)
    native_provider = _OptimizerBoundaryFixture(native)
    native_optimizer, native_specs = engine._make_optimizer(
        native, native_provider, "A2-D", 1008, source_moments,
    )
    assert native_specs == source_specs
    assert native.native_routing_state()["strength"] == 1.0
    assert native.native_routing_state()["exploration"] is False
    restored_moments = named_optimizer_state(native, native_optimizer)
    assert restored_moments["state_by_name"]
    _assert_named_optimizer_equal(source_moments, restored_moments)

    dataset_id = _sampling_dataset_id(native_provider.identity_payload())
    checkpoint = engine._checkpoint_payload(
        native,
        native_optimizer,
        identity={
            "workflow": "native_retrieval_exact_resume_cpu_fixture",
            "sampling_dataset_id": dataset_id,
            "sampling_identity": {
                "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
                "source_sampling_dataset_id": dataset_id,
                "matches_d0_e500_namespace": True,
                "arm": "A2-D",
                "fit_epoch_is_absolute": True,
                "sampler_epoch_mapping": "identity",
            },
        },
        arm="A2-D",
        epoch=1007,
        history=[{"epoch": 1007, "native_routing_state": native.native_routing_state()}],
        best_field=0.0,
        best_guarded=0.0,
        sampler_state={
            "completed_epoch": 1007,
            "absolute_sampling_epoch": 1007,
            "source_sampling_dataset_id": dataset_id,
        },
        group_specs=native_specs,
    )
    # Keep the serialization entirely in RAM; this is a checkpoint-contract test.
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    stream.seek(0)
    restored_payload = torch.load(stream, map_location="cpu", weights_only=False)

    resumed = NativeRetrievalCore.from_config(native.get_config()).train()
    resumed.load_state_dict(restored_payload["model_state_dict"], strict=True)
    assert resumed.native_epoch == restored_payload["epoch"] == 1007
    assert resumed.native_routing_state() == checkpoint["history"][0]["native_routing_state"]
    resumed_provider = _OptimizerBoundaryFixture(resumed)
    resumed_optimizer, resumed_specs = engine._make_optimizer(
        resumed, resumed_provider, "A2-D", 1008,
        restored_payload["optimizer_state_by_name"],
    )
    assert resumed_specs == native_specs
    _assert_named_optimizer_equal(
        named_optimizer_state(native, native_optimizer),
        named_optimizer_state(resumed, resumed_optimizer),
    )

    # The checkpoint's absolute case/query namespace and next age select the
    # same all-open exploration step (e1008) on both uninterrupted and resumed paths.
    restore_rng_state(checkpoint["rng_state"])
    uninterrupted = _synthetic_step(native, native_optimizer, engine, native_provider, 1008, dataset_id)
    uninterrupted_rng_after = torch.get_rng_state().clone()
    restore_rng_state(restored_payload["rng_state"])
    resumed_step = _synthetic_step(resumed, resumed_optimizer, engine, resumed_provider, 1008, dataset_id)
    resumed_rng_after = torch.get_rng_state().clone()
    assert native.native_routing_state()["exploration"] is True
    assert resumed.native_routing_state() == native.native_routing_state()
    assert uninterrupted[0:2] == resumed_step[0:2]
    assert torch.equal(uninterrupted[2], resumed_step[2])
    assert torch.equal(uninterrupted[3], resumed_step[3])
    assert torch.equal(uninterrupted[4], resumed_step[4])
    assert torch.equal(uninterrupted_rng_after, resumed_rng_after)
    for (left_name, left), (right_name, right) in zip(native.named_parameters(), resumed.named_parameters()):
        assert left_name == right_name
        assert torch.equal(left, right), left_name
    _assert_named_optimizer_equal(
        named_optimizer_state(native, native_optimizer),
        named_optimizer_state(resumed, resumed_optimizer),
    )


def test_training_engine_fit_advances_native_core_age_and_saves_it(tmp_path):
    torch.manual_seed(304)
    model = NativeRetrievalCore.from_a2d(_core("A2-D")).train()
    provider = _OneCaseNativeEngineProvider()
    engine = TrainingEngine(
        EngineConfig(
            seed=17, microbatch_cases=1, effective_cases=1,
            total_epochs=1, warmup_epochs=0, monitor_every=1,
            monitor_epochs=(1,), checkpoint_epochs=(1,), latest_every=1,
            curve_every=None, progress_enabled=False,
            sampling_version=SamplingKey.CASE_EPOCH_VERSION,
            training_mode="joint",
        ),
        device="cpu",
    )
    result = engine.fit(
        model, provider, tmp_path / "one_case_native_engine_fit",
        identity={"workflow": "native_engine_fit_cpu_fixture"},
        arm="A2-D", stop_after=1,
    )
    saved = torch.load(
        tmp_path / "one_case_native_engine_fit" / "checkpoints" / "latest_model.pt",
        map_location="cpu", weights_only=False,
    )
    assert result["completed_epoch"] == 1
    assert provider.seen_routing_states
    assert all(row["absolute_epoch"] == 1 for row in provider.seen_routing_states)
    assert model.native_routing_state()["absolute_epoch"] == 1
    assert int(saved["model_state_dict"]["native_training_epoch"]) == 1
    assert saved["history"][0]["optimizer_updates"] == 1
