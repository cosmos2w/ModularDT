"""Round8 TrainingEngine hook and separate research-admission contracts."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import content_hypergraph_evaluate as evaluation
import pytest
import torch
from content_hypergraph_evaluate import fixed_monitor_batches, state_value_equal

from honf_forward_core.native_retrieval.budgeted_core import BudgetedNativeCore
from honf_runtime import native_budgeted_child
from honf_runtime.native_budgeted_factory import _validate_source_sampler
from honf_runtime.native_budgeted_protocol import (
    _BUDGET_SCHEDULE,
    validate_round8_research_admission,
)
from honf_runtime.unified_training import (
    EngineConfig,
    LossTerm,
    TaskBatch,
    TrainingEngine,
)
from tests.test_content_hypergraph_v2_core import _core, _query, _scene


def test_state_dict_comparison_accepts_native_extra_state_mappings():
    class _Stateful(torch.nn.Module):
        def __init__(self, rng_identity):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.rng_identity = rng_identity

        def get_extra_state(self):
            return {"rng_identity": self.rng_identity, "schema": "fixture_v1"}

    left = _Stateful({"seed_namespace": [1, 2], "absolute_epoch": 750}).state_dict()
    same = _Stateful({"seed_namespace": [1, 2], "absolute_epoch": 750}).state_dict()
    different = _Stateful({"seed_namespace": [1, 2], "absolute_epoch": 751}).state_dict()

    assert state_value_equal(left["weight"], same["weight"])
    assert state_value_equal(left["_extra_state"], same["_extra_state"])
    assert not state_value_equal(left["_extra_state"], different["_extra_state"])


def test_round8_evaluation_loader_compares_serialized_extra_state(monkeypatch, tmp_path):
    class _Stateful(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0]))
            self.rng_identity = "fixture"

        def get_extra_state(self):
            return {"rng_identity": self.rng_identity, "schema": "fixture_v1"}

        def set_extra_state(self, state):
            self.rng_identity = state["rng_identity"]

    recipe = {"schema": "fixture_recipe"}
    state = _Stateful().state_dict()
    payload = {
        "workflow": "native_budgeted_dispatch_joint_v2",
        "experiment_identity": {
            "recipe": recipe,
            "recipe_sha256": evaluation.canonical_sha256(recipe),
            "provider_identity": {"fixture": True},
        },
        "model_state_dict": state,
    }
    model = _Stateful()
    standalone = _Stateful()

    class _Provider:
        adapter = _Stateful()

        def identity_payload(self):
            return {"fixture": True}

    monkeypatch.setattr(evaluation, "load_trusted_checkpoint", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(evaluation, "load_content_hypergraph_checkpoint",
                        lambda *_args, **_kwargs: (standalone, None, payload))
    monkeypatch.setattr(
        "honf_runtime.native_budgeted_protocol.validate_native_budgeted_recipe",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setitem(
        sys.modules,
        "honf_runtime.native_budgeted_factory",
        SimpleNamespace(build_native_budgeted_task=lambda *_args, **_kwargs: {
            "model": model, "provider": _Provider(),
        }),
    )

    loaded_model, loaded_provider, loaded_payload = evaluation.load_evaluation_components(
        tmp_path / "fixture.pt", "cpu",
    )

    assert loaded_model is model
    assert isinstance(loaded_provider, _Provider)
    assert loaded_payload is payload
    assert not model.training


class _Provider:
    def __init__(self):
        self.batch = TaskBatch(
            scene_inputs=_scene(1),
            receivers=_query(1, 3),
            targets=torch.zeros((1, 3, 3)),
            case_keys=("fixture-case",),
        )

    def loss_denominators(self, batches, _phase, _arm):
        return {"field": sum(batch.targets.numel() for batch in batches)}

    def make_scene(self, scene_inputs):
        return scene_inputs

    def predict_native(self, model, scene, receivers, _arm, _phase, **_kwargs):
        heat = torch.tensor([[1.0, -0.2, 0.5]], dtype=scene.module_features.dtype)
        return model.decode(model.prepare(scene), receivers, heat), {}

    def loss_terms(self, predictions, targets, _phase, _auxiliary):
        error = predictions.nonlinear_field - targets
        return {"field": LossTerm(error.square().sum(), targets.numel())}

    def work_counts(self, _batch, _predictions, _auxiliary):
        return {"fixture_native_calls": 1}


def _engine(effective_cases=1) -> TrainingEngine:
    return TrainingEngine(
        EngineConfig(
            seed=7,
            microbatch_cases=1,
            effective_cases=effective_cases,
            total_epochs=2500,
            warmup_epochs=500,
            training_mode="joint",
        ),
        device="cpu",
    )


def test_round8_recipe_pins_exact_expectation_and_post700_relaxation_temperatures():
    assert _BUDGET_SCHEDULE["exact_work_marginal_temperature_through_e700"] == 1.0
    assert _BUDGET_SCHEDULE["post700_work_relaxation_temperature"] == 0.5
    assert _BUDGET_SCHEDULE["relaxation_temperature"] == [1.0, 0.5]
    assert _BUDGET_SCHEDULE["relaxation_anneal_epochs"] == [501, 700]


def test_round8_wind_sampling_seed_remains_separate_from_initialization_seed():
    source_engine = SimpleNamespace(config=SimpleNamespace(
        seed=42, sampling_version="case_epoch_v1",
    ))
    provider_training = {"initialization_seed": 0, "sampling_seed": 42}
    rng_policy = {"sampling_version": "case_epoch_v1"}

    _validate_source_sampler(source_engine, provider_training, rng_policy)
    with pytest.raises(ValueError, match="sampler seed"):
        _validate_source_sampler(
            source_engine,
            {"initialization_seed": 0, "sampling_seed": 0},
            rng_policy,
        )


def test_round8_engine_adds_work_objective_only_when_native_dispatch_is_live():
    provider = _Provider()
    batch = provider.batch
    open_model = BudgetedNativeCore.from_a2d(_core("A2-D")).train()
    open_model._budgeted_all_open = True
    open_optimizer = torch.optim.AdamW(open_model.parameters(), lr=1.0e-4)
    losses, _, _, _ = _engine()._run_update(
        open_model,
        provider,
        open_optimizer,
        (batch,),
        phase="joint",
        arm="A2-D",
        epoch=501,
        update_index=0,
    )
    assert "budget_work_penalty" not in losses
    assert int(open_model.budgeted_macro_step) == 0
    assert all(parameter.grad is None for parameter in open_model.budgeted_router.parameters())

    model = BudgetedNativeCore.from_a2d(
        _core("A2-D"), membership_bias=0.0, lambda0=0.1,
    ).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    losses, work, _, _ = _engine()._run_update(
        model,
        provider,
        optimizer,
        (batch,),
        phase="joint",
        arm="A2-D",
        epoch=501,
        update_index=0,
    )
    assert "budget_work_penalty" in losses
    assert work["budget_work_case_requests"] == 1.0
    assert int(model.budgeted_macro_step) == 1
    assert any(parameter.grad is not None for parameter in model.budgeted_router.parameters())


class _MacroConstraintCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from honf_forward_core.native_retrieval.budget_state import BudgetDualState

        self.route = torch.nn.Parameter(torch.tensor(0.5))
        self.budgeted_router = torch.nn.Module()
        self.budgeted_router.dual = BudgetDualState(lambda0=1.0)
        self.residual = None

    def _uses_native_retrieval(self):
        return True

    def budgeted_begin_update(self, epoch, _update_index):
        return {"absolute_epoch": epoch}

    def budgeted_begin_microbatch(self):
        self.residual = None

    def budgeted_microbatch_result(self):
        assert self.residual is not None
        self.budgeted_residual_for_observation = self.residual.detach()
        return {
            "constraint_residual_numerator": self.residual,
            "work_penalty_denominator": 1,
            "relaxed_case_sum": self.residual.detach(),
            "controller_case_sum": self.residual.detach(),
            "target_case_sum": torch.zeros_like(self.residual.detach()),
            "telemetry": {},
        }

    def budgeted_after_update(self, _epoch, _update_index):
        return None


class _TwoCaseConstraintModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _MacroConstraintCore()
        self.task_weight = torch.nn.Parameter(torch.tensor(1.0))


class _TwoCaseConstraintProvider:
    def __init__(self):
        self.batches = (
            TaskBatch("over", None, torch.tensor(0.0), case_keys=("over",)),
            TaskBatch("under", None, torch.tensor(0.0), case_keys=("under",)),
        )

    def loss_denominators(self, batches, _phase, _arm):
        return {"field": sum(batch.targets.numel() for batch in batches)}

    def make_scene(self, scene_inputs):
        return scene_inputs

    def predict_native(self, model, scene, _receivers, _arm, _phase, **_kwargs):
        threshold = 0.25 if scene == "over" else 0.75
        model.core.residual = model.core.route - threshold
        return model.task_weight.reshape(1), {}

    def loss_terms(self, predictions, targets, _phase, _auxiliary):
        return {"field": LossTerm((predictions - targets).square().sum(), targets.numel())}

    def work_counts(self, _batch, _predictions, _auxiliary):
        return {}


def test_round8_macro_constraint_hinges_the_equal_case_mean_not_casewise_hinges():
    provider = _TwoCaseConstraintProvider()
    model = _TwoCaseConstraintModel()
    initial_route = model.core.route.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2, weight_decay=0.0)
    losses, work, _, _ = _engine(effective_cases=2)._run_update(
        model,
        provider,
        optimizer,
        provider.batches,
        phase="joint",
        arm="G-budgeted-M",
        epoch=501,
        update_index=0,
    )

    # The two case residuals are +0.25 and -0.25. Their equal-case macro mean
    # is exactly zero, while the former mean-of-hinges objective would be 0.15625.
    assert losses["budget_work_penalty"] == 0.0
    assert work["budget_macro_constraint_residual"] == 0.0
    torch.testing.assert_close(model.core.route.detach(), initial_route)
    assert model.core.route.grad is None


def test_round8_child_loader_validates_attachment_audit_extensions(monkeypatch):
    recipe = {"treatment": "D-control", "dataset": "thermal"}
    expected = {"schema": "fixture_child", "treatment": "D-control"}
    monkeypatch.setattr(native_budgeted_child, "validate_native_budgeted_recipe", lambda value, **_kwargs: value)
    monkeypatch.setattr(native_budgeted_child, "native_budgeted_child_declaration", lambda _value: expected)
    source_state = {"core.weight": torch.tensor([2.0])}
    source_groups = [{"name": "source", "parameter_names": ["core.weight"]}]
    source_payload = {
        "model_state_dict": source_state,
        "optimizer_state_by_name": {
            "state_by_name": {"core.weight": {"step": 3}},
            "groups": source_groups,
        },
        "history": [{"case_visits": 576, "optimizer_updates": 2000}],
    }
    amendment = {
        **expected,
        "exact_source_model_state_by_name": True,
        "added_state_keys": [],
        "source_optimizer_state_parameter_count": 1,
        "new_route_parameter_names": [],
        "new_route_moments_at_attachment": 0,
        "source_optimizer_groups_and_schedule_verified": True,
        "inherited_case_visits": 576,
        "inherited_optimizer_updates": 2000,
        "next_epoch": 501,
        "preserved_case_visits": 576,
        "preserved_optimizer_updates": 2000,
        "recorded_unix": 1.0,
        "exact_source_adamw_moments_verified": True,
        "fresh_route_optimizer_state_count_at_attachment": 0,
        "route_parameter_count": 0,
        "observed_named_groups": source_groups,
    }
    native_budgeted_child.verify_native_budgeted_child_amendment(
        amendment, recipe=recipe, source_payload=source_payload, model_state_dict=source_state,
        optimizer_state_by_name=source_payload["optimizer_state_by_name"],
    )
    amendment["unrecognized_extension"] = True
    with pytest.raises(ValueError, match="missing or unexpected attachment-audit"):
        native_budgeted_child.verify_native_budgeted_child_amendment(
            amendment, recipe=recipe, source_payload=source_payload, model_state_dict=source_state,
        )


def test_round8_train_monitor_uses_the_exact_d0_engine_recipe(monkeypatch):
    base_recipe = {"protocol": "source_read_development_v1", "architecture": "A2-D"}
    observed = {}

    class _MonitorEngine:
        def epoch_case_order(self, provider, epoch):
            observed["epoch"] = epoch
            return ["case-1"], "D0-namespace"

        def epoch_update_slices(self, count):
            return (slice(0, count),)

        def prepare_update_microbatches(self, provider, **kwargs):
            observed["sampling_dataset_id"] = kwargs["sampling_dataset_id"]
            yield kwargs["update_cases"]

    def engine_for(recipe, device):
        assert recipe == base_recipe
        observed["device"] = device
        return _MonitorEngine()

    monkeypatch.setitem(sys.modules, "content_hypergraph_train", SimpleNamespace(engine_for=engine_for))
    provider = SimpleNamespace(device=torch.device("cpu"))
    recipe = {"family": "native_budgeted_dispatch_v2", "base_recipe": base_recipe}
    batches, ordered = fixed_monitor_batches(provider, recipe, 701, "train")

    assert list(batches) == [["case-1"]]
    assert ordered == ["case-1"]
    assert observed == {"epoch": 701, "device": "cpu", "sampling_dataset_id": "D0-namespace"}


def _admission(startup_speed: str = "not_faster"):
    evidence = {
        name: {"passed": True}
        for name in (
            "all_open_baseline_recovery",
            "dense_binary_reference",
            "useful_router_gradient",
            "target_heat_isolation",
            "real_optimizer_memory",
            "reserved_resources",
        )
    }
    evidence["real_optimizer_memory"].update({
        "device": "cuda",
        "peak_allocated_bytes": 1024,
        "optimizer_updates": 1,
        "gpu_uuid": "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39",
    })
    evidence["useful_router_gradient"]["calibration"] = {
        "membership_bias": 5.8,
        "lambda0": 0.03,
        "path": "/data2/receipts/birth_native_qualification.json",
        "sha256": "c" * 64,
    }
    evidence["reserved_resources"].update({
        "resource_reservation_path": "/data2/receipts/resource_reservation.json",
        "resource_reservation_sha256": "b" * 64,
        "gpu_uuid": "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39",
        "gpu2_only": True,
        "minimum_free_gpu_bytes": 8 * 1024**3,
        "minimum_free_fraction": 0.20,
        "estimated_associated_seconds": 6000,
        "estimated_elapsed_seconds": 6000,
        "estimated_persistent_bytes": 3 * 1024**3,
        "estimated_temporary_bytes": 2 * 1024**3,
    })
    return {
        "schema": "round8_research_admission_v1",
        "status": "admitted",
        "dataset": "thermal",
        "source_checkpoint": "/data2/source/epoch_0500_model.pt",
        "source_checkpoint_sha256": "a" * 64,
        "native_backend": "grouped_native",
        "evidence": evidence,
        "resource_reservation": {
            "path": "/data2/receipts/resource_reservation.json",
            "sha256": "b" * 64,
        },
        "startup_speed": {"status": startup_speed},
    }


def test_round8_admission_accepts_not_faster_without_weakening_research_checks():
    receipt = _admission()
    assert validate_round8_research_admission(receipt)["startup_speed"]["status"] == "not_faster"
    receipt["evidence"]["useful_router_gradient"]["passed"] = False
    with pytest.raises(ValueError, match="useful_router_gradient"):
        validate_round8_research_admission(receipt)
