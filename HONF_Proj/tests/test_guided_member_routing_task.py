"""Focused adapter/provider contracts for the R13-4 guided member reader."""
from __future__ import annotations

import copy
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
for _directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from channelthermal.guided_member_routing import ThermalGuidedMemberRoutingAdapter
from channelthermal.training.guided_member_routing_task import (
    FULL_ARM,
    GUIDE_ARM,
    TRAINING_STATE_SCHEMA,
    GuidedMemberRoutingHealth,
    ThermalGuidedMemberRoutingTask,
    ThermalGuidedPairPredictions,
)
from channelthermal.training.member_routing_health import MemberRoutingHealth
from channelthermal.training.content_hypergraph_task import ThermalContentPredictions
from honf_forward_core.interaction_reused_member_routing.guided import GuidedMemberRoutingCore
from honf_runtime.unified_training import SamplingKey
from tests.test_interaction_reused_member_routing import _base_config, _query
from tests.test_native_budgeted_postfit import _thermal_stats


class _ReadCore:
    def __init__(self):
        self._read_execution_override = None

    @contextmanager
    def read_execution(self, mode):
        previous = self._read_execution_override
        self._read_execution_override = mode
        try:
            yield
        finally:
            self._read_execution_override = previous


class _PredictionModel:
    def __init__(self, architecture, *, training):
        self.architecture = architecture
        self.training = training
        self.core = _ReadCore()
        self.reads = []
        self.prepared = SimpleNamespace(
            query_batch=SimpleNamespace(
                num_queries=5,
                batch_size=1,
                local_weights=torch.ones(1, 4),
            ),
            fluid_xy=torch.zeros(1, 5, 2),
            union=SimpleNamespace(unique_counts=torch.tensor([3])),
            organization=SimpleNamespace(work_counts={"relation_prepare_rows": 7}),
        )

    def prepare_native(self, *args, **kwargs):
        del args, kwargs
        return self.prepared

    def apply_native_with_work_counts(self, prepared, physical_heat=None, *, return_trace=False):
        del prepared, physical_heat
        mode = self.core._read_execution_override
        self.reads.append((mode, return_trace))
        native = {"path": torch.tensor([1.0 if mode == "full" else 2.0])}
        counts = {"query_E_cheap_score_rows": 30 if mode == "full" else 12}
        return native, counts, SimpleNamespace(trace=None)


def _provider(model):
    provider = ThermalGuidedMemberRoutingTask.__new__(ThermalGuidedMemberRoutingTask)
    provider.model = model
    return provider


def test_v3_thermal_adapter_round_trips_versioned_guided_core_and_parameters():
    core = GuidedMemberRoutingCore(
        _base_config(),
        member_routing_mode="guide",
        input_dataset="synthetic-native-contract",
        prior_mass_tolerance=0.01,
    )
    model = ThermalGuidedMemberRoutingAdapter(_thermal_stats(), core=core)

    restored = ThermalGuidedMemberRoutingAdapter.from_state_payload(
        model.state_payload(), state_dict=model.state_dict(),
    )

    assert model.SCHEMA == "thermal_interaction_reused_member_routing_adapter_v3"
    assert model.CORE_SCHEMA == "interaction_reused_member_routing_v3"
    assert model.architecture == GUIDE_ARM
    assert restored.get_config() == model.get_config()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_provider_transfer_converts_only_the_state_schema_and_resets_old_health():
    provider = ThermalGuidedMemberRoutingTask.__new__(ThermalGuidedMemberRoutingTask)
    source = {
        "schema": "thermal_member_routing_native_training_state_v2",
        "primary_updates": 812,
        "training_samples": 19200,
        "validation_samples": 1500,
        "training_elapsed_seconds": 129.5,
        "member_routing_health": {"last_observed_epoch": 500, "clean_stop": None},
    }
    before = copy.deepcopy(source)

    converted = provider.source_training_state(source)

    assert source == before
    assert converted == {
        "schema": TRAINING_STATE_SCHEMA,
        "primary_updates": 812,
        "training_samples": 19200,
        "validation_samples": 1500,
        "training_elapsed_seconds": 129.5,
    }
    same_schema = {"schema": TRAINING_STATE_SCHEMA, "training_elapsed_seconds": 0.0}
    restored = provider.source_training_state(same_schema)
    restored["training_elapsed_seconds"] = 10.0
    assert same_schema["training_elapsed_seconds"] == 0.0
    with pytest.raises(ValueError, match="R13-3 Relative"):
        provider.source_training_state({"schema": "thermal_member_routing_native_training_state_v1"})


def test_transferred_health_baseline_is_recorded_at_parent_epoch():
    class Health:
        initial_observation = None

        def observe(self, model, *, training_epoch):
            return {"training_epoch": training_epoch, "model": model}

        def record_initial(self, observation):
            self.initial_observation = observation

    provider = _provider(object())
    provider.source_parent_age = 500
    provider._member_routing_health = Health()
    observed = provider.prepare_transfer_health(provider.model)
    assert observed["training_epoch"] == 500
    assert provider._member_routing_health.initial_observation is observed
    assert provider.prepare_transfer_health(provider.model, source_epoch=500) is None
    with pytest.raises(ValueError, match="source age"):
        provider.prepare_transfer_health(provider.model, source_epoch=501)


def test_guided_health_uses_restricted_native_path_and_parent_relative_schedule(monkeypatch):
    seen = []

    def observe(self, model, *, training_epoch):
        del self
        seen.append((model.core._read_execution_override, training_epoch))
        return {"execution_scope": {}}

    monkeypatch.setattr(MemberRoutingHealth, "observe", observe)
    guide_health = GuidedMemberRoutingHealth.__new__(GuidedMemberRoutingHealth)
    guide_model = _PredictionModel(GUIDE_ARM, training=False)
    guide_health.observe(guide_model, training_epoch=501)
    full_health = GuidedMemberRoutingHealth.__new__(GuidedMemberRoutingHealth)
    full_health.observe(_PredictionModel(FULL_ARM, training=False), training_epoch=501)
    assert seen == [("restricted", 501), ("full", 501)]

    guide_health.state = {"initial_observation": {"training_epoch": 500}}
    guide_health.observation_epochs = 12
    guide_health.later_every = 10
    assert all(guide_health.scheduled(epoch) for epoch in range(501, 513))
    assert not guide_health.scheduled(513)
    assert guide_health.scheduled(522)


def test_training_guide_pairs_full_and_restricted_while_eval_honors_outer_read_override():
    model = _PredictionModel(GUIDE_ARM, training=True)
    provider = _provider(model)
    scene = SimpleNamespace(structure=object(), case_ids=("case",))
    receivers = SimpleNamespace(
        fluid_xy=torch.zeros(1, 5, 2),
        heat=torch.ones(1, 2),
        local_query_points=None,
        ntheta=16,
    )

    paired, auxiliary = provider.predict_native(
        model, scene, receivers, GUIDE_ARM, "joint", epoch=501,
    )
    assert isinstance(paired, ThermalGuidedPairPredictions)
    assert [mode for mode, _trace in model.reads] == ["full", "restricted"]
    assert paired.work["native_fluid_queries"] == 5
    assert paired.work["core_decode_guide_full_query_E_cheap_score_rows"] == 30
    assert paired.work["core_decode_guide_restricted_query_E_cheap_score_rows"] == 12
    assert auxiliary["work"] == dict(paired.work)

    model.training = False
    model.reads.clear()
    prediction, _auxiliary = provider.predict_native(
        model, scene, receivers, GUIDE_ARM, "joint", epoch=501,
    )
    assert isinstance(prediction, ThermalContentPredictions)
    assert model.reads == [("restricted", False)]
    model.reads.clear()
    with model.core.read_execution("full"):
        provider.predict_native(model, scene, receivers, GUIDE_ARM, "joint", epoch=501)
    assert model.reads == [("full", False)]

    full_model = _PredictionModel(FULL_ARM, training=False)
    full_provider = _provider(full_model)
    prediction, _auxiliary = full_provider.predict_native(
        full_model, scene, receivers, FULL_ARM, "joint", epoch=501,
    )
    assert isinstance(prediction, ThermalContentPredictions)
    assert full_model.reads == [("full", False)]
    assert "core_decode_primary_query_E_cheap_score_rows" in prediction.work
    assert prediction.work["member_routing_primary_read_mode_full"] == 1
    assert prediction.work["member_routing_primary_read_mode_restricted"] == 0


def test_guide_native_objective_is_rolewise_mean_and_logs_separate_reader_losses():
    provider = _provider(_PredictionModel(GUIDE_ARM, training=True))
    provider.native_weights = {"flow/u": 2.5}

    def role_losses(self, prediction, targets):
        del self, targets
        return {"flow/u": prediction.native_main["role_loss"]}

    provider._role_losses = MethodType(role_losses, provider)
    predictions = ThermalGuidedPairPredictions(
        full=ThermalContentPredictions(None, {"role_loss": torch.tensor([1.0, 3.0])}),
        restricted=ThermalContentPredictions(None, {"role_loss": torch.tensor([3.0, 5.0])}),
        work={"core_decode_guide_full_query_E_cheap_score_rows": 20},
    )
    targets = SimpleNamespace(case_ids=("a", "b"))
    terms = provider.loss_terms(predictions, targets, "joint", {})

    assert set(terms) == {"flow/u"}
    assert float(terms["flow/u"].numerator) == 6.0
    assert terms["flow/u"].denominator == 2
    assert terms["flow/u"].weight == 2.5
    work = provider.work_counts(None, predictions, None)
    assert work["guide_reader_native_loss_metric__case_count"] == 2.0
    assert work["guide_reader_native_loss_metric__full__flow/u__sum"] == 4.0
    assert work["guide_reader_native_loss_metric__restricted__flow/u__sum"] == 8.0

    provider.source_kind = "relation_guided_e500_transfer"
    provider.source_parent_age = 500
    row = {"work_counts": work}
    provider.completed_training_epoch(provider.model, None, 501, None, row)
    assert row["reader_epoch_context"]["additional_epochs_after_parent"] == 1
    assert row["paired_reader_native_losses"]["per_case_mean_normalized_mse"] == {
        "full_reference": {"flow/u": 2.0},
        "restricted_guide": {"flow/u": 4.0},
    }
    assert row["paired_reader_native_losses"]["auxiliary_objective_weight"] == 0.0


def test_health_sampling_key_matches_case_epoch_draws_across_named_arms():
    full = SamplingKey(0, 503, 7, 1, "joint", FULL_ARM,
                       sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id="thermal-native")
    guide = SamplingKey(0, 503, 7, 1, "joint", GUIDE_ARM,
                        sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id="thermal-native")
    assert full.native_indices(4096, 8, "case-12", "fluid", "primary_native_queries").tolist() == guide.native_indices(
        4096, 8, "case-12", "fluid", "primary_native_queries",
    ).tolist()


def test_guided_health_probe_pairs_native_losses_heat_sensitivity_and_prior_capture(monkeypatch):
    import channelthermal.training.member_routing_health as health_module

    @dataclass
    class Receivers:
        fluid_xy: torch.Tensor
        heat: torch.Tensor
        local_query_points: torch.Tensor | None = None
        ntheta: int = 16

    @dataclass
    class Targets:
        field_targets: torch.Tensor
        point_weights: torch.Tensor
        case_ids: tuple[str, ...] = ("case",)

    @dataclass
    class Batch:
        receivers: Receivers
        targets: Targets

    batch = Batch(
        Receivers(torch.zeros(1, 1024, 2), torch.ones(1, 2)),
        Targets(torch.zeros(1, 1024, 5), torch.ones(1, 1024)),
    )

    class Provider:
        seed = 0
        sampling_version = SamplingKey.CASE_EPOCH_VERSION
        sampling_dataset_id = "thermal-fixed-train"
        device = "cpu"

        def make_batch(self, case_ids, key):
            assert tuple(case_ids) == ("case",)
            assert key.epoch == 1
            return batch

        def _prepare_for_probe(self, model, current_batch):
            del model
            assert current_batch.receivers.fluid_xy.shape[1] == 1024
            return SimpleNamespace(query_batch=_query(700), organization=object())

        @staticmethod
        def _query_subset(query, indices):
            return ThermalGuidedMemberRoutingTask._query_subset(query, indices)

        def _role_losses(self, prediction, targets):
            assert targets.field_targets.shape[1] == 1024
            return {"flow/u": prediction.native_main["role_loss"]}

    class TraceCore(_ReadCore):
        query_tile_size = 512

        def __init__(self):
            super().__init__()
            self.captured_queries = []

        def capture_interaction_routing_trace(self, organization, query, *, intervention):
            del organization
            assert query.num_queries <= 8
            assert intervention["read_mode"] == self._read_execution_override
            self.captured_queries.append((query.num_queries, intervention["read_mode"]))
            candidates = torch.zeros(1, query.num_queries, 4, dtype=torch.bool)
            candidates[..., :2] = True
            return {
                "conditional_preference_E": torch.tensor([[[0.4, 0.3, 0.2, 0.1]]]).expand(
                    1, query.num_queries, 4,
                ),
                "prior_P_E": torch.tensor([[[0.4, 0.3, 0.2, 0.1]]]).expand(1, query.num_queries, 4),
                "full_eligible_mask_E": torch.ones(1, query.num_queries, 4, dtype=torch.bool),
                "chosen_candidate_mask_E": (
                    candidates if intervention["read_mode"] == "restricted"
                    else torch.ones_like(candidates)
                ),
                "chosen_candidate_mask_M": torch.ones(1, query.num_queries, 2, dtype=torch.bool),
            }

    class ProbeModel(_PredictionModel):
        def apply_native_with_work_counts(self, prepared, physical_heat=None, *, return_trace=False):
            assert return_trace is False
            assert prepared.query_batch.num_queries == 700
            mode = self.core._read_execution_override
            self.reads.append((mode, return_trace))
            heat_value = float(torch.as_tensor(physical_heat).mean())
            native = {"role_loss": torch.tensor([heat_value])}
            rows = 700 * (4 if mode == "full" else 2)
            return native, {"query_E_cheap_score_rows": rows}, SimpleNamespace(trace=None)

    def field_delta(base, other, model, prepared):
        del model, prepared
        return {"temperature/fluid": float(other["role_loss"].mean() - base["role_loss"].mean())}

    monkeypatch.setattr(health_module, "_native_field_delta", field_delta)
    model = ProbeModel(GUIDE_ARM, training=False)
    model.core = TraceCore()
    health = GuidedMemberRoutingHealth.__new__(GuidedMemberRoutingHealth)
    health.provider = Provider()

    record = health._paired_probe(model, "case")

    assert model.reads == [
        ("full", False), ("restricted", False), ("full", False), ("restricted", False),
    ]
    assert model.core.captured_queries == [(8, "full"), (8, "restricted")]
    assert record["native_role_loss_normalized_mse"]["flow/u"] == {
        "full": 1.0,
        "restricted": 1.0,
        "restricted_minus_full": 0.0,
    }
    assert record["conditional_preference_mass_captured_mean"] == pytest.approx(0.7)
    assert record["relation_prior_mass_captured_mean"] == pytest.approx(0.7)
    assert record["native_primary_fluid_query_count"] == 1024
    assert record["route_trace_query_count"] == 8
    assert record["native_decode_E_member_score_rows"] == {
        "full": 2800, "restricted": 1400, "avoided_before_member_mlp": 1400,
    }
    assert record["route_trace_E_member_score_rows"] == {
        "query_count": 8,
        "full": 32,
        "restricted": 16,
        "avoided_before_member_mlp": 16,
        "scope": "bounded route trace only; native decoder totals are reported separately",
    }
    assert record["prescribed_heat_plus10_fieldwise_native_delta"]["full"]["temperature/fluid"] == pytest.approx(0.1)
    assert record["prescribed_heat_plus10_fieldwise_native_delta"]["restricted"]["temperature/fluid"] == pytest.approx(0.1)
