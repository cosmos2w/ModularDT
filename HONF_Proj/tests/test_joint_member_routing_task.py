from __future__ import annotations

import sys
import json
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
for _directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from channelthermal.training.content_hypergraph_task import (
    THERMAL_CONTENT_LOSS_WEIGHTS,
    ThermalContentPredictions,
    ThermalContentReceivers,
    ThermalContentTargets,
)
from channelthermal.training.joint_member_routing_task import (
    JOINT_BLOCKS,
    JointMemberRoutingHealth,
    ThermalJointMemberRoutingTask,
    ThermalJointPairPredictions,
)
from channelthermal.training.member_routing_task import ThermalInteractionReusedMemberRoutingTask
from honf_runtime.unified_training import LossTerm, TaskBatch
from honf_runtime.joint_member_routing_child import restore_joint_member_routing_transfer_rng_state


class _ReadCore:
    query_tile_size = 24

    def __init__(self):
        self._read_execution_override = None
        self._joint_epoch = 17
        self._joint_update_index = 9
        self.owner = None
        routes = {}
        for route in ("mm", "em", "ee", "me"):
            module = torch.nn.Module()
            module.register_parameter("gate", torch.nn.Parameter(torch.full((1, 2, 2), 0.1)))
            routes[route] = module
        self.applied_incidence = SimpleNamespace(routes=routes)
        self.decode_calls = []

    @contextmanager
    def read_execution(self, mode):
        previous = self._read_execution_override
        self._read_execution_override = mode
        try:
            yield
        finally:
            self._read_execution_override = previous

    def decode(self, organization, query, *, return_trace=False):
        owner = self.owner
        mode = self._read_execution_override
        query_count = int(query.num_queries)
        self.decode_calls.append((mode, return_trace, query_count))
        score = owner.score.expand(1, query_count, -1)
        scalar = score.sum(-1)
        for name in JOINT_BLOCKS + ("member_values_M", "member_values_E"):
            scalar = scalar + getattr(organization, name).mean() * 0.01
        fields = scalar[..., None].expand(-1, -1, 4)
        trace = None
        if return_trace:
            candidate_m = torch.ones((1, query_count, 3), dtype=torch.bool)
            candidate_e = torch.ones((1, query_count, 4), dtype=torch.bool)
            trace = {
                "proposal_scores_joint": score,
                "full_eligible_mask": torch.ones((1, query_count, 7), dtype=torch.bool),
                "joint_candidate_mask": torch.cat((candidate_m, candidate_e), dim=-1),
                "candidate_mask_M": candidate_m,
                "candidate_mask_E": candidate_e,
                "full_eligible_mask_M": candidate_m,
                "full_eligible_mask_E": candidate_e,
                "conditional_preference_M": torch.full((1, query_count, 3), 1.0 / 3.0),
                "conditional_preference_E": torch.full((1, query_count, 4), 0.25),
                "prior_role_mass_M": torch.full((1, query_count), 0.5),
                "prior_role_mass_E": torch.full((1, query_count), 0.5),
                "retained_joint_mass_M": torch.full((1, query_count), 0.5),
                "retained_joint_mass_E": torch.full((1, query_count), 0.5),
                "address_block_mass_MM": torch.full((1, query_count), 0.2),
                "address_block_mass_EM": torch.full((1, query_count), 0.2),
                "address_block_mass_EE": torch.full((1, query_count), 0.2),
                "address_block_mass_ME": torch.full((1, query_count), 0.2),
                "address_block_mass_self_M": torch.full((1, query_count), 0.2),
                "candidate_ids_M": torch.tensor([[0, 0, 0, 20]]),
                "candidate_ids_E": torch.tensor([[0, 0, 0, 100]]),
            }
        return SimpleNamespace(
            joint_fields=fields, work_counts={"query_rows": query_count}, trace=trace,
        )


class _PairModel:
    architecture = "R13-5-ME-Joint95"
    training = True

    def __init__(self):
        self.core = _ReadCore()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.reads = []

    def apply_native_with_work_counts(self, prepared, physical_heat, *, return_trace=False):
        del physical_heat
        mode = self.core._read_execution_override
        self.reads.append((mode, return_trace))
        multiplier = 1.0 if mode == "full" else 3.0
        return {"value": self.weight * multiplier}, {"mode_rows": int(multiplier)}, SimpleNamespace(trace=None)


def test_paired_native_task_averages_same_parameter_full_and_joint95_losses():
    model = _PairModel()
    provider = ThermalJointMemberRoutingTask.__new__(ThermalJointMemberRoutingTask)
    provider.model = model
    provider.native_weights = {"flow/u": 0.75}
    provider._role_losses = lambda prediction, targets: {  # noqa: E731 - focused method seam
        "flow/u": prediction.native_main["value"].reshape(1)
    }
    prepared = SimpleNamespace(
        fluid_xy=torch.zeros(1, 4, 2),
        query_batch=SimpleNamespace(local_weights=torch.ones(1, 3)),
        union=SimpleNamespace(unique_counts=torch.tensor([5])),
        organization=SimpleNamespace(work_counts={"prepare_rows": 8}),
    )

    paired = provider._paired_predictions(prepared, torch.ones(1, 2))
    losses = provider.loss_terms(
        paired,
        SimpleNamespace(case_ids=("case",)),
        "joint",
        {},
    )

    assert isinstance(paired, ThermalJointPairPredictions)
    assert model.reads == [("full", False), ("restricted", False)]
    assert model.core._read_execution_override is None
    assert losses["flow/u"].numerator.item() == pytest.approx(4.0)
    assert losses["flow/u"].denominator == 1
    assert losses["flow/u"].weight == pytest.approx(0.75)
    losses["flow/u"].numerator.backward()
    assert model.weight.grad.item() == pytest.approx(2.0)


def _assert_numpy_rng_state_equal(left, right):
    assert left[0] == right[0]
    assert (left[1] == right[1]).all()
    assert left[2:] == right[2:]


def test_joint_member_routing_transfer_maps_source_cuda1_to_single_visible_target_and_restores_host_rng(monkeypatch):
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    original_torch_cpu = torch.get_rng_state().clone()
    source = {
        "python": original_python,
        "numpy": original_numpy,
        "torch_cpu": original_torch_cpu,
        "torch_cuda": [torch.tensor([1, 2, 3], dtype=torch.uint8),
                       torch.tensor([4, 5, 6], dtype=torch.uint8)],
    }
    restored_cuda = []
    random.seed(919)
    np.random.seed(919)
    torch.random.manual_seed(919)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, device=None: restored_cuda.append((value.clone(), str(device))),
    )
    try:
        receipt = restore_joint_member_routing_transfer_rng_state(
            source, source_device="cuda:1", target_device=torch.device("cuda:0"),
        )
        assert random.getstate() == source["python"]
        _assert_numpy_rng_state_equal(np.random.get_state(), source["numpy"])
        assert torch.equal(torch.get_rng_state(), source["torch_cpu"])
        assert len(restored_cuda) == 1
        assert torch.equal(restored_cuda[0][0], source["torch_cuda"][1])
        assert restored_cuda[0][1] == "cuda:0"
        assert receipt["source_saved_cuda_device_count"] == 2
        assert receipt["target_visible_cuda_device_count"] == 1
        assert receipt["saved_cuda_streams_identical"] is False
        assert receipt["source_cuda_state_index"] == 1
        assert receipt["restored_cuda_rng_scope"] == "selected_training_device_only"
        assert receipt["cuda_rng_mapping"] == [{
            "source_logical_device": 1, "target_logical_device": 0,
        }]
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)
        torch.random.set_rng_state(original_torch_cpu)


def test_joint_member_routing_transfer_reports_identical_saved_streams(monkeypatch):
    source_stream = torch.tensor([4, 5, 6], dtype=torch.uint8)
    restored_cuda = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, device=None: restored_cuda.append((value.clone(), str(device))),
    )
    receipt = restore_joint_member_routing_transfer_rng_state(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [source_stream, source_stream.clone()],
        },
        source_device="cuda:1",
        target_device="cuda:0",
    )
    assert receipt["saved_cuda_streams_identical"] is True
    assert receipt["source_saved_cuda_device_count"] == 2
    assert receipt["target_visible_cuda_device_count"] == 1
    assert receipt["cuda_rng_mapping"] == [{
        "source_logical_device": 1, "target_logical_device": 0,
    }]
    assert torch.equal(restored_cuda[0][0], source_stream)


def test_joint_member_routing_transfer_accepts_one_saved_stream_after_device_remap(monkeypatch):
    saved = torch.tensor([7, 8, 9], dtype=torch.uint8)
    restored_cuda = []
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [saved],
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, device=None: restored_cuda.append((value.clone(), str(device))),
    )
    receipt = restore_joint_member_routing_transfer_rng_state(
        state, source_device="cuda:0", target_device="cuda:0",
    )
    assert len(restored_cuda) == 1
    assert torch.equal(restored_cuda[0][0], saved)
    assert restored_cuda[0][1] == "cuda:0"
    assert receipt["source_saved_cuda_device_count"] == 1
    assert receipt["target_visible_cuda_device_count"] == 1
    assert receipt["saved_cuda_streams_identical"] is True
    assert receipt["source_cuda_state_index"] == 0


@dataclass(frozen=True)
class _PreparedBatch:
    query_batch: SimpleNamespace
    organization: SimpleNamespace


class _HealthModel(torch.nn.Module):
    architecture = "R13-5-ME-Joint95"

    def __init__(self):
        super().__init__()
        self.core = _ReadCore()
        self.core.owner = self
        self.score = torch.nn.Parameter(torch.full((1, 1, 7), 0.2))
        self.calls = []

    def _extract_native_roles(self, prepared, temperature, flow):
        del prepared
        return {"pred_field": torch.cat((flow, temperature), dim=-1)}

    def apply_native_with_work_counts(self, prepared, physical_heat, *, return_trace=False):
        mode = self.core._read_execution_override
        query_count = int(prepared.query_batch.num_queries)
        assert query_count <= self.core.query_tile_size
        self.calls.append((mode, return_trace, query_count, tuple(physical_heat.shape)))
        score = self.score.expand(1, query_count, -1)
        organization = prepared.organization
        native_scalar = score.sum(-1)
        for name in JOINT_BLOCKS + ("member_values_M", "member_values_E"):
            native_scalar = native_scalar + getattr(organization, name).mean() * 0.01
        native = {"pred_field": native_scalar[..., None]}
        trace = None
        if return_trace:
            q, members = query_count, 7
            candidate_m = torch.ones((1, q, 3), dtype=torch.bool)
            candidate_e = torch.ones((1, q, 4), dtype=torch.bool)
            trace = {
                "proposal_scores_joint": score,
                "full_eligible_mask": torch.ones((1, q, members), dtype=torch.bool),
                "joint_candidate_mask": torch.cat((candidate_m, candidate_e), dim=-1),
                "candidate_mask_M": candidate_m,
                "candidate_mask_E": candidate_e,
                "full_eligible_mask_M": candidate_m,
                "full_eligible_mask_E": candidate_e,
                "conditional_preference_M": torch.full((1, q, 3), 1.0 / 3.0),
                "conditional_preference_E": torch.full((1, q, 4), 0.25),
                "prior_role_mass_M": torch.full((1, q), 0.5),
                "prior_role_mass_E": torch.full((1, q), 0.5),
                "retained_joint_mass_M": torch.full((1, q), 0.5),
                "retained_joint_mass_E": torch.full((1, q), 0.5),
                "address_block_mass_MM": torch.full((1, q), 0.2),
                "address_block_mass_EM": torch.full((1, q), 0.2),
                "address_block_mass_EE": torch.full((1, q), 0.2),
                "address_block_mass_ME": torch.full((1, q), 0.2),
                "address_block_mass_self_M": torch.full((1, q), 0.2),
                "candidate_ids_M": torch.tensor([[0, 0, 0, 20]]),
                "candidate_ids_E": torch.tensor([[0, 0, 0, 100]]),
            }
        return native, {"query_rows": query_count}, SimpleNamespace(trace=trace)


class _HealthProvider:
    seed = 0
    sampling_version = "case_epoch_v1"
    sampling_dataset_id = "thermal-fixed-train"
    training_sampling_arm = "R13-3-Relative"
    device = "cpu"
    primary_queries = 1024

    def __init__(self):
        self.batch = TaskBatch(
            scene_inputs=SimpleNamespace(structure={}, case_ids=("case",)),
            receivers=ThermalContentReceivers(
                fluid_xy=torch.zeros(1, 1024, 2),
                local_query_points=torch.zeros(1, 3, 4, 2),
                heat=torch.ones(1, 3),
                ntheta=16,
            ),
            targets=ThermalContentTargets(
                case_ids=("case",), case_indices=(0,), epoch=1,
                field_targets=torch.zeros(1, 1024, 5),
                point_weights=torch.ones(1, 1024),
                interface_target=torch.zeros(1, 3, 16, 2),
                interface_valid_mask=torch.ones(1, 3, 16),
                material_targets=torch.zeros(1, 3, 4),
            ),
            case_keys=("case",),
        )
        self.native_weights = {
            role: 1.0 for role in THERMAL_CONTENT_LOSS_WEIGHTS if role != "response"
        }
        self.saved_state = {"training_elapsed_seconds": 0.0}
        self.prepared = None

    def make_batch(self, case_ids, key):
        assert tuple(case_ids) == ("case",)
        assert key.arm == self.training_sampling_arm
        return self.batch

    def _prepare_for_probe(self, model, batch):
        assert batch.receivers.fluid_xy.shape == (1, 1024, 2)
        assert batch.receivers.local_query_points.shape == (1, 3, 4, 2)
        assert batch.targets.field_targets.shape == (1, 1024, 5)
        assert batch.targets.material_targets.shape == (1, 3, 4)
        from honf_forward_core.content_hypergraph.contracts import QueryBatch

        route_names = ("mm", "em", "ee", "me")
        block_names = dict(zip(route_names, JOINT_BLOCKS))
        blocks, stages = {}, {}
        for route in route_names:
            allowed = torch.ones((1, 2, 2), dtype=torch.bool)
            if route == "mm":
                allowed.zero_()
            gate = model.core.applied_incidence.routes[route].gate
            blocks[block_names[route]] = gate * allowed
            stages[route] = {"snapshot": SimpleNamespace(allowed=allowed)}
        blocks.update({
            "member_values_M": torch.nn.Parameter(torch.full((1, 3, 4), 0.2)),
            "member_values_E": torch.nn.Parameter(torch.full((1, 4, 4), 0.3)),
        })
        organization = SimpleNamespace(
            **blocks,
            scene=SimpleNamespace(
                num_modules=3,
                module_mask=torch.ones(1, 3, dtype=torch.bool),
            ),
            H_exchange=None,
            H_self_M=None,
            H_self=None,
            H=None,
            final_applied_context_stages=stages,
            H_route_provenance={name.removesuffix("_applied").removeprefix("H_").lower(): {
                "stage_identity": f"cycle1/{name}", "route": name,
                "receiver_role": "M", "donor_role": "E",
            } for name in JOINT_BLOCKS},
        )
        self.prepared = _PreparedBatch(
            query_batch=QueryBatch(coordinates=torch.zeros(1, 20, 2)),
            organization=organization,
        )
        return self.prepared

    def _role_losses(self, prediction, targets):
        del targets
        value = prediction.native_main["pred_field"].square().mean()
        return {role: value for role in self.native_weights}

    def training_state_dict(self):
        return dict(self.saved_state)

    def load_training_state_dict(self, state):
        self.saved_state = dict(state)


def test_health_uses_complete_native_objective_and_bounded_actual_trace(monkeypatch):
    provider = _HealthProvider()
    model = _HealthModel()
    model.train(True)
    health = JointMemberRoutingHealth.__new__(JointMemberRoutingHealth)
    health.provider = provider
    health.panel_case_ids = ("case",)
    health.state = {
        "schema": health.STATE_SCHEMA, "panel_case_ids": ["case"],
        "initial_observation": None, "last_observation": None,
        "last_observed_epoch": 0,
    }
    from channelthermal.training import member_routing_health as base_health
    monkeypatch.setattr(base_health, "_native_field_delta", lambda base, other, model, prepared: {
        role: 0.0 for role in THERMAL_CONTENT_LOSS_WEIGHTS if role != "response"
    })

    observation = health.observe(model, training_epoch=0)
    json.dumps(observation, allow_nan=False)
    health.record_initial(observation)
    initial_receipt = {"member_routing_health_initial": observation}
    epoch_observation = dict(observation, training_epoch=1)
    health.record_epoch(epoch_observation)
    epoch_receipt = {"member_routing_health": epoch_observation}
    json.dumps(
        {"stage_receipt": initial_receipt, "epoch_receipt": epoch_receipt,
         "health_state": health.state_dict()},
        allow_nan=False,
    )

    assert observation["all_four_applied_blocks_observed"] is True
    assert observation["mode_restored"] is True
    assert model.training is True
    assert model.core._read_execution_override is None
    assert (model.core._joint_epoch, model.core._joint_update_index) == (17, 9)
    assert model.core.decode_calls == [
        ("full", True, 8), ("full", False, 12),
        ("restricted", True, 8), ("restricted", False, 12),
    ]
    assert len(model.calls) == 4
    for mode in ("full", "restricted"):
        summary = observation["cases"][0]["modes"][mode]
        assert summary["prepared_union_query_count"] == 20
        assert summary["retained_trace_query_count"] == 8
        assert summary["native_primary_fluid_query_count"] == 1024
        assert "complete provider-sampled Q=1024" in summary["native_loss_query_scope"]
        assert set(JOINT_BLOCKS) <= set(summary["native_objective_dense_adjoint_norm_by_applied_block"])
        assert summary["native_objective_dense_adjoint_norm_by_applied_block"]["H_MM_applied"] > 0
        assert summary["native_objective_allowed_edge_adjoint_norm_by_applied_block"]["H_MM_applied"] == 0
        assert summary["native_objective_actual_positive_edge_adjoint_norm_by_applied_block"]["H_MM_applied"] == 0
        assert summary["native_objective_applied_route_parameter_gradient_norm_by_block"]["H_MM_applied"] == 0
        for block in JOINT_BLOCKS[1:]:
            assert summary["native_objective_actual_positive_edge_adjoint_norm_by_applied_block"][block] > 0
            assert summary["native_objective_applied_route_parameter_gradient_norm_by_block"][block] > 0
        assert summary["native_objective_proposal_address_score_gradient_norm_by_role"]["M"] > 0
        assert summary["native_objective_proposal_address_score_gradient_norm_by_role"]["E"] > 0
    assert provider.saved_state == {"training_elapsed_seconds": 0.0}


def test_joint_health_schedule_and_recording_are_relative_to_transfer_origin_500():
    health = JointMemberRoutingHealth.__new__(JointMemberRoutingHealth)
    health.observation_epochs = 12
    health.later_every = 10
    health.state = {
        "schema": health.STATE_SCHEMA,
        "panel_case_ids": ["low", "high"],
        "initial_observation": {"training_epoch": 500, "cases": []},
        "last_observation": {"training_epoch": 500, "cases": []},
        "last_observed_epoch": 500,
    }

    assert all(health.scheduled(epoch) for epoch in range(501, 513))
    assert not health.scheduled(521)
    assert health.scheduled(522)
    assert health.record_epoch({"training_epoch": 501, "cases": []}) is None
    assert health.state["last_observed_epoch"] == 501


def test_joint_epoch_callback_reuses_the_single_maintained_health_observer(monkeypatch):
    calls = []

    def base_completed(self, model, engine, epoch, output, row):
        del self, model, engine, output
        calls.append(int(epoch))
        row["member_routing_health"] = {"training_epoch": int(epoch)}

    monkeypatch.setattr(ThermalInteractionReusedMemberRoutingTask, "completed_training_epoch", base_completed)
    provider = ThermalJointMemberRoutingTask.__new__(ThermalJointMemberRoutingTask)
    provider.source_kind = "scratch_joint_member_routing"
    provider.source_parent_age = 0
    row = {"work_counts": {}}

    provider.completed_training_epoch(None, None, 1, None, row)

    assert calls == [1]
    assert row["member_routing_health"] == {"training_epoch": 1}
    assert "joint_member_routing_health" not in row
    assert row["reader_epoch_context"]["additional_epochs_after_parent"] == 1


def test_heat_collapse_review_requires_previously_resolved_low_and_high_cases_and_both_modes():
    roles = ("temperature/fluid", "temperature/surface", "temperature/material")

    def case(count, sensitivity, repeatability):
        return {
            "module_count": count,
            "modes": {
                mode: {
                    "heat_plus10_fieldwise_native_delta": {role: sensitivity for role in roles},
                    "repeatability_fieldwise_native_delta": {role: repeatability for role in roles},
                }
                for mode in ("full", "restricted")
            },
        }

    baseline = {"cases": [case(1, 0.1, 0.001), case(12, 0.2, 0.001)]}
    pure_role = {"cases": [case(1, 0.0, 0.001), case(12, 0.0, 0.001)]}
    assert JointMemberRoutingHealth.pause_reason(baseline, pure_role) is not None
    unresolved = {"cases": [case(1, 0.0, 0.001), case(12, 0.2, 0.001)]}
    assert JointMemberRoutingHealth.pause_reason(baseline, unresolved) is None
    not_previously_resolved = {"cases": [case(1, 0.0, 0.001), case(12, 0.0, 0.001)]}
    assert JointMemberRoutingHealth.pause_reason(not_previously_resolved, pure_role) is None


def test_named_optimizer_seed_loader_preserves_each_compatible_guide_moment():
    from honf_runtime.unified_training import _load_named_optimizer_state, named_optimizer_state

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW([{"params": model.parameters(), "group_name": "joint"}], lr=3e-4)
    loss = model(torch.tensor([[1.0, 2.0]])).square().sum()
    loss.backward()
    optimizer.step()
    source = named_optimizer_state(model, optimizer)
    replacement = torch.nn.Linear(2, 1)
    target = torch.optim.AdamW([{"params": replacement.parameters(), "group_name": "joint"}], lr=3e-4)
    _load_named_optimizer_state(replacement, target, source)
    restored = named_optimizer_state(replacement, target)

    assert restored["groups"] == source["groups"]
    assert set(restored["state_by_name"]) == set(source["state_by_name"])
    for name, source_state in source["state_by_name"].items():
        for key, value in source_state.items():
            torch.testing.assert_close(restored["state_by_name"][name][key], value, rtol=0, atol=0)
