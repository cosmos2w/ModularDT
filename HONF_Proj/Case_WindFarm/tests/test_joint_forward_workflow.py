"""Focused contracts for the matched WindFarm joint-forward workflow."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from honf_forward_core.config import BatchData, InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.adaptive_interaction_cover import INTERACTION_MECHANISMS
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer, OrganizerScores

from windfarm.workflows import joint_forward as joint


class _NativeRun:
    nx, ny, nz = 4, 3, 3
    x_m = np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float64)
    y_m = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)
    z_m = np.asarray([1.0, 3.0, 9.0], dtype=np.float64)
    cell_count = nx * ny * nz
    U = np.arange(cell_count * 3, dtype=np.float32).reshape(cell_count, 3)


class _NativeCase:
    index = 7
    layout_index = 7
    diameter_m = 2.0
    hub_height_m = 2.0
    module_centers = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
    module_present = np.asarray([1.0], dtype=np.float32)
    run = _NativeRun()


def test_cached_role_sampler_matches_direct_native_quadrature_draws() -> None:
    case = _NativeCase()
    coordinates, masks, quadrature = joint._native_role_masks(case)
    counts = {role: 13 for role in joint.ROLE_NAMES}
    rng_cached = np.random.default_rng(932)
    cached = joint.sample_native_role_queries(
        case, rng_cached, counts, catalogue_cache=joint.NativeRoleCatalogueCache(),
    )
    rng_direct = np.random.default_rng(932)
    direct_indices: list[np.ndarray] = []
    for role in joint.ROLE_NAMES:
        valid = np.flatnonzero(masks[role])
        cdf = np.cumsum(quadrature[valid] / quadrature[valid].sum(dtype=np.float64))
        cdf[-1] = 1.0
        positions = np.searchsorted(cdf, rng_direct.random(counts[role]), side="right")
        direct_indices.append(valid[np.minimum(positions, cdf.size - 1)])
    expected = np.concatenate(direct_indices)
    np.testing.assert_array_equal(cached.flat_indices, expected)
    np.testing.assert_array_equal(cached.coordinates_D, coordinates[expected])
    np.testing.assert_array_equal(cached.target_mps, case.run.U[expected])

    cache = joint.NativeRoleCatalogueCache()
    first = joint.sample_native_role_queries(case, np.random.default_rng(1), counts, catalogue_cache=cache)
    second = joint.sample_native_role_queries(case, np.random.default_rng(2), counts, catalogue_cache=cache)
    assert cache.build_count == 1
    assert first.geometry_sha256 == second.geometry_sha256
    assert not np.array_equal(first.flat_indices, second.flat_indices)
    changed_geometry = SimpleNamespace(
        index=case.index,
        layout_index=case.layout_index,
        diameter_m=case.diameter_m,
        hub_height_m=case.hub_height_m,
        module_centers=case.module_centers,
        module_present=case.module_present,
        run=SimpleNamespace(
            nx=case.run.nx,
            ny=case.run.ny,
            nz=case.run.nz,
            x_m=case.run.x_m + 0.25,
            y_m=case.run.y_m,
            z_m=case.run.z_m,
            cell_count=case.run.cell_count,
            U=case.run.U,
        ),
    )
    with pytest.raises(RuntimeError, match="native role row 7 changed geometry"):
        joint.sample_native_role_queries(
            changed_geometry, np.random.default_rng(3), counts, catalogue_cache=cache,
        )


def test_role_catalogue_cache_is_byte_bounded_lru() -> None:
    base = _NativeCase()
    case_a = SimpleNamespace(
        index=7,
        layout_index=base.layout_index,
        diameter_m=base.diameter_m,
        hub_height_m=base.hub_height_m,
        module_centers=base.module_centers,
        module_present=base.module_present,
        run=base.run,
    )
    case_b = SimpleNamespace(
        index=8,
        layout_index=base.layout_index,
        diameter_m=base.diameter_m,
        hub_height_m=base.hub_height_m,
        module_centers=base.module_centers + np.asarray([[0.1, 0.0, 0.0]], dtype=np.float32),
        module_present=base.module_present,
        run=SimpleNamespace(
            nx=base.run.nx,
            ny=base.run.ny,
            nz=base.run.nz,
            x_m=base.run.x_m + 0.1,
            y_m=base.run.y_m,
            z_m=base.run.z_m,
            cell_count=base.run.cell_count,
            U=base.run.U,
        ),
    )
    sizing_cache = joint.NativeRoleCatalogueCache()
    bytes_a = sizing_cache.get(case_a).cached_nbytes
    bytes_b = sizing_cache.get(case_b).cached_nbytes
    capacity = max(bytes_a, bytes_b)
    cache = joint.NativeRoleCatalogueCache(max_cached_bytes=capacity)

    first = cache.get(case_a)
    second = cache.get(case_b)
    assert first.geometry_sha256 != second.geometry_sha256
    summary = cache.summary()
    assert summary["catalogue_count"] == 1
    assert summary["cached_bytes"] == second.cached_nbytes
    assert summary["cached_bytes"] <= summary["cache_capacity_bytes"] == capacity
    assert summary["cache_eviction_count"] == 1

    assert cache.get(case_b) is second
    assert cache.summary()["cache_hit_count"] == 1
    assert cache.get(case_a).geometry_sha256 == first.geometry_sha256
    summary = cache.summary()
    assert summary["catalogue_count"] == 1
    assert summary["cached_bytes"] <= summary["cache_capacity_bytes"]
    assert summary["cache_eviction_count"] == 2
    assert summary["catalogue_build_count"] == 3


def test_actual_windfarm_same_layout_different_wind_directions_use_distinct_catalogues() -> None:
    from windfarm.data import COMPACT_GEOMETRY_KEYS, WindFarmNativeView

    project_root = Path(__file__).resolve().parents[2]
    compact_path = project_root / "Case_WindFarm/Dataset/links/wind_farm/family_tensor.npz"
    volume_path = project_root / "Case_WindFarm/Dataset/links/wind_farm/family_volume"
    if not compact_path.is_file() or not volume_path.is_dir():
        pytest.skip("local WindFarm native dataset links are not available")
    with np.load(compact_path, allow_pickle=False) as archive:
        compact = {name: np.asarray(archive[name]).copy() for name in COMPACT_GEOMETRY_KEYS}
    view = WindFarmNativeView(volume_path, compact_metadata=compact)
    layout_rows = np.flatnonzero(np.asarray(view.metadata["layout_index"]) == 70)
    assert layout_rows.tolist() == [210, 211, 212]
    cases = [view.run(int(row)) for row in layout_rows[:2]]
    assert [case.wind_direction_deg for case in cases] == [270.0, 285.0]
    assert cases[0].layout_index == cases[1].layout_index
    assert cases[0].run.shape_nxyz != cases[1].run.shape_nxyz
    assert not np.array_equal(cases[0].module_centers, cases[1].module_centers)

    role_counts = {role: 4 for role in joint.ROLE_NAMES}
    cache = joint.NativeRoleCatalogueCache(max_cached_bytes=1024**3)
    for case, seed in zip(cases, (270, 285)):
        catalogue = cache.get(case)
        rng = np.random.default_rng(seed)
        expected_indices: list[np.ndarray] = []
        for role in joint.ROLE_NAMES:
            cdf = catalogue.role_cdf[role]
            positions = np.searchsorted(cdf, rng.random(role_counts[role]), side="right")
            positions = np.minimum(positions, cdf.size - 1)
            valid = catalogue.role_indices.get(role)
            expected_indices.append(positions if valid is None else valid[positions])
        expected_flat = np.concatenate(expected_indices).astype(np.int64, copy=False)

        sample = joint.sample_native_role_queries(
            case,
            np.random.default_rng(seed),
            role_counts,
            catalogue_cache=cache,
        )
        np.testing.assert_array_equal(sample.flat_indices, expected_flat)
        np.testing.assert_array_equal(sample.coordinates_D, catalogue.coordinates_D[expected_flat])
        np.testing.assert_array_equal(sample.target_mps, np.asarray(case.run.U[expected_flat], dtype=np.float32))
        assert sample.geometry_sha256 == catalogue.geometry_sha256

    summary = cache.summary()
    assert summary["catalogue_count"] == 2
    assert summary["catalogue_build_count"] == 2
    assert summary["cache_capacity_bytes"] == 1024**3
    assert summary["cached_bytes"] <= summary["cache_capacity_bytes"]
    assert summary["cache_hit_count"] == 2


def test_budget_stop_waits_for_stage3_and_requires_stalled_failure() -> None:
    recovering = joint._budget_trend_stop_decision(
        [{"update": 150.0, "max_ratio": 1.50}],
        update=500,
        max_updates=3000,
        current_max_ratio=1.30,
    )
    assert not recovering["stop"]
    assert not recovering["eligible_after_stage3_review"]

    recovering_at_gate = joint._budget_trend_stop_decision(
        [{"update": 500.0, "max_ratio": 1.50}],
        update=1500,
        max_updates=3000,
        current_max_ratio=1.30,
    )
    assert not recovering_at_gate["stop"]

    stalled = joint._budget_trend_stop_decision(
        [{"update": 500.0, "max_ratio": 1.30}],
        update=1500,
        max_updates=3000,
        current_max_ratio=1.29,
    )
    assert stalled["stop"]
    assert stalled["persistent_failure"]
    assert stalled["stalled_or_worsening"]


def test_fidelity_work_gate_uses_training_risk_and_recovers_after_clear_audits() -> None:
    budgets = {role: 1.0 for role in joint.ROLE_NAMES}
    rows = [
        {
            "row_index": row,
            "reference_role_risk": joint.role_risk_report(
                {role: mse for role in joint.ROLE_NAMES}, budgets
            ),
        }
        for row, mse in ((11, 0.81), (12, 0.81), (99, 100.0))
    ]
    samples = {
        row: SimpleNamespace(role_sample_counts={role: 2 for role in joint.ROLE_NAMES})
        for row in (11, 12, 99)
    }
    training = joint._training_audit_role_risk(rows, [11, 12], samples, budgets)
    assert training["rows"] == [11, 12]
    assert training["development_rows_used_for_work_control"] is False
    assert training["role_risk"]["volume"]["budget_ratio"] == pytest.approx(0.9)

    gate = joint.FidelityWorkGate()
    assert not gate.observe(1.12)["work_incentive_suspended_for_reference_fidelity"]
    assert gate.observe(1.03)["work_incentive_suspended_for_reference_fidelity"]
    assert gate.observe(0.95)["work_incentive_suspended_for_reference_fidelity"]
    assert not gate.observe(0.90)["work_incentive_suspended_for_reference_fidelity"]


def test_schedule_cycles_every_post_exclusion_row_and_preserves_g2_alternation() -> None:
    rows = list(range(8))
    held_layout = min(joint.ORGANIZER_DEVELOPMENT_LAYOUT_INDICES)
    layouts = np.asarray([held_layout, 11, 12, 13, 14, 15, 16, 17], dtype=np.int64)
    train_rows = joint._training_rows_without_held_layouts(rows, layouts)
    assert 0 not in train_rows
    assert all(layouts[row] not in joint.ORGANIZER_DEVELOPMENT_LAYOUT_INDICES for row in train_rows)

    labels = train_rows[:2]
    first = joint._make_schedule(max_updates=1500, train_rows=train_rows, labeled_rows=labels, seed=1234)
    second = joint._make_schedule(max_updates=1500, train_rows=train_rows, labeled_rows=labels, seed=1234)
    assert first == second
    assert first[150]["stage"] == "fixed_g2_fidelity_recovery"
    assert first[150]["row_index"] in labels
    assert first[151]["stage"] == "fixed_g2_fidelity_recovery"
    assert first[151]["row_index"] in train_rows
    assert first[152]["row_index"] in labels

    stage3_rows = [
        record["row_index"] for record in first
        if record["stage"] == "joint_hard_packet_refinement"
    ]
    np.testing.assert_array_equal(sorted(stage3_rows[:len(train_rows)]), sorted(train_rows))
    np.testing.assert_array_equal(sorted(stage3_rows[len(train_rows):2 * len(train_rows)]), sorted(train_rows))
    coverage = joint._scheduled_row_visit_coverage(
        first,
        through_update=1500,
        train_rows=train_rows,
        layout_indices_by_row=layouts,
        stage="joint_hard_packet_refinement",
    )
    assert coverage["visited_training_row_count"] == len(train_rows)
    assert coverage["visited_training_layout_count"] == len(train_rows)
    assert coverage["unvisited_training_rows"] == []


def test_optimizer_attempt_schedule_is_arm_and_stage_specific() -> None:
    stage1 = "all_access_g2_warmup"
    stage2 = "fixed_g2_fidelity_recovery"
    stage3 = "joint_hard_packet_refinement"
    assert joint._optimizer_steps_for_stage("w_full", stage1, use_bridge=False) == (
        "physical_model",
    )
    assert joint._optimizer_steps_for_stage("w_packet", stage1, use_bridge=False) == (
        "physical_model", "organizer",
    )
    assert joint._optimizer_steps_for_stage("w_packet", stage2, use_bridge=False) == (
        "physical_model",
    )
    assert joint._optimizer_steps_for_stage("w_packet", stage3, use_bridge=True) == (
        "physical_model", "organizer",
    )
    assert joint._optimizer_steps_for_stage("w_packet", stage3, use_bridge=False) == (
        "physical_model",
    )


@dataclass
class _Encoded:
    module_tokens: torch.Tensor
    env_tokens: torch.Tensor
    global_token: torch.Tensor


class _FakeBackend:
    def build_case_trees(self, encoded):
        del encoded
        return (object(),)


class _FakeCore:
    backend = _FakeBackend()


class _FakeOrganizer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.2, device="cuda:0"))

    def score_cases(self, encoded, prepared_state, trees):
        assert len(trees) == 1
        assert not encoded.module_tokens.requires_grad
        assert not prepared_state["module_states"].requires_grad
        return (self.logit + prepared_state["module_states"].mean(),)


def _require_gpu2() -> torch.device:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "2" or not torch.cuda.is_available():
        pytest.skip("focused optimizer contract requires the scheduled physical GPU 2 slot")
    return torch.device("cuda:0")


def test_stage1_detached_label_loss_and_physical_backward_are_separate() -> None:
    device = _require_gpu2()
    physical = torch.nn.Parameter(torch.tensor(2.0, device=device))
    encoded = _Encoded(
        module_tokens=(physical * torch.ones((1, 2, 3), device=device)),
        env_tokens=(physical * torch.ones((1, 2, 3), device=device)),
        global_token=(physical * torch.ones((1, 1, 3), device=device)),
    )
    organizer = _FakeOrganizer().to(device)
    physical_optimizer = torch.optim.SGD([physical], lr=0.05)
    organizer_optimizer = torch.optim.SGD(organizer.parameters(), lr=0.05)
    scores = joint._detached_organizer_scores(_FakeCore(), organizer, encoded)
    supervised_loss = scores[0].square()
    physical_loss = (encoded.module_tokens.mean() - 0.5).square()

    physical_optimizer.zero_grad(set_to_none=True)
    organizer_optimizer.zero_grad(set_to_none=True)
    physical_loss.backward()
    physical_optimizer.step()
    assert physical.grad is not None and float(physical.grad.abs()) > 0.0

    physical_optimizer.zero_grad(set_to_none=True)
    before = organizer.logit.detach().clone()
    supervised_loss.backward()
    assert physical.grad is None
    assert organizer.logit.grad is not None and float(organizer.logit.grad.abs()) > 0.0
    organizer_optimizer.step()
    assert not torch.equal(before, organizer.logit.detach())


class _ControlledOrganizer(InputOnlyCoverOrganizer):
    def __init__(self) -> None:
        super().__init__(state_dim=8, module_feature_dim=2, environment_feature_dim=2, hidden_dim=12)
        self.qe_logit = torch.nn.Parameter(torch.tensor(-2.0))

    def score_cases(self, encoded, prepared_state, trees):
        del prepared_state
        scores = []
        for tree in trees:
            nodes = len(tree.nodes)
            modules = encoded.module_present.shape[1]
            environments = encoded.env_coords.shape[1]
            module_logits = torch.full((nodes, modules), 4.0, device=encoded.module_tokens.device)
            environment_logits = torch.full((nodes, environments), 4.0, device=encoded.module_tokens.device)
            qe_logits = torch.cat((self.qe_logit.expand(nodes, 1), environment_logits[:, 1:]), dim=1)
            typed = {
                mechanism: (module_logits if mechanism in {"MM", "EM", "QM"} else environment_logits)
                for mechanism in INTERACTION_MECHANISMS
            }
            typed["QE"] = qe_logits
            scores.append(OrganizerScores(
                torch.full((nodes,), -4.0, device=encoded.module_tokens.device),
                module_logits, qe_logits, typed,
            ))
        return tuple(scores)


def _shadow_case(device: torch.device):
    torch.manual_seed(3107)
    module_centers = torch.tensor([[
        [0.25, 0.5, 0.5], [1.5, 0.75, 0.5], [0.75, 1.25, 0.5],
    ]], device=device)
    env_coords = torch.tensor([[
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
    ]], device=device)
    batch = BatchData(
        module_centers=module_centers,
        module_present=torch.tensor([[1.0, 1.0, 0.0]], device=device),
        module_features=torch.randn((1, 3, 2), device=device),
        global_context=torch.randn((1, 5), device=device),
        query_xy=torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.25, 0.0], [1.0, 0.5, 0.0], [1.5, 0.75, 0.0]]], device=device),
        query_time=None,
        target_field=torch.zeros((1, 4, 3), device=device),
        case_name="joint-forward-work-proxy-test",
        env_coords=env_coords,
        env_features=torch.randn((1, 4, 2), device=device),
        env_weights=torch.tensor([[1.0, 2.0, 1.5, 0.5]], device=device),
        query_features=torch.randn((1, 4, 2), device=device),
        metadata=[{}],
    )
    config = UnifiedForwardConfig(
        forward_architecture="dense_pairwise_field",
        field_dim=3,
        spatial_dim=3,
        coordinate_scale=[2.0, 2.0, 1.0],
        hidden_dim=8,
        dropout=0.0,
        geometry_mode="nonperiodic",
        boundary_feature_mode="none",
        interface_model=InterfaceFieldConfig(
            message_hidden_dim=12,
            attention_heads=2,
            coarse_latent_count=2,
            coarse_blocks=1,
            relative_fourier_frequencies=1,
            receiver_chunk_size=2,
            activation_checkpointing=False,
        ),
    )
    core = InterfaceFieldCore(config).to(device).eval()
    core.backend.set_cover_mode("external")
    core.backend.set_cover_executor("dense_masked")
    return core, batch, _ControlledOrganizer().to(device).eval()


def test_stage3_bridge_preserves_hard_gradient_and_updates_soft_work_organizer() -> None:
    device = _require_gpu2()
    core, batch, organizer = _shadow_case(device)
    encoded = core.encode_case(batch)
    result = joint.hard_value_soft_organizer_forward(
        core, encoded, encoded.module_tokens, organizer,
        batch.query_xy, batch.query_features, receiver_chunk_size=2,
    )
    assert torch.equal(result.prediction, result.hard_prediction)
    work = joint._soft_work_proxy(result.soft_plans[0], encoded, batch.query_xy[0])
    target = result.hard_prediction.detach() + 0.1
    physical_loss = (result.prediction - target).square().mean()
    hard_loss = (result.hard_prediction - target).square().mean()
    physical = tuple(parameter for parameter in core.parameters() if parameter.requires_grad)
    joint_gradients = torch.autograd.grad(physical_loss, physical, retain_graph=True, allow_unused=True)
    hard_gradients = torch.autograd.grad(hard_loss, physical, retain_graph=True, allow_unused=True)
    for actual, expected in zip(joint_gradients, hard_gradients, strict=True):
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-7)
    work_gradient = torch.autograd.grad(work, organizer.qe_logit, retain_graph=True)[0]
    assert torch.isfinite(work_gradient) and abs(float(work_gradient)) > 1.0e-10

    optimizer = torch.optim.SGD(organizer.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    before = organizer.qe_logit.detach().clone()
    (physical_loss + 0.1 * work).backward()
    assert organizer.qe_logit.grad is not None and torch.isfinite(organizer.qe_logit.grad)
    optimizer.step()
    assert not torch.equal(before, organizer.qe_logit.detach())


def test_resource_ledger_blocks_before_an_optimizer_attempt(tmp_path) -> None:
    ledger = joint.RunResourceLedger(
        tmp_path,
        max_wall_seconds=30,
        max_attempted_optimizer_calls=1,
    )
    assert ledger.can_begin_update(1, arm="w_full") == (True, None)
    attempted, reason = ledger.record_attempt(arm="w_full", update=1, optimizer="physical_model")
    assert attempted and reason is None
    allowed, reason = ledger.can_begin_update(1, arm="w_full")
    assert not allowed
    assert reason and "attempted-optimizer-call ceiling" in reason
    attempted, reason = ledger.record_attempt(arm="w_full", update=2, optimizer="physical_model")
    assert not attempted
    assert reason and "attempted-optimizer-call ceiling" in reason
    assert ledger.attempted_optimizer_calls == 1
    assert len(ledger.attempts_path.read_text(encoding="utf-8").splitlines()) == 1


def test_resource_ledger_counts_physical_and_organizer_attempts_per_arm(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(joint, "MAX_OPTIMIZER_CALLS_PER_ARM", 2)
    ledger = joint.RunResourceLedger(
        tmp_path,
        max_wall_seconds=30,
        max_attempted_optimizer_calls=10,
    )
    assert ledger.record_attempt(arm="w_packet", update=1, optimizer="physical_model") == (True, None)
    ledger.record_completion(arm="w_packet")
    allowed, reason = ledger.can_begin_update(2, arm="w_packet")
    assert not allowed and reason and "w_packet attempted-optimizer-call ceiling" in reason
    assert ledger.record_attempt(arm="w_packet", update=1, optimizer="organizer") == (True, None)
    ledger.record_completion(arm="w_packet")
    assert ledger.record_attempt(arm="w_full", update=1, optimizer="physical_model") == (True, None)
    ledger.record_completion(arm="w_full")
    attempted, reason = ledger.record_attempt(arm="w_packet", update=2, optimizer="physical_model")
    assert not attempted and reason and "w_packet attempted-optimizer-call ceiling" in reason
    saved = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert saved["attempted_optimizer_calls_by_arm"] == {"w_full": 1, "w_packet": 2}
    assert saved["completed_optimizer_calls_by_arm"] == {"w_full": 1, "w_packet": 2}
    assert saved["optimizer_call_limit_per_arm"] == 2


def test_resource_ledger_terminal_status_is_persisted(tmp_path) -> None:
    ledger = joint.RunResourceLedger(tmp_path, max_wall_seconds=30)
    attempted, reason = ledger.record_attempt(arm="w_full", update=1, optimizer="physical_model")
    assert attempted and reason is None
    ledger.record_completion(arm="w_full")
    ledger.finalize("complete")

    saved = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert saved["status"] == "complete"
    assert saved["attempted_optimizer_calls"] == 1
    assert saved["completed_optimizer_calls"] == 1


def test_pair_terminal_status_requires_two_arms_at_the_same_cursor() -> None:
    full = {"status": "early_stopped", "scheduled_update_cursor": 500}
    packet = {"status": "complete", "scheduled_update_cursor": 500}
    assert joint._pair_terminal_status({"w_full": full, "w_packet": packet}) == "early_stopped"
    assert joint._pair_terminal_status({"w_full": full}) == "resource_stopped_or_incomplete"
    assert joint._pair_terminal_status({
        "w_full": full,
        "w_packet": {"status": "resource_stopped", "scheduled_update_cursor": 499},
    }) == "resource_stopped_or_incomplete"
    assert joint._pair_terminal_status({
        "w_full": full,
        "w_packet": {"status": "complete", "scheduled_update_cursor": 499},
    }) == "resource_stopped_or_incomplete"


def test_full_grid_review_stops_before_native_setup_when_pair_wall_expires(
    tmp_path, monkeypatch
) -> None:
    """A late review cannot start a long grid after its shared cap expires."""

    ledger = joint.RunResourceLedger(tmp_path / "late_grid", max_wall_seconds=10)
    monkeypatch.setattr(ledger, "elapsed_seconds", lambda: 10.0)
    with pytest.raises(joint._PairWallLimitReached, match="full-grid row 7 setup"):
        joint._stream_full_grid(
            None,
            SimpleNamespace(index=7),
            None,
            torch.device("cuda:0"),
            organizer=None,
            fixed_plan=None,
            chunk_size=8192,
            resource_ledger=ledger,
        )


def test_soft_qe_audit_stops_before_row_setup_when_pair_wall_expires(tmp_path, monkeypatch) -> None:
    ledger = joint.RunResourceLedger(tmp_path / "late_soft_audit", max_wall_seconds=10)
    monkeypatch.setattr(ledger, "elapsed_seconds", lambda: 10.0)
    with pytest.raises(joint._PairWallLimitReached, match="organizer soft-QE audit row 7"):
        joint._soft_qe_audit(
            SimpleNamespace(eval=lambda: None),
            None,
            {7: object()},
            {},
            None,
            torch.device("cuda:0"),
            resource_ledger=ledger,
        )


def test_full_grid_review_keeps_completed_stream_metrics_if_later_stream_stops(
    tmp_path, monkeypatch
) -> None:
    ledger = joint.RunResourceLedger(tmp_path / "ledger", max_wall_seconds=30)
    monkeypatch.setattr(joint, "_model_state_sha256", lambda model: "frozen-state")
    calls = 0

    def stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise joint._PairWallLimitReached("later full-grid stream reached wall cap")
        return {"decode_calls": 3, "wall_seconds": 0.25, "role_metrics": {"wake": 0.1}}

    monkeypatch.setattr(joint, "_stream_full_grid", stream)
    output_dir = tmp_path / "grids"
    with pytest.raises(joint._PairWallLimitReached, match="later full-grid stream"):
        joint._full_grid_review(
            arm="w_packet",
            update=500,
            model=None,
            organizer=object(),
            view=SimpleNamespace(run=lambda row: SimpleNamespace(index=row)),
            train_rows=[1, 2],
            dev_rows=[3, 4],
            g2_documents={},
            include_fixed_g2_grid=False,
            fixed_g2_grid_skip_reason="test",
            normalizer=None,
            device=torch.device("cuda:0"),
            chunk_size=8192,
            output_dir=output_dir,
            resource_ledger=ledger,
        )
    saved = json.loads((output_dir / "full_grid_progress.json").read_text(encoding="utf-8"))
    assert saved["completed_native_grid_decode_calls"] == 3
    assert saved["completed_native_grid_results"]["1"]["organizer_selected_hard_plan"]["role_metrics"]["wake"] == 0.1
