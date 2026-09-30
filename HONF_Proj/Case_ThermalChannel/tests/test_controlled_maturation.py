from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from channelthermal.response_control.active_packet import RouteWorkRecord, ThermalHardValueSoftOperator
from channelthermal.response_control.contracts import AbsolutePrediction
from channelthermal.response_control.losses import ThermalLossScales
from channelthermal.response_control.maturation import (
    THERMAL_ACTION_PATHS,
    ThermalMaturationSchedule,
    available_frontier_for_paths,
    complementary_soft_shadow_variants,
    summarize_realized_cut_records,
)
from channelthermal.response_control.training import (
    StagedTrainingConfig,
    TrainingStage,
    run_staged_fit,
)
from test_response_control import _AbsoluteField, _stencil


def test_resumed_direct_scorer_requires_bound_execution_mode() -> None:
    driver_path = Path(__file__).resolve().parents[1] / "scripts" / "run_controlled_maturation.py"
    spec = importlib.util.spec_from_file_location("thermal_maturation_driver_mode_test", driver_path)
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    factorized = SimpleNamespace(factorized_first_layer=True)
    assert driver._validate_p_execution_mode(factorized, {}, requested_factorized=True)
    assert driver._validate_p_execution_mode(
        factorized, {"execution_mode": {"thermal_factor_direct_scorer": True}},
        requested_factorized=True,
    )
    with pytest.raises(RuntimeError, match="checkpoint scorer execution mode"):
        driver._validate_p_execution_mode(
            factorized, {"execution_mode": {"thermal_factor_direct_scorer": False}},
            requested_factorized=True,
        )
    with pytest.raises(RuntimeError, match="constructed P scorer"):
        driver._validate_p_execution_mode(factorized, {}, requested_factorized=False)
    with pytest.raises(ValueError, match="invalid execution-mode"):
        driver._validate_p_execution_mode(
            factorized, {"execution_mode": {"thermal_factor_direct_scorer": "1"}},
            requested_factorized=True,
        )


def _tree(*children: tuple[int | None, int | None]):
    return SimpleNamespace(nodes=tuple(SimpleNamespace(left=left, right=right) for left, right in children))


def test_maturation_uses_two_scaffold_passes_then_two_passes_per_action() -> None:
    families = tuple(f"family-{index}" for index in range(8))
    schedule = ThermalMaturationSchedule(families, start_update=200, seed=11)

    plans = [schedule.plan(200 + update) for update in range(100)]
    sparse = [plan for plan in plans if not plan.full_access_replay]
    assert len(sparse) == 80
    assert sum(plan.full_access_replay for plan in plans) == 20
    assert [plan.relative_update for plan in plans if plan.full_access_replay] == list(range(4, 100, 5))

    for primary_pass in range(8):
        pass_rows = [plan for plan in sparse if plan.primary_pass == primary_pass]
        assert {plan.family_id for plan in pass_rows} == set(families)
        assert len(pass_rows) == len(families)
    assert all(plan.action == "two_packet_scaffold" for plan in sparse if plan.primary_pass < 2)
    for primary_pass, expected_action in zip(range(2, 8), ("root", "two_packet", "four_packet") * 2, strict=True):
        pass_rows = [plan for plan in sparse if plan.primary_pass == primary_pass]
        assert {plan.action for plan in pass_rows} == {expected_action}
        assert all(plan.requested_cut_paths == THERMAL_ACTION_PATHS[expected_action] for plan in pass_rows)


def test_full_replay_reuses_prior_family_and_does_not_consume_sparse_pass() -> None:
    schedule = ThermalMaturationSchedule(("a", "b", "c"), start_update=200)
    sparse_before = [schedule.plan(200 + update) for update in range(4)]
    replay = schedule.plan(204)
    next_sparse = schedule.plan(205)

    assert replay.full_access_replay is True
    assert replay.capacity_fraction == 1.0
    assert replay.family_id == sparse_before[-1].family_id
    assert replay.relative_sparse_update == sparse_before[-1].relative_sparse_update
    assert next_sparse.full_access_replay is False
    assert next_sparse.relative_sparse_update == 4


def test_soft_shadow_variants_are_complementary_per_sparse_pair() -> None:
    state_labels = (
        "baseline",
        "i_minus",
        "i_plus",
        "j_minus",
        "j_plus",
        "mm",
        "mp",
        "pm",
        "pp",
        "heat_transfer_plus",
        "heat_transfer_minus",
    )
    first = complementary_soft_shadow_variants(
        state_labels, relative_sparse_update=0, seed=19
    )
    second = complementary_soft_shadow_variants(
        state_labels, relative_sparse_update=1, seed=19
    )
    assert len(first) == len(second) == 5
    assert set(first).isdisjoint(second)
    assert set(first) | set(second) == set(state_labels[1:])
    assert first == complementary_soft_shadow_variants(
        state_labels, relative_sparse_update=0, seed=19
    )
    third = complementary_soft_shadow_variants(
        state_labels, relative_sparse_update=2, seed=19
    )
    assert len(third) == 5


def test_realized_cut_summary_binds_sparse_baseline_and_variant_state_attempt() -> None:
    expected_states = ("baseline", "i_minus", "historical_value_replay")
    records = [
        {
            "state": state,
            "case_index": 0,
            "frontier": frontier,
            "frontier_paths": paths,
            "raw_frontier_k": len(frontier),
            "nonredundant_k": k,
            "nonredundant_k_status": "measured_joint_support",
            "completed_updates_before_attempt": 300,
            "attempted_optimizer_step_including_old_branch": 311,
        }
        for state, frontier, paths, k in (
            ("baseline", (3, 4, 2), ("LL", "LR", "R"), 2),
            ("i_minus", (1, 2), ("L", "R"), 2),
            ("historical_value_replay", (1,), ("",), 1),
        )
    ]
    result = summarize_realized_cut_records(
        records,
        expected_states=expected_states,
        completed_updates_before_attempt=300,
        attempted_optimizer_step_including_old_branch=311,
        full_access_replay=False,
    )
    assert result["status"] == "realized_sparse_cut"
    assert result["baseline"]["frontier_paths"] == ["LL", "LR", "R"]
    assert result["baseline"]["raw_frontier_k"] == 3
    assert result["baseline"]["nonredundant_k"] == 2
    assert [row["state"] for row in result["variant_cut_distribution"]] == ["i_minus"]


def test_realized_cut_summary_rejects_orphan_duplicate_and_missing_cut_rows() -> None:
    expected_states = ("baseline", "historical_value_replay")
    baseline = {
        "state": "baseline",
        "frontier": [1, 2],
        "frontier_paths": ["L", "R"],
        "raw_frontier_k": 2,
        "nonredundant_k": 2,
        "completed_updates_before_attempt": 200,
        "attempted_optimizer_step_including_old_branch": 211,
    }
    historical = {
        **baseline,
        "state": "historical_value_replay",
        "frontier": [1],
        "frontier_paths": [""],
        "raw_frontier_k": 1,
        "nonredundant_k": 1,
    }
    kwargs = {
        "expected_states": expected_states,
        "completed_updates_before_attempt": 200,
        "attempted_optimizer_step_including_old_branch": 211,
        "full_access_replay": False,
    }
    with pytest.raises(ValueError, match="different optimizer attempt"):
        summarize_realized_cut_records(
            [{**baseline, "attempted_optimizer_step_including_old_branch": 210}, historical],
            **kwargs,
        )
    with pytest.raises(ValueError, match="exactly one"):
        summarize_realized_cut_records([baseline, baseline, historical], **kwargs)
    with pytest.raises(ValueError, match="expected hard-call order"):
        summarize_realized_cut_records([baseline], **kwargs)


def test_realized_cut_summary_marks_full_access_without_sparse_k() -> None:
    result = summarize_realized_cut_records(
        [],
        expected_states=("baseline", "historical_value_replay"),
        completed_updates_before_attempt=204,
        attempted_optimizer_step_including_old_branch=215,
        full_access_replay=True,
    )
    assert result["status"] == "full_access_no_cut"
    assert result["baseline"] is None
    assert result["by_state"] == {}
    with pytest.raises(ValueError, match="must not claim sparse"):
        summarize_realized_cut_records(
            [{"state": "baseline"}],
            expected_states=("baseline", "historical_value_replay"),
            completed_updates_before_attempt=204,
            attempted_optimizer_step_including_old_branch=215,
            full_access_replay=True,
        )


def test_scheduled_operator_stamps_only_complete_attempt_cut_rows(tmp_path: Path) -> None:
    driver_path = Path(__file__).resolve().parents[1] / "scripts" / "run_active_packet_forward.py"
    spec = importlib.util.spec_from_file_location("thermal_scheduled_operator_test", driver_path)
    assert spec is not None and spec.loader is not None
    forward = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forward)

    class _Schedule:
        current_update = 300

        @staticmethod
        def budgets():
            return {"QE": 0.90, "MM": 0.90}

    class _Builder:
        def __init__(self):
            self.exact_full_access = False
            self.last_records = ()
            self.budget_fractions = {}

    class _Paired:
        def __init__(self, builder):
            self.builder = builder

        def __call__(self, *_args, **_kwargs):
            work = RouteWorkRecord("QE", 0.90, 1.0, 1.0, 2.0, 1, 2, True, "test")
            mm_work = RouteWorkRecord("MM", 0.90, 1.0, 1.0, 2.0, 1, 2, True, "test")
            self.builder.last_records = (
                {
                    "frontier": [1],
                    "frontier_paths": [""],
                    "raw_frontier_k": 1,
                    "nonredundant_k": 1,
                    "nonredundant_k_status": "measured_joint_support",
                    "joint_support_mechanisms": ["MM", "QE"],
                    "joint_support_signature_hex_by_frontier": {"1": "01"},
                    "collapsed_cut_rows": [],
                    "source_mask_evidence": {},
                    "direct_scorer_factorized_first_layer": False,
                    "routes": {"QE": work, "MM": mm_work},
                    "full_access_bypass_routes": [],
                },
            )
            return object()

    builder = _Builder()
    log_path = tmp_path / "route_work.jsonl"
    operator = forward._ScheduledThermalOperator(
        paired=_Paired(builder),
        schedule=_Schedule(),
        hard_builder=builder,
        soft_builder=_Builder(),
        state_labels=("baseline", "variant"),
        arm="G",
        route_log=log_path,
        sparse_route_verified={"QE": False, "MM": False},
    )
    for _ in range(3):
        operator(None, {}, {})
    assert not log_path.exists()
    operator.optimizer_attempt(300, 311)
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 3
    assert {row["state"] for row in rows} == {"baseline", "variant", "historical_value_replay"}
    assert all(row["completed_updates_before_attempt"] == 300 for row in rows)
    assert all(row["attempted_optimizer_step_including_old_branch"] == 311 for row in rows)
    assert all(row["optimizer_update"] == 301 for row in rows)
    assert operator.realized_cut_records_this_update == []


def test_scheduled_operator_rejects_incomplete_native_state_sequence(tmp_path: Path) -> None:
    driver_path = Path(__file__).resolve().parents[1] / "scripts" / "run_active_packet_forward.py"
    spec = importlib.util.spec_from_file_location("thermal_scheduled_operator_incomplete_test", driver_path)
    assert spec is not None and spec.loader is not None
    forward = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forward)
    schedule = SimpleNamespace(current_update=300, budgets=lambda: {"QE": 0.90, "MM": 0.90})
    work = RouteWorkRecord("QE", 0.90, 1.0, 1.0, 2.0, 1, 2, True, "test")
    builder = SimpleNamespace(
        exact_full_access=False,
        last_records=(
            {
                "frontier": [1],
                "frontier_paths": [""],
                "raw_frontier_k": 1,
                "nonredundant_k": 1,
                "nonredundant_k_status": "measured_joint_support",
                "joint_support_mechanisms": ["QE"],
                "joint_support_signature_hex_by_frontier": {"1": "01"},
                "collapsed_cut_rows": [],
                "source_mask_evidence": {},
                "routes": {"QE": work},
                "full_access_bypass_routes": [],
            },
        ),
        budget_fractions={},
    )
    operator = forward._ScheduledThermalOperator(
        paired=lambda *_args, **_kwargs: object(),
        schedule=schedule,
        hard_builder=builder,
        soft_builder=SimpleNamespace(budget_fractions={}),
        state_labels=("baseline", "variant"),
        arm="G",
        route_log=tmp_path / "incomplete.jsonl",
        sparse_route_verified={"QE": False, "MM": False},
    )
    operator(None, {}, {})
    with pytest.raises(RuntimeError, match="exactly one native call per stencil state"):
        operator.optimizer_attempt(300, 311)


def _u300_amendment_fixture(tmp_path: Path, runner_path: Path):
    output_dir = tmp_path / "controlled_run"
    output_dir.mkdir()
    checkpoint_paths = {}
    checkpoint_hashes = {}
    attempted = {"G": 310, "P": 303}
    arms = {}
    for arm in ("G", "P"):
        checkpoint = output_dir / f"{arm}_u0300.pt"
        checkpoint.write_bytes(f"checkpoint-{arm}".encode())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        checkpoint_paths[arm] = checkpoint
        checkpoint_hashes[arm] = digest
        arms[arm] = {
            "latest_update": 300,
            "final_update": 300,
            "status": "passed_timing_gate",
            "latest_checkpoint": str(checkpoint),
            "latest_checkpoint_sha256": digest,
            "attempted_optimizer_steps": attempted[arm],
        }
    manifest = {
        "status": "passed_timing_gate_u300",
        "driver_sha256": "old-driver-sha",
        "physical_gpu_uuid": "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39",
        "driver_source_amendments": [
            {
                "kind": "post_u300_execution_mode_guard_source_amendment",
                "new_driver_sha256": "old-driver-sha",
            }
        ],
        "arms": arms,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (output_dir / "optimizer_attempts.jsonl").write_text("", encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    source_path = str(runner_path.resolve())
    parity = {
        "status": "passed",
        "controlled_manifest_sha256": manifest_sha,
        "checkpoint_sha256": checkpoint_hashes,
        "optimizer_calls": 0,
        "reference_solver_calls": 0,
        "changed_model_weights": False,
        "physical_gpu": {"uuid": "GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39"},
        "source_sha256": {source_path: "parity-runner-sha"},
        "train_only_input": {
            "checkpoint_bindings": {
                arm: {
                    "path": str(checkpoint_paths[arm]),
                    "sha256": checkpoint_hashes[arm],
                    "attempted_optimizer_steps": attempted[arm],
                }
                for arm in ("G", "P")
            }
        },
        "arms": {arm: {"gate": {"all_gates_pass": True}} for arm in ("G", "P")},
    }
    parity_path = tmp_path / "parity.json"
    parity_path.write_text(json.dumps(parity), encoding="utf-8")
    return manifest_path, parity_path, manifest_sha


def test_u300_protocol_amendment_records_three_source_identities_atomically(tmp_path: Path) -> None:
    driver_path = Path(__file__).resolve().parents[1] / "scripts" / "run_controlled_maturation.py"
    spec = importlib.util.spec_from_file_location("thermal_u300_amendment_test", driver_path)
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver._gpu2_compute_process_ids = lambda: set()
    manifest_path, parity_path, old_manifest_sha = _u300_amendment_fixture(tmp_path, driver_path)
    original_bytes = manifest_path.read_bytes()

    assert not torch.cuda.is_initialized()
    result = driver.run(
        SimpleNamespace(
            output_dir=manifest_path.parent,
            u300_protocol_amendment=parity_path,
            resume=True,
            preflight_only=True,
            arms=("G", "P"),
        )
    )
    assert not torch.cuda.is_initialized()
    amended = json.loads(manifest_path.read_text(encoding="utf-8"))
    backup_path = manifest_path.with_name("run_manifest_before_u301_protocol_amendment.json")
    lineage = amended["u300_protocol_amendment"]
    assert result["status"] == "u300_protocol_amended_no_training"
    assert backup_path.read_bytes() == original_bytes
    assert lineage["old_manifest_sha256"] == old_manifest_sha
    assert lineage["old_driver_sha256"] == "old-driver-sha"
    assert lineage["parity_runner_sha256"] == "parity-runner-sha"
    assert lineage["final_driver_sha256"] == driver._sha256(driver_path)
    assert amended["driver_sha256"] == lineage["final_driver_sha256"]
    assert amended["status"] == "passed_timing_gate_u300"
    assert lineage["first_new_update"] == 301


def test_u300_protocol_amendment_rejects_unbound_or_active_run(tmp_path: Path) -> None:
    driver_path = Path(__file__).resolve().parents[1] / "scripts" / "run_controlled_maturation.py"
    spec = importlib.util.spec_from_file_location("thermal_u300_amendment_reject_test", driver_path)
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    manifest_path, parity_path, _ = _u300_amendment_fixture(tmp_path, driver_path)
    with pytest.raises(RuntimeError, match="GPU2 is not idle"):
        driver._apply_u300_protocol_amendment(
            manifest_path,
            parity_path,
            active_gpu2_pids={87654},
        )

    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["controlled_manifest_sha256"] = "mismatched-manifest"
    parity_path.write_text(json.dumps(parity), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound to this unchanged"):
        driver._apply_u300_protocol_amendment(
            manifest_path,
            parity_path,
            active_gpu2_pids=set(),
        )


def test_complementary_soft_shadow_average_matches_full_route_gradient() -> None:
    variant_labels = (
        "i_minus",
        "i_plus",
        "j_minus",
        "j_plus",
        "mm",
        "mp",
        "pm",
        "pp",
        "heat_transfer_plus",
        "heat_transfer_minus",
    )
    state_labels = ("baseline", *variant_labels, "historical_value_replay")
    first_half = set(
        complementary_soft_shadow_variants(
            ("baseline", *variant_labels), relative_sparse_update=0, seed=29
        )
    )
    second_half = set(
        complementary_soft_shadow_variants(
            ("baseline", *variant_labels), relative_sparse_update=1, seed=29
        )
    )

    class _PlanBuilder:
        def __init__(self, *, full_access: bool = False) -> None:
            fraction = 1.0 if full_access else 0.9
            self.mode = "P"
            self.budget_fractions = {"QE": fraction, "MM": fraction}
            self.extra_route = "MM"
            self.exact_full_access = full_access
            self.hard = True

    class _TinyNative:
        def __init__(self) -> None:
            self.physical = torch.nn.Parameter(torch.tensor(1.25, dtype=torch.float64))
            self.route = torch.nn.Parameter(torch.tensor(-0.4, dtype=torch.float64))
            self.state_coefficient = 1.0
            self.calls: list[tuple[bool, bool]] = []

        def __call__(
            self,
            _design,
            _context,
            _queries,
            *,
            cover_plan_builder=None,
            detach_model_parameters: bool = False,
        ) -> AbsolutePrediction:
            hard = cover_plan_builder is None or bool(cover_plan_builder.hard)
            self.calls.append((hard, detach_model_parameters))
            physical = self.physical.detach() if detach_model_parameters else self.physical
            if hard:
                value = physical + self.route.detach() * (0.25 * self.state_coefficient)
            else:
                value = physical + self.route * self.state_coefficient
            role_values = {
                "fluid_fields": value.reshape(1, 1),
                "interface": (value + 0.3).reshape(1, 1),
                "solid_temperature": (value - 0.2).reshape(1, 1),
            }
            return AbsolutePrediction(role_values)

    def evaluate(selected: set[str] | None):
        native = _TinyNative()
        paired = ThermalHardValueSoftOperator(
            native, _PlanBuilder(), _PlanBuilder()
        )
        predictions: dict[str, dict[str, torch.Tensor]] = {}
        for index, state in enumerate(state_labels, start=1):
            native.state_coefficient = index / 7.0
            if state in {"baseline", "historical_value_replay"} or selected is None:
                scale = 1.0
            else:
                scale = 2.0 if state in selected else 0.0
            prediction = paired(None, {}, {}, soft_shadow_scale=scale)
            predictions[state] = {
                role: value.reshape(())
                for role, value in prediction.role_values.items()
            }
        loss = sum(
            (role_index + 1.0)
            * value.square()
            for role_index, role in enumerate(
                ("fluid_fields", "interface", "solid_temperature")
            )
            for value in (predictions[state][role] for state in state_labels)
        )
        mixed = (
            predictions["mm"]["fluid_fields"]
            - predictions["mp"]["fluid_fields"]
            - predictions["pm"]["fluid_fields"]
            + predictions["pp"]["fluid_fields"]
        )
        pressure_pair = (
            predictions["heat_transfer_plus"]["interface"]
            - predictions["heat_transfer_minus"]["interface"]
        )
        loss = loss + 0.17 * mixed.square() + 0.11 * pressure_pair.square()
        physical_gradient, route_gradient = torch.autograd.grad(
            loss, (native.physical, native.route), allow_unused=True
        )
        return {
            "loss": loss.detach(),
            "physical_gradient": physical_gradient.detach(),
            "route_gradient": route_gradient.detach(),
            "predictions": {
                state: {role: value.detach() for role, value in roles.items()}
                for state, roles in predictions.items()
            },
            "call_count": len(native.calls),
        }

    full = evaluate(None)
    first = evaluate(first_half)
    second = evaluate(second_half)
    assert full["call_count"] == 2 * len(state_labels)
    assert first["call_count"] == second["call_count"] == len(state_labels) + 7
    for candidate in (first, second):
        torch.testing.assert_close(candidate["loss"], full["loss"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            candidate["physical_gradient"],
            full["physical_gradient"],
            rtol=0.0,
            atol=0.0,
        )
        for state in state_labels:
            for role in ("fluid_fields", "interface", "solid_temperature"):
                torch.testing.assert_close(
                    candidate["predictions"][state][role],
                    full["predictions"][state][role],
                    rtol=0.0,
                    atol=0.0,
                )
    mean_route_gradient = 0.5 * (first["route_gradient"] + second["route_gradient"])
    torch.testing.assert_close(
        mean_route_gradient,
        full["route_gradient"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_full_access_soft_shadow_still_bypasses_the_soft_native_call() -> None:
    class _PlanBuilder:
        def __init__(self) -> None:
            self.mode = "P"
            self.budget_fractions = {"QE": 1.0, "MM": 1.0}
            self.extra_route = "MM"
            self.exact_full_access = True
            self.hard = True

    class _Native:
        def __init__(self) -> None:
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.calls = 0

        def __call__(self, *_args, **_kwargs):
            self.calls += 1
            value = self.weight.reshape(1, 1)
            return AbsolutePrediction(
                {
                    "fluid_fields": value,
                    "interface": value,
                    "solid_temperature": value,
                }
            )

    native = _Native()
    paired = ThermalHardValueSoftOperator(native, _PlanBuilder(), _PlanBuilder())
    prediction = paired(None, {}, {}, soft_shadow_scale=2.0)
    assert native.calls == 1
    assert paired.last_soft is None
    assert all(value.item() == 2.0 for value in prediction.role_values.values())


def test_custom_family_sampler_resumes_with_empty_partial_epoch_tail() -> None:
    stencils = [_stencil(), _stencil()]
    config = StagedTrainingConfig(
        arm="R_response",
        max_optimizer_updates=3,
        max_epochs=2,
        total_optimizer_update_ceiling=3,
        checkpoint_every_updates=1,
        review_updates=(3,),
        stages=(TrainingStage("custom_family_schedule", 0, 3, ("value",)),),
    )
    scales = ThermalLossScales(
        value={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        finite={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        mixed={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
    )
    selector = lambda completed_update: completed_update % len(stencils)
    saved: list[dict[str, object]] = []

    model = _AbsoluteField()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)
    first = run_staged_fit(
        model,
        model,
        optimizer,
        stencils,
        scales=scales,
        config=config,
        stop_at_update=1,
        device="cpu",
        training_stencil_index_for_update=selector,
        on_checkpoint=lambda payload, _label: saved.append(dict(payload)),
    )

    assert first.final_update == 1
    assert saved[-1]["actual_optimizer_updates"] == 1
    assert saved[-1]["sampler_remaining_order"] == []
    assert 1 % len(stencils) != 0

    resumed_model = _AbsoluteField()
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=1.0e-4)
    resumed = run_staged_fit(
        resumed_model,
        resumed_model,
        resumed_optimizer,
        stencils,
        scales=scales,
        config=config,
        initial_update=1,
        resume_payload=saved[-1],
        stop_at_update=2,
        device="cpu",
        training_stencil_index_for_update=selector,
    )

    assert resumed.initial_update == 1
    assert resumed.final_update == 2
    assert resumed.actual_optimizer_updates == 1
    assert len(resumed.history) == 1
    assert resumed.history[0].training_stencil_index == 1


def test_early_leaf_resolves_to_available_complete_cut_without_increasing_k() -> None:
    # Root splits, but its left branch is already a physical leaf.
    early_left = _tree((1, 2), (None, None), (3, 4), (None, None), (None, None))
    frontier, paths = available_frontier_for_paths(
        early_left, THERMAL_ACTION_PATHS["four_packet"]
    )
    assert frontier == (1, 3, 4)
    assert paths == ("L", "RL", "RR")
    assert len(frontier) < len(THERMAL_ACTION_PATHS["four_packet"])

    root_only = _tree((None, None))
    frontier, paths = available_frontier_for_paths(
        root_only, THERMAL_ACTION_PATHS["four_packet"]
    )
    assert frontier == (0,)
    assert paths == ("",)


def test_early_leaf_left_refined_right_coarse_preserves_left_to_right_cut_order() -> None:
    # Root splits; its left child has two children while its right child is a
    # terminal leaf. Native node order is LL=3, LR=4, R=2, not numeric order.
    left_refined = _tree(
        (1, 2),
        (3, 4),
        (None, None),
        (None, None),
        (None, None),
    )
    frontier, paths = available_frontier_for_paths(
        left_refined, THERMAL_ACTION_PATHS["four_packet"]
    )
    assert frontier == (3, 4, 2)
    assert paths == ("LL", "LR", "R")


def test_duplicate_family_ids_are_rejected() -> None:
    try:
        ThermalMaturationSchedule(("same", "same"))
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate response families must not silently collapse the exposure denominator")


def test_auxiliary_anchor_changes_update_in_blockwise_projection_branch() -> None:
    scales = ThermalLossScales(
        value={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        finite={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        mixed={"fluid_fields": 1.0, "interface": 1.0, "solid_temperature": 1.0},
        pressure_value=1.0,
        pressure_response=1.0,
        pressure_limit=3.0,
        pressure_boundary=0.5,
        solid_temperature=1.0,
        smooth_peak_beta=1.0,
        near_limit_band=0.5,
    )
    config = StagedTrainingConfig(
        arm="R_response",
        max_optimizer_updates=1,
        max_epochs=1,
        total_optimizer_update_ceiling=1,
        checkpoint_every_updates=1,
        review_updates=(1,),
        project_response_gradient_blockwise=True,
        stages=(TrainingStage("projected_response", 0, 1, ("value", "finite")),),
    )
    starting_state = _AbsoluteField().state_dict()

    def fit(with_anchor: bool) -> float:
        model = _AbsoluteField()
        model.load_state_dict(starting_state)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-5)

        def anchor_loss(_completed, _stencil, _predictions, _terms):
            if not with_anchor:
                return None, {}
            # Nonzero train-only incumbent-distortion gradient at this start.
            return 100.0 * torch.square(model.gain - 0.1), {"anchor_loss": 100.0 * float((model.gain.detach() - 0.1).square())}

        run_staged_fit(
            model,
            model,
            optimizer,
            [_stencil()],
            scales=scales,
            loss_weights={"value": 1.0, "finite": 1.0},
            config=config,
            auxiliary_loss_fn=anchor_loss,
        )
        return float(model.gain.detach())

    without_anchor = fit(False)
    with_anchor = fit(True)
    assert with_anchor != without_anchor
