"""Small CPU algebra contracts for the opt-in Run-1409 backend."""

from __future__ import annotations

import torch
import pytest

from honf_forward_core.interface_fields.budgeted_group_control import (
    BudgetedGroupControlPairwiseField,
)
from honf_forward_core.interface_fields.case_group_budget import (
    CaseGroupGate,
    routing_logit_scale,
    sparsification_continuation,
)
from honf_forward_core.interface_fields.group_control_pairwise import (
    GroupControlPairwiseField,
)
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _encoded_case(batch: int = 1, modules: int = 3, envs: int = 4, hidden: int = 8) -> EncodedInterfaceCase:
    return EncodedInterfaceCase(
        module_tokens=torch.randn(batch, modules, hidden),
        env_tokens=torch.randn(batch, envs, hidden),
        global_token=torch.randn(batch, hidden),
        module_centers=torch.rand(batch, modules, 2) * 2.0,
        env_coords=torch.rand(batch, envs, 2) * 2.0,
        module_present=torch.ones(batch, modules),
        module_features=torch.randn(batch, modules, 2),
        env_features=None,
        env_weights=torch.rand(batch, envs) + 0.1,
        coordinate_scale=torch.tensor([2.0, 2.0]),
    )


def _read(field, state, encoded, receivers):
    features = field.router.query_fourier(receivers / field.router._scale(encoded))
    return field.read(state, encoded, receivers, features)[0]


def _copy_matching_state(source, target) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name not in target_state:
            continue
        if name.endswith("router.group_codes") and target_state[name].ndim == 2:
            target_state[name][: value.shape[0]].copy_(value)
            continue
        try:
            same_shape = target_state[name].shape == value.shape
        except RuntimeError:
            same_shape = False
        if same_shape:
            target_state[name].copy_(value)
    target.load_state_dict(target_state, strict=False)


def test_gate_reference_limit_and_mixed_stable_packing() -> None:
    gate6 = CaseGroupGate(16, 6)
    all_open = gate6.build_budget(
        torch.full((1, 5), 100.0, requires_grad=True),
        deterministic=True,
    )
    torch.testing.assert_close(all_open.eta, torch.full((1, 6), 1.0 / 6.0))
    torch.testing.assert_close(all_open.kappa, torch.tensor([6.0]))
    assert all_open.live_count.tolist() == [6]

    gate12 = CaseGroupGate(16, 12)
    logits = torch.tensor(
        [
            [100.0, 100.0, 100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0],
            [100.0, 100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0],
        ],
        requires_grad=True,
    )
    budget = gate12.build_budget(logits, deterministic=True)
    assert budget.live_count.tolist() == [6, 3]
    assert budget.packed_ids.tolist() == [[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 0, 0]]
    assert budget.packed_valid.tolist() == [[True] * 6, [True, True, True, False, False, False]]
    packed_z, packed_valid = budget.packed_gate()
    torch.testing.assert_close(packed_z[1, 3:], torch.zeros(3))
    assert not packed_valid[1, 3:].any()
    (budget.expected_optional_count.sum() + budget.kappa.sum()).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_six_open_budget_matches_run1406_reader() -> None:
    torch.manual_seed(1409)
    encoded = _encoded_case(hidden=8)
    module_states = torch.randn(1, 3, 8)
    receivers = torch.rand(1, 5, 2) * 2.0
    parent = GroupControlPairwiseField(8, 8, 2, 2, group_count=6, group_control_dim=4).eval()
    candidate = BudgetedGroupControlPairwiseField(
        8,
        8,
        2,
        2,
        group_count=6,
        group_control_dim=4,
    ).eval()
    budget = candidate.router.case_gate.build_budget(
        torch.full((1, 5), 100.0), deterministic=True
    )
    # Materialize all lazy projections before copying the shared Run-1406 state.
    parent_state = parent.prepare(encoded, module_states)
    candidate_state = candidate.prepare(encoded, module_states, phase_shared_state=budget)
    _read(parent, parent_state, encoded, receivers)
    _read(candidate, candidate_state, encoded, receivers)
    _copy_matching_state(parent, candidate)
    parent_state = parent.prepare(encoded, module_states)
    candidate_state = candidate.prepare(encoded, module_states, phase_shared_state=budget)
    parent_output = _read(parent, parent_state, encoded, receivers)
    candidate_output = _read(candidate, candidate_state, encoded, receivers)
    torch.testing.assert_close(candidate_output, parent_output, rtol=2e-5, atol=2e-5)


def test_mixed_case_full_compact_output_and_gradients_match() -> None:
    torch.manual_seed(1410)
    encoded = _encoded_case(batch=2, hidden=8)
    module_states = torch.randn(2, 3, 8)
    receivers = torch.rand(2, 5, 2) * 2.0
    field = BudgetedGroupControlPairwiseField(
        8,
        8,
        2,
        2,
        group_count=12,
        group_control_dim=4,
    ).eval()
    logits = torch.tensor(
        [
            [100.0, 100.0, 100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0],
            [100.0, 100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0],
        ],
        requires_grad=True,
    )
    budget = field.router.case_gate.build_budget(logits, deterministic=True)
    full_state = field.prepare(encoded, module_states, phase_shared_state=budget)
    full_output = _read(field, full_state, encoded, receivers)
    field.set_execution_mode("compact")
    compact_state = field.prepare(encoded, module_states, phase_shared_state=budget)
    compact_output = _read(field, compact_state, encoded, receivers)
    torch.testing.assert_close(compact_output, full_output, rtol=2e-5, atol=2e-5)
    full_grads = torch.autograd.grad(
        full_output.square().mean(), (field.router.group_codes, logits), retain_graph=True
    )
    compact_grads = torch.autograd.grad(
        compact_output.square().mean(), (field.router.group_codes, logits)
    )
    torch.testing.assert_close(compact_grads[0], full_grads[0], rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(compact_grads[1], full_grads[1], rtol=2e-4, atol=2e-5)


def test_closed_capacity_and_phase_plan_invariance() -> None:
    torch.manual_seed(1411)
    encoded = _encoded_case(hidden=8)
    module_states = torch.randn(1, 3, 8)
    receivers = torch.rand(1, 5, 2) * 2.0
    field12 = BudgetedGroupControlPairwiseField(8, 8, 2, 2, group_count=12, group_control_dim=4).eval()
    field32 = BudgetedGroupControlPairwiseField(8, 8, 2, 2, group_count=32, group_control_dim=4).eval()
    logits12 = torch.tensor([[100.0] * 5 + [-100.0] * 6])
    logits32 = torch.tensor([[100.0] * 5 + [-100.0] * 26])
    budget12 = field12.router.case_gate.build_budget(logits12, deterministic=True)
    budget32 = field32.router.case_gate.build_budget(logits32, deterministic=True)
    state12 = field12.prepare(encoded, module_states, phase_shared_state=budget12)
    state32 = field32.prepare(encoded, module_states, phase_shared_state=budget32)
    _read(field12, state12, encoded, receivers)
    _read(field32, state32, encoded, receivers)
    _copy_matching_state(field12, field32)
    for mode in ("full_width", "compact"):
        field12.set_execution_mode(mode)
        field32.set_execution_mode(mode)
        state12 = field12.prepare(encoded, module_states, phase_shared_state=budget12)
        state32 = field32.prepare(encoded, module_states, phase_shared_state=budget32)
        output12 = _read(field12, state12, encoded, receivers)
        output32 = _read(field32, state32, encoded, receivers)
        torch.testing.assert_close(output32, output12, rtol=2e-5, atol=2e-5)

    refreshed = field12.prepare(
        encoded,
        module_states + 0.2,
        phase_shared_state=state12["case_group_budget"],
    )
    assert refreshed["case_group_budget"] is state12["case_group_budget"]
    assert refreshed["group_control_state"] is not state12["group_control_state"]
    assert not torch.equal(
        refreshed["module_first_affine"],
        state12["module_first_affine"],
    )


def test_v1_gate_head_shape_and_probability_contract_remain_compatible() -> None:
    torch.manual_seed(1412)
    gate = CaseGroupGate(4, 12)
    assert tuple(gate.network.net[-1].weight.shape) == (1, gate.hidden_dim)
    context = torch.randn(2, 13)
    prototypes = torch.randn(12, 4)
    logits = gate.logits(context, prototypes)
    assert tuple(logits.shape) == (2, 11)
    budget = gate.build_budget(logits, deterministic=True)
    assert tuple(budget.positive_probability.shape) == (2, 12)
    assert tuple(budget.optional_open_probability.shape) == (2, 11)
    restored = CaseGroupGate(4, 12)
    restored.load_state_dict(gate.state_dict(), strict=True)
    torch.testing.assert_close(restored.logits(context, prototypes), logits)


def test_rescue_schedule_formulas_are_exact() -> None:
    assert routing_logit_scale(1) == 0.1
    assert routing_logit_scale(25) == 1.0
    assert routing_logit_scale(13) == 0.55
    assert routing_logit_scale(0) == 0.1
    assert sparsification_continuation(1) == 0.0
    assert sparsification_continuation(25) == 0.0
    assert sparsification_continuation(26) == pytest.approx(1.0 / 125.0)
    assert sparsification_continuation(150) == 1.0
    assert sparsification_continuation(500) == 1.0


def test_rescue_gate_is_symmetric_and_all_open_during_warmup() -> None:
    torch.manual_seed(1413)
    gate = CaseGroupGate(4, 12, hidden_dim=7, rescue_mode=True)
    context = torch.randn(2, 13)
    prototypes = torch.randn(12, 4)
    permutation = torch.tensor([7, 2, 11, 0, 4, 9, 1, 8, 3, 10, 6, 5])
    logits = gate.logits(context, prototypes)
    permuted_logits = gate.logits(context, prototypes[permutation])
    torch.testing.assert_close(permuted_logits, logits[:, permutation])
    warm = gate.build_budget(logits, deterministic=True, continuation=0.0)
    torch.testing.assert_close(warm.z, torch.ones_like(warm.z))
    torch.testing.assert_close(warm.eta, torch.full_like(warm.eta, 1.0 / 12.0))
    torch.testing.assert_close(warm.kappa, torch.full((2,), 12.0))
    assert warm.live_count.tolist() == [12, 12]
    assert warm.packed_width == 12
    permuted = gate.build_budget(permuted_logits, deterministic=True, continuation=1.0)
    reference = gate.build_budget(logits, deterministic=True, continuation=1.0)
    torch.testing.assert_close(permuted.z, reference.z[:, permutation])
    torch.testing.assert_close(permuted.raw_z, reference.raw_z[:, permutation])
    torch.testing.assert_close(permuted.eta, reference.eta[:, permutation])
    torch.testing.assert_close(permuted.support, reference.support[:, permutation])
    torch.testing.assert_close(permuted.kappa, reference.kappa)


def test_rescue_continuation_keeps_closed_raw_groups_executable() -> None:
    gate = CaseGroupGate(4, 12, rescue_mode=True)
    logits = torch.full((1, 12), -100.0)
    budget = gate.build_budget(logits, deterministic=True, continuation=0.4)
    assert not budget.raw_support.any()
    assert budget.support.all()
    torch.testing.assert_close(budget.z, torch.full_like(budget.z, 0.6))
    torch.testing.assert_close(budget.kappa, torch.full((1,), 12.0))
    torch.testing.assert_close(
        budget.expected_optional_count,
        budget.expected_excess_group_count,
    )


def test_rescue_full_hardening_fallback_uses_highest_logit() -> None:
    gate = CaseGroupGate(4, 12, rescue_mode=True)
    logits = torch.full((1, 12), -100.0)
    logits[0, 7] = -99.0
    budget = gate.build_budget(logits, deterministic=True, continuation=1.0)
    assert budget.fallback_used.tolist() == [True]
    assert not budget.raw_support.any()
    assert budget.support[0, 7]
    assert budget.live_count.tolist() == [1]
    torch.testing.assert_close(
        budget.eta[0],
        torch.nn.functional.one_hot(torch.tensor(7), 12).float(),
    )
    torch.testing.assert_close(budget.kappa, torch.ones(1))
    assert budget.packed_ids.tolist() == [[7]]


def test_rescue_mixed_batch_full_compact_parity_and_gradients() -> None:
    torch.manual_seed(1414)
    encoded = _encoded_case(batch=2, hidden=8)
    module_states = torch.randn(2, 3, 8)
    receivers = torch.rand(2, 5, 2) * 2.0
    field = BudgetedGroupControlPairwiseField(
        8,
        8,
        2,
        2,
        group_count=12,
        group_control_dim=4,
        rescue_mode=True,
        schedule="dense_to_sparse_v2",
    ).eval()
    logits = torch.tensor(
        [
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0] + [-100.0] * 6,
            [100.0, 100.0, 100.0] + [-100.0] * 9,
        ],
        requires_grad=True,
    )
    budget = field.router.case_gate.build_budget(logits, deterministic=True)
    assert budget.packed_ids.tolist() == [[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 0, 0]]
    assert budget.packed_valid.tolist() == [[True] * 6, [True, True, True, False, False, False]]
    features = field.router.query_fourier(receivers / field.router._scale(encoded))
    full_state = field.prepare(encoded, module_states, phase_shared_state=budget)
    full_output = field.read(full_state, encoded, receivers, features)[0]
    field.set_execution_mode("compact")
    compact_state = field.prepare(encoded, module_states, phase_shared_state=budget)
    compact_output = field.read(compact_state, encoded, receivers, features)[0]
    torch.testing.assert_close(compact_output, full_output, rtol=2e-5, atol=2e-5)
    full_grads = torch.autograd.grad(
        full_output.square().mean(), (field.router.group_codes, logits), retain_graph=True
    )
    compact_grads = torch.autograd.grad(
        compact_output.square().mean(), (field.router.group_codes, logits)
    )
    torch.testing.assert_close(compact_grads[0], full_grads[0], rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(compact_grads[1], full_grads[1], rtol=2e-4, atol=2e-5)


def test_rescue_progress_reuses_gate_plan_and_refreshes_controls() -> None:
    torch.manual_seed(1415)
    encoded = _encoded_case(hidden=8)
    module_states = torch.randn(1, 3, 8)
    field = BudgetedGroupControlPairwiseField(
        8,
        8,
        2,
        2,
        group_count=12,
        group_control_dim=4,
        rescue_mode=True,
        schedule="dense_to_sparse_v2",
    ).train()
    field.set_training_progress(epoch=1, total_epochs=50)
    p0 = field.prepare(encoded, module_states)
    budget = p0["case_group_budget"]
    assert budget.continuation == 0.0
    assert budget.route_logit_scale == 0.1
    p1 = field.prepare(encoded, module_states + 0.2, phase_shared_state=budget)
    assert p1["case_group_budget"] is budget
    assert p1["case_group_budget_gate_reused"].tolist() == [1.0]
    assert not torch.equal(p0["group_control_state"].group_control, p1["group_control_state"].group_control)
