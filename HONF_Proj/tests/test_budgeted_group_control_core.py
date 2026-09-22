"""Small CPU algebra contracts for the opt-in Run-1409 backend."""

from __future__ import annotations

import torch

from honf_forward_core.interface_fields.budgeted_group_control import (
    BudgetedGroupControlPairwiseField,
)
from honf_forward_core.interface_fields.case_group_budget import CaseGroupGate
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
