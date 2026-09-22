"""Focused algebra checks for Run-1409 occupancy evidence."""

from __future__ import annotations

import torch

from honf_forward_core.evaluation.occupancy_adaptive import (
    audit_assignment,
    derive_occupancy_plan,
    exact_empty_columns,
    mask_query_assignments,
    occupancy_kappa,
    pack_columns,
    parity_report,
    permutation_report,
    support_metrics,
)
from tools.diagnostics.occupancy_adaptive_evidence import canonicalize_case


def _assignments() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    module = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 0.7, 0.3, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    environment = torch.tensor(
        [[[0.6, 0.0, 0.4, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    # The final assignment is proposal-masked and moves only mass among the
    # proposal occupied columns 0, 1, and 2.
    final_module = torch.tensor(
        [[[0.9, 0.1, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    final_environment = torch.tensor(
        [[[0.5, 0.0, 0.5, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    return module, environment, final_module, final_environment


def test_proposal_and_refined_rows_are_normalized_and_empty_columns_are_exact() -> None:
    module, environment, final_module, _final_environment = _assignments()
    module_valid = torch.tensor([[True, True, False]])
    proposal_audit = audit_assignment(module, row_valid=module_valid)
    final_audit = audit_assignment(final_module, row_valid=module_valid)
    environment_audit = audit_assignment(environment)
    assert proposal_audit.active_rows_normalized
    assert proposal_audit.inactive_rows_zero
    assert final_audit.active_rows_normalized
    assert environment_audit.active_rows_normalized
    assert exact_empty_columns(module, row_valid=module_valid).tolist() == [[False, False, False, True]]
    assert exact_empty_columns(final_module).tolist() == [[False, False, False, True]]


def test_occupancy_plan_preserves_ids_computes_kappa_and_pads_without_capacity_effect() -> None:
    module, environment, final_module, final_environment = _assignments()
    module_valid = torch.tensor([[True, True, False]])
    module_measure = torch.tensor([[0.5, 0.5, 0.0]])
    environment_measure = torch.tensor([[0.2, 0.3, 0.5]])
    module_coordinates = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [9.0, 9.0]]])
    environment_coordinates = torch.tensor([[[0.0, 1.0], [1.0, 1.0], [4.0, 4.0]]])
    plan = derive_occupancy_plan(
        module,
        environment,
        final_module,
        final_environment,
        module_measure,
        environment_measure,
        module_coordinates,
        environment_coordinates,
        module_valid=module_valid,
    )
    assert plan.kmax == 4
    assert plan.k_plan.tolist() == [3]
    assert plan.prototype_ids.tolist() == [[0, 1, 2, -1]]
    assert plan.packed_valid.tolist() == [[True, True, True, False]]
    expected_kappa = occupancy_kappa(plan.module_mass, plan.environment_mass)
    torch.testing.assert_close(plan.kappa, expected_kappa)
    packed = pack_columns(final_environment, plan.prototype_ids, packed_valid=plan.packed_valid)
    torch.testing.assert_close(packed.values[..., :3], final_environment[..., :3])
    assert torch.equal(packed.values[..., 3], torch.zeros_like(packed.values[..., 3]))


def test_plan_reuse_ids_can_refresh_current_assignments_and_mask_empty_phase_sources() -> None:
    module, environment, final_module, final_environment = _assignments()
    module_valid = torch.tensor([[True, True, False]])
    plan = derive_occupancy_plan(
        module,
        environment,
        final_module,
        final_environment,
        torch.tensor([[0.5, 0.5, 0.0]]),
        torch.tensor([[0.2, 0.3, 0.5]]),
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [9.0, 9.0]]]),
        torch.tensor([[[0.0, 1.0], [1.0, 1.0], [4.0, 4.0]]]),
        module_valid=module_valid,
    )
    # P1/P2 can change source memberships while retaining P0 original IDs.
    refreshed = torch.tensor([[[0.2, 0.8, 0.0, 0.0], [0.0, 0.2, 0.8, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    assert torch.equal(plan.prototype_ids, torch.tensor([[0, 1, 2, -1]]))
    assert not torch.equal(refreshed, final_module)
    phase_source_mass = torch.tensor([[0.7, 0.3, 0.0, 0.0]])
    logits = torch.zeros(1, 3, 4)
    query = mask_query_assignments(logits, phase_source_mass)
    assert torch.equal(query[..., 2:], torch.zeros_like(query[..., 2:]))
    torch.testing.assert_close(query.sum(dim=-1), torch.ones(1, 3))


def test_full_and_packed_control_parity_and_source_permutation() -> None:
    module, environment, final_module, final_environment = _assignments()
    plan = derive_occupancy_plan(
        module,
        environment,
        final_module,
        final_environment,
        torch.tensor([[0.5, 0.5, 0.0]]),
        torch.tensor([[0.2, 0.3, 0.5]]),
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [9.0, 9.0]]]),
        torch.tensor([[[0.0, 1.0], [1.0, 1.0], [4.0, 4.0]]]),
        module_valid=torch.tensor([[True, True, False]]),
    )
    packed = pack_columns(final_environment, plan.prototype_ids, packed_valid=plan.packed_valid)
    full_control = (final_environment * torch.arange(1, 5, dtype=torch.float32)).sum(dim=-1)
    packed_control = (packed.values * torch.arange(1, 5, dtype=torch.float32)).sum(dim=-1)
    assert parity_report(full_control, packed_control)["pass"]
    permutation = torch.tensor([2, 0, 1])
    report = permutation_report(
        environment[..., :3],
        torch.tensor([[0.2, 0.3, 0.5]]),
        torch.tensor([[[0.0, 1.0], [1.0, 1.0], [4.0, 4.0]]]),
        permutation,
    )
    assert float(report["mass_max_abs_error"]) < 1.0e-6
    assert float(report["centre_max_abs_error"]) < 1.0e-6


def test_nonconsecutive_original_ids_gather_matching_controls_for_packed_parity() -> None:
    values = torch.tensor([[[0.25, 0.0, 0.5, 0.0, 0.0, 0.25], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]])
    prototype_ids = torch.tensor([[0, 2, 5, 0, 0, 0]], dtype=torch.long)
    packed_valid = torch.tensor([[True, True, True, False, False, False]])
    packed = pack_columns(values, prototype_ids, packed_valid=packed_valid)
    controls = torch.arange(1.0, 7.0).reshape(1, 6, 1)
    gathered_controls = controls.gather(
        1, prototype_ids.clamp_min(0).unsqueeze(-1).expand(-1, -1, 1)
    )
    gathered_controls = gathered_controls * packed_valid.unsqueeze(-1)
    full_output = torch.einsum("bsk,bkd->bsd", values, controls)
    packed_output = torch.einsum("bsp,bpd->bsd", packed.values, gathered_controls)
    torch.testing.assert_close(full_output, packed_output)


def test_support_metrics_keep_logical_paths_unique_pairs_and_actual_rows_separate() -> None:
    module, environment, _final_module, _final_environment = _assignments()
    query = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.4, 0.6, 0.0]]])
    metrics = support_metrics(
        module,
        environment,
        query,
        module_valid=torch.tensor([[True, True, False]]),
        actual_module_rows=torch.tensor([4]),
        actual_environment_rows=torch.tensor([6]),
        padded_module_rows=torch.tensor([8]),
        padded_environment_rows=torch.tensor([6]),
    )
    assert metrics["module_logical_paths"].item() >= metrics["module_unique_pairs"].item()
    assert metrics["environment_logical_paths"].item() >= metrics["environment_unique_pairs"].item()
    assert metrics["actual_module_rows"].item() == 4
    assert metrics["padded_module_rows"].item() == 8


def test_population_case_preserves_per_source_actual_row_ledger() -> None:
    _module, environment, final_module, _final_environment = _assignments()
    payload = {
        "module_coords": torch.tensor([[[0.0, 0.0], [1.0, 0.0], [9.0, 9.0]]]),
        "environment_coords": torch.tensor([[[0.0, 1.0], [1.0, 1.0], [4.0, 4.0]]]),
        "query_xy": torch.tensor([[[0.25, 0.25], [2.0, 2.0]]]),
        "module_present": torch.tensor([[1.0, 1.0, 0.0]]),
        "module_assignment": final_module,
        "environment_assignment": environment,
        "query_assignment": torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]]]
        ),
        "module_measure": torch.tensor([[0.5, 0.5, 0.0]]),
        "environment_measure": torch.tensor([[0.2, 0.3, 0.5]]),
        "occupancy_group_phase_ledger": {
            "P2": {
                "module": {"actual_rows": 24, "padded_rows": 24},
                "environment": {"actual_rows": 32, "padded_rows": 32},
            }
        },
    }
    record = canonicalize_case(payload, query_count=2, kmax=4)
    assert record["case"]["p2_module_actual_rows"] == 24
    assert record["case"]["p2_environment_actual_rows"] == 32
    assert record["case"]["module_assignment_numerical_rank"] == 2
    assert record["case"]["environment_assignment_numerical_rank"] == 3
    assert record["case"]["query_assignment_numerical_rank"] == 2
    assert record["case"]["joint_source_assignment_numerical_rank"] == 4
