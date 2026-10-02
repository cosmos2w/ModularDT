"""Family-split and state-version checks for measured action fitting."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.action_risk_fit import (
    ActionEvidenceRow,
    crossfit_action_risk,
    crossfit_ridge_action_differences,
    evaluate_action_policy,
    fit_action_risk_head,
    fit_ridge_action_baseline,
    fit_ridge_action_differences,
)


def _rows() -> tuple[ActionEvidenceRow, ...]:
    rows = []
    for family, bias in (("fit", 0.0), ("held", 0.4)):
        for action, work, trained, full in (
            ("root", 0.6, True, False), ("two", 0.8, True, False),
            ("full", 1.0, False, True),
        ):
            value = bias + {"root": 0.1, "two": 0.2, "full": 0.3}[action]
            rows.append(ActionEvidenceRow(
                family_key=family, case_key=f"{family}-case", action_key=action,
                forward_sha256="fixed-forward-v1",
                packet_rows=torch.tensor([[value, 0.2], [0.3, value]]),
                budget_vector=torch.tensor([0.9, 0.9]),
                receiver_role_features=torch.tensor([[1.0, 0.1], [0.2, 1.0]]),
                candidate_role_error=torch.tensor([1.0 + value, 2.0 + value]),
                incumbent_role_error=torch.tensor([1.0, 2.0]),
                numerical_floor=torch.tensor([0.001, 0.001]),
                exact_work=work, nonredundant_k=1 if full else 2,
                trained_sparse=trained, full_access=full,
            ))
    return tuple(rows)


def test_held_family_labels_cannot_affect_neural_or_ridge_fit() -> None:
    rows = _rows()
    poisoned = tuple(
        replace(row, candidate_role_error=torch.tensor([float("nan"), float("nan")]))
        if row.family_key == "held" else row for row in rows
    )
    kwargs = {"current_forward_sha256": "fixed-forward-v1", "train_families": ("fit",)}
    first = fit_action_risk_head(rows, updates=8, seed=73, **kwargs)
    second = fit_action_risk_head(poisoned, updates=8, seed=73, **kwargs)
    assert first.train_case_keys == ("fit-case",)
    assert first.first_loss == second.first_loss
    assert first.last_loss == second.last_loss
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name], rtol=0, atol=0)
    ridge = fit_ridge_action_baseline(rows, **kwargs)
    poisoned_ridge = fit_ridge_action_baseline(poisoned, **kwargs)
    torch.testing.assert_close(ridge.coefficients, poisoned_ridge.coefficients, rtol=0, atol=0)


def test_action_table_rejects_stale_forward_weights_and_missing_full_fallback() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="stale"):
        fit_action_risk_head(rows, current_forward_sha256="later-forward-v2", train_families=("fit",), updates=1)
    no_full = tuple(row for row in rows if not row.full_access)
    with pytest.raises(ValueError, match="full-access"):
        fit_ridge_action_baseline(
            no_full, current_forward_sha256="fixed-forward-v1", train_families=("fit",),
        )


def test_grouped_crossfit_uses_only_named_fit_families() -> None:
    rows = _rows()
    outside = tuple(
        replace(row, family_key="outside", case_key="outside-case")
        for row in rows if row.family_key == "held"
    )
    clean = rows + outside
    poisoned = rows + tuple(
        replace(row, candidate_role_error=torch.full((2,), float("nan")))
        for row in outside
    )
    kwargs = {
        "current_forward_sha256": "fixed-forward-v1",
        "train_families": ("fit", "held"),
        "folds": 2,
        "updates": 2,
        "seed": 19,
    }
    first = crossfit_action_risk(clean, **kwargs)
    second = crossfit_action_risk(poisoned, **kwargs)
    assert len(first.row_keys) == 6
    assert set(first.family_keys) == {"fit", "held"}
    assert first.neural.shape == first.ridge.shape == first.measured.shape == (6, 2)
    assert torch.all(first.neural_upper_margin >= 0)
    torch.testing.assert_close(first.neural, second.neural, rtol=0, atol=0)
    torch.testing.assert_close(first.ridge, second.ridge, rtol=0, atol=0)


def test_action_difference_ridge_uses_same_case_full_and_training_families_only() -> None:
    rows = _rows()
    fit = fit_ridge_action_differences(
        rows, current_forward_sha256="fixed-forward-v1", train_families=("fit",),
    )
    assert fit.train_families == ("fit",)
    assert fit.train_case_keys == ("fit-case",)
    fit_case = [row for row in rows if row.case_key == "fit-case"]
    full = next(row for row in fit_case if row.full_access)
    sparse = next(row for row in fit_case if row.trained_sparse)
    assert fit.predict(sparse, full).shape == (2,)

    changed_held = tuple(
        replace(row, candidate_role_error=row.candidate_role_error + 25.0)
        if row.family_key == "held" else row for row in rows
    )
    changed_fit = fit_ridge_action_differences(
        changed_held, current_forward_sha256="fixed-forward-v1", train_families=("fit",),
    )
    torch.testing.assert_close(fit.coefficients, changed_fit.coefficients, rtol=0, atol=0)

    crossfit = crossfit_ridge_action_differences(
        rows, current_forward_sha256="fixed-forward-v1", train_families=("fit", "held"),
    )
    assert crossfit.row_keys == tuple(sorted(crossfit.row_keys, key=lambda key: (key[0].split("-")[0], *key)))
    assert len(crossfit.fold_family_splits) == 2
    for training, held in crossfit.fold_family_splits:
        assert len(held) == 1
        assert set(training).isdisjoint(held)
        assert set(training) | set(held) == {"fit", "held"}
    held_rows = [row for row in rows if row.family_key == "held" and row.trained_sparse]
    for row in held_rows:
        index = crossfit.row_keys.index((row.case_key, row.action_key))
        full = next(item for item in rows if item.case_key == row.case_key and item.full_access)
        expected = torch.log(
            (row.candidate_role_error.double() + row.numerical_floor.double())
            / (full.candidate_role_error.double() + full.numerical_floor.double())
        )
        torch.testing.assert_close(crossfit.measured[index], expected, rtol=0, atol=0)

    changed_crossfit = crossfit_ridge_action_differences(
        changed_held, current_forward_sha256="fixed-forward-v1", train_families=("fit", "held"),
    )
    held_indices = [index for index, family in enumerate(crossfit.family_keys) if family == "held"]
    torch.testing.assert_close(
        crossfit.predicted[held_indices], changed_crossfit.predicted[held_indices], rtol=0, atol=0,
    )


def test_action_difference_ridge_detaches_fixed_action_descriptors() -> None:
    features = torch.tensor([[0.2, 0.1], [0.4, 0.3]], requires_grad=True)
    rows = tuple(
        replace(row, packet_rows=features) if row.family_key == "fit" else row
        for row in _rows()
    )
    fit = fit_ridge_action_differences(
        rows, current_forward_sha256="fixed-forward-v1", train_families=("fit",),
    )
    assert not fit.coefficients.requires_grad
    assert not fit.feature_mean.requires_grad
    assert features.grad is None


def test_action_risk_fit_does_not_backpropagate_into_frozen_action_inputs() -> None:
    features = torch.tensor([[0.2, 0.1], [0.4, 0.3]], requires_grad=True)
    rows = tuple(
        replace(row, packet_rows=features) if row.family_key == "fit" else row
        for row in _rows()
    )
    fit_action_risk_head(
        rows, current_forward_sha256="fixed-forward-v1",
        train_families=("fit",), updates=2,
    )
    assert features.grad is None


def test_policy_audit_reports_oracle_and_false_choices_without_fallback_success() -> None:
    rows = tuple(
        replace(row, candidate_role_error=torch.tensor([1.08, 2.08]))
        if row.family_key == "fit" and row.action_key == "root" else row
        for row in _rows()
    )
    predictions = {
        (row.case_key, row.action_key): torch.tensor(
            [0.20, 0.20] if row.family_key == "held" or row.action_key == "root"
            else [0.0, 0.0]
        ) for row in rows
    }
    results = evaluate_action_policy(
        rows, predictions, current_forward_sha256="fixed-forward-v1",
        fixed_role_log_limits=torch.tensor([0.1, 0.1]),
        absolute_allowance_by_case={
            "fit-case": torch.zeros(2), "held-case": torch.zeros(2),
        },
        empirical_margin=torch.zeros(2),
    )
    fit, held = results
    assert fit.measured_oracle_action_key == "root"
    assert fit.selected_action_key == "two"
    assert fit.false_safe_sparse == ("two",)
    assert fit.false_reject_sparse == ("root",)
    assert not fit.selected_supported_sparse
    assert held.measured_oracle_action_key is None
    assert held.unsupported_at_budget and held.selected_action_key == "full"
    assert not held.selected_supported_sparse and held.selected_nonredundant_k is None


def test_held_incumbent_changes_oracle_but_not_deployment_selection() -> None:
    rows = _rows()
    predictions = {
        (row.case_key, row.action_key): torch.tensor([0.20, 0.20])
        for row in rows
    }
    limits = torch.tensor([0.1, 0.1])
    allowances = {"fit-case": torch.zeros(2), "held-case": torch.zeros(2)}
    original = evaluate_action_policy(
        rows, predictions, current_forward_sha256="fixed-forward-v1",
        fixed_role_log_limits=limits,
        absolute_allowance_by_case=allowances,
        empirical_margin=torch.zeros(2),
    )
    changed_held_incumbent = tuple(
        replace(row, incumbent_role_error=torch.tensor([2.0, 3.0]))
        if row.family_key == "held" else row
        for row in rows
    )
    changed = evaluate_action_policy(
        changed_held_incumbent, predictions, current_forward_sha256="fixed-forward-v1",
        fixed_role_log_limits=limits,
        absolute_allowance_by_case=allowances,
        empirical_margin=torch.zeros(2),
    )
    original_held = next(row for row in original if row.case_key == "held-case")
    changed_held = next(row for row in changed if row.case_key == "held-case")
    assert original_held.selected_action_key == changed_held.selected_action_key == "full"
    assert original_held.measured_oracle_action_key is None
    assert changed_held.measured_oracle_action_key == "root"
