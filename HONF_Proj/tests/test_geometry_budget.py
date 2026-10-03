"""Source geometry controls preserve joint action budgets and physical guards."""

import pytest
import torch

from honf_forward_core.evaluation.geometry_budget import geometry_action_budget


def run(weight, *, measures=None, near=None, valid=None, **budgets):
    weight = torch.tensor(weight, dtype=torch.float32)[None]
    support = weight > 0
    control = torch.arange(weight.numel() * 2, dtype=torch.float32).reshape(*weight.shape, 2) * support[..., None]
    receivers = torch.tensor([[[0.0, 0.0], [10.0, 0.0]]])[:, : weight.shape[1]]
    sources = torch.tensor([[[10.0, 0.0], [0.0, 0.0], [5.0, 0.0]]])[:, : weight.shape[2]]
    measures = torch.ones(1, weight.shape[-1]) if measures is None else torch.tensor([measures])
    valid = torch.ones_like(support) if valid is None else torch.tensor([valid])
    near = None if near is None else torch.tensor([near])
    density = weight.clone()
    original = (density.clone(), weight.clone(), support.clone(), control.clone())
    result, stats = geometry_action_budget(
        density,
        weight,
        support,
        control,
        receivers,
        sources,
        measures,
        valid,
        near,
        torch.arange(weight.shape[-1])[None],
        **budgets,
    )
    for before, value in zip(original, (density, weight, support, control), strict=True):
        torch.testing.assert_close(before, value, rtol=0, atol=0)
    return original, result, stats, measures


def joint_rows(tensors):
    density, weight, support, control = [tensor.numpy() for tensor in tensors]
    return [
        [
            sorted(
                (float(density[b, r, s]), float(weight[b, r, s]), bool(support[b, r, s]), *control[b, r, s])
                for s in range(weight.shape[-1])
            )
            for r in range(weight.shape[1])
        ]
        for b in range(weight.shape[0])
    ]


def test_two_by_two_changes_support_preserving_both_degrees_and_joint_row_tuples():
    before, after, stats, measures = run([[2.0, 0.0, 1.0], [0.0, 1.0, 2.0]], max_active_swaps=0)
    assert stats["support_switches"] == 1 and stats["changed_support_pairs"] == 4
    assert stats["geometry_cost_after"] < stats["geometry_cost_before"]
    assert not stats["source_weighted_sums_preserved"]
    assert joint_rows(before) == joint_rows(after)
    torch.testing.assert_close(before[2].sum(-1), after[2].sum(-1))
    torch.testing.assert_close(before[2].sum(-2), after[2].sum(-2))
    for index in (0, 1):
        torch.testing.assert_close(
            (before[index] * measures[:, None]).sum(-1), (after[index] * measures[:, None]).sum(-1)
        )
    torch.testing.assert_close(
        (before[0][..., None] * before[3] * measures[:, None, :, None]).sum(-2),
        (after[0][..., None] * after[3] * measures[:, None, :, None]).sum(-2),
    )


def test_dense_active_swaps_change_weight_positions_without_support_claim():
    before, after, stats, _ = run([[1.5, 0.5]])
    assert stats["support_switches"] == 0 and stats["active_tuple_swaps"] == 1
    assert stats["changed_support_pairs"] == 0 and not stats["effective_support_control"]
    assert stats["changed_weight_positions"] == 2
    assert stats["geometry_cost_after"] < stats["geometry_cost_before"]
    assert joint_rows(before) == joint_rows(after)


@pytest.mark.parametrize("guard", ["measure", "near", "eligibility"])
def test_unequal_measure_near_and_ineligible_crosses_do_not_move(guard):
    options = (
        {"measures": [1.0, 2.0]}
        if guard == "measure"
        else {"near": [[1.0, 0.0]]}
        if guard == "near"
        else {"valid": [[True, False]]}
    )
    before, after, stats, _ = run([[1.5, 0.5]], **options)
    for original, changed in zip(before, after, strict=True):
        torch.testing.assert_close(original, changed, rtol=0, atol=0)
    assert stats["geometry_cost_before"] == stats["geometry_cost_after"]


def test_deterministic_budget_and_no_action_degeneracy():
    a = run([[2.0, 0.0, 1.0], [0.0, 1.0, 2.0]], max_support_switches=1, max_active_swaps=1)
    b = run([[2.0, 0.0, 1.0], [0.0, 1.0, 2.0]], max_support_switches=1, max_active_swaps=1)
    assert a[2] == b[2]
    for first, second in zip(a[1], b[1], strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    before, after, stats, _ = run([[1.0, 1.0]], max_candidate_checks=0)
    assert joint_rows(before) == joint_rows(after)
    assert not stats["effective_support_control"]
    with pytest.raises(ValueError, match="nonnegative"):
        run([[1.0, 1.0]], max_support_switches=-1)


def test_nan_padding_cost_masked_and_valid_nan_geometry_rejected():
    density = weight = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]])
    support = weight > 0
    control = weight[..., None].expand(-1, -1, -1, 2).clone()
    receivers = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [float("nan"), float("nan")]]])
    sources = torch.tensor([[[10.0, 0.0], [0.0, 0.0], [float("nan"), float("nan")]]])
    measures = torch.tensor([[1.0, 1.0, float("nan")]])
    valid = torch.zeros_like(support)
    valid[:, :2, :2] = True
    after, stats = geometry_action_budget(
        density, weight, support, control, receivers, sources, measures, valid, None, torch.arange(3)[None]
    )
    assert stats["geometry_cost_before"] == 400.0 and stats["geometry_cost_after"] == 0.0
    for original, changed in zip((density, weight, support, control), after, strict=True):
        torch.testing.assert_close(original[:, 2], changed[:, 2], rtol=0, atol=0)
        torch.testing.assert_close(original[:, :, 2], changed[:, :, 2], rtol=0, atol=0)
    receivers[:, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="geometry must be finite"):
        geometry_action_budget(
            density, weight, support, control, receivers, sources, measures, valid, None, torch.arange(3)[None]
        )
