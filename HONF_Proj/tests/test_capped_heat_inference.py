"""Frozen train-support bounds, public starts and charged constrained trails."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from honf_inverse_core.heat_inference import (
    UnsupportedHeatTotal,
    bounded_trust_heat_inference,
    project_capped_heat,
    public_heat_starts,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from thermal_campaign_heat_inference import persist_heat_progress, selected_training_heat_bounds


def test_capped_projection_has_the_euclidean_water_filling_solution():
    active = torch.tensor([True, False, True, True])
    values = torch.tensor([2., 999., -1., 2.], dtype=torch.float64)
    total = torch.tensor(3., dtype=torch.float64)
    result = project_capped_heat(values, active, total, (.5, 1.5))
    torch.testing.assert_close(result, torch.tensor([1.25, 0., .5, 1.25], dtype=torch.float64))
    shifted = values + 7
    torch.testing.assert_close(project_capped_heat(shifted, active, total, (.5, 1.5)), result)
    torch.testing.assert_close(result.sum(), total, rtol=0, atol=1e-13)
    torch.testing.assert_close(project_capped_heat(result, active, total, (.5, 1.5)), result)


@pytest.mark.parametrize("total", [1.49, 4.51])
def test_infeasible_public_totals_are_unsupported_before_any_predictor_call(total):
    calls = []

    def predictor(heat):
        calls.append(heat)
        raise AssertionError("Unsupported task must not call the surrogate")

    trail = bounded_trust_heat_inference(predictor, torch.ones(2), torch.ones(1),
        torch.ones(3, dtype=torch.bool), torch.tensor(total), torch.ones(3) / 3,
        heat_bounds=(.5, 1.5))
    assert trail.status == "unsupported"
    assert "Public total" in trail.unsupported_reason
    assert trail.optimizer_steps == 0 and trail.heat == ()
    assert calls == [] and trail.forward_calls == () and trail.vjp_calls == ()
    with pytest.raises(UnsupportedHeatTotal):
        public_heat_starts(torch.ones(3, dtype=torch.bool), torch.tensor(total), heat_bounds=(.5, 1.5))


def test_seeded_public_starts_preserve_caps_total_and_inactive_rows():
    active = torch.tensor([True, False, True, True, True])
    total = torch.tensor(4.8)
    first = public_heat_starts(active, total, seed=731, heat_bounds=(.5, 1.5))
    repeated = public_heat_starts(active, total, seed=731, heat_bounds=(.5, 1.5))
    assert len(first) == 2
    assert not torch.equal(first[0], first[1])
    for a, b in zip(first, repeated):
        assert torch.equal(a, b)
        assert (a[active] >= .5).all() and (a[active] <= 1.5).all()
        assert (a[~active] == 0).all()
        torch.testing.assert_close(a.sum(), total)


@pytest.mark.parametrize("mode", ["joint", "graph", "ungrouped"])
def test_capped_trust_retains_observed_only_selection_and_block_mass(mode):
    def predictor(heat):
        return {"observed": heat[:2], "held": heat[2:], "groups": ([0, 1],),
                "peaks": heat + 1, "pressure": heat.sum()}

    kwargs = {"predictor": predictor, "observed": torch.tensor([.1, 2.]), "held": torch.tensor([.8]),
        "active": torch.ones(3, dtype=torch.bool), "total": torch.tensor(3.),
        "initial_fraction": torch.tensor([0., 1., 0.]), "mode": mode, "steps": 5,
        "heat_bounds": (.5, 1.5), "permutation_stream": (torch.tensor([1, 2, 0]),) * 5}
    first = bounded_trust_heat_inference(**kwargs)
    second = bounded_trust_heat_inference(**{**kwargs, "held": torch.tensor([1000.])})
    assert first.status == "completed"
    assert torch.equal(torch.stack(first.heat), torch.stack(second.heat))
    assert first.optimizer_steps == 5 and max(first.trial_evaluations) <= 2
    assert first.forward_calls[-1] == 1 + 5 + sum(first.trial_evaluations)
    for step, heat in enumerate(first.heat):
        assert (heat >= .5 - 1e-6).all() and (heat <= 1.5 + 1e-6).all()
        torch.testing.assert_close(heat.sum(), torch.tensor(3.))
        if step:
            assert ((heat - first.heat[step - 1]) / 3).abs().max() <= .0500001
            outside = torch.ones(3, dtype=torch.bool)
            outside[first.selected_modules[step - 1]] = False
            torch.testing.assert_close(heat[outside], first.heat[step - 1][outside])
    assert all(a >= b for a, b in zip(first.observed_rmse, first.observed_rmse[1:]))


def test_each_accepted_state_is_persisted_before_next_gradient(tmp_path):
    rows, calls = [], []
    path = tmp_path / "progress.npz"

    def predictor(heat):
        calls.append(1)
        return {"observed": heat[:2], "held": heat[2:]}

    def persist(row):
        persist_heat_progress(path, rows, row)
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["heat"][-1], row["heat"].cpu().numpy())
            assert saved["charged_forward_calls"][-1] == len(calls)
            assert saved["accepted_steps"][-1] == row["accepted_steps"]

    trail = bounded_trust_heat_inference(predictor, torch.tensor([.6, 1.4]), torch.tensor([1.]),
        torch.ones(3, dtype=torch.bool), torch.tensor(3.), torch.ones(3) / 3,
        steps=3, heat_bounds=(.5, 1.5), state_callback=persist)
    assert len(rows) == 4 and trail.accepted_steps > 0
    assert rows[-1]["accepted_steps"] == trail.accepted_steps


def test_training_heat_caps_ignore_inactive_padding_and_validation_outliers(tmp_path):
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as f:
        f["case_ids"] = np.array(["train_a", "train_b", "test_a"], dtype=h5py.string_dtype())
        f["splits"] = np.array(["train", "train", "test"], dtype=h5py.string_dtype())
        for name, values, active in (("train_a", [.7, 999.], [1., 0.]),
                                     ("train_b", [1.8, 1.2], [1., 1.]),
                                     ("test_a", [.01, 10.], [1., 1.])):
            group = f.create_group(f"cases/{name}")
            group["heat_powers"] = values
            group["module_present"] = active
    manifest = {"partitions": {"train": {"case_ids": ["train_a", "train_b"]}}, "manifest_sha256": "frozen"}
    bounds = selected_training_heat_bounds(path, manifest)
    assert bounds["minimum"] == .7 and bounds["maximum"] == 1.8
    assert bounds["selected_train_cases"] == 2 and bounds["active_heat_values"] == 3
    with pytest.raises(ValueError, match="nontraining"):
        selected_training_heat_bounds(path, {**manifest, "partitions": {"train": {"case_ids": ["test_a"]}}})


@pytest.mark.parametrize("count", [2, 3, 7, 12])
@pytest.mark.parametrize("cap", [.5036706328392029, 1.999153733253479])
def test_native_fp32_boundary_sum_is_feasible_without_widening_caps(count, cap):
    from honf_inverse_core.heat_inference import heat_feasibility_tolerance
    bounds = (.5036706328392029, 1.999153733253479)
    heat = torch.full((count,), cap)
    total = heat.sum()
    active = torch.ones(count, dtype=torch.bool)
    projected = project_capped_heat(heat, active, total, bounds)
    assert torch.equal(projected, heat)
    tolerance = heat_feasibility_tolerance(total, count, bounds, dtype=heat.dtype)
    assert abs(float(projected.double().sum() - total.double())) <= tolerance
    outside = total - 8 * tolerance if cap == bounds[0] else total + 8 * tolerance
    with pytest.raises(UnsupportedHeatTotal):
        project_capped_heat(heat, active, outside, bounds)


def test_repeated_graph_proposals_at_native_heat_caps_report_restoration_residuals():
    bounds = (.5036706328392029, 1.999153733253479)
    heat = torch.tensor([bounds[0], bounds[0], bounds[1], bounds[1]])
    total = heat.sum()
    active = torch.ones(4, dtype=torch.bool)
    checked = []

    def predictor(values):
        assert values.dtype == torch.float32
        assert (values >= bounds[0]).all() and (values <= bounds[1]).all()
        checked.append(values.detach().clone())
        return {"observed": values[:2], "held": values[2:], "groups": ([0, 1],)}

    trail = bounded_trust_heat_inference(predictor, torch.full((2,), bounds[1]), heat[2:],
        active, total, heat / total, mode="graph", steps=10, heat_bounds=bounds)
    assert trail.status == "completed" and trail.optimizer_steps == 10
    assert len(trail.total_residual) == len(trail.heat) == 11
    assert max(map(abs, trail.total_residual)) <= trail.feasibility_tolerance
    assert max(trail.bound_excess) == 0.
    for index, values in enumerate(trail.heat):
        assert (values >= bounds[0]).all() and (values <= bounds[1]).all()
        assert trail.total_residual[index] == float(values.double().sum() - total.double())
        if index:
            assert torch.equal(values[2:], trail.heat[index - 1][2:])
    assert len(checked) == trail.forward_calls[-1]
