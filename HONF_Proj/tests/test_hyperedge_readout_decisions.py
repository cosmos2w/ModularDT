"""TRAIN-only gate boundaries, unavailable ratios and parent-deterioration controls."""

import math

import pytest

from honf_runtime.hyperedge_readout_decisions import (
    common_extension_decision,
    common_extension_signal,
    full_bank_signal,
    local_readout_signal,
    priority_contrast,
)


def thermal(v=1., p=1., **changes):
    return {"v": v, "p": p, "temperature_fluid": 1., "temperature_surface": 1.,
            "temperature_material": 1., "q_proxy": 1., "u": 1., "omega": 1., **changes}


def local(edge, pool, parent, detailed=None, objective=.9):
    return local_readout_signal("thermal", edge=edge, pool=pool, parent=parent,
                                detailed_edge=detailed or [edge] * 4, detailed_pool=[pool] * 4,
                                edge_native_objective=objective, parent_native_objective=1.)


def test_predeclared_signal_cannot_be_produced_only_by_deteriorating_pool():
    result = local(thermal(), thermal(v=2., p=2.), thermal())
    assert result["edge_over_pool"]["ratio"] == .5
    assert not result["useful_signal"]
    assert result["misses"] == ["parent_priority_reduction_at_least_5pct"]


def test_signal_requires_native_guards_parent_fit_and_three_original_fixtures():
    parent = pool = thermal()
    edge = thermal(v=.81, p=.98)
    assert math.isclose(priority_contrast("thermal", edge, pool)["ratio"], math.sqrt(.81 * .98))
    assert local(edge, pool, parent)["useful_signal"]
    assert not local(edge, pool, parent, detailed=[edge, edge, pool, pool])["useful_signal"]
    assert not local(thermal(v=.7, p=1.051), pool, parent)["useful_signal"]
    assert not local({**edge, "q_proxy": 1.051}, pool, parent)["useful_signal"]
    assert not local(edge, pool, parent, objective=1.0001)["useful_signal"]


def test_zero_error_reference_remains_unavailable_and_retains_absolute_errors():
    result = local(thermal(v=.1, p=.1), thermal(v=0., p=0.), thermal())
    assert not result["useful_signal"]
    assert result["edge_over_pool"]["ratio"] is None
    assert result["edge_over_pool"]["native_means"]["v"] == {"candidate": .1, "reference": 0.}
    missing = thermal(v=.1, p=.1)
    missing.pop("temperature_material")
    assert not local(missing, thermal(), thermal())["useful_signal"]


def test_extension_requires_full_windows_finite_gradients_and_three_percent():
    kwargs = {"priority_fit300": 1., "priority_fit500": .98,
              "objective_epochs301_400": [2.] * 100, "objective_epochs401_500": [1.94] * 100,
              "finite_gradients": True}
    assert common_extension_signal(**kwargs)["train_signal"]
    assert not common_extension_signal(**{**kwargs, "objective_epochs401_500": [1.95] * 100})["train_signal"]
    assert not common_extension_signal(**{**kwargs, "finite_gradients": False})["train_signal"]
    assert common_extension_signal(**{**kwargs, "priority_fit500": .96,
                                      "objective_epochs401_500": [2.] * 100})["train_signal"]
    with pytest.raises(ValueError, match="complete 100-epoch"):
        common_extension_signal(**{**kwargs, "objective_epochs301_400": [2.] * 99})


def test_global_signal_needs_both_comparators_and_nonpriority_guards():
    kwargs = {"dataset": "thermal", "global_board": thermal(v=.8, p=.8),
              "edge": thermal(v=.9, p=.9), "parent": thermal(),
              "global_native_objective": .8, "parent_native_objective": 1.}
    assert full_bank_signal(**kwargs)["useful_signal"]
    assert not full_bank_signal(**{**kwargs, "edge": thermal(v=.85, p=.85)})["useful_signal"]
    assert not full_bank_signal(**{**kwargs, "global_board": thermal(v=.8, p=.8, u=1.11)})["useful_signal"]


def test_wind_priority_is_ux_and_vector_error_stays_a_guard():
    parent = {key: 1. for key in ("near_turbine/Ux", "downstream_envelope/Ux", "near_turbine/vector",
                                  "downstream_envelope/vector", "volume/Ux", "volume/vector",
                                  "background/Ux", "background/vector")}
    edge = {**parent, "near_turbine/Ux": .8, "downstream_envelope/Ux": .9}
    kwargs = {"dataset": "wind", "edge": edge, "pool": parent, "parent": parent,
              "detailed_edge": [edge] * 4, "detailed_pool": [parent] * 4,
              "edge_native_objective": .9, "parent_native_objective": 1.}
    assert local_readout_signal(**kwargs)["useful_signal"]
    assert not local_readout_signal(**{**kwargs, "edge": {**edge, "near_turbine/vector": 1.051}})["useful_signal"]


def test_extension_reserves_both_arms_and_rejects_an_unhealthy_counterpart():
    improving = common_extension_signal(priority_fit300=1., priority_fit500=.96,
                                        objective_epochs301_400=[1.] * 100,
                                        objective_epochs401_500=[1.] * 100, finite_gradients=True)
    flat = {**improving, "train_signal": False}
    kwargs = {"dataset": "thermal", "arms": {"pool_first": flat, "edge_first": improving},
              "source_checkpoint_sha256": "a" * 64, "audit_panel_sha256": "b" * 64,
              "paired_resource_reserved": True}
    assert common_extension_decision(**kwargs)["approved"]
    assert not common_extension_decision(**{**kwargs, "paired_resource_reserved": False})["approved"]
    assert not common_extension_decision(**{**kwargs, "arms": {"pool_first": {**flat, "finite_gradients": False},
                                                              "edge_first": improving}})["approved"]
