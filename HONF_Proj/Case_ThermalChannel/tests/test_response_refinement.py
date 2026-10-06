"""TRAIN eligibility, role-normalized responses, null steps and accumulated updates."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from channelthermal.interaction_evidence.response_dataset import ResponseStencil
from channelthermal.interaction_evidence.types import (
    DesignState,
    EvidenceSource,
    EvidenceSplit,
    MeasuredQuantity,
    ModuleState,
    OperatingContext,
    PhysicalSolveOutput,
    RoleOutput,
    SolveRecord,
    SolveStatus,
)
from channelthermal.training.campaign_response import NativeCampaignResponse
from channelthermal.training.response_refinement import (
    FIT_ANCHORS,
    HEAT_VARIANTS,
    NativeResponseRefinement,
    feasible_heat_transfer,
    fit_response_scales,
    gradient_component_norms,
    normalized_role_error,
    pooled_gradient_coefficients,
    response_pair_losses,
    response_refinement_declaration,
    validate_response_addendum,
    validate_response_refinement_declaration,
)
from torch import nn


def _stencil(anchor, *, delta=2., split=EvidenceSplit.TRAIN, rows=2):
    modules = (ModuleState(f"{anchor}:module:0", (1., 1.), 1.),)
    design = DesignState(anchor, f"family:{anchor}", split, modules)
    roles = {}
    for name, channels in {
        "fluid_fields": ("u", "v", "p", "omega", "temperature"),
        "interface": ("T_surface", "q_normal"), "solid_temperature": ("temperature",),
    }.items():
        roles[name] = RoleOutput(name, np.zeros((rows, 2)), np.zeros((rows, len(channels))), channels,
            tuple("native_unit" for _ in channels), np.ones((rows, len(channels)), bool), np.ones(rows),
            tuple(f"{anchor}:{name}:{i}" for i in range(rows)),
            None if name == "fluid_fields" else tuple(modules[0].module_id for _ in range(rows)))
    output = PhysicalSolveOutput(roles, {"pressure_drop": MeasuredQuantity(0., "native_unit")},
                                 design.active_module_ids, {modules[0].module_id: 0.}, {})
    baseline = SolveRecord(f"{anchor}:baseline", design, OperatingContext({}), EvidenceSource.REFERENCE_SOLVER,
                           SolveStatus.CONVERGED, 0., {}, output)
    variants = {}
    for label, sign in zip(HEAT_VARIANTS, (-1., 1.)):
        trial_roles = {}
        for name, role in roles.items():
            values = np.array(role.values, copy=True)
            values[:, -1] = sign * delta
            if name == "interface":
                values[:, 0] = sign * delta
            trial_roles[name] = replace(role, values=values)
        variants[label] = replace(baseline, record_id=f"{anchor}:{label}", output=replace(output, roles=trial_roles))
    return ResponseStencil(baseline, variants)


def _stats():
    return {"field_std_by_channel": np.array([1., 2., 3., 4., 10.]),
            "interface_target_std": np.array([20., 30.]), "internal_temperature_std": np.array([40.])}


def test_addendum_requires_named_train_fit_families_and_exact_heat_labels(tmp_path):
    settings = response_refinement_declaration(tmp_path)
    validate_response_refinement_declaration(settings)
    with pytest.raises(ValueError, match="ThermalChannel"):
        validate_response_refinement_declaration(settings, SimpleNamespace())
    for key, replacement in (
        ("fit_anchor_ids", ["0304", *FIT_ANCHORS[1:]]),
        ("fixed_audit_anchor_ids", ["0001", "0291", "0294", "0687"]),
        ("variant_labels", ["i_minus", "i_plus"]),
    ):
        changed = copy.deepcopy(settings)
        changed["response_addendum"][key] = replacement
        with pytest.raises(ValueError, match=key):
            validate_response_addendum(changed)
    with pytest.raises(ValueError, match="coefficients"):
        settings["response_refinement"]["calibration"] = {"response_coefficient": 0., "null_coefficient": 1.}
        validate_response_refinement_declaration(settings)


def test_historical_selected_anchor_guard_is_unchanged(monkeypatch, tmp_path):
    import channelthermal.training.campaign_response as historical

    stencil = _stencil("0001")
    monkeypatch.setattr(historical, "load_response_atlas_stencil", lambda path: (stencil, {"anchor_id": "0001"}))
    dataset_config = {"development_subset": {"partitions": {"train": {"case_ids": ["0348"]}}}}
    with pytest.raises(ValueError, match="selected training anchors"):
        NativeCampaignResponse(nn.Linear(1, 1), None, dataset_config, {"response_stencils": [str(tmp_path / "family.npz")]})


def test_response_scales_are_equal_family_direction_and_exclude_withheld_labels():
    stencils = [_stencil(anchor, delta=float(index + 1), rows=2 + index * 20) for index, anchor in enumerate(FIT_ANCHORS)]
    scales = fit_response_scales(stencils, _stats())
    expected = np.sqrt(np.mean(np.square([1., 2., 3., 4.])))
    for role in scales["roles"].values():
        assert role["response_rms"] == pytest.approx(expected)
        assert role["scale"] == pytest.approx(expected)
        assert role["floor"] >= role["primary_std_floor"]
        assert role["floor"] >= role["storage_roundoff_indicator"]
    tiny = fit_response_scales([_stencil(anchor, delta=1e-9) for anchor in FIT_ANCHORS], _stats())
    assert tiny["roles"]["solid_temperature"]["scale"] == pytest.approx(.04)
    for bad in (_stencil("0304"), _stencil("0001", split=EvidenceSplit.DEVELOPMENT)):
        with pytest.raises(ValueError, match="fit anchors|exclude"):
            fit_response_scales([bad, *stencils[1:]], _stats())
    geometric = replace(stencils[0], variants={"i_plus": next(iter(stencils[0].variants.values()))})
    with pytest.raises(ValueError, match="non-heat"):
        fit_response_scales([geometric, *stencils[1:]], _stats())


def test_receiver_identity_and_coordinate_joins_are_exact():
    stencil = _stencil("0001")
    trial = stencil.variants[HEAT_VARIANTS[0]]
    roles = dict(trial.output.roles)
    role = roles["solid_temperature"]
    roles["solid_temperature"] = replace(role, query_ids=tuple(reversed(role.query_ids)))
    broken = ResponseStencil(stencil.baseline, {HEAT_VARIANTS[0]: replace(trial, output=replace(trial.output, roles=roles))})
    with pytest.raises(ValueError, match="identities"):
        broken.finite_change(HEAT_VARIANTS[0], "solid_temperature")


def test_both_states_differentiate_and_absolute_means_do_not_grow_with_receivers():
    stencils = [_stencil(anchor) for anchor in FIT_ANCHORS]
    scales = fit_response_scales(stencils, _stats())
    pair = ResponseStencil(stencils[0].baseline, {HEAT_VARIANTS[0]: stencils[0].variants[HEAT_VARIANTS[0]]})
    before = {name: torch.ones(role.values.shape, requires_grad=True) for name, role in pair.baseline.output.roles.items()}
    after = {name: torch.zeros(role.values.shape, requires_grad=True) for name, role in pair.baseline.output.roles.items()}
    losses = response_pair_losses(pair, before, after, _stats(), scales)
    (losses["absolute"] + losses["response"]).backward()
    assert all(value.grad is not None and bool(value.grad.abs().sum() > 0) for value in (*before.values(), *after.values()))
    small = pair.baseline.output.roles["solid_temperature"]
    assert normalized_role_error(torch.ones(2, 1), small.values, small, [1.]).item() == 1.
    large = _stencil("0001", rows=200).baseline.output.roles["solid_temperature"]
    assert normalized_role_error(torch.ones(200, 1), large.values, large, [1.]).item() == 1.


def test_diverse_null_transfers_are_feasible_deterministic_and_support_m1():
    heat, present = np.array([.5, 1., 2., 0.]), [1, 1, 1, 0]
    for seed in range(5):
        direction, receipt = feasible_heat_transfer(heat, present, seed=seed, fraction=.05, lower_heat=.5, upper_heat=2.)
        np.testing.assert_array_equal(direction, feasible_heat_transfer(heat, present, seed=seed, fraction=.05, lower_heat=.5, upper_heat=2.)[0])
        assert direction.sum() == pytest.approx(0.) and direction[3] == 0
        assert np.all((heat + direction)[:3] >= .5) and np.all((heat + direction)[:3] <= 2.)
        assert receipt["fraction"] == .05 and receipt["amplitude"] > 1e-8
    single, receipt = feasible_heat_transfer([1., 0.], [1, 0], seed=1, fraction=.10, lower_heat=.5, upper_heat=2.)
    assert single[0] == pytest.approx(.1) and single[1] == 0 and not receipt["fixed_total"]
    assert feasible_heat_transfer([1.], [1], seed=1, fraction=.05, lower_heat=1., upper_heat=1.) is None


def test_train_only_shared_gradient_calibration_rejects_disconnected_terms():
    model = nn.Linear(2, 1)
    primary = model(torch.tensor([[1., 2.]])).square().sum()
    components = {"response": model(torch.tensor([[2., 3.]])).square().sum(),
                  "null": model(torch.tensor([[3., 5.]])).square().sum()}
    norms = gradient_component_norms(primary, components, model)
    assert all(value > 0 for value in norms.values())
    assert all(parameter.grad is None for parameter in model.parameters())
    rows = [{"arm": arm, "anchor_id": anchor, **norms} for arm in ("separable", "joint") for anchor in FIT_ANCHORS]
    calibration = pooled_gradient_coefficients(rows)
    assert calibration["response_coefficient"] == pytest.approx(.25 * norms["primary"] / norms["response"])
    assert calibration["null_coefficient"] == pytest.approx(.10 * norms["primary"] / norms["null"])
    changed = copy.deepcopy(rows)
    changed[0]["anchor_id"] = "0304"
    with pytest.raises(ValueError, match="withheld"):
        pooled_gradient_coefficients(changed)
    with pytest.raises(ValueError, match="Disconnected"):
        gradient_component_norms(primary, {**components, "null": model.weight.sum() * 0.}, model)


def test_once_per_epoch_auxiliary_accumulation_matches_literal_update():
    torch.manual_seed(41)
    literal = nn.Linear(2, 1)
    accumulated = copy.deepcopy(literal)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.001) for model in (literal, accumulated)]
    x = torch.tensor([[1., 2.], [2., 3.], [3., 4.]])
    auxiliary_inputs = torch.tensor([[4., 5.]])
    callback = NativeResponseRefinement.__new__(NativeResponseRefinement)
    callback.calibration = {"response_coefficient": .2, "null_coefficient": .3}
    callback.settings = {"auxiliary_absolute_weight": .25}

    def components(model):
        value = model(auxiliary_inputs)
        return {"absolute": (value - .1).square().mean(), "response": (value - .2).square().mean(),
                "null": (value - .3).square().mean()}

    values = components(literal)
    (literal(x).square().mean() + .25 * values["absolute"] + .2 * values["response"] + .3 * values["null"]).backward()
    optimizers[0].step()
    callback.components = lambda epoch: (components(accumulated), {})
    for part, weight in ((x[:2], 2 / 3), (x[2:], 1 / 3)):
        primary = accumulated(part).square().mean()
        if weight == 1 / 3:
            aux, _ = callback(1, primary, weight)
            primary = primary + aux
        (primary * weight).backward()
    optimizers[1].step()
    for actual, expected in zip(accumulated.parameters(), literal.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_unsealed_callback_and_recalibration_are_rejected():
    callback = NativeResponseRefinement.__new__(NativeResponseRefinement)
    callback.calibration = None
    with pytest.raises(ValueError, match="Freeze pooled"):
        callback(1, torch.tensor(1.), 1.)
    callback.freeze_calibration({"response_coefficient": 1., "null_coefficient": 2.})
    with pytest.raises(ValueError, match="frozen"):
        callback.freeze_calibration({"response_coefficient": 2., "null_coefficient": 2.})
