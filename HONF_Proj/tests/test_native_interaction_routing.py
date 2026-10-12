"""Focused CPU contracts for the opt-in typed native routing modes."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from channelthermal.flow_curl import NATIVE_CURL_READOUT_LAW
from channelthermal.joint_regional import (
    InteractionPreservingThermalAdapter,
    load_interaction_preserving_thermal_checkpoint,
)
from windfarm.interaction_preserving import (
    NATIVE_ROUTING_WIND_FAMILY,
    InteractionPreservingWindModel,
)
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline

from honf_forward_core.interface_fields.interaction_core import InteractionContextCore
from honf_forward_core.interface_fields.interaction_preserving_joint import (
    InteractionPreservingJointCore,
)
from honf_forward_core.interface_fields.native_interaction_routing import (
    density_normalize,
    reassign_equal_measure_densities,
)


def _core(mode: str, *, seed: int = 73) -> InteractionPreservingJointCore:
    return InteractionPreservingJointCore(
        4, 3, 5, spatial_dim=2, hidden=12, message=8, mode=mode,
        collective_width=6, regional_anchors=0, field_outputs=2,
        affine_outputs=0, initialization_seed=seed,
        native_control_hidden=(5 if mode == "P-C" else None),
    ).double()


def _inputs(*, all_equal_measures: bool = False) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(8301)
    random = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    environment_measures = ([1.0, 1.0] if all_equal_measures else [0.4, 1.6])
    return {
        "sources": random(1, 4, 4),
        "context": random(1, 3),
        "centers": random(1, 4, 2),
        "present": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float64),
        "lengths": torch.tensor([[2.0, 1.5]], dtype=torch.float64),
        "source_lengths": torch.full((1, 4), 0.2, dtype=torch.float64),
        "source_measures": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float64),
        "environment_tokens": random(1, 2, 5),
        "environment_coords": random(1, 2, 2),
        "environment_present": torch.ones(1, 2, dtype=torch.float64),
        "environment_measures": torch.tensor([environment_measures], dtype=torch.float64),
        "source_ids": torch.tensor([[31, 7, 19, -1]], dtype=torch.int64),
        "environment_ids": torch.tensor([[53, 11]], dtype=torch.int64),
        "domain_origin": torch.tensor([[-1.0, -0.5]], dtype=torch.float64),
    }


def _prepare(model, values, **kwargs):
    return model.prepare_context(**values, **kwargs)


def _shared_parameters(model):
    return {name: value for name, value in model.state_dict().items()
            if not name.startswith("native_routing.")}


def _thermal_stats() -> dict[str, np.ndarray]:
    return {
        "field_mean_by_channel": np.zeros(5, dtype=np.float32),
        "field_std_by_channel": np.ones(5, dtype=np.float32),
        "interface_targets_mean": np.zeros(2, dtype=np.float32),
        "interface_targets_std": np.ones(2, dtype=np.float32),
        "internal_temperature_std": np.ones(1, dtype=np.float32),
    }


def _thermal_architecture_child_payload(mode: str) -> tuple[InteractionPreservingThermalAdapter, dict]:
    recipe_path = Path(__file__).resolve().parents[1] / "tools" / "joint_regional_train.py"
    if str(recipe_path.parent) not in sys.path:
        sys.path.insert(0, str(recipe_path.parent))
    module_name = "native_routing_child_receipt_recipe_test"
    recipe_module = sys.modules.get(module_name)
    if recipe_module is None or not hasattr(recipe_module, "_recovery_model_contract"):
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, recipe_path)
        assert spec is not None and spec.loader is not None
        recipe_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = recipe_module
        spec.loader.exec_module(recipe_module)

    def make_model(model_mode: str) -> InteractionPreservingThermalAdapter:
        return InteractionPreservingThermalAdapter(
            mode=model_mode,
            normalization_stats=_thermal_stats(),
            hidden=128,
            message=128,
            regional_anchors=0,
            depth=2,
            receiver_tile=512,
            nx=128,
            ny=64,
            environment_nx=24,
            environment_ny=8,
            forcing_scale=1.0,
            collective_width=64,
            seed=0,
            max_sources=12,
            flow_readout_law=NATIVE_CURL_READOUT_LAW,
        )

    parent = make_model("P")
    child = make_model(mode)
    child_state = child.state_dict()
    child.load_state_dict({**child_state, **parent.state_dict()}, strict=True)
    parent_provider = {
        "task": "ThermalChannel",
        "arm": "P",
        "model_family": parent.family_id,
        "model_config": parent.model_config(),
        "adapter_config": parent.adapter_config(),
        "normalization_stats": copy.deepcopy(parent.normalization_stats),
    }
    recipe = {
        "task": "thermal",
        "mode": "P",
        "seed": 0,
        "regional_anchors": 0,
        "flow_readout_law": NATIVE_CURL_READOUT_LAW,
        "model_contract": recipe_module._recovery_model_contract(
            "thermal", "P", NATIVE_CURL_READOUT_LAW,
        ),
    }
    parent_identity = {
        "recipe": recipe,
        "provider_identity": parent_provider,
        "external_learned_field_files": [],
    }
    child_provider = copy.deepcopy(parent_provider)
    child_provider.update({
        "arm": mode,
        "model_family": child.family_id,
        "model_config": child.model_config(),
    })
    routed_names = sorted(
        name for name in child.state_dict() if name.startswith("core.native_routing.")
    )

    def inventory(names):
        return {
            name: {
                "shape": list(child.state_dict()[name].shape),
                "dtype": str(child.state_dict()[name].dtype),
                "elements": child.state_dict()[name].numel(),
            }
            for name in names
        }

    parent_state_names = sorted(set(child.state_dict()) - set(routed_names))
    source_sha256 = hashlib.sha256(f"test-parent-{mode}".encode()).hexdigest()
    receipt = {
        "schema": "native_interaction_routing_child_v1",
        "source_checkpoint_sha256": source_sha256,
        "source_epoch": 1000,
        "source_arm": "P",
        "target_mode": mode,
        "added_parameter_names": routed_names,
        "review_epochs": [1001, 1002],
        "final_epoch": 1002,
        "scientific_update_limit": 8,
        "reason": "CPU loader contract test; no optimizer steps.",
        "dataset": "ThermalChannel",
        "routing_configuration": copy.deepcopy(child.model_config()["native_routing"]),
        "inherited_parameter_inventory": inventory(parent_state_names),
        "new_parameter_inventory": inventory(routed_names),
        "new_parameter_initial_state_sha256": "1" * 64,
        "inherited_state_sha256": "2" * 64,
        "child_initial_state_sha256": "3" * 64,
        "exact_parent_state_copied": True,
        "all_parameters_trainable": True,
        "optimizer_mapping": "exact inherited named moments; new states initially absent",
        "parent_provider_identity": copy.deepcopy(parent_provider),
    }
    payload = {
        "checkpoint_schema_version": 1,
        "workflow": "joint_regional_fields_v1",
        "arm": mode,
        "experiment_identity": {
            "recipe": copy.deepcopy(recipe),
            "provider_identity": child_provider,
            "external_learned_field_files": [],
        },
        "model_state_dict": copy.deepcopy(child.state_dict()),
        "branch_parent": {
            "path": f"/sealed/parent_{mode}.pt",
            "sha256": source_sha256,
            "identity": copy.deepcopy(parent_identity),
            "epoch": 1000,
            "architecture_child_initialization": receipt,
        },
    }
    return child, payload


@pytest.mark.parametrize("mode", ["P-R", "P-C"])
def test_thermal_loader_accepts_verified_engine_architecture_child(mode):
    child, payload = _thermal_architecture_child_payload(mode)
    restored = load_interaction_preserving_thermal_checkpoint(payload)
    assert restored.mode == mode
    assert restored.model_config() == child.model_config()
    assert all(torch.equal(restored.state_dict()[name], value)
               for name, value in child.state_dict().items())


@pytest.mark.parametrize("mutation", ["missing_receipt", "parent_checksum", "target_mode", "parent_recipe"])
def test_thermal_loader_rejects_forged_or_missing_engine_child_receipt(mutation):
    _child, payload = _thermal_architecture_child_payload("P-R")
    payload = copy.deepcopy(payload)
    receipt = payload["branch_parent"]["architecture_child_initialization"]
    if mutation == "missing_receipt":
        payload["branch_parent"].pop("architecture_child_initialization")
    elif mutation == "parent_checksum":
        receipt["source_checkpoint_sha256"] = "f" * 64
    elif mutation == "target_mode":
        receipt["target_mode"] = "P-C"
    else:
        payload["branch_parent"]["identity"]["recipe"]["mode"] = "P-C"
    with pytest.raises((ValueError, TypeError)):
        load_interaction_preserving_thermal_checkpoint(payload)


def _wind_transforms():
    normalizer = VelocityNormalizer(
        mean=np.asarray([0.3, -0.1, 0.0]),
        std=np.asarray([0.4, 0.3, 0.2]),
        safe_std=np.asarray([0.4, 0.3, 0.2]),
        u_ref_mps=10.0,
        sample_count_per_row=8192,
        source_rows=72,
        seed=42,
    )
    z = np.linspace(0.0, 6.25, 8, dtype=np.float64)
    values = np.column_stack((8.0 + 0.1 * z, 1.0 - 0.05 * z, 0.02 * z))
    profile = VerticalProfileBaseline(z, values, np.full(8, 100, dtype=np.int64))
    return normalizer, profile


def _wind_model(mode: str):
    normalizer, profile = _wind_transforms()
    return InteractionPreservingWindModel(
        velocity_transform=normalizer,
        background_profile=profile,
        mode=mode,
        hidden=128,
        message=128,
        regional_anchors=0,
        depth=2,
        collective_width=64,
        receiver_tile=512,
        seed=42,
        max_sources=30,
    )


def test_density_normalization_is_exact_at_zero_and_preserves_actual_rowmass():
    base = torch.tensor([[[0.0, 0.2, 0.8], [0.0, 0.0, 0.0]]], dtype=torch.float64)
    zero = torch.zeros_like(base)
    rho0, weights0 = density_normalize(base, zero)
    assert torch.equal(rho0, torch.ones_like(base))
    assert torch.equal(weights0, base)

    scores = torch.tensor([[[1.2, -0.4, 0.7], [0.3, -0.2, 0.8]]], dtype=torch.float64,
                          requires_grad=True)
    rho, weights = density_normalize(base, scores)
    torch.testing.assert_close(weights.sum(-1), base.sum(-1), rtol=0, atol=2e-16)
    assert torch.equal(weights[0, 1], torch.zeros_like(weights[0, 1]))
    assert torch.isfinite(rho).all() and torch.isfinite(weights).all()
    assert torch.equal(rho[0, 1], torch.ones_like(rho[0, 1]))
    torch.autograd.grad((weights.square()).sum(), scores)


@pytest.mark.parametrize("dtype,rtol,atol", [
    (torch.float64, 2.0e-14, 2.0e-14),
    (torch.float32, 1.0e-6, 1.0e-7),
])
def test_density_normalization_matches_independent_exp_reference(dtype, rtol, atol):
    base = torch.tensor([[[1.5, 0.7, 0.0, 2.3],
                          [0.0, 0.0, 0.0, 0.0],
                          [0.2, 0.0, 0.3, 0.0]]], dtype=dtype)
    scores = torch.tensor([[[0.8, -0.6, 0.3, 1.1],
                            [-0.2, 0.4, 0.7, -1.0],
                            [0.5, -0.3, 0.9, 0.1]]], dtype=dtype)
    mass = base.sum(-1, keepdim=True)
    unnormalized = base * scores.exp()
    denominator = unnormalized.sum(-1, keepdim=True)
    safe_denominator = torch.where(denominator > 0, denominator,
                                   torch.ones_like(denominator))
    expected = torch.where(denominator > 0,
                           unnormalized * mass / safe_denominator,
                           torch.zeros_like(unnormalized))

    density, weights = density_normalize(base, scores)
    torch.testing.assert_close(weights, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(weights.sum(-1), mass.squeeze(-1), rtol=rtol, atol=atol)
    assert torch.equal(weights[0, 1], torch.zeros_like(weights[0, 1]))
    assert torch.isfinite(density).all() and torch.isfinite(weights).all()


def test_density_reassignment_only_permutes_equal_measure_present_donors():
    density = torch.tensor([[[0.5, 0.75, 0.6, 0.9, 0.25]]], dtype=torch.float64)
    base = torch.tensor([[[0.25, 0.25, 0.25, 0.25, 0.0]]], dtype=torch.float64)
    measures = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    present = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float64)
    ids = torch.tensor([[31, 7, 19, 2, -1]], dtype=torch.int64)
    moved, receipt = reassign_equal_measure_densities(density, base, measures, present, ids)
    assert not torch.equal(moved[..., :4], density[..., :4])
    assert torch.equal(moved[..., 4], density[..., 4])
    assert receipt == {
        "changed_donors": 4,
        "skipped_singleton_buckets": 0,
        "eligible_buckets": 1,
        "eligible_donors": 4,
    }
    torch.testing.assert_close((moved * base).sum(-1), (density * base).sum(-1), rtol=0, atol=0)


def test_native_module_initialization_uses_a_private_cpu_rng_stream():
    torch.manual_seed(191)
    before = torch.random.get_rng_state().clone()
    model = _core("P-R")
    assert torch.equal(torch.random.get_rng_state(), before)
    assert model.native_routing.added_parameter_count() > 0


@pytest.mark.parametrize("mode", ["P-R", "P-C"])
def test_zero_initialized_native_modes_recover_full_core_input_vjps(mode):
    """Zero-start recovery includes every floating native context and query input."""
    for source_count in (1, 4):
        baseline, routed = _core("P"), _core(mode)
        input_sets = []
        for _ in range(2):
            current = {}
            for name, value in _inputs().items():
                if name in ("sources", "centers", "present", "source_lengths",
                            "source_measures", "source_ids"):
                    value = value[:, :source_count]
                value = value.detach().clone()
                if value.is_floating_point():
                    value = value.requires_grad_(True)
                current[name] = value
            input_sets.append(current)
        receivers = [torch.tensor([[[0.1, 0.2], [0.8, -0.4], [1.2, 0.9]]],
                                  dtype=torch.float64, requires_grad=True)
                     for _ in range(2)]
        output_weights = torch.tensor([[[0.5, -0.7], [0.2, 0.4], [-0.3, 0.8]]],
                                      dtype=torch.float64)

        def evaluate(model, current, query, weights=output_weights):
            prepared = _prepare(model, current)
            fields = model.predict_fields(prepared, query)
            loss = (fields * weights).sum()
            differentiable_inputs = tuple(value for value in current.values()
                                          if value.is_floating_point()) + (query,)
            gradients = torch.autograd.grad(loss, differentiable_inputs, allow_unused=True)
            return fields, gradients

        baseline_fields, baseline_gradients = evaluate(baseline, input_sets[0], receivers[0])
        routed_fields, routed_gradients = evaluate(routed, input_sets[1], receivers[1])
        torch.testing.assert_close(routed_fields, baseline_fields, rtol=0, atol=0)
        for actual, expected in zip(routed_gradients, baseline_gradients, strict=True):
            assert (actual is None) == (expected is None)
            if actual is not None:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["P-R", "P-C"])
def test_native_routing_one_source_has_finite_empty_mm_self_exclusion(mode):
    values = _inputs()
    for name in ("sources", "centers", "present", "source_lengths", "source_measures", "source_ids"):
        values[name] = values[name][:, :1]
    model = _core(mode)
    prepared = _prepare(model, values, routing_trace=True)
    mm_rows = [item for item in prepared.native_routing_trace if item["legacy_route"] == "MM"]
    assert len(mm_rows) == 2
    for item in mm_rows:
        assert item["self_exclusion_mask"].shape == (1, 1, 1)
        assert not bool(item["self_exclusion_mask"].any())
        assert torch.equal(item["w0"], torch.zeros_like(item["w0"]))
        assert torch.equal(item["w"], torch.zeros_like(item["w"]))
        assert torch.isfinite(item["rho"]).all()
        assert torch.isfinite(item["aggregate"]).all()
        assert torch.isfinite(item["weighted_message_norm"]).all()


@pytest.mark.parametrize("mode", ["P-R", "P-C"])
def test_zero_initialized_native_modes_preserve_direct_outputs_and_common_weights(mode):
    values = _inputs()
    direct, routed = _core("P"), _core(mode)
    for name, parameter in direct.named_parameters():
        torch.testing.assert_close(parameter, dict(routed.named_parameters())[name], rtol=0, atol=0)

    direct_context = _prepare(direct, values)
    routed_context = _prepare(routed, values)
    for name in ("source_states", "environment_states", "global_state"):
        torch.testing.assert_close(getattr(routed_context, name), getattr(direct_context, name),
                                   rtol=0, atol=0)
    receivers = torch.tensor([[[0.1, 0.2], [0.8, -0.4], [1.2, 0.9]]], dtype=torch.float64)
    direct_fields = direct.predict_fields(direct_context, receivers)
    routed_fields = routed.predict_fields(routed_context, receivers)
    torch.testing.assert_close(routed_fields, direct_fields, rtol=0, atol=0)

    output_weights = torch.tensor([[[0.5, -0.7], [0.2, 0.4], [-0.3, 0.8]]], dtype=torch.float64)
    (routed_fields * output_weights).sum().backward()
    modules = (routed.native_routing.score_mlps if mode == "P-R"
               else routed.native_routing.aggregate_modulators)
    for route in ("MM", "ME", "EM"):
        final = modules[route][-1]
        assert final.bias is None if mode == "P-R" else final.bias is not None
        assert final.weight.grad is not None and torch.isfinite(final.weight.grad).all()
        assert final.weight.grad.abs().sum() > 0


def test_p_r_trace_keeps_literal_base_rows_detached_and_interventions_rebuild_restore():
    model = _core("P-R")
    with torch.no_grad():
        for index, route in enumerate(("MM", "ME", "EM")):
            model.native_routing.score_mlps[route][-1].weight.copy_(
                torch.linspace(-0.08, 0.09, 32, dtype=torch.float64)[None] * (index + 1))
    values = _inputs(all_equal_measures=True)
    for value in values.values():
        if value.is_floating_point():
            value.requires_grad_(True)
    normal = _prepare(model, values, routing_trace=True)
    assert len(normal.native_routing_trace) == 6
    mm_first = next(item for item in normal.native_routing_trace
                    if item["round"] == 1 and item["legacy_route"] == "MM")
    assert not bool(mm_first["destination_present_mask"][0, -1])
    # Legacy w0 is preserved on padded destination rows; destination presence
    # is recorded independently so dense padded work remains visible.
    torch.testing.assert_close(mm_first["w0"][0, -1].sum(), torch.tensor(1.0, dtype=torch.float64))
    assert "donor_active_mask" in mm_first and "self_exclusion_mask" in mm_first
    assert mm_first["destination_ids"].tolist() == [[31, 7, 19, -1]]
    assert all(not value.requires_grad for item in normal.native_routing_trace
               for value in item.values() if torch.is_tensor(value))

    reset = model.intervene_context(
        normal, native_route_intervention={"kind": "density_reset", "route_types": ["MM"]}
    )
    for item in reset.native_routing_trace:
        if item["legacy_route"] == "MM":
            assert torch.equal(item["rho"], torch.ones_like(item["rho"]))
            assert item["intervention_receipt"]["changed_donors"] > 0
    reassigned = model.intervene_context(
        normal,
        native_route_intervention={"kind": "equal_measure_reassignment", "route_types": ["MM"]},
    )
    mm_receipts = [item["intervention_receipt"] for item in reassigned.native_routing_trace
                   if item["legacy_route"] == "MM"]
    assert sum(receipt["changed_donors"] for receipt in mm_receipts) > 0
    for item in reassigned.native_routing_trace:
        if item["legacy_route"] == "MM":
            torch.testing.assert_close(item["w"].sum(-1), item["w0"].sum(-1), rtol=0, atol=2e-16)

    restored = model.restore(reassigned)
    assert restored.native_route_intervention == (None, ())
    assert restored.native_routing_trace is None
    receivers = torch.tensor([[[0.1, 0.2], [0.8, -0.4]]], dtype=torch.float64)
    torch.testing.assert_close(model.predict_fields(restored, receivers),
                               model.predict_fields(normal, receivers), rtol=0, atol=0)


def test_native_routing_rebuild_is_source_environment_permutation_equivariant():
    model = _core("P-R")
    with torch.no_grad():
        for index, route in enumerate(("MM", "ME", "EM")):
            model.native_routing.score_mlps[route][-1].weight.copy_(
                torch.linspace(-0.07, 0.08, 32, dtype=torch.float64)[None] * (index + 1))
    values = _inputs()
    receivers = torch.tensor([[[0.1, 0.2], [0.8, -0.4]]], dtype=torch.float64)
    direct = _prepare(model, values)
    direct_fields = model.predict_fields(direct, receivers)

    source_order = torch.tensor([2, 0, 3, 1])
    environment_order = torch.tensor([1, 0])
    permuted_values = dict(values)
    for name in ("sources", "centers", "present", "source_lengths", "source_measures", "source_ids"):
        permuted_values[name] = values[name].index_select(1, source_order)
    for name in ("environment_tokens", "environment_coords", "environment_present",
                 "environment_measures", "environment_ids"):
        permuted_values[name] = values[name].index_select(1, environment_order)
    permuted = _prepare(model, permuted_values)
    torch.testing.assert_close(model.predict_fields(permuted, receivers), direct_fields,
                               rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(permuted.source_states,
                               direct.source_states.index_select(1, source_order),
                               rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(permuted.environment_states,
                               direct.environment_states.index_select(1, environment_order),
                               rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(permuted.global_state, direct.global_state,
                               rtol=1.0e-12, atol=1.0e-12)


def test_native_routing_environment_quadrature_duplicate_with_split_measure():
    model = _core("P-R")
    with torch.no_grad():
        for index, route in enumerate(("MM", "ME", "EM")):
            model.native_routing.score_mlps[route][-1].weight.copy_(
                torch.linspace(-0.06, 0.09, 32, dtype=torch.float64)[None] * (index + 1))
    values = _inputs()
    refined = dict(values)
    refined["environment_tokens"] = torch.cat(
        (values["environment_tokens"][:, :1], values["environment_tokens"][:, :1],
         values["environment_tokens"][:, 1:]), dim=1)
    refined["environment_coords"] = torch.cat(
        (values["environment_coords"][:, :1], values["environment_coords"][:, :1],
         values["environment_coords"][:, 1:]), dim=1)
    refined["environment_present"] = torch.cat(
        (values["environment_present"][:, :1], values["environment_present"][:, :1],
         values["environment_present"][:, 1:]), dim=1)
    refined["environment_measures"] = torch.tensor([[0.2, 0.2, 1.6]], dtype=torch.float64)
    refined["environment_ids"] = torch.tensor([[53, 54, 11]], dtype=torch.int64)
    receivers = torch.tensor([[[0.1, 0.2], [0.8, -0.4]]], dtype=torch.float64)
    original_context = _prepare(model, values)
    refined_context = _prepare(model, refined)
    torch.testing.assert_close(model.predict_fields(refined_context, receivers),
                               model.predict_fields(original_context, receivers),
                               rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(refined_context.source_states, original_context.source_states,
                               rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(refined_context.global_state, original_context.global_state,
                               rtol=1.0e-12, atol=1.0e-12)


def test_p_r_score_order_can_reverse_when_only_destination_changes():
    model = _core("P-R")
    scorer = model.native_routing.score_mlps["MM"]
    with torch.no_grad():
        scorer[0].weight.zero_()
        scorer[0].bias.zero_()
        scorer[-1].weight.zero_()
        scorer[0].weight[0, 0] = 1.0
        scorer[0].weight[1, 0] = 1.0
        scorer[0].weight[0, model.hidden] = 1.0
        scorer[0].weight[1, model.hidden] = -1.0
        scorer[-1].weight[0, 0] = 1.0
        scorer[-1].weight[0, 1] = -1.0
    pair_features = torch.zeros((1, 2, 2, 2 * model.hidden + model.spatial_dim * 9),
                                dtype=torch.float64)
    pair_features[0, 0, :, 0] = -2.0
    pair_features[0, 1, :, 0] = 2.0
    pair_features[0, :, 0, model.hidden] = -1.0
    pair_features[0, :, 1, model.hidden] = 1.0
    scores = model.native_routing.score("MM", pair_features)
    base = torch.ones_like(scores)
    _, weights = density_normalize(base, scores)
    assert scores[0, 0, 0] > scores[0, 0, 1]
    assert scores[0, 1, 0] < scores[0, 1, 1]
    assert weights[0, 0, 0] > weights[0, 0, 1]
    assert weights[0, 1, 0] < weights[0, 1, 1]


def test_routing_generation_guard_rejects_toggle_back_and_non_pr_intervention():
    values = _inputs()
    model = _core("P-R")
    prepared = _prepare(model, values)
    model.set_native_routing_enabled(False)
    model.set_native_routing_enabled(True)
    with pytest.raises(ValueError, match="routing policy changed"):
        model.predict_fields(prepared, torch.zeros(1, 1, 2, dtype=torch.float64))

    changed_config_model = _core("P-R")
    changed_config_context = _prepare(changed_config_model, values)
    changed_config_model.config["native_routing"]["policy"] = "destination_gain"
    with pytest.raises(ValueError, match="native routing config changed"):
        changed_config_model.predict_fields(
            changed_config_context, torch.zeros(1, 1, 2, dtype=torch.float64)
        )

    direct_toggle_model = _core("P-R")
    direct_toggle_context = _prepare(direct_toggle_model, values)
    direct_toggle_model.native_routing_enabled = False
    direct_toggle_model.native_routing_enabled = True
    with pytest.raises(ValueError, match="routing policy changed"):
        direct_toggle_model.predict_fields(
            direct_toggle_context, torch.zeros(1, 1, 2, dtype=torch.float64)
        )

    for mode in ("P", "P-C"):
        direct = _core(mode)
        with pytest.raises(ValueError, match="P-R"):
            _prepare(direct, values, native_route_intervention={"kind": "density_reset"})


@pytest.mark.parametrize("mode", ["P-R", "P-C"])
@pytest.mark.parametrize("mutation", ["owner", "weights", "dtype", "input"])
def test_prepared_native_routing_context_rejects_stale_owner_dtype_weights_and_inputs(mode, mutation):
    model = _core(mode)
    values = _inputs()
    prepared = _prepare(model, values)
    receivers = torch.zeros((1, 1, 2), dtype=torch.float64)
    owner = model
    if mutation == "owner":
        owner = _core(mode)
    elif mutation == "weights":
        with torch.no_grad():
            next(model.parameters()).add_(0.01)
    elif mutation == "dtype":
        model.float()
    else:
        with torch.no_grad():
            values["centers"][0, 0, 0].add_(0.01)
    with pytest.raises(ValueError, match="Prepared interaction|Prepared native|Prepared response"):
        owner.predict_fields(prepared, receivers)


def test_default_measure_core_keeps_legacy_float_id_admission():
    values = _inputs()
    values["source_ids"] = values["source_ids"].to(torch.float64)
    values["environment_ids"] = values["environment_ids"].to(torch.float64)
    values.pop("domain_origin")
    model = InteractionContextCore(4, 3, 5, spatial_dim=2, hidden=12, message=8).double()
    _prepare(model, values)
    with pytest.raises(TypeError, match="integer tensor dtype"):
        _prepare(model, values, routing_trace=True)


def test_dataset_native_modes_seal_families_widths_and_round_trip_configs():
    thermal = {}
    for mode in ("P-R", "P-C"):
        model = InteractionPreservingThermalAdapter(
            mode=mode, normalization_stats=_thermal_stats(), hidden=128, message=128,
            regional_anchors=0, depth=2, receiver_tile=512, nx=128, ny=64,
            environment_nx=24, environment_ny=8, forcing_scale=1.0,
            collective_width=64, seed=0, max_sources=12,
        )
        config = model.model_config()
        thermal[mode] = model.core.native_routing.added_parameter_count()
        assert config["family"] == "thermal_native_interaction_routing_v1"
        assert config["native_routing"]["mode"] == mode
        assert config["native_routing"]["initialization_seed"] == 0
        assert all(name.startswith("core.native_routing.")
                   for name, _ in model.named_parameters() if "native_routing" in name)
        loaded = load_interaction_preserving_thermal_checkpoint(model.checkpoint_payload())
        assert loaded.model_config() == config
        loaded.load_state_dict(model.state_dict(), strict=True)
    assert abs(thermal["P-R"] - thermal["P-C"]) / thermal["P-R"] < 0.05
    assert thermal["P-R"] == 26496 and thermal["P-C"] == 26400

    wind = {}
    for mode in ("P-R", "P-C"):
        model = _wind_model(mode)
        config = model.export_config()
        wind[mode] = model.core.native_routing.added_parameter_count()
        assert config["family"] == NATIVE_ROUTING_WIND_FAMILY
        assert config["core"]["native_routing"]["mode"] == mode
        restored = InteractionPreservingWindModel.from_config(config)
        restored.load_state_dict(model.state_dict(), strict=True)
        assert restored.export_config() == config
    assert abs(wind["P-R"] - wind["P-C"]) / wind["P-R"] < 0.05
    assert wind["P-R"] == 27360 and wind["P-C"] == 27720


def test_wind_loader_rejects_native_policy_or_family_drift():
    config = _wind_model("P-C").export_config()
    malformed = dict(config)
    malformed["core"] = dict(config["core"])
    malformed["core"]["native_routing"] = dict(config["core"]["native_routing"])
    malformed["core"]["native_routing"]["control_hidden"] = 35
    with pytest.raises(ValueError, match="sealed width"):
        InteractionPreservingWindModel.from_config(malformed)
    malformed = dict(config)
    malformed["family"] = "wind_interaction_preserving_joint_v1"
    with pytest.raises(ValueError, match="family ID"):
        InteractionPreservingWindModel.from_config(malformed)
