"""Focused CPU qualification for the opt-in P-family native adapters."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from channelthermal.joint_regional import (
    INTERACTION_PRESERVING_THERMAL_MODES,
    JOINT_THERMAL_CHANNEL_ORDER,
    InteractionPreservingThermalAdapter,
    load_interaction_preserving_thermal_checkpoint,
)
from channelthermal.training.joint_task import JointThermalTask, _case_ids_hash
from torch import nn
from windfarm.geometry import POSITIONAL_SCALE_D
from windfarm.interaction_preserving import InteractionPreservingWindModel
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline
from windfarm.training.joint_task import WIND_JOINT_DEPENDENCY

from honf_forward_core.interface_fields.interaction_core import InteractionScene
from honf_runtime import unified_training as training_runtime
from honf_runtime.unified_training import (
    EngineConfig,
    LossTerm,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    TaskBatch,
    TrainingEngine,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _thermal_stats() -> dict[str, np.ndarray]:
    return {
        "field_mean_by_channel": np.asarray([1.0, -2.0, 0.5, 3.0, 10.0], dtype=np.float32),
        "field_std_by_channel": np.asarray([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "interface_targets_mean": np.asarray([10.0, 0.0], dtype=np.float32),
        "interface_targets_std": np.asarray([4.0, 5.0], dtype=np.float32),
        "internal_temperature_std": np.asarray([7.0], dtype=np.float32),
    }


def _thermal_model(mode: str = "P") -> InteractionPreservingThermalAdapter:
    torch.manual_seed(17)
    graph = mode in ("P-G", "P-H")
    return InteractionPreservingThermalAdapter(
        mode=mode,
        normalization_stats=_thermal_stats(),
        hidden=128,
        message=128,
        regional_anchors=16 if graph else 0,
        depth=2,
        receiver_tile=512,
        nx=128,
        ny=64,
        environment_nx=24,
        environment_ny=8,
        forcing_scale=1.0,
        collective_width=64,
        locality_prior_strength=1.0 if graph else None,
        seed=0,
        max_sources=12,
    )


def _thermal_inputs():
    structure = {
        "module_centers": torch.tensor([[[2.0, 2.0], [7.5, 4.0]]], dtype=torch.float32),
        "module_present": torch.ones(1, 2, dtype=torch.float32),
        "module_source_ids": torch.tensor([[31, 47]], dtype=torch.int64),
        "material_params": torch.tensor([[0.02, 0.01, 0.02, 1.0, 2.0, 0.22]], dtype=torch.float32),
        "re": torch.tensor([[50.0]], dtype=torch.float32),
        "u_in": torch.ones(1, 1, dtype=torch.float32),
        "domain_length_x": torch.tensor([[12.0]], dtype=torch.float32),
        "domain_length_y": torch.tensor([[6.0]], dtype=torch.float32),
    }
    receivers = torch.tensor([[[0.75, 0.75], [3.75, 2.25], [8.25, 3.75]]], dtype=torch.float32)
    local = torch.tensor(
        [[[[0.0, 0.0], [0.35, 0.0]], [[0.0, 0.0], [0.35, 0.0]]]], dtype=torch.float32
    )
    return structure, receivers, local


def _thermal_engine_payload(model: InteractionPreservingThermalAdapter) -> dict:
    cli = _joint_recipe_module()
    identity = {
        "recipe": {
            "task": "thermal",
            "mode": "P",
            "seed": 0,
            "regional_anchors": 0,
            "model_contract": cli._recovery_model_contract("thermal", "P"),
        },
        "provider_identity": {
            "task": "ThermalChannel",
            "model_family": model.FAMILY,
            "model_config": model.model_config(),
            "adapter_config": model.adapter_config(),
            "normalization_stats": copy.deepcopy(model.normalization_stats),
        },
        "external_learned_field_files": [],
    }
    return {
        "checkpoint_schema_version": 1,
        "workflow": "joint_regional_fields_v1",
        "arm": "P",
        "experiment_identity": identity,
        "model_state_dict": copy.deepcopy(model.state_dict()),
    }


def _joint_recipe_module():
    path = REPO_ROOT / "tools" / "joint_regional_train.py"
    name = "interaction_preserving_runtime_recipe_test"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["P", "P-R", "P-C"])
def test_thermal_native_heads_keep_flow_separate_and_reach_both_read_paths(mode):
    model = _thermal_model(mode).eval()
    structure, receivers, local = _thermal_inputs()
    prepared = model.prepare_native(structure, receivers, local_query_points=local, ntheta=4)
    heat_a = torch.tensor([[0.5, 0.9]], dtype=torch.float64)
    heat_b = torch.tensor([[1.1, 0.2]], dtype=torch.float64)
    delta = torch.tensor([[0.15, -0.2]], dtype=torch.float64)
    output_a = model.apply_native(prepared, heat_a, accumulation_dtype=torch.float64)
    output_b = model.apply_native(prepared, heat_b, accumulation_dtype=torch.float64)
    output_increment = model.apply_native(
        prepared, delta, increment=True, accumulation_dtype=torch.float64
    )
    output_shifted = model.apply_native(prepared, heat_a + delta, accumulation_dtype=torch.float64)

    assert output_a["pred_field"].shape == (1, 3, 5)
    assert torch.equal(output_a["pred_field"][..., :4], output_b["pred_field"][..., :4])
    assert torch.equal(output_a["pred_field"][..., :4], output_shifted["pred_field"][..., :4])
    for name in ("fluid_temperature", "pred_interface", "pred_internal_temperature"):
        torch.testing.assert_close(
            output_shifted[name] - output_a[name], output_increment[name], rtol=1e-8, atol=1e-11
        )
    normalized_flow = model.core.predict_fields(prepared.context, receivers)
    expected_flow = normalized_flow * normalized_flow.new_tensor(
        _thermal_stats()["field_std_by_channel"][:4]
    ) + normalized_flow.new_tensor(_thermal_stats()["field_mean_by_channel"][:4])
    assert torch.equal(output_a["pred_field"][..., :4], expected_flow.to(output_a["pred_field"].dtype))
    assert not torch.allclose(output_a["fluid_temperature"], output_b["fluid_temperature"])

    source_order = torch.tensor([1, 0])
    permuted_structure = dict(structure)
    for name in ("module_centers", "module_present", "module_source_ids"):
        permuted_structure[name] = structure[name].index_select(1, source_order)
    permuted_local = local.index_select(1, source_order)
    permuted = model.prepare_native(
        permuted_structure, receivers, local_query_points=permuted_local, ntheta=4
    )
    reordered_output = model.apply_native(
        permuted, heat_a.index_select(1, source_order), accumulation_dtype=torch.float64
    )
    torch.testing.assert_close(reordered_output["pred_field"], output_a["pred_field"],
                               rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(reordered_output["pred_interface"],
                               output_a["pred_interface"].index_select(1, source_order),
                               rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(reordered_output["pred_internal_temperature"],
                               output_a["pred_internal_temperature"].index_select(1, source_order),
                               rtol=2.0e-5, atol=2.0e-6)

    flow_loss = output_a["pred_field"][..., :4].square().mean()
    affine_loss = output_a["fluid_temperature"].square().mean()
    parameters = dict(model.named_parameters())
    groups = {
        "flow_head": [name for name in parameters if name.startswith("core.field_head.")],
        "flow_read": [name for name in parameters if name.startswith("core.source_read.")],
        "affine_head": [name for name in parameters if name.startswith("core.affine_head.")],
        "affine_read": [name for name in parameters if name.startswith("core.affine_source_read.")],
        "shared_pair": [name for name in parameters if name.startswith("core.module_messages.0.")],
    }
    assert all(groups.values())
    for loss, required in (
        (flow_loss, ("flow_head", "flow_read", "shared_pair")),
        (affine_loss, ("affine_head", "affine_read", "shared_pair")),
    ):
        for group in required:
            names = groups[group]
            gradients = torch.autograd.grad(
                loss, tuple(parameters[name] for name in names), retain_graph=True
            )
            assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
            assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


@pytest.mark.parametrize("mode", ["P-R", "P-C"])
def test_thermal_native_routing_zero_start_matches_p_outputs_and_physical_input_vjps(mode):
    baseline, routed = _thermal_model("P").eval(), _thermal_model(mode).eval()

    def evaluate(model):
        structure, receivers, local = _thermal_inputs()
        differentiable = {}
        for name, value in structure.items():
            value = value.detach().clone()
            if value.is_floating_point():
                value.requires_grad_(True)
            differentiable[name] = value
        receivers = receivers.detach().clone().requires_grad_(True)
        local = local.detach().clone().requires_grad_(True)
        heat = torch.tensor([[0.5, 0.9]], dtype=torch.float64, requires_grad=True)
        prepared = model.prepare_native(
            differentiable, receivers, local_query_points=local, ntheta=4
        )
        output = model.apply_native(prepared, heat, accumulation_dtype=torch.float64)
        loss = (output["pred_field"].square().sum()
                + output["pred_interface"].square().sum()
                + output["pred_internal_temperature"].square().sum())
        physical_inputs = tuple(value for value in differentiable.values()
                                if value.is_floating_point()) + (receivers, local, heat)
        gradients = torch.autograd.grad(loss, physical_inputs, allow_unused=True)
        return output, gradients

    baseline_output, baseline_gradients = evaluate(baseline)
    routed_output, routed_gradients = evaluate(routed)
    assert set(routed_output) == set(baseline_output)
    for name in routed_output:
        if torch.is_tensor(routed_output[name]):
            torch.testing.assert_close(routed_output[name], baseline_output[name],
                                       rtol=2.0e-5, atol=2.0e-6)
    for actual, expected in zip(routed_gradients, baseline_gradients, strict=True):
        assert (actual is None) == (expected is None)
        if actual is not None:
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=3.0e-5, atol=3.0e-6)


def test_thermal_p_checkpoint_roundtrips_standalone_and_engine_envelopes():
    model = _thermal_model().eval()
    structure, receivers, local = _thermal_inputs()
    with torch.no_grad():
        reference = model.apply_native(
            model.prepare_native(structure, receivers, local_query_points=local, ntheta=4),
            torch.tensor([[0.5, 0.9]]),
        )["pred_field"]

    standalone = model.checkpoint_payload()
    assert standalone["channel_order"] == list(JOINT_THERMAL_CHANNEL_ORDER)
    restored = load_interaction_preserving_thermal_checkpoint(standalone).eval()
    engine_restored = load_interaction_preserving_thermal_checkpoint(_thermal_engine_payload(model)).eval()
    for candidate in (restored, engine_restored):
        with torch.no_grad():
            result = candidate.apply_native(
                candidate.prepare_native(structure, receivers, local_query_points=local, ntheta=4),
                torch.tensor([[0.5, 0.9]]),
            )["pred_field"]
        torch.testing.assert_close(result, reference, rtol=0.0, atol=0.0)
        assert all(name.startswith("core.") for name in candidate.state_dict())


@pytest.mark.parametrize("mutation", ["frame", "grid", "head", "seed", "parent"])
def test_thermal_loader_rejects_changed_standalone_contract(mutation):
    model = _thermal_model()
    payload = copy.deepcopy(model.checkpoint_payload())
    if mutation == "frame":
        payload["channel_order"] = list(reversed(JOINT_THERMAL_CHANNEL_ORDER))
    elif mutation == "grid":
        payload["model_config"]["nx"] = 64
    elif mutation == "head":
        payload["model_config"]["field_outputs"] = 3
    elif mutation == "seed":
        payload["model_config"]["seed"] = 1
    else:
        payload["parent_checkpoint"] = "external-flow.pt"
    with pytest.raises((ValueError, TypeError)):
        load_interaction_preserving_thermal_checkpoint(payload)


@pytest.mark.parametrize("mutation", ["coordinate_frame", "shared_grid", "output_law", "seed", "parent"])
def test_thermal_loader_rejects_changed_engine_contract(mutation):
    model = _thermal_model()
    payload = _thermal_engine_payload(model)
    if mutation == "coordinate_frame":
        payload["experiment_identity"]["recipe"]["model_contract"]["input_frame"] = "shifted coordinates"
    elif mutation == "shared_grid":
        payload["experiment_identity"]["recipe"]["model_contract"]["native_shared_grid_shape"] = [64, 128]
    elif mutation == "output_law":
        payload["experiment_identity"]["recipe"]["model_contract"]["readouts"]["temperature_head"]["law"] = "nonlinear"
    elif mutation == "seed":
        payload["experiment_identity"]["recipe"]["seed"] = 42
    else:
        payload["experiment_identity"]["external_learned_field_files"] = ["legacy-flow.pt"]
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


def _wind_model(*, mode="P-G", factorized_affine_execution=False):
    normalizer, profile = _wind_transforms()
    torch.manual_seed(42)
    direct = mode in ("P", "P-R", "P-C")
    return InteractionPreservingWindModel(
        velocity_transform=normalizer,
        background_profile=profile,
        mode=mode,
        hidden=128,
        message=128,
        regional_anchors=0 if direct else 32,
        depth=2,
        collective_width=64,
        receiver_tile=512,
        seed=42,
        max_sources=30,
        locality_prior_strength=None if direct else 1.0,
        factorized_affine_execution=factorized_affine_execution,
    )


def _wind_scene() -> InteractionScene:
    dtype = torch.float32
    scale = torch.tensor(POSITIONAL_SCALE_D, dtype=dtype)
    lower = torch.tensor([[-3.0, -2.0, 0.0]], dtype=dtype)
    extent = torch.tensor([[10.0, 8.0, 6.25]], dtype=dtype)
    upper = lower + extent
    centers = torch.tensor([[[-1.0, 0.0, 0.875], [3.0, 2.0, 0.875]]], dtype=dtype)
    sources = torch.tensor([[[0.5, 0.875], [0.5, 0.875]]], dtype=dtype)
    present = torch.ones(1, 2, dtype=dtype)
    source_ids = torch.tensor([[4, 11]], dtype=torch.int64)
    context = torch.zeros(1, 11, dtype=dtype)
    context[0, 0] = 1.0
    context[0, 3] = 2.0 / 30.0
    context[0, 4] = 1.0
    context[0, 5:8] = lower[0] / scale
    context[0, 8:11] = extent[0] / scale

    axes = [torch.linspace(0.5, 1.0, 2) for _ in range(3)]
    unit = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, 8, 3)
    environment_coords = lower[:, None] + unit * extent[:, None]
    low_features = (environment_coords - lower[:, None]) / scale
    high_features = (upper[:, None] - environment_coords) / scale
    absolute_height = environment_coords[..., 2:3] / scale[2]
    environment_tokens = torch.cat((low_features, high_features, absolute_height), dim=-1)
    environment_measure = torch.full((1, 8), float(torch.prod(extent).item()) / 8.0, dtype=dtype)
    return InteractionScene(
        sources=sources,
        context=context,
        centers=centers,
        present=present,
        lengths=extent,
        source_lengths=2.0 * sources[..., 0],
        dependency=WIND_JOINT_DEPENDENCY,
        source_measures=present.clone(),
        environment_tokens=environment_tokens,
        environment_coords=environment_coords,
        environment_present=torch.ones_like(environment_measure),
        environment_measures=environment_measure,
        source_ids=source_ids,
    )


@pytest.mark.parametrize("mode", ["P-R", "P-C"])
def test_wind_native_routing_zero_start_matches_p_outputs_and_physical_input_vjps(mode):
    baseline, routed = _wind_model(mode="P").eval(), _wind_model(mode=mode).eval()
    receivers = torch.tensor([[[-2.5, -1.0, 1.0], [0.0, 1.0, 2.5],
                               [3.0, 3.0, 4.5]]], dtype=torch.float32)

    def evaluate(model):
        source_scene = _wind_scene()
        scene_inputs = []
        scene_kwargs = {}
        for name in source_scene.__dataclass_fields__:
            value = getattr(source_scene, name)
            if torch.is_tensor(value):
                value = value.detach().clone()
                if value.is_floating_point():
                    value.requires_grad_(True)
                    scene_inputs.append(value)
            scene_kwargs[name] = value
        scene = InteractionScene(**scene_kwargs)
        query = receivers.detach().clone().requires_grad_(True)
        prepared = model.prepare_scene(scene)
        output = model.predict_physical(prepared, query)
        gradients = torch.autograd.grad(output.square().sum(), (*scene_inputs, query),
                                        allow_unused=True)
        return output, gradients

    baseline_output, baseline_gradients = evaluate(baseline)
    routed_output, routed_gradients = evaluate(routed)
    torch.testing.assert_close(routed_output, baseline_output, rtol=2.0e-5, atol=2.0e-6)
    for actual, expected in zip(routed_gradients, baseline_gradients, strict=True):
        assert (actual is None) == (expected is None)
        if actual is not None:
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=3.0e-5, atol=3.0e-6)


@pytest.mark.parametrize("mode", ["P-R", "P-C"])
@pytest.mark.parametrize("dataset", ["thermal", "wind"])
def test_native_prepared_scene_roundtrips_source_and_environment_ids(mode, dataset):
    if dataset == "thermal":
        model = _thermal_model(mode).eval()
        structure, receivers, local = _thermal_inputs()
        structure["module_source_ids"] = ["module-west", "module-east"]
        prepared = model.prepare_native(structure, receivers, local_query_points=local, ntheta=4)
        assert prepared.source_id_catalogue == ("module-west", "module-east")
        assert prepared.context.source_id_catalogue == prepared.source_id_catalogue
        assert torch.equal(prepared.context.source_ids, torch.tensor([[0, 1]], dtype=torch.int64))
        tensors = model.context_tensors(structure)
    else:
        model = _wind_model(mode=mode).eval()
        scene = _wind_scene()
        prepared = model.prepare_scene(scene)
        assert torch.equal(prepared.scene.source_ids, torch.tensor([[4, 11]], dtype=torch.int64))
        assert torch.equal(prepared.core_context.source_ids, prepared.scene.source_ids)
        tensors = scene.tensors()
        tensors["domain_origin"] = scene.context[:, 5:8] * scene.centers.new_tensor(POSITIONAL_SCALE_D)

    traced = model.core.prepare_context(**tensors, routing_trace=True)
    expected_sources = (prepared.context.source_ids if dataset == "thermal"
                        else prepared.core_context.source_ids)
    expected_environments = torch.arange(
        tensors["environment_tokens"].shape[1], dtype=torch.int64,
        device=tensors["environment_tokens"].device,
    )[None]
    for item in traced.native_routing_trace:
        assert torch.equal(item["destination_ids"],
                           expected_sources if item["destination_type"] == "physical_source"
                           else expected_environments)
        assert torch.equal(item["donor_ids"],
                           expected_sources if item["donor_type"] == "physical_source"
                           else expected_environments)


def test_wind_config_roundtrip_preserves_physical_frame_and_scaling():
    model = _wind_model().eval()
    scene = _wind_scene()
    receivers = torch.tensor([[[-2.5, -1.0, 1.0], [0.0, 1.0, 2.5], [3.0, 3.0, 4.5]]])
    prepared = model.prepare_scene(scene)
    residual = model.predict_standardized(prepared, receivers)
    expected = model.profile_at_receivers(receivers) + residual * torch.tensor(
        model.velocity_transform.safe_std, dtype=receivers.dtype
    ) * model.velocity_transform.u_ref_mps
    actual = model.predict_physical(prepared, receivers)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    config = model.export_config()
    restored = InteractionPreservingWindModel.from_config(config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(
        restored.predict_physical(restored.prepare_scene(scene), receivers), actual, rtol=0.0, atol=0.0
    )
    assert restored.export_config() == config
    factored = _wind_model(factorized_affine_execution=True).eval()
    factored.load_state_dict(model.state_dict(), strict=True)
    factored_config = factored.export_config()
    assert factored_config["core"]["factorized_affine_execution"] is True
    factored_restored = InteractionPreservingWindModel.from_config(factored_config).eval()
    factored_restored.load_state_dict(factored.state_dict(), strict=True)
    torch.testing.assert_close(
        factored_restored.predict_physical(factored_restored.prepare_scene(scene), receivers),
        factored.predict_physical(factored.prepare_scene(scene), receivers), rtol=0.0, atol=0.0
    )
    with pytest.raises(ValueError, match="frame|feature inventory|output law"):
        broken = copy.deepcopy(config)
        broken["native_coordinate_frame"] = "shifted-origin"
        InteractionPreservingWindModel.from_config(broken)
    with pytest.raises(ValueError, match="nonlinear"):
        restored.apply_increment(torch.ones(1, 2))
    restored_prepared = restored.prepare_scene(scene)
    with pytest.raises(ValueError, match="affine output capability"):
        restored.core.prepare_receivers(restored_prepared.core_context, receivers)


def test_wind_prepared_state_rejects_transform_mutation():
    model = _wind_model().eval()
    scene = _wind_scene()
    receivers = torch.tensor([[[0.0, 1.0, 2.5]]])
    prepared = model.prepare_scene(scene)
    model.velocity_transform.safe_std[0] += 0.01
    with pytest.raises(ValueError, match="velocity transform changed"):
        model.predict_physical(prepared, receivers)


@pytest.mark.parametrize("mode", ["P-R", "P-C"])
def test_wind_native_routing_still_rejects_exact_affine_increments(mode):
    model = _wind_model(mode=mode)
    with pytest.raises(ValueError, match="nonlinear|exact affine"):
        model.apply_increment(torch.ones(1, 2))


def test_six_recovery_recipes_validate_and_seal_graph_interventions():
    cli = _joint_recipe_module()
    config_dir = REPO_ROOT / "src" / "config_core" / "forward" / "joint_regional"
    paths = [
        config_dir / name
        for name in (
            "thermal_interaction_preserving_p.json",
            "thermal_interaction_preserving_p_g.json",
            "thermal_interaction_preserving_p_h.json",
            "wind_interaction_preserving_p.json",
            "wind_interaction_preserving_p_g.json",
            "wind_interaction_preserving_p_h.json",
        )
    ]
    recipes = [cli.read_recipe(path) for path in paths]
    assert len(recipes) == 6
    assert {(recipe["task"], recipe["mode"]) for recipe in recipes} == {
        (task, mode)
        for task in ("thermal", "wind")
        for mode in INTERACTION_PRESERVING_THERMAL_MODES
    }
    for recipe in recipes:
        if recipe["mode"] == "P":
            assert recipe["regional_anchors"] == 0
        else:
            assert recipe["regional_anchors"] > 0
        assert recipe["locality_prior_strength"] == (0.0 if recipe["mode"] == "P" else 1.0)
        assert recipe["model_contract"]["collective"]["enabled"] is (recipe["mode"] != "P")

    p_g = copy.deepcopy(next(row for row in recipes if row["mode"] == "P-G"))
    p_g.pop("locality_prior_strength")
    with pytest.raises(ValueError, match="lambda=1"):
        cli.validate_recipe(p_g)
    p_h = copy.deepcopy(next(row for row in recipes if row["mode"] == "P-H"))
    p_h["locality_prior_strength"] = 0.5
    with pytest.raises(ValueError, match="lambda=1"):
        cli.validate_recipe(p_h)
    p = copy.deepcopy(next(row for row in recipes if row["task"] == "wind" and row["mode"] == "P"))
    p["primary_queries"] += 1
    with pytest.raises(ValueError, match="primary_queries"):
        cli.validate_recipe(p)


def test_thermal_calibration_receipt_is_required_digest_bound_and_train_source_bound(tmp_path, monkeypatch):
    cli = _joint_recipe_module()
    recipe = cli.read_recipe(
        REPO_ROOT / "src" / "config_core" / "forward" / "joint_regional" / "thermal_interaction_preserving_p.json"
    )
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    receipt_path = tmp_path / recipe["calibration_receipt"]["receipt_path"]
    with pytest.raises(FileNotFoundError, match="TRAIN calibration receipt"):
        cli._bind_calibration_receipt(recipe)

    receipt_path.parent.mkdir(parents=True)
    active_identity = {"source_git_commit": "train-source", "source_set_sha256": "a" * 64}
    monkeypatch.setattr(cli, "training_source_identity", lambda *, require_clean: active_identity)
    receipt = {
        "training_source_identity": {"source_git_commit": "other-source", "source_set_sha256": "b" * 64},
        "response_coefficient": 0.25,
        "operator_coefficient": 0.5,
    }
    receipt["payload_sha256"] = cli._canonical_payload_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="different committed training source"):
        cli._bind_calibration_receipt(recipe)

    receipt["training_source_identity"] = active_identity
    receipt["payload_sha256"] = "f" * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="digest is missing or invalid"):
        cli._bind_calibration_receipt(recipe)


def test_legacy_j_calibration_dispatch_is_not_rejected_by_the_fresh_p_contract():
    task = object.__new__(JointThermalTask)
    task.joint_mode = "J-H"
    receipt = {"method": "legacy_joint_thermal_gradient_ratio"}
    task._calibrate_legacy_auxiliary_coefficients = lambda: receipt
    task._calibrate_interaction_preserving_auxiliary_coefficients = lambda: pytest.fail(
        "A J-family calibration must not enter the fresh-P path."
    )
    assert task.calibrate_auxiliary_coefficients() is receipt


def test_fresh_p_calibration_rejects_a_different_training_case_membership():
    task = object.__new__(JointThermalTask)
    task.joint_mode = "P"
    task._train_ids = ["case-one", "case-three", "case-ten", "case-twelve"]
    task.training_cases = [
        {"structure": {"module_present": np.ones(count, dtype=np.float32)}}
        for count in (1, 3, 10, 12)
    ]
    receipt = {
        "schema_version": 1,
        "method": "fresh_joint_init_fixed_train_gradient_ratio_v1",
        "calibration_mode": "P",
        "compatible_modes": list(INTERACTION_PRESERVING_THERMAL_MODES),
        "dataset_protocol": "fixed25_v1",
        "training_membership_sha256": "0" * 64,
    }
    assert receipt["training_membership_sha256"] != _case_ids_hash(task._train_ids)
    with pytest.raises(ValueError, match="not the current fresh-P, fixed25 TRAIN artifact"):
        task._validate_auxiliary_calibration(receipt)


class _ExactResumeToyProvider:
    def identity_payload(self):
        return {"dataset": "tiny-resume-test", "membership": "fixed"}

    def epoch_cases(self, epoch: int, seed: int):
        del epoch, seed
        return (0, 1)

    def make_batch(self, case_keys, key: SamplingKey):
        del key
        x = torch.tensor(case_keys, dtype=torch.float32).reshape(-1, 1)
        return TaskBatch(scene_inputs=x, receivers=None, targets=2.0 * x + 1.0,
                         auxiliary={}, case_keys=tuple(case_keys))

    def loss_denominators(self, batches, phase, arm):
        del phase, arm
        return {"native": float(sum(batch.targets.numel() for batch in batches))}

    def make_scene(self, scene_inputs):
        return scene_inputs

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch, temperature):
        del receivers, execution_mode, phase, epoch, temperature
        return model(scene), None

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        del phase, auxiliary_state
        return {"native": LossTerm((predictions - targets).square().sum(), targets.numel())}

    def validation_batches(self):
        key = SamplingKey(0, 1, 0, 0, "joint", "joint")
        yield self.make_batch((0, 1), key)

    def validation_metrics(self, predictions, targets, auxiliary_state):
        del auxiliary_state
        return {"squared_error": float((predictions - targets).square().sum()), "count": targets.numel()}

    def reduce_native_metrics(self, records):
        return {"field_score": sum(row["squared_error"] for row in records)
                / sum(row["count"] for row in records)}

    def optimizer_groups(self, model, arm, stage):
        del model, arm, stage
        schedule = ScheduleSpec(
            peak_lr=0.02, warmup_start_lr=0.02, warmup_epochs=1,
            hold_through_epoch=1, total_epochs=2, final_lr=0.001,
        )
        return (OptimizerGroupSpec("predictor", ("weight", "bias"), schedule, weight_decay=0.0),)

    def work_counts(self, batch, predictions, auxiliary_state):
        del predictions, auxiliary_state
        return {"query_rows": batch.targets.numel()}


def test_exact_resume_rejects_a_changed_pair_core_mathematics(tmp_path, monkeypatch):
    monkeypatch.setattr(training_runtime, "_render_loss_curves", lambda *args: None)
    config = EngineConfig(
        seed=0,
        microbatch_cases=1,
        effective_cases=1,
        total_epochs=2,
        warmup_epochs=1,
        open_through_epoch=1,
        soft_through_epoch=1,
        monitor_every=1,
        checkpoint_epochs=(1, 2),
        latest_every=1,
        curve_every=1,
        training_mode="joint",
    )
    engine = TrainingEngine(config, device="cpu")
    identity = {
        "workflow": "joint_regional_fields_v1",
        "recipe": {"task": "thermal", "mode": "P", "model_contract": {"math": "two-round-source-read"}},
    }
    output = tmp_path / "exact-resume"
    model = nn.Linear(1, 1)
    engine.fit(model, _ExactResumeToyProvider(), output, identity=identity, arm="P", stop_after=1)
    changed = copy.deepcopy(identity)
    changed["recipe"]["model_contract"]["math"] = "pooled-context-control"
    with pytest.raises(ValueError, match="identity, engine config"):
        engine.fit(
            model,
            _ExactResumeToyProvider(),
            output,
            identity=changed,
            arm="P",
            stop_after=2,
            resume_checkpoint=output / "latest_model.pt",
        )
