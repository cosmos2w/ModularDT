"""Focused CPU contracts for the fresh joint Thermal regional adapter/provider."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from channelthermal.joint_regional import (
    JOINT_THERMAL_CHANNEL_ORDER,
    JOINT_THERMAL_MODES,
    JointThermalRegionalAdapter,
    load_joint_thermal_checkpoint,
)
from channelthermal.training import joint_task as joint_task_module
from channelthermal.training.joint_task import JointThermalTask


def _stats():
    return {
        "field_mean_by_channel": np.asarray([1.0, -2.0, 0.5, 3.0, 10.0], dtype=np.float32),
        "field_std_by_channel": np.asarray([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "interface_targets_mean": np.asarray([10.0, 0.0], dtype=np.float32),
        "interface_targets_std": np.asarray([4.0, 5.0], dtype=np.float32),
        "internal_temperature_std": np.asarray([7.0], dtype=np.float32),
    }


def _structure(module_count: int):
    centers = []
    for index in range(module_count):
        centers.append([1.0 + 1.75 * (index % 6), 1.0 + 3.7 * (index // 6)])
    return {
        "module_centers": torch.tensor([centers], dtype=torch.float32),
        "module_present": torch.ones(1, module_count, dtype=torch.float32),
        "material_params": torch.tensor([[0.02, 0.01, 0.02, 1.0, 2.0, 0.22]], dtype=torch.float32),
        "re": torch.tensor([[50.0]], dtype=torch.float32),
        "u_in": torch.ones(1, 1, dtype=torch.float32),
        "domain_length_x": torch.tensor([[12.0]], dtype=torch.float32),
        "domain_length_y": torch.tensor([[6.0]], dtype=torch.float32),
        "module_source_ids": tuple(f"physical-{index}" for index in range(module_count)),
    }


def _adapter(mode="J-H", locality_prior_strength=0.0):
    torch.manual_seed(91)
    return JointThermalRegionalAdapter(
        mode=mode,
        normalization_stats=_stats(),
        hidden=12,
        message=12,
        regional_anchors=6,
        depth=2,
        receiver_tile=3,
        nx=8,
        ny=4,
        environment_nx=4,
        environment_ny=2,
        locality_prior_strength=locality_prior_strength,
        seed=91,
    )


def _prepare(model, module_count):
    structure = _structure(module_count)
    xy = torch.tensor([[[0.75, 0.75], [3.75, 2.25], [8.25, 3.75]]], dtype=torch.float32)
    local = torch.tensor([[[[0.0, 0.0], [0.35, 0.0]], [[0.0, 0.0], [0.35, 0.0]]]], dtype=torch.float32)
    if module_count != 2:
        local = torch.zeros(1, module_count, 2, 2, dtype=torch.float32)
    prepared = model.prepare_native(structure, xy, local_query_points=local, ntheta=4, chunk_size=3)
    return structure, xy, local, prepared


def test_native_subset_and_dense_truncation_preserve_affine_roles_and_heat_null_flow():
    model = _adapter(locality_prior_strength=1.0).eval()
    structure, xy, local, _ = _prepare(model, 2)
    dense, subset = [model.prepare_native(
        structure, xy, local_query_points=local, ntheta=4,
        retained_access_mass=.99, receiver_edge_executor=executor)
        for executor in ('dense', 'subset')]
    heat = torch.tensor([[.7, 1.1]], dtype=torch.float64)
    delta = torch.tensor([[.125, -.25]], dtype=torch.float64)
    outputs = [model.apply_native(state, heat, accumulation_dtype=torch.float64) for state in (dense, subset)]
    for name in ('pred_field', 'pred_interface', 'pred_internal_temperature'):
        torch.testing.assert_close(outputs[0][name], outputs[1][name], rtol=3e-6, atol=3e-6)
    changed = model.apply_native(subset, heat + delta, accumulation_dtype=torch.float64)
    increment = model.apply_native(subset, delta, increment=True, accumulation_dtype=torch.float64)
    assert torch.equal(changed['pred_field'][..., :4], outputs[1]['pred_field'][..., :4])
    for name in ('fluid_temperature', 'pred_interface', 'pred_internal_temperature'):
        torch.testing.assert_close(changed[name] - outputs[1][name], increment[name], rtol=1e-8, atol=1e-12)
    assert subset.response.retention_receipt['executor_receipt']['executors'] == ['packed-subset']
    subset.joint_read_options['retained_access_mass'] = .90
    with pytest.raises(ValueError, match='receiver execution changed'):
        model.apply_native(subset, heat)


class _ForbiddenTarget:
    def __getattribute__(self, name):
        raise AssertionError(f"forbidden target/input access: {name}")


@pytest.mark.parametrize("mode", JOINT_THERMAL_MODES)
def test_joint_native_heat_null_affinity_and_target_poisoning(mode):
    model = _adapter(mode)
    structure, xy, local, _ = _prepare(model, 2)
    structure.update(
        heat_powers=_ForbiddenTarget(),
        steady_field=_ForbiddenTarget(),
        predicted_ports=_ForbiddenTarget(),
        observed_temperature=_ForbiddenTarget(),
    )
    prepared = model.prepare_native(structure, xy, local_query_points=local, ntheta=4, chunk_size=3)
    assert prepared.context.dependency.output_law == "affine"
    assert prepared.context.dependency.output_roles == ("temperature",)
    assert prepared.context.model_output_laws == {"flow": "nonlinear", "temperature_response": "affine"}

    heat = torch.tensor([[0.7, 1.1]], dtype=torch.float64)
    delta = torch.tensor([[0.125, -0.25]], dtype=torch.float64)
    baseline = model.apply_native(prepared, heat, accumulation_dtype=torch.float64)
    changed = model.apply_native(prepared, heat + delta, accumulation_dtype=torch.float64)
    increment = model.apply_native(
        prepared,
        delta,
        increment=True,
        accumulation_dtype=torch.float64,
    )
    assert torch.equal(baseline["pred_field"][..., :4], changed["pred_field"][..., :4])
    assert torch.equal(increment["pred_field"][..., :4], torch.zeros_like(increment["pred_field"][..., :4]))
    torch.testing.assert_close(
        changed["fluid_temperature"] - baseline["fluid_temperature"],
        increment["fluid_temperature"],
        atol=1e-13,
        rtol=1e-8,
    )
    torch.testing.assert_close(
        changed["pred_interface"] - baseline["pred_interface"],
        increment["pred_interface"],
        atol=1e-13,
        rtol=1e-8,
    )
    assert increment["fluid_temperature"].dtype == torch.float64
    assert "module_material_peak" not in increment
    assert "pred_port_condition" not in increment


@pytest.mark.parametrize("module_count", [1, 12])
def test_low_and_high_source_count_both_heads_reach_shared_hypergraph_parameters(module_count):
    model = _adapter("J-H")
    _, _, _, prepared = _prepare(model, module_count)
    heat = torch.linspace(0.2, 1.0, module_count)[None]
    output = model.apply_native(prepared, heat)
    response = model.apply_native(prepared, torch.full_like(heat, 0.17), increment=True)
    flow_loss = output["pred_field"][..., :4].square().mean()
    thermal_loss = (
        response["fluid_temperature"].square().mean()
        + response["pred_interface"].square().mean()
        + response["pred_internal_temperature"].square().mean()
    )
    parameters = dict(model.named_parameters())
    shared_names = [
        "core.source_encoder.0.weight",
        "core.source_read.0.weight",
        "core.edge_updates.0.0.weight",
        "core.receiver_query.0.weight",
    ]
    # With one physical source, a normalized source-incidence distribution has
    # no source competition and its membership-score derivative is exactly 0.
    # The high-M panel tests that learned incidence is live in both heads.
    if module_count == 12:
        shared_names.append("core.source_membership_score.0.weight")
    for loss in (flow_loss, thermal_loss):
        gradients = torch.autograd.grad(loss, tuple(parameters[name] for name in shared_names), retain_graph=True)
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)


def test_joint_checkpoint_roundtrip_is_self_contained_and_stale_context_is_rejected():
    model = _adapter("J-geometry")
    _, _, _, prepared = _prepare(model, 3)
    before = model.apply_native(prepared, torch.tensor([[0.5, 0.8, 1.1]]))["pred_field"].detach()
    payload = model.checkpoint_payload(provider_identity={"scope": "focused_cpu_test"})
    assert payload["channel_order"] == list(JOINT_THERMAL_CHANNEL_ORDER)
    assert all(name.startswith("core.") for name in payload["model_state_dict"])
    assert not any(name.startswith(("parent.", "flow.", "thermal.")) for name in payload["model_state_dict"])
    restored = load_joint_thermal_checkpoint(payload)
    _, _, _, restored_prepared = _prepare(restored, 3)
    after = restored.apply_native(restored_prepared, torch.tensor([[0.5, 0.8, 1.1]]))["pred_field"].detach()
    torch.testing.assert_close(after, before, atol=0, rtol=0)

    with torch.no_grad():
        next(model.parameters()).add_(1.0e-3)
    with pytest.raises(ValueError, match="weights changed"):
        model.apply_native(prepared, torch.ones(1, 3))


@pytest.mark.parametrize(
    ("strength", "error"),
    [
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (float("-inf"), ValueError),
        (True, TypeError),
        ("bad", ValueError),
    ],
)
def test_locality_prior_strength_requires_finite_nonnegative_number(strength, error):
    with pytest.raises(error, match="locality prior"):
        _adapter("J-H", locality_prior_strength=strength)


def test_locality_prior_zero_preserves_v1_config_and_self_contained_heat_null():
    default = _adapter("J-H")
    explicit_zero = _adapter("J-H", locality_prior_strength=0.0)
    assert default.model_config() == explicit_zero.model_config()
    assert "locality_prior_strength" not in default.model_config()
    assert default.core_config == explicit_zero.core_config
    assert all(torch.equal(left, right) for left, right in zip(default.state_dict().values(), explicit_zero.state_dict().values(), strict=True))

    _, _, _, prepared = _prepare(default, 3)
    payload = default.checkpoint_payload(provider_identity={"scope": "locality_prior_default_test"})
    assert "locality_prior_strength" not in payload["model_config"]
    restored = load_joint_thermal_checkpoint(payload)
    _, _, _, restored_prepared = _prepare(restored, 3)
    zero_increment = restored.apply_native(
        restored_prepared,
        torch.zeros(1, 3),
        increment=True,
    )
    assert torch.equal(zero_increment["pred_field"], torch.zeros_like(zero_increment["pred_field"]))
    assert torch.equal(zero_increment["pred_interface"], torch.zeros_like(zero_increment["pred_interface"]))
    assert torch.equal(zero_increment["pred_internal_temperature"], torch.zeros_like(zero_increment["pred_internal_temperature"]))
    baseline = default.apply_native(prepared, torch.tensor([[0.5, 0.8, 1.1]]))
    restored_baseline = restored.apply_native(restored_prepared, torch.tensor([[0.5, 0.8, 1.1]]))
    for key in ("pred_field", "pred_interface", "pred_internal_temperature"):
        torch.testing.assert_close(restored_baseline[key], baseline[key], atol=0, rtol=0)


def test_locality_prior_jh_roundtrip_and_flow_response_gradients_are_finite():
    default = _adapter("J-H")
    revised = _adapter("J-H", locality_prior_strength=1.0)
    assert revised.model_config()["locality_prior_strength"] == 1.0
    assert "locality_prior_strength" not in default.model_config()
    assert list(default.state_dict()) == list(revised.state_dict())
    assert all(torch.equal(left, right) for left, right in zip(default.state_dict().values(), revised.state_dict().values(), strict=True))

    structure, xy, local, _ = _prepare(revised, 12)
    prepared = revised.prepare_native(structure, xy, local_query_points=local, ntheta=4, chunk_size=3)
    heat = torch.linspace(0.2, 1.0, 12)[None]
    field_output = revised.apply_native(prepared, heat)
    response_output = revised.apply_native(prepared, torch.full_like(heat, 0.17), increment=True)
    shared_names = (
        "core.source_encoder.0.weight",
        "core.source_read.0.weight",
        "core.edge_updates.0.0.weight",
        "core.source_membership_score.0.weight",
        "core.receiver_query.0.weight",
    )
    parameters = dict(revised.named_parameters())
    for loss in (
        field_output["pred_field"][..., :4].square().mean(),
        response_output["fluid_temperature"].square().mean()
        + response_output["pred_interface"].square().mean()
        + response_output["pred_internal_temperature"].square().mean(),
    ):
        gradients = torch.autograd.grad(loss, tuple(parameters[name] for name in shared_names), retain_graph=True)
        assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)

    payload = revised.checkpoint_payload(provider_identity={"scope": "locality_prior_roundtrip_test"})
    assert payload["model_config"]["locality_prior_strength"] == 1.0
    restored = load_joint_thermal_checkpoint(payload)
    assert restored.model_config()["locality_prior_strength"] == 1.0
    _, _, _, restored_prepared = _prepare(restored, 12)
    restored_output = restored.apply_native(restored_prepared, heat)
    torch.testing.assert_close(restored_output["pred_field"], field_output["pred_field"], atol=0, rtol=0)


def test_locality_prior_is_restricted_to_jh():
    with pytest.raises(ValueError, match="only supported by J-H"):
        _adapter("J-geometry", locality_prior_strength=1.0)


def test_joint_factory_passes_locality_prior_to_fresh_model(monkeypatch, tmp_path):
    data_path = tmp_path / "packed_dataset.h5"
    data_path.write_bytes(b"synthetic factory binding")
    manifest = {"manifest_sha256": "a" * 64, "source": {"metadata_sha256": "b" * 64}}
    monkeypatch.setattr(
        joint_task_module,
        "_load_data_binding",
        lambda manifest_arg, data_arg: (manifest, data_path, None),
    )
    monkeypatch.setattr(
        joint_task_module,
        "development_case_ids",
        lambda current_manifest, split: tuple(
            f"train-{index}" for index in range(150)
        ) if split == "train" else tuple(f"dev-{index}" for index in range(22)),
    )
    monkeypatch.setattr(
        joint_task_module,
        "fit_global_normalizer",
        lambda current_path, case_ids: SimpleNamespace(stats=_stats()),
    )
    monkeypatch.setattr(
        joint_task_module,
        "_read_selected_cases",
        lambda *args, **kwargs: tuple({} for _ in range(150 if kwargs["split"] == "train" else 22)),
    )
    monkeypatch.setattr(
        joint_task_module,
        "_load_recipe_helpers",
        lambda: (lambda cases: (), lambda families: {}),
    )
    monkeypatch.setattr(joint_task_module, "_read_response_families", lambda *args: ())
    captured = {}

    class CapturingTask:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(joint_task_module, "JointThermalTask", CapturingTask)
    model, _provider = joint_task_module.build_thermal_joint_task(
        mode="J-H",
        device="cpu",
        hidden=12,
        message=12,
        regional_anchors=6,
        locality_prior_strength=1.0,
    )
    assert model.locality_prior_strength == 1.0
    assert model.model_config()["locality_prior_strength"] == 1.0
    assert captured["model"] is model
    assert captured["model"].core.locality_prior_strength == 1.0


@pytest.mark.parametrize(
    "physical_ids",
    [
        [117, 904],
        np.asarray([117, 904], dtype=np.int64),
        torch.tensor([117, 904], dtype=torch.int64),
    ],
    ids=("list", "numpy", "tensor"),
)
def test_numeric_physical_source_ids_survive_slot_mapping(physical_ids):
    model = _adapter("J-H")
    structure = _structure(2)
    structure["module_source_ids"] = physical_ids
    tensors = model.context_tensors(structure)
    assert tensors["source_ids"].tolist() == [[117, 904]]

    prepared_structure = _structure(2)
    prepared_structure["module_source_ids"] = physical_ids
    xy = torch.tensor([[[0.75, 0.75], [3.75, 2.25]]], dtype=torch.float32)
    local = torch.zeros(1, 2, 2, 2, dtype=torch.float32)
    prepared = model.prepare_native(prepared_structure, xy, local_query_points=local, ntheta=4)
    assert prepared.context.source_ids.tolist() == [[117, 904]]
    assert prepared.source_id_catalogue == (117, 904)
    assert prepared.context.source_id_catalogue == (117, 904)
    assert prepared.response.source_id_catalogue == (117, 904)


@pytest.mark.parametrize(
    ("physical_ids", "error"),
    [
        ([17, 17], ValueError),
        ([-1, 18], ValueError),
        ([17.5, 18.0], ValueError),
        ([float("nan"), 18.0], ValueError),
        ([2**63, 18], ValueError),
        ([True, False], TypeError),
    ],
)
def test_numeric_physical_source_ids_reject_invalid_active_catalogues(physical_ids, error):
    model = _adapter("J-H")
    structure = _structure(2)
    structure["module_source_ids"] = physical_ids
    with pytest.raises(error):
        model.context_tensors(structure)


def test_mutated_numpy_source_identity_invalidates_prepared_native_state():
    model = _adapter("J-H")
    structure = _structure(2)
    physical_ids = np.asarray([117, 904], dtype=np.int64)
    structure["module_source_ids"] = physical_ids
    xy = torch.tensor([[[0.75, 0.75], [3.75, 2.25]]], dtype=torch.float32)
    local = torch.zeros(1, 2, 2, 2, dtype=torch.float32)
    prepared = model.prepare_native(structure, xy, local_query_points=local, ntheta=4)
    physical_ids[0] = 118
    with pytest.raises(ValueError, match="Geometry/context changed"):
        model.apply_native(prepared, torch.ones(1, 2))


def test_inactive_source_identity_may_use_reserved_sentinel():
    model = _adapter("J-H")
    structure = _structure(2)
    structure["module_present"] = torch.tensor([[1.0, 0.0]])
    structure["module_source_ids"] = np.asarray([2**40 + 117, -1], dtype=np.int64)
    tensors = model.context_tensors(structure)
    assert tensors["source_ids"].tolist() == [[2**40 + 117, -1]]


def test_predict_native_sample_preserves_large_numpy_physical_ids(monkeypatch):
    model = _adapter("J-H")
    base_structure = _structure(2)
    structure = {
        "module_centers": base_structure["module_centers"][0].numpy(),
        "module_present": base_structure["module_present"][0].numpy(),
        "material_params": base_structure["material_params"][0].numpy(),
        "re": np.asarray([50.0], dtype=np.float32),
        "u_in": np.asarray([1.0], dtype=np.float32),
        "domain_length_x": np.asarray([12.0], dtype=np.float32),
        "domain_length_y": np.asarray([6.0], dtype=np.float32),
        "module_source_ids": np.asarray([2**40 + 117, 2**40 + 904], dtype=np.int64),
        "heat_powers": np.asarray([0.75, 1.0], dtype=np.float32),
    }
    x = (np.arange(8, dtype=np.float32) + 0.5) * 1.5
    y = (np.arange(4, dtype=np.float32) + 0.5) * 1.5
    x_grid, y_grid = np.meshgrid(x, y)
    observed_ids = []
    prepare_native = model.prepare_native

    def capture_ids(current_structure, fluid_xy, **kwargs):
        prepared = prepare_native(current_structure, fluid_xy, **kwargs)
        observed_ids.extend(prepared.context.source_ids[0].cpu().tolist())
        return prepared

    monkeypatch.setattr(model, "prepare_native", capture_ids)
    sample = {
        "structure": structure,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "module_internal_query_points": np.zeros((2, 2), dtype=np.float32),
    }
    result = model.predict_native_sample(sample, device="cpu")
    assert observed_ids == [2**40 + 117, 2**40 + 904]
    assert result["pred_field_grid"].shape == (4, 8, 5)
    assert result["pred_interface"].shape == (2, 64, 2)
    theta = np.arange(64, dtype=np.float32) * (2 * np.pi / 64)
    expected_angles = np.stack((theta, np.cos(theta), np.sin(theta)), axis=-1)
    np.testing.assert_allclose(result["pred_port_condition"][0, :, :3], expected_angles, atol=1e-6)

    observed_ids.clear()
    explicit = model.predict_native_sample(sample, device="cpu", ntheta=16)
    assert observed_ids == [2**40 + 117, 2**40 + 904]
    assert explicit["pred_interface"].shape == (2, 16, 2)
    theta = np.arange(16, dtype=np.float32) * (2 * np.pi / 16)
    expected_angles = np.stack((theta, np.cos(theta), np.sin(theta)), axis=-1)
    np.testing.assert_allclose(explicit["pred_port_condition"][0, :, :3], expected_angles, atol=1e-6)


@pytest.mark.parametrize("ntheta", [0, -1])
def test_predict_native_sample_rejects_nonpositive_interface_count(ntheta):
    with pytest.raises(ValueError, match="ntheta must be positive"):
        _adapter().predict_native_sample({}, ntheta=ntheta)


@pytest.mark.parametrize("ntheta", [16.0, True, "64"])
def test_predict_native_sample_requires_integer_interface_count(ntheta):
    with pytest.raises(TypeError, match="ntheta must be a positive integer"):
        _adapter().predict_native_sample({}, ntheta=ntheta)


def test_normalization_mutation_invalidates_prepared_native_state():
    model = _adapter("J-H")
    _, _, _, prepared = _prepare(model, 2)
    model.normalization_stats["field_mean_by_channel"][0] += 0.25
    with pytest.raises(ValueError, match="normalization changed"):
        model.apply_native(prepared, torch.ones(1, 2))


def test_native_record_increment_returns_flow_null_and_precise_temperature_roles():
    model = _adapter("J-H")
    module = SimpleNamespace(
        module_id="module-0",
        position_xy=(3.0, 2.0),
        active=True,
        heating=0.75,
    )
    roles = {
        "fluid_fields": SimpleNamespace(
            query_features=np.asarray([[0.75, 0.75], [3.75, 2.25], [8.25, 3.75]], dtype=np.float32)
        ),
        "interface": SimpleNamespace(
            receiver_module_ids=np.asarray(["module-0"] * 4),
            query_features=np.stack((np.arange(4) * (2 * np.pi / 4), np.zeros(4)), axis=-1).astype(np.float32),
        ),
        "solid_temperature": SimpleNamespace(
            receiver_module_ids=np.asarray(["module-0"] * 2),
            query_features=np.asarray([[0.0, 0.0], [0.35, 0.0]], dtype=np.float32),
        ),
    }
    record = SimpleNamespace(
        design=SimpleNamespace(modules=[module]),
        context=SimpleNamespace(values={
            "nu": 0.02,
            "solid_alpha": 0.01,
            "fluid_alpha": 0.02,
            "solid_k": 1.0,
            "fluid_k": 2.0,
            "module_radius": 0.22,
            "re": 50.0,
            "u_in": 1.0,
            "domain_length_x": 12.0,
            "domain_length_y": 6.0,
        }),
        output=SimpleNamespace(roles=roles),
    )
    prepared = model.prepare_record(record)
    base = model.apply_record(prepared, accumulation_dtype=torch.float64)
    delta = torch.tensor([0.125], dtype=torch.float64)
    plus = model.apply_record(
        prepared,
        torch.tensor([0.875], dtype=torch.float64),
        accumulation_dtype=torch.float64,
    )
    increment = model.apply_record_increment(prepared, delta, accumulation_dtype=torch.float64)
    assert torch.equal(increment["fluid_fields"][:, :4], torch.zeros_like(increment["fluid_fields"][:, :4]))
    torch.testing.assert_close(
        plus["fluid_fields"] - base["fluid_fields"], increment["fluid_fields"], atol=1e-13, rtol=1e-8
    )
    torch.testing.assert_close(plus["interface"] - base["interface"], increment["interface"], atol=1e-13, rtol=1e-8)
    torch.testing.assert_close(
        plus["solid_temperature"] - base["solid_temperature"],
        increment["solid_temperature"],
        atol=1e-13,
        rtol=1e-8,
    )


def test_joint_provider_has_one_all_parameter_group_and_balanced_native_loss_families():
    model = _adapter("J-H")
    provider = object.__new__(JointThermalTask)
    provider.model = model
    provider.joint_mode = "J-H"
    provider.total_epochs = 2500
    provider.flow_role_weights = {role: 0.5 / 4 for role in ("u", "v", "p", "omega")}
    provider.thermal_role_weights = {role: 0.5 / 3 for role in ("fluid", "surface", "material")}
    provider.q_proxy_weight = 0.05
    provider.response_weight = 0.1
    provider.operator_weight = 0.2
    provider.use_operator = True
    provider._joint_loss_metadata = provider._build_loss_metadata()

    first = provider.optimizer_groups(model, "J-H", "joint")
    second = provider.optimizer_groups(model, "J-H", "warmup")
    assert first == second
    assert len(first) == 1
    assert set(first[0].parameter_names) == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert all(name.startswith("core.") for name in first[0].parameter_names)
    assert first[0].schedule.total_epochs == 2500
    assert first[0].schedule.peak_lr == pytest.approx(3.0e-4)
    assert first[0].schedule.final_lr == pytest.approx(3.0e-6)

    values = {
        **{f"flow/{role}": torch.tensor([1.0, 2.0]) for role in ("u", "v", "p", "omega")},
        **{f"temperature/{role}": torch.tensor([3.0, 4.0]) for role in ("fluid", "surface", "material")},
        "q_proxy": torch.tensor([5.0, 6.0]),
    }
    values["flow_group"] = torch.stack([values[f"flow/{role}"] for role in ("u", "v", "p", "omega")]).mean(0)
    values["thermal_group"] = torch.stack(
        [values[f"temperature/{role}"] for role in ("fluid", "surface", "material")]
    ).mean(0)
    provider._native_role_losses = lambda *args: values
    provider._operator_loss = lambda *args: (
        torch.tensor([7.0, 8.0]),
        {"operator_rows": 256, "unique_stencil_receiver_rows": 64},
    )
    predictions = SimpleNamespace(
        model=model,
        execution_mode="J-H",
        native_main={},
        native_prepared=None,
        work={},
    )
    targets = SimpleNamespace(case_ids=("a", "b"), case_indices=(0, 1), response=None)
    terms = provider.loss_terms(predictions, targets, "joint", {})
    assert sum(terms[f"flow/{role}"].weight for role in ("u", "v", "p", "omega")) == pytest.approx(0.5)
    assert sum(terms[f"temperature/{role}"].weight for role in ("fluid", "surface", "material")) == pytest.approx(0.5)
    assert terms["q_proxy"].weight == pytest.approx(0.05)
    assert terms["operator_residual"].weight == pytest.approx(0.2)
    metadata = provider.loss_metadata()
    assert metadata["q_proxy"]["weight"] == "0.05 outside the equal flow/thermal group balance"
    assert metadata["flow/u"]["panel"] == "native_prediction"
    assert metadata["operator_residual"]["panel"] == "physics"


def test_joint_provider_identity_is_strict_json_serializable():
    provider = object.__new__(JointThermalTask)
    provider._train_ids = ("train-a", "train-b")
    provider._validation_ids = ("dev-a",)
    provider.manifest = {
        "manifest_sha256": "a" * 64,
        "partitions": {"train": {"case_ids": ["train-a", "train-b"]}},
    }
    provider.formal_full = False
    provider.source_binding = {"dataset_id": "synthetic_identity_test", "metadata_sha256": "b" * 64}
    provider.model = _adapter("J-H")
    provider.stats = _stats()
    provider.joint_mode = "J-H"
    provider.budget = {"fluid_queries": 1024, "material_queries_per_module": 32}
    provider.effective_batch_size = 48
    provider.operator_rows_per_case = 128
    provider.train_families = ({"family_id": "0001"},)
    provider.response_scales = {"fluid": 1.0, "surface": 1.0, "material": 1.0}
    provider.flow_role_weights = {role: 0.5 / 4 for role in ("u", "v", "p", "omega")}
    provider.thermal_role_weights = {role: 0.5 / 3 for role in ("fluid", "surface", "material")}
    provider.q_proxy_weight = 0.05
    provider.response_weight = 1.0
    provider.operator_weight = 1.0
    provider.auxiliary_calibration = None
    provider.total_epochs = 2500

    identity = provider.identity_payload()
    json.dumps(identity, allow_nan=False)
    assert identity["normalization_stats"]["field_mean_by_channel"] == [1.0, -2.0, 0.5, 3.0, 10.0]
    assert "locality_prior_strength" not in identity["model_config"]
    provider.model = _adapter("J-H", locality_prior_strength=1.0)
    revised_identity = provider.identity_payload()
    json.dumps(revised_identity, allow_nan=False)
    assert revised_identity["model_config"]["locality_prior_strength"] == 1.0


def test_native_metrics_report_near_far_query_counts_and_sampled_module_peaks():
    model = _adapter("J-H")
    _, _, _, prepared = _prepare(model, 3)
    output = model.apply_native(prepared, torch.tensor([[0.7, 0.8, 0.9]]))
    provider = object.__new__(JointThermalTask)
    provider.stats = _stats()
    provider.formal_full = False
    provider._validation_ids = ("synthetic-train-preflight",)
    predictions = SimpleNamespace(native_prepared=prepared, native_main=output)
    targets = SimpleNamespace(
        case_ids=("synthetic-train-preflight",),
        field_targets=torch.zeros_like(output["pred_field"]),
        point_weights=torch.ones(output["pred_field"].shape[:2]),
        interface_target=torch.zeros_like(output["pred_interface"]),
        material_targets=torch.zeros_like(output["pred_internal_temperature"][..., 0]),
    )
    record = provider.validation_metrics(predictions, targets, {})
    row = record["case_rows"][0]
    assert row["near_fluid_temperature_query_count"] > 0
    assert row["far_fluid_temperature_query_count"] > 0
    assert row["near_fluid_temperature_query_count"] + row["far_fluid_temperature_query_count"] == 3
    assert len(row["material_peak_abs_error_by_source"]) == 3
    assert np.isfinite(row["sampled_material_peak_rmse"])
    reduced = provider.reduce_native_metrics([record])
    assert reduced["near_fluid_temperature_query_count"] == row["near_fluid_temperature_query_count"]
    assert reduced["far_fluid_temperature_query_count"] == row["far_fluid_temperature_query_count"]
    assert reduced["sampled_material_peak_rmse_mean"] == pytest.approx(row["sampled_material_peak_rmse"])
