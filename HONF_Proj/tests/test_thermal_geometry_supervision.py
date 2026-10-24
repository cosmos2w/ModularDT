from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from channelthermal.interaction_evidence.response_dataset import ResponseBlock
import honf_runtime.thermal_geometry_supervision as geometry_module
from honf_runtime.thermal_geometry_supervision import (
    DEVELOPMENT_GEOMETRY_FAMILIES,
    MIXED_GEOMETRY_LABELS,
    SINGLE_GEOMETRY_LABELS,
    TRAIN_GEOMETRY_FAMILIES,
    GeometryResponsePair,
    GeometrySupervision,
    _component_sum,
    _component_scales,
    _make_pair,
    _normalize_atlas_source,
    _pair_panel_loss,
    geometry_response_due,
    validate_module_major_receivers,
    validate_training_family_metadata,
)


@dataclass(frozen=True)
class _Module:
    module_id: str
    position_xy: tuple[float, float]
    heating: float


@dataclass(frozen=True)
class _Design:
    modules: tuple[_Module, ...]

    @property
    def active_modules(self):
        return self.modules

    @property
    def active_module_ids(self):
        return tuple(module.module_id for module in self.modules)


def _record(modules: tuple[_Module, ...]):
    context = {
        "nu": 0.018,
        "solid_alpha": 0.01,
        "fluid_alpha": 0.02,
        "solid_k": 1.0,
        "fluid_k": 1.0,
        "module_radius": 0.45,
        "re": 50.0,
        "u_in": 1.0,
        "domain_length_x": 12.0,
        "domain_length_y": 6.0,
    }
    return SimpleNamespace(design=_Design(modules), context=SimpleNamespace(values=context))


def _block(role, values, mask, features, channel_names, query_ids, receiver_ids=None):
    return ResponseBlock(
        role=role,
        label="i_plus",
        query_features=features,
        delta=values,
        valid_mask=mask,
        quadrature_weights=np.ones(values.shape[0]),
        channel_names=channel_names,
        channel_units=tuple("native" for _ in channel_names),
        query_ids=query_ids,
        receiver_module_ids=receiver_ids,
    )


def _pair(family="0001", label="i_minus", *, target_scale=0.0):
    module_ids = tuple(f"{family}:module:{index}" for index in range(3))
    base_modules = tuple(_Module(module_id, (1.0 + index, 2.0 + index), 1.0 + index) for index, module_id in enumerate(module_ids))
    moved = list(base_modules)
    moved[0] = _Module(moved[0].module_id, (moved[0].position_xy[0] - 0.1, moved[0].position_xy[1]), moved[0].heating)
    baseline, trial = _record(base_modules), _record(tuple(moved))

    fluid_xy = np.stack((np.linspace(0.1, 1.0, 12), np.linspace(0.2, 0.9, 12)), axis=1)
    fluid_delta = np.full((12, 5), target_scale, dtype=np.float64)
    fluid_mask = np.ones_like(fluid_delta, dtype=bool)
    fluid_mask[2, :] = False
    fluid_mask[5, 1] = False
    fluid = _block("fluid_fields", fluid_delta, fluid_mask, fluid_xy,
                   ("u", "v", "p", "omega", "temperature"), tuple(f"f:{i}" for i in range(12)))

    theta = np.arange(64, dtype=np.float64) * (2.0 * np.pi / 64)
    interface_features_one = np.stack((theta, np.cos(theta), np.sin(theta)), axis=1)
    interface_features = np.concatenate([interface_features_one for _ in module_ids], axis=0)
    interface_rows = len(module_ids) * 64
    interface_delta = np.full((interface_rows, 2), target_scale, dtype=np.float64)
    interface_mask = np.ones_like(interface_delta, dtype=bool)
    interface_mask[3, 1] = False
    interface = _block(
        "interface", interface_delta, interface_mask, interface_features,
        ("T_surface", "q_normal"), tuple(f"port:{i}" for i in range(interface_rows)),
        tuple(module_id for module_id in module_ids for _ in range(64)),
    )

    local_one = np.stack((np.linspace(-0.8, 0.8, 10), np.zeros(10)), axis=1)
    local_features = np.concatenate([local_one for _ in module_ids], axis=0)
    solid_rows = len(module_ids) * 10
    solid_delta = np.full((solid_rows, 1), target_scale, dtype=np.float64)
    solid_mask = np.ones_like(solid_delta, dtype=bool)
    solid_mask[4, 0] = False
    solid = _block(
        "solid_temperature", solid_delta, solid_mask, local_features, ("temperature",),
        tuple(f"solid:{i}" for i in range(solid_rows)),
        tuple(module_id for module_id in module_ids for _ in range(10)),
    )
    return GeometryResponsePair(
        family_id=family,
        physical_family_id="duplicate_family:0001+0273" if family == "0001" else f"stored_family:{family}",
        label=label,
        module_ids=module_ids,
        baseline=baseline,
        trial=trial,
        roles={"fluid_fields": fluid, "interface": interface, "solid_temperature": solid},
        identity={"family_id": family, "label": label, "geometry_pair_sha256": "test"},
    )


def _scales():
    return {
        "fluid_fields/u": 1.0,
        "fluid_fields/v": 1.0,
        "fluid_fields/p": 1.0,
        "fluid_fields/omega": 1.0,
        "fluid_fields/temperature": 1.0,
        "interface/T_surface": 1.0,
        "interface/q_normal": 1.0,
        "solid_temperature/temperature": 1.0,
    }


class _Prepared:
    def __init__(self, structure, fluid_xy, local_query_points, ntheta):
        self.structure = structure
        self.fluid_xy = fluid_xy
        self.local_query_points = local_query_points
        self.ntheta = ntheta


class _DifferentiableNativeAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude = nn.Parameter(torch.tensor(0.7))
        self.observed_modes = []

    def prepare_native(self, structure, fluid_xy, *, local_query_points, ntheta, scene_key):
        del scene_key
        self.observed_modes.append(self.training)
        return _Prepared(structure, fluid_xy, local_query_points, ntheta)

    def apply_native(self, prepared, heat):
        self.observed_modes.append(self.training)
        centers = prepared.structure["module_centers"]
        geometry = centers[..., 0].mean(dim=1) + 0.3 * centers[..., 1].mean(dim=1)
        fluid_base = (prepared.fluid_xy[..., 0:1] + 0.5 * prepared.fluid_xy[..., 1:2]) * geometry[:, None, None]
        channel_gain = torch.arange(1, 6, device=centers.device, dtype=centers.dtype)[None, None, :]
        field = self.amplitude * fluid_base * channel_gain
        module_x = centers[..., 0][:, :, None, None]
        interface = self.amplitude * module_x.expand(-1, -1, prepared.ntheta, 2)
        local_x = prepared.local_query_points[..., 0][:, None, :, None]
        solid = self.amplitude * centers[..., 0][:, :, None, None] * local_x
        return {
            "pred_field": field,
            "pred_interface": interface,
            "pred_internal_temperature": solid,
        }


def _supervisor(pairs, *, chunk_size=8):
    supervisor = GeometrySupervision.__new__(GeometrySupervision)
    supervisor.family_ids = TRAIN_GEOMETRY_FAMILIES
    supervisor._pairs = tuple(pairs)
    supervisor._pair_by_key = {(pair.family_id, pair.label): pair for pair in pairs}
    supervisor.scales = _scales()
    supervisor.fluid_queries = 8
    supervisor.local_queries_per_module = 3
    supervisor.ntheta = 16
    supervisor.fluid_chunk_size = chunk_size
    supervisor.seed = 17
    return supervisor


def test_due_schedule_and_pair_cycle_are_bounded_to_single_geometry_labels():
    assert geometry_response_due(4)
    assert geometry_response_due(8)
    assert not geometry_response_due(1)
    assert not geometry_response_due(5)
    assert set(SINGLE_GEOMETRY_LABELS).isdisjoint(MIXED_GEOMETRY_LABELS)
    with pytest.raises(ValueError, match="positive one-based"):
        geometry_response_due(0)
    with pytest.raises(ValueError, match="restricted"):
        validate_training_family_metadata({"anchor_id": "0304"}, "0304")


def test_module_major_receiver_identity_is_checked_without_reindexing():
    ids = ("0001:module:0", "0001:module:1", "0001:module:2")
    expected = tuple(module_id for module_id in ids for _ in range(4))
    validate_module_major_receivers(ids, expected, 4)
    swapped = list(expected)
    swapped[4], swapped[8] = swapped[8], swapped[4]
    with pytest.raises(ValueError, match="module-major"):
        validate_module_major_receivers(ids, swapped, 4)
    with pytest.raises(ValueError, match="explicit receiver module IDs"):
        validate_module_major_receivers(ids, None, 4)


def test_common_masks_exclude_invalid_rows_and_component_scales_are_train_only():
    train_pairs = tuple(_pair(family, label, target_scale=0.25) for family in TRAIN_GEOMETRY_FAMILIES for label in SINGLE_GEOMETRY_LABELS)
    scales, detail = _component_scales(train_pairs)
    assert set(scales) == set(_scales())
    assert detail["fluid_fields/u"]["pair_count"] == 16
    assert detail["fluid_fields/u"]["equal_family_direction_rms"] == pytest.approx(0.25)
    assert train_pairs[0].roles["fluid_fields"].valid_mask.sum() < train_pairs[0].roles["fluid_fields"].valid_mask.size

    dev_pair = _pair(DEVELOPMENT_GEOMETRY_FAMILIES[0], SINGLE_GEOMETRY_LABELS[0], target_scale=100.0)
    with pytest.raises(ValueError, match="exactly four single-difference labels"):
        _component_scales((*train_pairs, dev_pair))


def test_invalid_masked_nan_targets_and_predictions_do_not_enter_loss_or_gradient():
    prediction = torch.tensor([[2.0], [float("nan")]], requires_grad=True)
    component_losses, counts = _component_sum(
        prediction,
        np.asarray([[1.0], [np.nan]]),
        np.asarray([[True], [False]]),
        {"fluid_fields/u": 1.0},
        "fluid_fields",
    )
    loss = component_losses["fluid_fields/u"] / counts["fluid_fields/u"]
    assert torch.isfinite(loss) and float(loss) == pytest.approx(1.0)
    loss.backward()
    assert torch.equal(prediction.grad, torch.tensor([[2.0], [0.0]]))


def test_mixed_geometry_label_is_rejected_by_the_training_pair_builder():
    with pytest.raises(ValueError, match="only i-/j-axis single-difference"):
        _make_pair(None, "mm", {}, {}, None, None, "", "")


def test_provider_atlas_source_and_material_query_alias(monkeypatch):
    pairs = tuple(_pair(family, label) for family in TRAIN_GEOMETRY_FAMILIES for label in SINGLE_GEOMETRY_LABELS)
    family_rows = tuple({"physical_family_id": pair.physical_family_id} for pair in pairs[::4])
    provider = SimpleNamespace(atlas_directory="atlas-root")
    assert _normalize_atlas_source(provider) == "atlas-root"
    assert _normalize_atlas_source("atlas-root") == "atlas-root"
    monkeypatch.setattr(geometry_module, "_build_pair_index", lambda source, family_ids=TRAIN_GEOMETRY_FAMILIES: (pairs, family_rows))
    supervisor = GeometrySupervision(provider, material_queries_per_module=3)
    assert supervisor.local_queries_per_module == 3


def test_native_loss_is_differentiable_in_eval_mode_and_restores_training_mode():
    pair = _pair()
    supervisor = _supervisor((pair,))
    model = _DifferentiableNativeAdapter().train()
    model.rarely_training_child = nn.Dropout()
    model.rarely_training_child.eval()
    loss, detail = supervisor.loss(model, 0)
    assert torch.isfinite(loss)
    assert model.training
    assert not model.rarely_training_child.training
    assert model.observed_modes and not any(model.observed_modes)
    assert detail["autograd_enabled"]
    assert detail["model_training_mode_restored"]
    assert detail["exposure"]["pair_count"] == 1
    assert detail["exposure"]["scene_count"] == 2
    assert detail["scene_visits"] == 2
    assert detail["query_rows"] == sum(detail["exposure"]["supervised_receiver_rows"].values())
    assert detail["native_query_rows"] == sum(detail["exposure"]["native_receiver_rows_evaluated_across_calls"].values())
    assert detail["loss_reduction"]["component_denominator"] == 8
    assert {f"fluid_fields/{name}" for name in ("u", "v", "p", "omega", "temperature")} <= set(detail["loss_reduction"]["components"])
    assert all("fluid_query_ids_sha256" in pair["panel"] for pair in detail["pairs"])
    loss.backward()
    assert model.amplitude.grad is not None
    assert torch.isfinite(model.amplitude.grad)
    assert float(model.amplitude.grad.abs()) > 0.0


def test_query_chunking_preserves_loss_and_parameter_gradient():
    pair = _pair()
    scales = _scales()
    whole_model, chunked_model = _DifferentiableNativeAdapter(), _DifferentiableNativeAdapter()
    chunked_model.load_state_dict(whole_model.state_dict())
    common = dict(
        schedule_index=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        fluid_queries=8,
        local_queries_per_module=3,
        ntheta=16,
        seed=17,
    )
    whole, *_ = _pair_panel_loss(whole_model, pair, scales, fluid_chunk_size=8, **common)
    chunked, *_ = _pair_panel_loss(chunked_model, pair, scales, fluid_chunk_size=3, **common)
    whole.backward()
    chunked.backward()
    assert torch.allclose(whole, chunked, rtol=1.0e-6, atol=1.0e-7)
    assert torch.allclose(whole_model.amplitude.grad, chunked_model.amplitude.grad, rtol=1.0e-6, atol=1.0e-7)


def test_calibration_uses_one_single_difference_per_train_family():
    pairs = tuple(_pair(family, label) for family in TRAIN_GEOMETRY_FAMILIES for label in SINGLE_GEOMETRY_LABELS)
    supervisor = _supervisor(pairs, chunk_size=8)
    model = _DifferentiableNativeAdapter()
    loss, detail = supervisor.loss(model, 2, calibration=True)
    assert torch.isfinite(loss)
    assert detail["family_ids"] == list(TRAIN_GEOMETRY_FAMILIES)
    assert detail["variant_labels"] == [SINGLE_GEOMETRY_LABELS[i % 4] for i in range(2, 6)]
    assert detail["exposure"]["pair_count"] == 4
    assert detail["exposure"]["scene_count"] == 8
    assert detail["optimizer_steps"] == 0
    assert detail["loss_reduction"]["pair_denominator"] == 4
    assert len(detail["loss_reduction"]["per_pair_component_valid_scalar_denominators"]) == 4
