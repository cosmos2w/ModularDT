from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from channelthermal.interaction_evidence.response_dataset import ResponseBlock
from channelthermal.interaction_evidence.types import RoleOutput
from torch import nn

from honf_runtime import contextual_frontier_responses as responses

_FAMILIES = ("0001", "0318", "0333", "0348", "0304", "0320", "0335", "0350")
_LABELS = (
    "i_minus", "i_plus", "j_minus", "j_plus", "mm", "mp", "pm", "pp",
    "heat_transfer_plus", "heat_transfer_minus",
)
_CHANNELS = {
    "fluid_fields": (("u", "v", "p", "omega", "temperature"),
                     ("velocity", "velocity", "pressure", "vorticity", "temperature")),
    "interface": (("T_surface", "q_normal"), ("temperature", "heat flux")),
    "solid_temperature": (("temperature",), ("temperature",)),
}


class _FakeStencil:
    def __init__(self, family_id: str):
        self.anchor_id = family_id
        self.physical_family_id = "duplicate_family:0001+0273" if family_id == "0001" else f"stored_family:{family_id}"
        self.split = "train"
        self.source = "reference_solver"
        self.module_ids = (f"{family_id}:module:0",)
        self.fluid_xy = np.stack((np.linspace(0.0, 1.0, 300), np.linspace(1.0, 0.0, 300)), axis=-1)
        self.theta = (np.arange(64, dtype=np.float64) * 2.0 * np.pi / 64.0)[:, None]
        local_axis = np.linspace(-0.8, 0.8, 8)
        local_x, local_y = np.meshgrid(local_axis, local_axis, indexing="xy")
        self.local_xy = np.stack((local_x.reshape(-1), local_y.reshape(-1)), axis=-1)
        self.baseline = self._record("baseline", 0.0, 0.0, 0.0)
        if family_id == "0335":
            changes = {
                "i_minus": (0.0, -0.15, 0.0), "i_plus": (0.0, 0.15, 0.0),
                "j_minus": (-0.15, 0.0, 0.0), "j_plus": (0.15, 0.0, 0.0),
                "mm": (-0.15, -0.15, 0.0), "mp": (0.15, -0.15, 0.0),
                "pm": (-0.15, 0.15, 0.0), "pp": (0.15, 0.15, 0.0),
                "heat_transfer_plus": (0.0, 0.0, 0.2), "heat_transfer_minus": (0.0, 0.0, -0.2),
            }
        else:
            changes = {
                "i_minus": (-0.15, 0.0, 0.0), "i_plus": (0.15, 0.0, 0.0),
                "j_minus": (0.0, -0.15, 0.0), "j_plus": (0.0, 0.15, 0.0),
                "mm": (-0.15, -0.15, 0.0), "mp": (-0.15, 0.15, 0.0),
                "pm": (0.15, -0.15, 0.0), "pp": (0.15, 0.15, 0.0),
                "heat_transfer_plus": (0.0, 0.0, 0.2), "heat_transfer_minus": (0.0, 0.0, -0.2),
            }
        self.variants = {
            label: self._record(label, dx, dy, dh)
            for label, (dx, dy, dh) in changes.items()
        }

    @staticmethod
    def _context():
        return SimpleNamespace(values={
            "nu": 1.0, "solid_alpha": 1.0, "fluid_alpha": 1.0,
            "solid_k": 1.0, "fluid_k": 1.0, "module_radius": 0.1,
            "re": 100.0, "u_in": 1.0, "domain_length_x": 10.0, "domain_length_y": 5.0,
        })

    def _record(self, label: str, dx: float, dy: float, dh: float):
        center = np.asarray([[3.0 + dx, 2.0 + dy]], dtype=np.float32)
        heating = np.asarray([1.0 + dh], dtype=np.float32)
        module = SimpleNamespace(position_xy=tuple(center[0]), heating=float(heating[0]))
        design = SimpleNamespace(active_modules=(module,), active_module_ids=self.module_ids)
        roles = self._roles(center[0], float(heating[0]), label)
        peaks = {self.module_ids[0]: float(np.max(roles["solid_temperature"].values[:, 0]))}
        output = SimpleNamespace(roles=roles, module_peak_temperature=peaks)
        return SimpleNamespace(design=design, context=self._context(), output=output)

    def _roles(self, center: np.ndarray, heat: float, label: str):
        fluid = _native_formula(self.fluid_xy, center, heat)["fluid_fields"]
        interface = _native_formula(self.theta, center, heat, interface=True)["interface"]
        material = _native_formula(self.local_xy, center, heat, material=True)["solid_temperature"]
        if label == "i_plus":
            fluid = fluid.copy()
            fluid[:, 0] += 0.5
        role_values = {
            "fluid_fields": (self.fluid_xy, fluid, None),
            "interface": (self.theta, interface, self.module_ids * 64),
            "solid_temperature": (self.local_xy, material, self.module_ids * len(self.local_xy)),
        }
        result = {}
        for role, (features, values, receiver_ids) in role_values.items():
            names, units = _CHANNELS[role]
            valid = np.ones_like(values, dtype=bool)
            if role == "fluid_fields":
                valid[-1, :] = False
            result[role] = RoleOutput(
                role=role,
                query_features=features,
                values=values,
                channel_names=names,
                channel_units=units,
                valid_mask=valid,
                quadrature_weights=np.ones(len(features), dtype=np.float64),
                query_ids=tuple(f"{role}:{index}" for index in range(len(features))),
                receiver_module_ids=receiver_ids,
                coordinate_kind={
                    "fluid_fields": "eulerian",
                    "interface": "interface_material_angle",
                    "solid_temperature": "solid_material_normalized_xy",
                }[role],
            )
        return result

    def common_mask(self, role: str):
        return np.logical_and.reduce([record.output.roles[role].valid_mask for record in (self.baseline, *self.variants.values())])

    def finite_change(self, label: str, role: str):
        before = self.baseline.output.roles[role]
        after = self.variants[label].output.roles[role]
        valid = self.common_mask(role)
        delta = after.values - before.values
        delta = np.where(valid, delta, np.nan)
        return ResponseBlock(
            role=role, label=label, query_features=before.query_features,
            delta=delta, valid_mask=valid, quadrature_weights=before.quadrature_weights,
            channel_names=before.channel_names, channel_units=before.channel_units,
            query_ids=before.query_ids, receiver_module_ids=before.receiver_module_ids,
        )

    def centered_mixed_difference(self, *, role, joint_pp, joint_pm, joint_mp, joint_mm,
                                  first_step, second_step):
        outputs = [self.variants[name].output.roles[role] for name in (joint_pp, joint_pm, joint_mp, joint_mm)]
        pp, pm, mp, mm = [output.values for output in outputs]
        valid = self.common_mask(role)
        delta = np.where(valid, (pp - pm - mp + mm) / (4.0 * first_step * second_step), np.nan)
        baseline = self.baseline.output.roles[role]
        return ResponseBlock(
            role=role, label="centered_mixed_derivative", query_features=baseline.query_features,
            delta=delta, valid_mask=valid, quadrature_weights=baseline.quadrature_weights,
            channel_names=baseline.channel_names, channel_units=baseline.channel_units,
            query_ids=baseline.query_ids, receiver_module_ids=baseline.receiver_module_ids,
        )


def _native_formula(coordinates: np.ndarray, center: np.ndarray, heat: float, *, interface=False, material=False):
    if interface:
        theta = coordinates[:, 0]
        cx, cy = center
        return {"interface": np.stack((heat + cx + np.cos(theta), heat * cx + cy + np.sin(theta)), axis=-1)}
    if material:
        local_x, local_y = coordinates.T
        cx, cy = center
        temperature = heat + cx + 2.0 * cy + local_x + local_y + cx * cy
        return {"solid_temperature": temperature[:, None]}
    x, y = coordinates.T
    cx, cy = center
    return {"fluid_fields": np.stack((
        x + cx + 0.5 * heat,
        y + cy,
        x * y + cx * cy + heat,
        cx * cx + cy * cy + x - y,
        heat + cx + cy + 0.2 * x + 0.3 * y,
    ), axis=-1)}


class _FakeModel(nn.Module):
    requires_control_in_prepare = True

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.register_buffer("field_std", torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.register_buffer("interface_target_std", torch.tensor([6.0, 7.0]))
        self.register_buffer("internal_temperature_std", torch.tensor([8.0]))
        self.prepared_labels = []
        self.core = SimpleNamespace(set_organization_clock=lambda *_args: None)

    def prepare_native(self, structure, fluid_xy, *, physical_heat, local_query_points, ntheta, scene_key):
        self.prepared_labels.append(scene_key[-1])
        return {
            "structure": structure,
            "heat": physical_heat,
            "fluid_xy": fluid_xy,
            "local_xy": local_query_points,
            "ntheta": ntheta,
        }

    def apply_native(self, prepared, physical_heat):
        center = prepared["structure"]["module_centers"][0]
        heat = physical_heat[0]
        fluid = _torch_formula(prepared["fluid_xy"][0], center[0], heat[0], role="fluid")
        theta = torch.arange(prepared["ntheta"], dtype=center.dtype, device=center.device) * (2.0 * torch.pi / prepared["ntheta"])
        surface = _torch_formula(theta[:, None], center[0], heat[0], role="surface")
        local = _torch_formula(prepared["local_xy"][0], center[0], heat[0], role="material")
        return {
            "pred_field": fluid[None] * self.weight,
            "pred_interface": surface[None, None] * self.weight,
            "pred_internal_temperature": local[None, None] * self.weight,
        }


def _torch_formula(coordinates, center, heat, *, role):
    if role == "surface":
        theta = coordinates[:, 0]
        cx, cy = center
        return torch.stack((heat + cx + torch.cos(theta), heat * cx + cy + torch.sin(theta)), dim=-1)
    if role == "material":
        local_x, local_y = coordinates.T
        cx, cy = center
        return (heat + cx + 2.0 * cy + local_x + local_y + cx * cy)[:, None]
    x, y = coordinates.T
    cx, cy = center
    return torch.stack((
        x + cx + 0.5 * heat,
        y + cy,
        x * y + cx * cy + heat,
        cx * cx + cy * cy + x - y,
        heat + cx + cy + 0.2 * x + 0.3 * y,
    ), dim=-1)


def test_physical_response_evaluation_keeps_cohorts_and_endpoint_support_separate(tmp_path, monkeypatch):
    atlas_root = tmp_path / "atlas"
    atlas_root.mkdir()
    stencils = {}
    for family_id in _FAMILIES:
        (atlas_root / f"train_{family_id}_responses.npz").touch()
        stencils[family_id] = _FakeStencil(family_id)
    monkeypatch.setattr(responses, "load_response_atlas_stencil", lambda path: (
        stencils[Path(path).stem.split("_")[1]], {
            "anchor_id": Path(path).stem.split("_")[1],
            "family_id": stencils[Path(path).stem.split("_")[1]].physical_family_id,
            "split": "train",
        },
    ))
    monkeypatch.setattr(responses, "_filesystem_free_bytes", lambda _path: 2**40)

    model = _FakeModel()
    provider = SimpleNamespace(
        atlas_directory=atlas_root,
        responses=SimpleNamespace(
            family_ids=("0348",),
            floors={
                "fluid_fields/u": 1.0e-3,
                "fluid_fields/v": 2.0e-3,
                "fluid_fields/p": 3.0e-3,
                "fluid_fields/omega": 4.0e-3,
                "fluid_fields/temperature": 5.0e-3,
                "interface/T_surface": 6.0e-3,
                "interface/q_normal": 7.0e-3,
                "solid_temperature/temperature": 8.0e-3,
            },
        ),
    )
    payload = {
        "epoch": 100,
        "arm": "J-D",
        "experiment_identity": {
            "workflow": "contextual_frontier_joint_training_v1",
            "config_name": "test",
            "provider_identity": {
                "architecture": "J-D",
                "recipe": {"storage": {"minimum_free_headroom_bytes": 0}},
                "subset_manifest_sha256": "test-manifest",
            },
        },
    }
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    monkeypatch.setattr(responses, "load_contextual_checkpoint", lambda *_args: (model, provider, payload))

    result = responses.evaluate_responses(checkpoint, "cpu", tmp_path / "response-eval")

    assert result["status"] == "completed"
    assert result["response_cohort_contract"]["fit_train_family_ids"] == ["0348"]
    assert result["native_floor_basis"]["admitted_response_fit_family_ids"] == ["0348"]
    assert result["native_floor_basis"]["validation_response_atlases_used_for_floor"] is False
    assert result["response_summaries"]["train_reference_only"]["family_ids"] == ["0001", "0318", "0333"]
    assert result["response_summaries"]["response_validation_only"]["family_ids"] == ["0304", "0320", "0335", "0350"]
    assert set(result["representative_family_ids"]).intersection({"0304", "0320", "0335", "0350"})
    fit = result["families"]["0348"]
    assert fit["query_panel"]["fluid_query_count"] == 256
    assert fit["query_panel"]["surface_theta_per_module"] == 16
    assert fit["query_panel"]["material_local_points_per_module"] == 64
    assert fit["responses"]["i_plus"]["components"]["fluid_fields/u"]["absolute_rmse"] == pytest.approx(0.5)
    assert fit["responses"]["mixed_central_derivative"]["kind"] == "centered_mixed_geometry_derivative"
    assert fit["responses"]["mixed_raw_four_corner_contrast"]["kind"] == "centered_mixed_geometry_four_corner_contrast"
    assert fit["responses"]["mixed_central_derivative"]["mixed_step_denominator"]["value"] == pytest.approx(0.09)
    derivative_unit = fit["responses"]["mixed_central_derivative"]["components"]["fluid_fields/u"]["unit"]
    contrast_unit = fit["responses"]["mixed_raw_four_corner_contrast"]["components"]["fluid_fields/u"]["unit"]
    assert "per saved position-coordinate unit squared" in derivative_unit
    assert contrast_unit == "velocity"
    assert result["families"]["0335"]["mixed_geometry_steps"]["i_coordinate_axis"] == "y"
    assert result["families"]["0335"]["mixed_geometry_steps"]["j_coordinate_axis"] == "x"
    assert fit["responses"]["i_plus"]["components"]["module_peak_endpoint/temperature"]["endpoint_basis"].startswith(
        "max over all common-valid saved local material points"
    )
    assert "zero_heat" not in fit["responses"]
    assert len(model.prepared_labels) == 8 * (1 + len(_LABELS))

    summary_path = tmp_path / "response-eval" / "response_summary.json"
    panel_path = tmp_path / "response-eval" / "response_delta_panels.npz"
    saved = json.loads(summary_path.read_text())
    assert saved["optimizer_updates"] == 0
    assert any(item["path"].startswith("mixed_central_") for item in saved["plots"])
    with np.load(panel_path, allow_pickle=False) as archive:
        assert "family_0001__response_i_plus__fluid_fields_prediction_delta" in archive
        assert "family_0304__response_mixed_central_derivative__fluid_fields_reference_delta" in archive
        assert "family_0001__response_mixed_central_derivative__module_peak_endpoint_reference_delta" in archive
        assert "family_0001__response_mixed_raw_four_corner_contrast__fluid_fields_reference_delta" in archive


def test_component_metric_suppresses_relative_error_below_native_floor():
    metric = responses._component_metric(
        np.asarray([1.0, 1.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([True, True]),
        unit="native",
        native_floor=1.0e-4,
    )
    assert metric["absolute_mae"] == pytest.approx(1.0)
    assert metric["absolute_rmse"] == pytest.approx(1.0)
    assert metric["relative_rmse_to_reference_rms"] is None
    assert metric["relative_metric_applicable"] is False


def test_development_relative_floors_reject_response_validation_families():
    stencil = _FakeStencil("0348")
    provider = SimpleNamespace(
        protocol="development",
        responses=SimpleNamespace(
            family_ids=("0001",),
            floors={
                "fluid_fields/u": 1.0e-3,
                "fluid_fields/v": 2.0e-3,
                "fluid_fields/p": 3.0e-3,
                "fluid_fields/omega": 4.0e-3,
                "fluid_fields/temperature": 5.0e-3,
                "interface/T_surface": 6.0e-3,
                "interface/q_normal": 7.0e-3,
                "solid_temperature/temperature": 8.0e-3,
            },
        ),
    )
    with pytest.raises(ValueError, match="selected-TRAIN family 0348 only"):
        responses._response_native_floors(_FakeModel(), provider, stencil.baseline.output.roles)


def test_full_train_response_cohorts_follow_all_admitted_train_families():
    full_fit_ids = ("0001", "0318", "0333", "0348")
    stencil = _FakeStencil("0348")
    provider = SimpleNamespace(
        protocol="contextual_full_train100",
        responses=SimpleNamespace(
            family_ids=full_fit_ids,
            floors={
                "fluid_fields/u": 1.0e-3,
                "fluid_fields/v": 2.0e-3,
                "fluid_fields/p": 3.0e-3,
                "fluid_fields/omega": 4.0e-3,
                "fluid_fields/temperature": 5.0e-3,
                "interface/T_surface": 6.0e-3,
                "interface/q_normal": 7.0e-3,
                "solid_temperature/temperature": 8.0e-3,
            },
        ),
    )
    _floors, admitted_fit_ids, _source = responses._response_native_floors(
        _FakeModel(), provider, stencil.baseline.output.roles,
    )
    assert admitted_fit_ids == full_fit_ids
    cohorts = responses._cohort_plan(admitted_fit_ids)
    assert cohorts == {
        "fit_train": list(full_fit_ids),
        "response_validation_only": ["0304", "0320", "0335", "0350"],
    }
    family_results = {
        family_id: {"module_count": index + 1}
        for index, family_id in enumerate((*full_fit_ids, "0304", "0320", "0335", "0350"))
    }
    family_metrics = {
        family_id: {"i_plus": {"fluid_fields/u": {
            "status": "valid",
            "absolute_mae": 0.1,
            "absolute_rmse": 0.2,
            "reference_rms": 1.0,
            "prediction_rms": 1.0,
            "relative_rmse_to_reference_rms": 0.2,
            "relative_metric_applicable": True,
        }}}
        for family_id in family_results
    }
    summary = responses._summarize_families(family_results, family_metrics, full_fit_ids)
    assert tuple(summary) == ("fit_train", "response_validation_only")
    assert summary["fit_train"]["family_ids"] == list(full_fit_ids)
    assert summary["response_validation_only"]["family_ids"] == ["0304", "0320", "0335", "0350"]
    assert responses._cohort_for_family("0348", full_fit_ids) == "fit_train"
    assert responses._cohort_for_family("0350", full_fit_ids) == "response_validation_only"
    assert "train_reference_only" not in summary
    representatives = responses._representative_families(family_results, full_fit_ids, limit=3)
    assert set(representatives).intersection(full_fit_ids)
    assert set(representatives).intersection({"0304", "0320", "0335", "0350"})
    assert responses._cohort_plan(("0348",))["train_reference_only"] == ["0001", "0318", "0333"]


def test_nonfinite_prediction_invalidates_metrics_instead_of_shrinking_denominator():
    metric = responses._component_metric(
        np.asarray([1.0, np.nan]),
        np.asarray([1.0, 1.0]),
        np.asarray([True, True]),
        unit="native",
        native_floor=1.0e-4,
    )
    assert metric["status"] == "invalid_nonfinite_prediction"
    assert metric["valid_scalar_count"] == 2
    assert metric["nonfinite_prediction_count"] == 1
    assert metric["absolute_mae"] is None
    assert metric["absolute_rmse"] is None
    assert metric["relative_rmse_to_reference_rms"] is None
    assert responses._jsonable({"invalid_value": float("nan")}) == {"invalid_value": None}


def test_saved_zero_heat_control_is_included_only_when_its_input_is_actually_zero():
    stencil = _FakeStencil("0348")
    stencil.variants["zero_heat"] = stencil._record("zero_heat", 0.0, 0.0, -1.0)
    details = responses._saved_control_details(stencil)
    assert details["zero_heat_controls"] == {"zero_heat": {"all_module_heating_is_zero": True}}

    stencil.variants["zero_heat"] = stencil._record("zero_heat", 0.0, 0.0, -0.5)
    with pytest.raises(ValueError, match="zero-heat"):
        responses._saved_control_details(stencil)


def test_response_output_budget_is_checked_before_creating_the_output(tmp_path):
    output = tmp_path / "out"
    with pytest.raises(OSError, match="cap"):
        responses._write_artifacts(
            output,
            {"summary": True, "representative_family_ids": []},
            b"x" * 32,
            {},
            max_output_bytes=8,
        )
    assert not output.exists()
