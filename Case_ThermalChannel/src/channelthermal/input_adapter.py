"""CHANNELTHERMAL-SPECIFIC physical-input adapter.

Inputs are ChannelThermal physical tensors: Reynolds number, inlet velocity,
module centers, module heat powers, module-present mask, and material
parameters. Outputs are generic HONF `global_context`, `module_features`,
module centers, and module-present tensors. This module is specific to
ChannelThermal and is not reusable across domains without replacing the
feature definitions below.

Module feature columns:
0. dataset-scaled heat power as provided by the dataset
1. absolute dataset-scaled heat power
2. signed case-relative heat, divided by max active absolute heat in the case
3. absolute case-relative heat
4. active module flag
5. solid thermal diffusivity descriptor
6. fluid thermal diffusivity descriptor
7. solid conductivity descriptor
8. fluid conductivity descriptor
9. module radius descriptor
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class ChannelThermalAdapterOutput:
    """Generic HONF tensors derived from one batch of physical case inputs."""

    global_context: torch.Tensor
    module_features: torch.Tensor
    module_centers: torch.Tensor
    module_present: torch.Tensor
    heat_powers: torch.Tensor


class ChannelThermalInputAdapter:
    """Convert physical ChannelThermal inputs to generic HONF tensors."""

    feature_names = (
        "heat_dataset_scaled",
        "abs_heat_dataset_scaled",
        "heat_case_relative",
        "abs_heat_case_relative",
        "active_flag",
        "solid_alpha",
        "fluid_alpha",
        "solid_k",
        "fluid_k",
        "module_radius",
    )

    legacy_global_context_names = (
        "re",
        "u_in",
        "active_module_fraction",
        "total_dataset_scaled_heat",
        "mean_active_dataset_scaled_heat",
        "max_abs_dataset_scaled_heat",
        "domain_length_x",
        "domain_length_y",
        "nu",
        "solid_alpha",
        "fluid_alpha",
        "solid_k",
        "fluid_k",
        "module_radius",
    )

    padding_invariant_global_context_names = (
        "re",
        "u_in",
        "active_module_count",
        "log1p_active_module_count",
        "module_number_density",
        "occupied_area_fraction",
        "total_dataset_scaled_heat",
        "total_dataset_scaled_heat_per_domain_area",
        "mean_active_dataset_scaled_heat",
        "max_abs_dataset_scaled_heat",
        "domain_length_x",
        "domain_length_y",
        "nu",
        "solid_alpha",
        "fluid_alpha",
        "solid_k",
        "fluid_k",
        "module_radius",
    )

    global_context_names = padding_invariant_global_context_names
    source_local_feature_names = tuple(
        {"heat_case_relative": "heat_train_scaled",
         "abs_heat_case_relative": "abs_heat_train_scaled"}.get(name, name)
        for name in feature_names
    )
    # Keep the existing input width so common physical tensors can receive the
    # same initializer. These four named slots are exact zero, never heat.
    source_local_global_context_names = tuple(
        "reserved_no_heat_" + name if "heat" in name else name
        for name in padding_invariant_global_context_names
    )

    def __init__(
        self,
        *,
        global_feature_schema: str = "padding_invariant_v2",
        legacy_active_fraction_reference_slots: int | None = None,
        fixed_heat_scale: float | None = None,
    ) -> None:
        """Select the padding-invariant schema or the legacy checkpoint transform."""

        self.global_feature_schema = str(global_feature_schema)
        self.legacy_active_fraction_reference_slots = legacy_active_fraction_reference_slots
        self.fixed_heat_scale = fixed_heat_scale
        if self.global_feature_schema == "legacy_v1":
            if (
                legacy_active_fraction_reference_slots is None
                or int(legacy_active_fraction_reference_slots) <= 0
            ):
                raise ValueError("legacy_v1 requires a positive fixed active-fraction reference.")
            self.global_context_names = self.legacy_global_context_names
        elif self.global_feature_schema == "padding_invariant_v2":
            self.global_context_names = self.padding_invariant_global_context_names
        elif self.global_feature_schema == "source_local_v3":
            if fixed_heat_scale is None or not math.isfinite(float(fixed_heat_scale)) or float(fixed_heat_scale) <= 0:
                raise ValueError("source_local_v3 requires a finite positive training-fitted fixed_heat_scale.")
            self.global_context_names = self.source_local_global_context_names
            self.feature_names = self.source_local_feature_names
        else:
            raise ValueError(f"Unsupported ChannelThermal global feature schema: {self.global_feature_schema!r}.")
        self.content_context_names = tuple(name for name in self.global_context_names if not name.startswith("reserved_no_heat_"))

    def __call__(
        self,
        *,
        re: torch.Tensor,
        u_in: torch.Tensor,
        module_centers: torch.Tensor,
        heat_powers: torch.Tensor,
        module_present: torch.Tensor,
        material_params: torch.Tensor,
        domain_length_x: torch.Tensor | None = None,
        domain_length_y: torch.Tensor | None = None,
    ) -> ChannelThermalAdapterOutput:
        """Map physical inputs to module features and padding-invariant global context."""

        module_centers = module_centers.float()
        heat_powers = heat_powers.float()
        module_present = module_present.float()
        re = self._as_batch_column(re, module_centers).float()
        u_in = self._as_batch_column(u_in, module_centers).float()
        material_params = self._as_material(material_params, module_centers)
        batch, num_modules = heat_powers.shape
        active = module_present.clamp(0.0, 1.0)
        active_count_raw = active.sum(dim=1, keepdim=True)
        active_count = active_count_raw.clamp_min(1.0)
        heat_active = heat_powers * active
        max_abs = heat_active.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-6)
        denominator = heat_powers.new_tensor(float(self.fixed_heat_scale)) if self.global_feature_schema == "source_local_v3" else max_abs
        heat_case_relative = heat_powers / denominator
        abs_heat_case_relative = heat_case_relative.abs()

        mat = self._pad_material(material_params, 6)
        descriptors = torch.stack([mat[:, 1], mat[:, 2], mat[:, 3], mat[:, 4], mat[:, 5]], dim=-1)
        descriptor_features = descriptors[:, None, :].expand(batch, num_modules, -1)
        module_features = torch.cat(
            [
                heat_powers[..., None],
                heat_powers.abs()[..., None],
                heat_case_relative[..., None],
                abs_heat_case_relative[..., None],
                active[..., None],
                descriptor_features,
            ],
            dim=-1,
        )
        module_features = module_features * active[..., None]

        lx = self._optional_batch_column(domain_length_x, module_centers, fallback=12.0)
        ly = self._optional_batch_column(domain_length_y, module_centers, fallback=4.0)
        total_heat = heat_active.sum(dim=1, keepdim=True)
        mean_heat = total_heat / active_count
        if self.global_feature_schema == "legacy_v1":
            reference = active.new_tensor(float(self.legacy_active_fraction_reference_slots))
            global_context = torch.cat(
                [re, u_in, active_count_raw / reference, total_heat, mean_heat, max_abs, lx, ly, mat[:, 0:6]],
                dim=-1,
            )
        else:
            domain_area = (lx * ly).clamp_min(1.0e-6)
            module_radius = mat[:, 5:6].clamp_min(0.0)
            occupied_area_fraction = active_count_raw * torch.pi * module_radius.square() / domain_area
            global_context = torch.cat(
                [
                    re,
                    u_in,
                    active_count_raw,
                    torch.log1p(active_count_raw),
                    active_count_raw / domain_area,
                    occupied_area_fraction,
                    total_heat,
                    total_heat / domain_area,
                    mean_heat,
                    max_abs,
                    lx,
                    ly,
                    mat[:, 0:6],
                ],
                dim=-1,
            )
            if self.global_feature_schema == "source_local_v3":
                # Select by the public schema, not fragile positional guesses.
                allowed = global_context.new_tensor([
                    float("heat" not in name) for name in self.padding_invariant_global_context_names
                ])
                global_context = global_context * allowed
        return ChannelThermalAdapterOutput(
            global_context=global_context,
            module_features=module_features,
            module_centers=module_centers,
            module_present=module_present,
            heat_powers=heat_powers,
        )

    @staticmethod
    def _as_batch_column(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Convert a scalar or ``[B]`` value to a device-matched ``[B,1]`` column."""

        value = value.to(device=like.device, dtype=like.dtype)
        if value.ndim == 0:
            value = value.view(1, 1).expand(like.shape[0], 1)
        elif value.ndim == 1:
            value = value[:, None]
        return value

    @staticmethod
    def _optional_batch_column(value: torch.Tensor | None, like: torch.Tensor, fallback: float) -> torch.Tensor:
        """Convert an optional case value to ``[B,1]``, filling a default if absent."""

        if value is None:
            return like.new_full((like.shape[0], 1), float(fallback))
        return ChannelThermalInputAdapter._as_batch_column(value, like)

    @staticmethod
    def _as_material(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Broadcast a shared material vector to ``[B,D]`` and match dtype/device."""

        value = value.to(device=like.device, dtype=like.dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0).expand(like.shape[0], -1)
        return value

    @staticmethod
    def _pad_material(value: torch.Tensor, width: int) -> torch.Tensor:
        """Truncate or right-pad material features to the requested width."""

        if value.shape[-1] >= width:
            return value[..., :width]
        pad = value.new_zeros(*value.shape[:-1], width - value.shape[-1])
        return torch.cat([value, pad], dim=-1)


def fit_source_local_heat_scale(packed_h5_path, train_case_ids, normalizer=None) -> float:
    """Maximum active absolute heat on the declared training membership only.

    The adapter consumes dataset-scaled heat; provide the training normalizer
    exactly when the dataset normalizes inputs. Validation never enters this
    fit. The resulting scalar belongs in the run/checkpoint configuration.
    """
    import h5py
    import numpy as np

    ids = tuple(str(case_id) for case_id in train_case_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("A nonempty unique training membership is required to fit heat scale.")
    maximum = 0.0
    with h5py.File(packed_h5_path, "r") as handle:
        for case_id in ids:
            group = handle["cases"][case_id]
            heat = np.asarray(group["heat_powers"][...], dtype=np.float64)
            active = np.asarray(group["module_present"][...]) > .5
            if normalizer is not None:
                heat = normalizer.normalize_heat_power(heat)
            values = np.abs(heat[active])
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite training heat in selected case {case_id}.")
            if values.size:
                maximum = max(maximum, float(values.max()))
    return max(maximum, 1.0e-6)
