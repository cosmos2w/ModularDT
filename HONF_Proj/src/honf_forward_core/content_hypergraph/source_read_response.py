"""Source-factorized affine Thermal response for Round 3 source reads.

The ordinary application and transpose paths use selected flat pairs and, for
completion readers, rank-16 query/source factors. They never construct a
``[B,Q,M]`` kernel. ``explicit_dense_kernel`` is the opt-in materialization
path for export and numerical comparison only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor


_RANK = 16
_SCHEMA = "source_read_affine_factors_v1"


def _require(condition: Tensor, message: str) -> None:
    if condition.is_cuda:
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _is_float_dtype(dtype: torch.dtype) -> bool:
    return bool(getattr(dtype, "is_floating_point", False))


def _tensor_mutation_version(value: Tensor) -> int:
    try:
        return int(value._version)
    except RuntimeError:
        # One-shot inference tensors have no mutation counter. Retained plan
        # owners must require versioned inputs for reusable response plans.
        return -1


def _positive_scalar(value: float | Tensor, name: str, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    scalar = torch.as_tensor(value, device=device, dtype=dtype)
    if scalar.numel() != 1 or not _is_float_dtype(scalar.dtype):
        raise ValueError(f"{name} must be a floating-point scalar.")
    scalar = scalar.reshape(())
    _require(torch.isfinite(scalar) & (scalar > 0), f"{name} must be finite and positive.")
    return scalar


def _scalar(value: float | Tensor, name: str, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    scalar = torch.as_tensor(value, device=device, dtype=dtype)
    if scalar.numel() != 1 or not _is_float_dtype(scalar.dtype):
        raise ValueError(f"{name} must be a floating-point scalar.")
    scalar = scalar.reshape(())
    _require(torch.isfinite(scalar), f"{name} must be finite.")
    return scalar


@dataclass(frozen=True)
class SourceReadAffineFactorsV1:
    """A sparse selected-pair response with an optional rank-16 completion.

    ``residual_coefficients`` stores the final signed coefficient for each
    selected ``(batch, query, source)`` row. In S-Utility it is formed as
    ``g * (wK_fine + bK)``. In S-Complete it is the residual
    ``g * wK * (fine - u0)``; the completion factors carry the base response,
    and ``scalar_bias`` contributes ``bK * sum(present_heat)`` exactly once.

    Controls are physical endpoint heat values satisfying ``abs(h) <= heating_envelope``
    for present sources. Set ``check_envelope=False`` only when applying a
    linear tangent/JVP, whose values need not lie in the endpoint envelope.
    Absent sources and invalid queries contribute exactly zero.
    """

    batch_size: int
    query_count: int
    source_count: int
    selected_batch: Tensor
    selected_query: Tensor
    selected_source: Tensor
    residual_coefficients: Tensor
    present_mask: Tensor
    heating_envelope: float | Tensor
    query_mask: Tensor | None = None
    completion_query: Tensor | None = None
    completion_source: Tensor | None = None
    thermal_rank_weight: Tensor | None = None
    scalar_bias: float | Tensor = 0.0

    def __post_init__(self) -> None:
        dimensions = (self.batch_size, self.query_count, self.source_count)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in dimensions):
            raise ValueError("batch_size, query_count, and source_count must be nonnegative integers.")
        if self.residual_coefficients.ndim != 1 or not self.residual_coefficients.is_floating_point():
            raise ValueError("residual_coefficients must be a floating tensor with shape [P].")
        _require(torch.isfinite(self.residual_coefficients).all(), "Response coefficients must be finite.")
        device = self.residual_coefficients.device
        pair_count = self.residual_coefficients.numel()
        for name, indices, bound in (
            ("selected_batch", self.selected_batch, self.batch_size),
            ("selected_query", self.selected_query, self.query_count),
            ("selected_source", self.selected_source, self.source_count),
        ):
            if indices.ndim != 1 or indices.dtype != torch.long or indices.numel() != pair_count:
                raise ValueError(f"{name} must be rank-1 int64 with the same length as residual_coefficients.")
            if indices.device != device:
                raise ValueError("Selected indices and response coefficients must share a device.")
            if indices.numel():
                _require(((indices >= 0) & (indices < bound)).all(),
                         f"{name} contains an index outside its declared axis.")
        if tuple(self.present_mask.shape) != (self.batch_size, self.source_count):
            raise ValueError("present_mask must have shape [B,M].")
        if self.present_mask.dtype != torch.bool or self.present_mask.device != device:
            raise ValueError("present_mask must be boolean and share the response device.")
        if self.query_mask is not None:
            if tuple(self.query_mask.shape) != (self.batch_size, self.query_count):
                raise ValueError("query_mask must have shape [B,Q].")
            if self.query_mask.dtype != torch.bool or self.query_mask.device != device:
                raise ValueError("query_mask must be boolean and share the response device.")

        completion_values = (self.completion_query, self.completion_source, self.thermal_rank_weight)
        has_completion = tuple(value is not None for value in completion_values)
        if any(has_completion) and not all(has_completion):
            raise ValueError("Completion query, source, and thermal rank factors must be supplied together.")
        if all(has_completion):
            assert self.completion_query is not None
            assert self.completion_source is not None
            assert self.thermal_rank_weight is not None
            expected_shapes = (
                (self.completion_query, (self.batch_size, self.query_count, _RANK), "completion_query"),
                (self.completion_source, (self.batch_size, self.source_count, _RANK), "completion_source"),
                (self.thermal_rank_weight, (_RANK,), "thermal_rank_weight"),
            )
            for value, shape, name in expected_shapes:
                if tuple(value.shape) != shape or not value.is_floating_point():
                    raise ValueError(f"{name} must be a floating tensor with shape {shape}.")
                if value.device != device:
                    raise ValueError("All response factors must share a device.")
                _require(torch.isfinite(value).all(), f"{name} must be finite.")
        else:
            # Utility coefficients already include their per-selected-source
            # scalar output bias; a second full-source bias would double count.
            raw_bias = torch.as_tensor(self.scalar_bias, device=device, dtype=self.residual_coefficients.dtype)
            if raw_bias.numel() != 1:
                raise ValueError(
                    "Without completion factors, scalar_bias must be zero; "
                    "utility bias is in the selected coefficients."
                )
            _require((torch.isfinite(raw_bias) & (raw_bias == 0)).all(),
                     "Without completion factors, scalar_bias must be zero; utility bias is in the selected coefficients.")

        bias = _scalar(
            self.scalar_bias,
            "scalar_bias",
            device=device,
            dtype=self.residual_coefficients.dtype,
        )
        envelope = _positive_scalar(
            self.heating_envelope,
            "heating_envelope",
            device=device,
            dtype=self.residual_coefficients.dtype,
        )
        object.__setattr__(self, "scalar_bias", bias)
        object.__setattr__(self, "heating_envelope", envelope)

        if self.selected_batch.numel():
            _require(self.present_mask[self.selected_batch, self.selected_source].all(),
                     "Selected response rows must refer to present sources.")
            if self.query_mask is not None:
                _require(self.query_mask[self.selected_batch, self.selected_query].all(),
                         "Selected response rows must refer to valid queries.")

    @property
    def has_completion(self) -> bool:
        return self.completion_query is not None

    @property
    def device(self) -> torch.device:
        return self.residual_coefficients.device

    def _tensor_items(self) -> Iterator[tuple[str, Tensor]]:
        for name in (
            "selected_batch",
            "selected_query",
            "selected_source",
            "residual_coefficients",
            "present_mask",
            "query_mask",
            "completion_query",
            "completion_source",
            "thermal_rank_weight",
            "scalar_bias",
            "heating_envelope",
        ):
            value = getattr(self, name)
            if isinstance(value, Tensor):
                yield name, value

    def tensor_values(self) -> dict[str, Tensor]:
        """Return a deterministic name-to-tensor mapping for version binding."""
        return dict(self._tensor_items())

    def version(self) -> tuple[object, ...]:
        """Return a stable structural and in-place-mutation signature."""
        tensor_versions = tuple(
            (
                name,
                tuple(value.shape),
                str(value.dtype),
                value.device.type,
                value.device.index,
                id(value),
                int(value.data_ptr()),
                _tensor_mutation_version(value),
            )
            for name, value in self._tensor_items()
        )
        return (_SCHEMA, self.batch_size, self.query_count, self.source_count, tensor_versions)

    def _accumulation_dtype(self, requested: torch.dtype | None, fallback: torch.dtype) -> torch.dtype:
        dtype = fallback if requested is None else requested
        if not _is_float_dtype(dtype):
            raise ValueError("accumulation_dtype must be a floating-point torch dtype.")
        return dtype

    def _control_values(
        self,
        controls: Tensor,
        accumulation_dtype: torch.dtype | None,
        check_envelope: bool,
    ) -> tuple[Tensor, Tensor]:
        if tuple(controls.shape) != (self.batch_size, self.source_count):
            raise ValueError("controls must have shape [B,M].")
        if not controls.is_floating_point() or controls.device != self.device:
            raise ValueError("controls must be floating-point and share the response device.")
        dtype = self._accumulation_dtype(accumulation_dtype, controls.dtype)
        heat = controls.to(dtype=dtype)
        _require(torch.isfinite(heat).all(), "controls must be finite.")
        present_heat = heat * self.present_mask.to(dtype=dtype)
        if check_envelope and present_heat.numel():
            present_values = heat[self.present_mask]
            envelope = self.heating_envelope.to(dtype=dtype)
            _require((present_values.abs() <= envelope).all(),
                     "Present-source controls must satisfy abs(h) <= heating_envelope.")
        return heat, present_heat

    def _masked_query(self, values: Tensor) -> Tensor:
        if self.query_mask is None:
            return values
        return values * self.query_mask.to(dtype=values.dtype)

    def apply(
        self,
        controls: Tensor,
        accumulation_dtype: torch.dtype | None = None,
        check_envelope: bool = True,
    ) -> Tensor:
        """Apply the affine-in-factors, linear-in-controls response as ``[B,Q]``."""
        heat, present_heat = self._control_values(controls, accumulation_dtype, check_envelope)
        dtype = heat.dtype
        result = heat.new_zeros((self.batch_size, self.query_count))

        if self.has_completion:
            assert self.completion_query is not None
            assert self.completion_source is not None
            assert self.thermal_rank_weight is not None
            query = self.completion_query.to(dtype=dtype)
            source = self.completion_source.to(dtype=dtype)
            rank_weight = self.thermal_rank_weight.to(dtype=dtype)
            # Contract source heat into rank before query expansion. No [B,Q,M]
            # activation is allocated on this ordinary application path.
            source_mass = torch.einsum("bm,bmr->br", present_heat, source)
            query_weight = query * rank_weight[None, None, :]
            result = torch.einsum("bqr,br->bq", query_weight, source_mass)
            result = result + self.scalar_bias.to(dtype=dtype) * present_heat.sum(dim=1, keepdim=True)

        if self.residual_coefficients.numel():
            destination = self.selected_batch * self.query_count + self.selected_query
            selected = (
                self.residual_coefficients.to(dtype=dtype)
                * heat[self.selected_batch, self.selected_source]
            )
            result = result.reshape(-1).index_add(0, destination, selected).reshape(
                self.batch_size, self.query_count
            )
        return self._masked_query(result)

    def transpose(
        self,
        cotangent: Tensor,
        accumulation_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Apply the exact response transpose to a ``[B,Q]`` cotangent."""
        if tuple(cotangent.shape) != (self.batch_size, self.query_count):
            raise ValueError("cotangent must have shape [B,Q].")
        if not cotangent.is_floating_point() or cotangent.device != self.device:
            raise ValueError("cotangent must be floating-point and share the response device.")
        dtype = self._accumulation_dtype(accumulation_dtype, cotangent.dtype)
        query_cotangent = self._masked_query(cotangent.to(dtype=dtype))
        _require(torch.isfinite(query_cotangent).all(), "cotangent must be finite.")
        result = query_cotangent.new_zeros((self.batch_size, self.source_count))

        if self.has_completion:
            assert self.completion_query is not None
            assert self.completion_source is not None
            assert self.thermal_rank_weight is not None
            query = self.completion_query.to(dtype=dtype)
            source = self.completion_source.to(dtype=dtype)
            rank_weight = self.thermal_rank_weight.to(dtype=dtype)
            query_rank_cotangent = torch.einsum(
                "bq,bqr,r->br", query_cotangent, query, rank_weight
            )
            result = torch.einsum("br,bmr->bm", query_rank_cotangent, source)
            result = result + (
                self.scalar_bias.to(dtype=dtype)
                * query_cotangent.sum(dim=1, keepdim=True)
            )

        if self.residual_coefficients.numel():
            destination = self.selected_batch * self.source_count + self.selected_source
            selected = (
                self.residual_coefficients.to(dtype=dtype)
                * query_cotangent[self.selected_batch, self.selected_query]
            )
            result = result.reshape(-1).index_add(0, destination, selected).reshape(
                self.batch_size, self.source_count
            )
        return result * self.present_mask.to(dtype=dtype)

    def explicit_dense_kernel(self, accumulation_dtype: torch.dtype | None = None) -> Tensor:
        """Explicitly materialize ``[B,Q,M]`` for export or numerical comparison."""
        fallback = self.residual_coefficients.dtype
        dtype = self._accumulation_dtype(accumulation_dtype, fallback)
        kernel = torch.zeros(
            (self.batch_size, self.query_count, self.source_count),
            dtype=dtype,
            device=self.device,
        )

        if self.has_completion:
            assert self.completion_query is not None
            assert self.completion_source is not None
            assert self.thermal_rank_weight is not None
            kernel = torch.einsum(
                "bqr,bmr,r->bqm",
                self.completion_query.to(dtype=dtype),
                self.completion_source.to(dtype=dtype),
                self.thermal_rank_weight.to(dtype=dtype),
            )
            kernel = kernel + self.scalar_bias.to(dtype=dtype) * self.present_mask[:, None, :].to(dtype=dtype)

        if self.residual_coefficients.numel():
            destination = (
                (self.selected_batch * self.query_count + self.selected_query)
                * self.source_count
                + self.selected_source
            )
            kernel = kernel.reshape(-1).index_add(
                0, destination, self.residual_coefficients.to(dtype=dtype)
            ).reshape(self.batch_size, self.query_count, self.source_count)
        kernel = kernel * self.present_mask[:, None, :].to(dtype=dtype)
        if self.query_mask is not None:
            kernel = kernel * self.query_mask[:, :, None].to(dtype=dtype)
        return kernel


__all__ = ["SourceReadAffineFactorsV1"]
