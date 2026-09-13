"""Thin WindFarm facade over the reusable HONF field cores.

The case package owns the input adapter and target transform, while the
reusable cores own all learned interaction layers.  This module deliberately
does not contain turbine physics or a second model implementation: it only
keeps the two supported core call sequences and their prepared state in one
small, checkpoint-friendly object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields import InterfaceFieldCore
from honf_forward_core.model import HONFNeuralField
from torch import nn

from .normalization import VelocityNormalizer


def _as_batch(value: BatchData | Mapping[str, Any]) -> BatchData:
    """Convert a mapping produced by a case collator to ``BatchData``."""

    if isinstance(value, BatchData):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"WindFarm model expects BatchData or a mapping, got {type(value)!r}.")
    payload = dict(value)
    # The native sampler keeps these two bounded host-side measures next to
    # the canonical tensors for the case loss and native diagnostics.  They
    # are deliberately not model inputs and are not fields of BatchData.
    payload.pop("query_loss_weight", None)
    payload.pop("query_measure_m3", None)
    payload.pop("module_count", None)
    tensor_fields = {
        "module_centers",
        "module_present",
        "module_features",
        "global_context",
        "query_xy",
        "query_time",
        "target_field",
        "env_coords",
        "env_features",
        "query_features",
        "env_weights",
    }
    for name in tensor_fields:
        item = payload.get(name)
        if item is not None and not torch.is_tensor(item):
            payload[name] = torch.as_tensor(item)
    # DataLoader metadata often uses a list of records.  The reusable core
    # treats it as opaque host metadata, so no conversion is needed here.
    return BatchData.from_dict(payload)


def _inverse_velocity(transform: VelocityNormalizer | None, standardized: torch.Tensor) -> torch.Tensor:
    """Apply the explicit training-owned velocity transform on any device."""

    if transform is None:
        raise ValueError("A fitted VelocityNormalizer is required for physical velocity predictions.")
    mean_t = torch.as_tensor(transform.mean, device=standardized.device, dtype=standardized.dtype)
    std_t = torch.as_tensor(transform.safe_std, device=standardized.device, dtype=standardized.dtype)
    return (standardized * std_t + mean_t) * float(transform.u_ref_mps)


@dataclass(frozen=True)
class PreparedWindFarmCase:
    """Case-static state that can be reused for arbitrary query chunks."""

    architecture: str
    encoded: Any
    dense_prepared: Any = None


class WindFarmForwardModel(nn.Module):
    """Shared field-only wrapper for classic K=6 and dense pairwise models."""

    def __init__(self, config: UnifiedForwardConfig, *, velocity_transform: VelocityNormalizer | None = None):
        super().__init__()
        self.config = config
        self.velocity_transform = velocity_transform
        architecture = str(config.forward_architecture)
        if architecture == "legacy_honf":
            self.core: nn.Module = HONFNeuralField(config)
        elif architecture == "dense_pairwise_field":
            self.core = InterfaceFieldCore(config)
        else:
            raise ValueError(
                "WindFarm supports only legacy_honf and dense_pairwise_field; "
                f"got {architecture!r}."
            )

    @property
    def architecture(self) -> str:
        return str(self.config.forward_architecture)

    def set_training_progress(self, *, epoch: int, total_epochs: int | None = None) -> None:
        setter = getattr(self.core, "set_training_progress", None)
        if callable(setter):
            setter(epoch=int(epoch), total_epochs=None if total_epochs is None else int(total_epochs))

    def selection_state(self) -> dict[str, int | None]:
        getter = getattr(self.core, "selection_state", None)
        if callable(getter):
            return dict(getter())
        return {"epoch": None, "total_epochs": None}

    def prepare_case(self, batch_value: BatchData | Mapping[str, Any]) -> PreparedWindFarmCase:
        """Encode one batch of layouts and prepare its reusable case state."""

        batch = _as_batch(batch_value)
        if self.architecture == "legacy_honf":
            encoded = self.core.encode_and_organize(batch)  # type: ignore[attr-defined]
            return PreparedWindFarmCase(self.architecture, encoded)
        encoded = self.core.encode_case(batch)  # type: ignore[attr-defined]
        # DensePairwiseField uses the encoded module tokens as its module state;
        # no thermal local module or fabricated port state belongs here.
        dense_prepared = self.core.prepare(encoded, encoded.module_tokens)  # type: ignore[attr-defined]
        return PreparedWindFarmCase(self.architecture, encoded, dense_prepared)

    def decode(
        self,
        prepared: PreparedWindFarmCase,
        query_coords: torch.Tensor,
        query_features: torch.Tensor | None = None,
        *,
        receiver_chunk_size: int | None = None,
        return_routing_maps: bool = False,
    ) -> dict[str, Any]:
        """Decode standardized velocity at arbitrary native or generated points."""

        queries = query_coords.float()
        features = query_features
        if features is not None:
            features = features.float()
        if self.architecture == "legacy_honf":
            outputs: list[dict[str, Any]] = []
            chunk_size = 128 if receiver_chunk_size is None else int(receiver_chunk_size)
            if chunk_size <= 0:
                raise ValueError("receiver_chunk_size must be positive.")
            for start in range(0, int(queries.shape[1]), chunk_size):
                chunk_features = None if features is None else features[:, start : start + chunk_size]
                outputs.append(self.core.decode_queries(  # type: ignore[attr-defined]
                    query_xy=queries[:, start : start + chunk_size],
                    query_time=None,
                    organizer_output=prepared.encoded,
                    global_token=prepared.encoded["global_token"],
                    query_features=chunk_features,
                    return_routing_maps=bool(return_routing_maps),
                ))
            if not outputs:
                raise ValueError("query_coords must contain at least one receiver.")
            query_local_keys = {
                "pred_field",
                "pred_mean",
                "pred_residual",
                "query_hyper_attention",
                "dominant_hyperedge",
                "hyper_attention_entropy_map",
                "c_H_norm",
                "c_pair_norm",
                "pairwise_edge_contribution",
            }
            result: dict[str, Any] = {}
            for key in outputs[0]:
                values = [item[key] for item in outputs]
                if key in query_local_keys and torch.is_tensor(values[0]):
                    result[key] = torch.cat(values, dim=1)
                else:
                    result[key] = values[0]
            return result
        return self.core.decode_queries(  # type: ignore[attr-defined]
            prepared.dense_prepared,
            queries,
            query_features=features,
            return_routing_maps=bool(return_routing_maps),
            receiver_chunk_size=receiver_chunk_size,
        )

    def predict_standardized(
        self,
        prepared: PreparedWindFarmCase,
        query_coords: torch.Tensor | None = None,
        query_features: torch.Tensor | None = None,
        *,
        receiver_chunk_size: int | None = None,
        return_routing_maps: bool = False,
    ) -> torch.Tensor:
        """Return the standardized three-channel field prediction."""

        if query_coords is None:
            raise ValueError("query_coords are required because prepared state excludes target/query metadata.")
        output = self.decode(
            prepared,
            query_coords,
            query_features=query_features,
            receiver_chunk_size=receiver_chunk_size,
            return_routing_maps=return_routing_maps,
        )
        prediction = output.get("pred_field")
        if prediction is None:
            raise KeyError("Reusable core output did not contain 'pred_field'.")
        if prediction.shape[-1] != int(self.config.field_dim):
            raise ValueError(
                f"WindFarm velocity output must have field_dim={self.config.field_dim}, "
                f"got {tuple(prediction.shape)}."
            )
        return prediction

    def predict_physical(
        self,
        prepared: PreparedWindFarmCase,
        query_coords: torch.Tensor | None = None,
        query_features: torch.Tensor | None = None,
        *,
        receiver_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Return velocity in m/s using only the training-owned transform."""

        standardized = self.predict_standardized(
            prepared,
            query_coords,
            query_features,
            receiver_chunk_size=receiver_chunk_size,
        )
        return _inverse_velocity(self.velocity_transform, standardized)

    def forward(self, batch_value: BatchData | Mapping[str, Any]) -> dict[str, Any]:
        """Run a complete sampled batch through its selected core."""

        batch = _as_batch(batch_value)
        prepared = self.prepare_case(batch)
        output = self.decode(
            prepared,
            batch.query_xy,
            query_features=batch.query_features,
        )
        output["prepared_case"] = prepared
        return output

    def materialize(self, batch_value: BatchData | Mapping[str, Any]) -> PreparedWindFarmCase:
        """Materialize lazy projections on real geometry before optimizer setup.

        The extra hub query ensures the dense local correction sees a genuine
        module-neighbour path even when a random sampled batch happens to lie
        outside every rotor-radius neighbourhood.  It carries no target and
        therefore cannot leak solved fields into model inputs.
        """

        batch = _as_batch(batch_value)
        centers = batch.module_centers.float()
        present = batch.module_present.float()
        hub_query = centers.clone()
        if hub_query.ndim != 3:
            raise ValueError("module_centers must have shape [B,M,3] for WindFarm materialization.")
        # Replace padded slots with a finite case-local point.  They are only
        # materialization receivers; module_present still masks their source
        # tokens in both reusable cores.
        fallback = centers.new_zeros((centers.shape[0], 1, centers.shape[-1]))
        hub_query = torch.where(present[..., None] > 0.5, hub_query, fallback.expand_as(hub_query))
        # The global context contract stores lower support coordinates and
        # support extents in its final six entries, each divided by the
        # fixed positional scale.  Rebuild the documented seven features for
        # the geometry-only materialization receivers; never borrow target
        # query features from the sampled supervision batch.
        scale = centers.new_tensor([50.0, 38.0, 6.25]).reshape(1, 1, 3)
        context = batch.global_context.float()
        if context.shape[-1] < 11:
            raise ValueError("WindFarm global_context must contain the documented width-11 support features.")
        lower = context[:, 5:8].unsqueeze(1) * scale
        extent = context[:, 8:11].unsqueeze(1) * scale
        upper = lower + extent
        hub_features = torch.cat(((hub_query - lower) / scale, (upper - hub_query) / scale, hub_query[..., 2:3] / scale[..., 2:3]), dim=-1)
        with torch.no_grad():
            prepared = self.prepare_case(batch)
            self.predict_standardized(prepared, hub_query, query_features=hub_features)
        uninitialized: list[str] = []
        for name, parameter in self.named_parameters():
            if isinstance(parameter, torch.nn.parameter.UninitializedParameter):
                uninitialized.append(name)
        if uninitialized:
            raise RuntimeError(f"WindFarm model still has lazy parameters after materialization: {uninitialized}")
        return prepared


__all__ = ["PreparedWindFarmCase", "WindFarmForwardModel"]
