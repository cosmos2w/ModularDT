"""Native-token retrieval contracts.

The objects here carry original IDs and one-to-one token records. They do not
contain group values or pooled decoder payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor

TokenKind = Literal["module", "environment"]
_ORGANIZATION_STORE_UNSET = object()


def _require(condition: Tensor, message: str) -> None:
    condition = condition.all()
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def tensor_version(value: Tensor) -> tuple[int, int, int]:
    """Return the repository's lightweight identity/storage/in-place stamp."""
    try:
        version = int(value._version)
    except RuntimeError:  # inference tensors have no version counter
        version = -1
    return id(value), int(value.data_ptr()), version


def tensor_versions(*values: Tensor | None) -> tuple[tuple[int, int, int], ...]:
    return tuple(tensor_version(value) for value in values if value is not None)


def query_batch_tensor_versions(query_batch: Any) -> tuple[tuple[int, int, int], ...]:
    """Stamp all receiver tensors that can change query retrieval semantics."""
    return tensor_versions(
        query_batch.coordinates,
        query_batch.features,
        query_batch.query_mask,
        query_batch.query_ids,
        query_batch.local_batch_indices,
        query_batch.local_query_indices,
        query_batch.local_module_indices,
        query_batch.local_weights,
    )


def query_batch_identity(query_batch: Any) -> tuple[Any, Any, Any]:
    """Return stable non-tensor receiver identity metadata."""
    return (
        query_batch.scene_key,
        query_batch.coordinate_frame_key,
        query_batch.receiver_key,
    )


def _validate_ids(ids: Tensor, valid: Tensor, name: str) -> None:
    if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must be an integer [B,N] tensor.")
    if valid.shape != ids.shape or valid.dtype != torch.bool:
        raise TypeError(f"{name} validity must be boolean with the same [B,N] shape.")
    same = ids[:, :, None] == ids[:, None, :]
    duplicate = same & valid[:, :, None] & valid[:, None, :]
    duplicate &= ~torch.eye(ids.shape[1], dtype=torch.bool, device=ids.device)[None, :, :]
    _require(~duplicate, f"Valid {name} values must be unique within each batch item.")


@dataclass(frozen=True)
class NativeTokenRecords:
    """Token-aligned native data for one physical record type.

    Every tensor keeps the native record axis. Decoder projections, when
    supplied, have shape ``[B,H,N,D]`` and are not contracted over ``N``.
    """

    kind: TokenKind
    native_ids: Tensor  # integer [B,N]
    coordinates: Tensor  # [B,N,C]
    characteristic_lengths: Tensor  # positive [B,N,C]
    valid: Tensor  # bool [B,N]
    measure: Tensor  # nonnegative native measure [B,N]
    raw_features: Tensor  # [B,N,F]
    contextualized_states: Tensor  # [B,N,W]
    decoder_keys: Tensor | None = None  # [B,H,N,D]
    decoder_values: Tensor | None = None  # [B,H,N,D]

    def __post_init__(self) -> None:
        if self.kind not in ("module", "environment"):
            raise ValueError("kind must be 'module' or 'environment'.")
        if self.coordinates.ndim != 3:
            raise ValueError("coordinates must have shape [B,N,C].")
        batch, count, coordinate_dim = self.coordinates.shape
        if self.characteristic_lengths.shape != (batch, count, coordinate_dim):
            raise ValueError("characteristic_lengths must match [B,N,C].")
        if self.raw_features.ndim != 3 or self.raw_features.shape[:2] != (batch, count):
            raise ValueError("raw_features must preserve the [B,N] native-record axis.")
        if self.contextualized_states.ndim != 3 or self.contextualized_states.shape[:2] != (batch, count):
            raise ValueError("contextualized_states must preserve the [B,N] native-record axis.")
        _validate_ids(self.native_ids, self.valid, f"{self.kind} native IDs")
        if self.measure.shape != (batch, count):
            raise ValueError("measure must have shape [B,N].")
        _require(torch.isfinite(self.measure) & (self.measure >= 0), "measure must be finite and nonnegative.")
        _require(torch.isfinite(self.coordinates), "coordinates must be finite.")
        _require(
            torch.isfinite(self.characteristic_lengths) & (self.characteristic_lengths > 0),
            "characteristic_lengths must be finite and positive.",
        )
        for name, value in (("decoder_keys", self.decoder_keys), ("decoder_values", self.decoder_values)):
            if value is not None and (value.ndim != 4 or value.shape[0] != batch or value.shape[2] != count):
                raise ValueError(f"{name} must have shape [B,H,N,D] and preserve every native record.")
        if (
            self.decoder_keys is not None
            and self.decoder_values is not None
            and self.decoder_keys.shape != self.decoder_values.shape
        ):
            raise ValueError("decoder key/value shapes must match.")


@dataclass
class NativeTokenStore:
    """Live per-scene M/E tokens plus the owner/version binding for reuse.

    ``mark_backward_consumed`` is called by the training boundary after a
    backward pass. A store whose prepared graph has been consumed must be
    rebuilt before another query or optimizer step uses it.
    """

    modules: NativeTokenRecords
    environments: NativeTokenRecords
    model_owner: int
    model_parameter_versions: tuple[tuple[int, int], ...]
    model_buffer_versions: tuple[tuple[int, int], ...]
    scene_tensor_versions: tuple[tuple[int, int, int], ...]
    scene_identity: tuple[Any, Any, Any]
    model_config_key: tuple[Any, ...]
    frame_key: Any = None
    normalizer_key: Any = None
    live_autograd_graph: bool = True
    backward_consumed: bool = False
    module_tensor_versions: tuple[tuple[int, int, int], ...] = ()
    environment_tensor_versions: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.modules.kind != "module" or self.environments.kind != "environment":
            raise ValueError("NativeTokenStore requires module and environment records in the matching slots.")
        if self.modules.native_ids.shape[0] != self.environments.native_ids.shape[0]:
            raise ValueError("Module and environment records must share the batch axis.")
        if not self.module_tensor_versions:
            self.module_tensor_versions = tensor_versions(
                self.modules.native_ids,
                self.modules.coordinates,
                self.modules.characteristic_lengths,
                self.modules.valid,
                self.modules.measure,
                self.modules.raw_features,
                self.modules.contextualized_states,
                self.modules.decoder_keys,
                self.modules.decoder_values,
            )
        if not self.environment_tensor_versions:
            self.environment_tensor_versions = tensor_versions(
                self.environments.native_ids,
                self.environments.coordinates,
                self.environments.characteristic_lengths,
                self.environments.valid,
                self.environments.measure,
                self.environments.raw_features,
                self.environments.contextualized_states,
                self.environments.decoder_keys,
                self.environments.decoder_values,
            )

    @property
    def scene_key(self) -> Any:
        return self.scene_identity[0]

    def mark_backward_consumed(self) -> None:
        """Invalidate this prepared graph after its backward use."""
        if self.live_autograd_graph:
            self.backward_consumed = True

    def assert_current(
        self,
        *,
        model_owner: int,
        model_parameter_versions: tuple[tuple[int, int], ...],
        model_buffer_versions: tuple[tuple[int, int], ...],
        scene_tensor_versions: tuple[tuple[int, int, int], ...],
        scene_identity: tuple[Any, Any, Any],
        model_config_key: tuple[Any, ...],
    ) -> None:
        """Reject stale weights, inputs, preprocessing, configuration, or graph lifetime."""
        if self.backward_consumed:
            raise ValueError("Native token store autograd graph was consumed by backward; rebuild the store.")
        if self.model_owner != model_owner:
            raise ValueError("Native token store belongs to a different model instance.")
        if self.model_parameter_versions != model_parameter_versions:
            raise ValueError("Model parameters changed after native token preparation.")
        if self.model_buffer_versions != model_buffer_versions:
            raise ValueError("Model buffers changed after native token preparation.")
        if self.scene_tensor_versions != scene_tensor_versions:
            raise ValueError("Scene geometry, measures, masks, features, or token order changed.")
        if self.scene_identity != scene_identity:
            raise ValueError("Scene, normalization, or coordinate-frame identity changed.")
        if self.model_config_key != model_config_key:
            raise ValueError("Native token store configuration is stale.")
        if self.module_tensor_versions != tensor_versions(
            self.modules.native_ids,
            self.modules.coordinates,
            self.modules.characteristic_lengths,
            self.modules.valid,
            self.modules.measure,
            self.modules.raw_features,
            self.modules.contextualized_states,
            self.modules.decoder_keys,
            self.modules.decoder_values,
        ):
            raise ValueError("Prepared module token records changed in place.")
        if self.environment_tensor_versions != tensor_versions(
            self.environments.native_ids,
            self.environments.coordinates,
            self.environments.characteristic_lengths,
            self.environments.valid,
            self.environments.measure,
            self.environments.raw_features,
            self.environments.contextualized_states,
            self.environments.decoder_keys,
            self.environments.decoder_values,
        ):
            raise ValueError("Prepared environment token records changed in place.")


@dataclass(frozen=True)
class NativeQueryRetrievalPlan:
    """Query-owned native-ID permissions and candidate rows.

    Candidate indices use ``(batch, query, token_axis_index)`` order. They are
    unique per receiver and token type; repeated group/head paths are already
    combined by max and never imply repeated value evaluation.
    """

    seed_permission: Tensor  # [B,Hd,Q,S]
    module_permission: Tensor  # [B,Hd,Q,M]
    module_support: Tensor  # bool [B,Q,M]
    unique_module_indices: Tensor  # integer [N,3] = (b,q,module_axis_index)
    module_ids: Tensor  # integer [B,M]
    seed_ids: Tensor  # integer [B,S]
    query_ids: Tensor | None  # integer [B,Q]
    query_mask: Tensor  # bool [B,Q]
    eligible_module_rows: Tensor  # bool [B,Q,M]
    environment_permission: Tensor | None = None  # [B,Hd,Q,E], opt-in to avoid primary QxE metadata
    environment_support: Tensor | None = None  # bool [B,Q,E]
    unique_environment_indices: Tensor | None = None  # integer [N,3]
    environment_ids: Tensor | None = None  # integer [B,E]
    eligible_environment_rows: Tensor | None = None  # bool [B,Q,E]
    include_environment: bool = False
    environment_value_consumer: bool = False
    retrieved_module_pairs: Tensor | None = None  # scalar tensor; support count, not executor evidence
    retrieved_environment_pairs: Tensor | None = None  # scalar tensor; metadata count only
    query_tensor_versions: tuple[tuple[int, int, int], ...] | None = None
    query_identity: tuple[Any, Any, Any] | None = None
    model_owner: int | None = None
    graph_binding: tuple[Any, ...] | None = None
    _hypergraph: Any = field(default=None, repr=False, compare=False)
    _token_store: Any = field(default=None, repr=False, compare=False)
    _owned_tensor_versions: tuple[tuple[int, int, int], ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.seed_permission.ndim != 4 or self.module_permission.ndim != 4:
            raise ValueError("Seed and module permissions must be [B,Hd,Q,S/M].")
        batch, heads, queries, seeds = self.seed_permission.shape
        if self.module_permission.shape[:3] != (batch, heads, queries):
            raise ValueError("Module permission batch/head/query axes must match seed permissions.")
        modules = self.module_permission.shape[-1]
        if self.module_support.shape != (batch, queries, modules) or self.module_support.dtype != torch.bool:
            raise ValueError("module_support must be boolean [B,Q,M].")
        if self.module_ids.ndim != 2 or self.module_ids.shape != (batch, modules):
            raise ValueError("module_ids must match the plan's module token axis.")
        if self.seed_ids.shape != (batch, seeds):
            raise ValueError("seed_ids must match the query-seed score axis.")
        if self.query_mask.shape != (batch, queries) or self.query_mask.dtype != torch.bool:
            raise ValueError("query_mask must be boolean [B,Q].")
        if self.query_ids is not None and self.query_ids.shape != (batch, queries):
            raise ValueError("query_ids must have shape [B,Q].")
        if self.eligible_module_rows.shape != (batch, queries, modules):
            raise ValueError("eligible_module_rows must have shape [B,Q,M].")
        if self.unique_module_indices.ndim != 2 or self.unique_module_indices.shape[-1] != 3:
            raise ValueError("unique_module_indices must have shape [N,3].")
        if self.unique_module_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("unique_module_indices must be integer-valued.")
        if self.include_environment:
            if self.environment_permission is None or self.environment_support is None:
                raise ValueError("Environment permission/support is required when include_environment=True.")
            if self.environment_ids is None or self.unique_environment_indices is None:
                raise ValueError("Environment IDs and candidate indices are required when included.")
            if self.environment_permission.ndim != 4 or self.environment_permission.shape[:3] != (
                batch,
                heads,
                queries,
            ):
                raise ValueError("environment_permission must be [B,Hd,Q,E].")
            environments = self.environment_permission.shape[-1]
            if self.environment_support.shape != (batch, queries, environments):
                raise ValueError("environment_support must be [B,Q,E].")
            if self.environment_ids.shape != (batch, environments):
                raise ValueError("environment_ids must match the E token axis.")
            if self.eligible_environment_rows is None or self.eligible_environment_rows.shape != (
                batch,
                queries,
                environments,
            ):
                raise ValueError("eligible_environment_rows must be [B,Q,E].")
            if self.unique_environment_indices.ndim != 2 or self.unique_environment_indices.shape[-1] != 3:
                raise ValueError("unique_environment_indices must have shape [N,3].")
        elif any(
            value is not None
            for value in (
                self.environment_permission,
                self.environment_support,
                self.unique_environment_indices,
                self.environment_ids,
                self.eligible_environment_rows,
            )
        ):
            raise ValueError("Environment query tensors must be absent when include_environment=False.")
        if self.environment_value_consumer and not self.include_environment:
            raise ValueError("A direct environment consumer requires an E retrieval plan.")
        if (self.query_tensor_versions is None) != (self.query_identity is None):
            raise ValueError("Bound query versions and identity must be supplied together.")
        if self.query_tensor_versions is not None and self.model_owner is None:
            raise ValueError("A query-bound native plan must carry its model owner.")
        object.__setattr__(self, "_owned_tensor_versions", tensor_versions(*self._owned_tensors()))

    def _owned_tensors(self) -> tuple[Tensor | None, ...]:
        """Tensors whose mutation would invalidate this plan's public traces."""
        return (
            self.seed_permission,
            self.module_permission,
            self.module_support,
            self.unique_module_indices,
            self.module_ids,
            self.seed_ids,
            self.query_ids,
            self.query_mask,
            self.eligible_module_rows,
            self.environment_permission,
            self.environment_support,
            self.unique_environment_indices,
            self.environment_ids,
            self.eligible_environment_rows,
            self.retrieved_module_pairs,
            self.retrieved_environment_pairs,
        )

    def assert_query_current(
        self,
        query_batch: Any | None = None,
        *,
        organization_model_owner: int | None = None,
        organization_token_store: Any = _ORGANIZATION_STORE_UNSET,
    ) -> None:
        """Validate the receiver, graph, and public permission/index tensors.

        Plans built without ``query_batch`` are low-level numerical fixtures:
        their own tensors and any graph binding are checked, but they carry no
        claim that a later receiver has the same geometry, identities, or order.
        If the caller supplies ``organization_token_store``, a store-bound plan
        must be evaluated against that exact prepared token store.
        """
        if self.query_tensor_versions is not None:
            if query_batch is None:
                raise ValueError("A query-bound native plan requires the original query batch for validation.")
            if query_batch_tensor_versions(query_batch) != self.query_tensor_versions:
                raise ValueError("Query geometry, features, masks, IDs, or local support changed after planning.")
            identity = query_batch_identity(query_batch)
            if identity != self.query_identity:
                raise ValueError("Query scene, frame, or receiver identity changed after planning.")
        if tensor_versions(*self._owned_tensors()) != self._owned_tensor_versions:
            raise ValueError("Native query plan permissions, support, IDs, or indices changed in place.")
        if organization_model_owner is not None and self.model_owner is not None and organization_model_owner != self.model_owner:
            raise ValueError("Native query plan belongs to a different model instance.")
        if self.graph_binding is not None and self._hypergraph is not None:
            owner, parameter_versions, buffer_versions, scene_identity, scene_tensor_versions, model_config_key = (
                self.graph_binding
            )
            self._hypergraph.assert_current(
                owner_token=owner,
                parameter_versions=parameter_versions,
                buffer_versions=buffer_versions,
                scene_identity=scene_identity,
                scene_tensor_versions=scene_tensor_versions,
                model_config_key=model_config_key,
            )
        if self._token_store is not None:
            if self.graph_binding is None:
                raise ValueError("A token-store-bound query plan is missing its graph binding.")
            if (
                organization_token_store is not _ORGANIZATION_STORE_UNSET
                and organization_token_store is not self._token_store
            ):
                raise ValueError("Native query plan is bound to a different prepared token store.")
            owner, parameter_versions, buffer_versions, scene_identity, scene_tensor_versions, model_config_key = (
                self.graph_binding
            )
            self._token_store.assert_current(
                model_owner=owner,
                model_parameter_versions=parameter_versions,
                model_buffer_versions=buffer_versions,
                scene_tensor_versions=scene_tensor_versions,
                scene_identity=scene_identity,
                model_config_key=model_config_key,
            )
