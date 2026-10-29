"""Standard model interfaces for the HET-MF electro-thermal case.

These facades connect the case-owned inputs to the generic HONF cores. They do
not contain a thermal wrapper, a staged surrogate, or a physics correction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .processing import FIELD_ORDER


SIX_HYPEREDGE = "six_hyperedge"
DENSE_PAIRWISE = "dense_pairwise"
SIX_HYPEREDGE_DISPLAY_NAME = "Six-hyperedge organization controlling contextualized member reads"
DENSE_PAIRWISE_DISPLAY_NAME = "Dense MM/ME/EM interaction and direct field reads"
CLASSIC_ADAPTER_SCHEMA = "hetmf_classic_batch_v1"
COORDINATE_FRAME = "hetmf_unit_square_xy_v1"


@dataclass(frozen=True)
class ClassicReadyBatch:
    """Core inputs, held-out labels, and physical identity/measures for one batch.

    ``scene.target_field`` is always ``None``. Targets, query weights, source
    identities, and physical measures remain case-owned fields alongside the
    generic ``BatchData`` so neither core can consume labels as model inputs.
    """

    scene: Any
    targets: torch.Tensor
    query_weights: torch.Tensor
    module_measures: torch.Tensor
    environment_measures: torch.Tensor
    module_characteristic_lengths: torch.Tensor
    environment_characteristic_lengths: torch.Tensor
    contact_ids: tuple[tuple[str | None, ...], ...]
    environment_ids: tuple[tuple[int, ...], ...]
    query_ids: torch.Tensor | None
    state_ids: tuple[str, ...]
    metadata: tuple[Mapping[str, Any], ...]

    def to(self, device: str | torch.device) -> ClassicReadyBatch:
        """Move numeric tensors and the generic scene to a model device."""
        return replace(
            self,
            scene=self.scene.to(device),
            targets=self.targets.to(device),
            query_weights=self.query_weights.to(device),
            module_measures=self.module_measures.to(device),
            environment_measures=self.environment_measures.to(device),
            module_characteristic_lengths=self.module_characteristic_lengths.to(device),
            environment_characteristic_lengths=self.environment_characteristic_lengths.to(device),
            query_ids=None if self.query_ids is None else self.query_ids.to(device),
        )


class HETMFClassicAdapter:
    """Map HET-MF archive samples to generic ``BatchData`` and five targets.

    The prescribed voltage stays in each contact's feature row before the
    organizer or dense interaction core runs. Input transforms are fixed
    physical logs; targets are returned separately and keep the dataset's
    TRAIN-only normalization unchanged.
    """

    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    def collate(self, samples: Sequence[Mapping[str, Any]]) -> ClassicReadyBatch:
        if not samples:
            raise ValueError("sample batch cannot be empty")

        rows: list[dict[str, Any]] = []
        query_count: int | None = None
        for row_index, sample in enumerate(samples):
            if not isinstance(sample, Mapping) or not isinstance(sample.get("inputs"), Mapping):
                raise TypeError("each sample must contain an input mapping")
            inputs = sample["inputs"]
            try:
                environment = np.asarray(inputs["environment_records"], dtype=np.float64)
                contacts = np.asarray(inputs["contacts"], dtype=np.float64)
                contact_mask = np.asarray(inputs["contact_mask"])
                query_xy = np.asarray(sample["query_xy"], dtype=np.float64)
                targets = np.asarray(sample["fields"], dtype=np.float32)
                query_weights = np.asarray(sample["query_weight"], dtype=np.float64)
            except KeyError as exc:
                raise ValueError(f"sample is missing required HET-MF field {exc.args[0]!r}") from exc

            if environment.shape != (144, 8) or not np.isfinite(environment).all():
                raise ValueError("environment_records must be finite with shape [144,8]")
            if np.any(environment[:, 2:5] <= 0) or np.any(environment[:, 5:8] <= 0):
                raise ValueError("environment spacings, measures, and physical coefficients must be positive")
            if np.any((environment[:, :2] < 0) | (environment[:, :2] > 1)):
                raise ValueError("environment coordinates must lie in the unit square")
            if contacts.ndim != 2 or contacts.shape[1] != 5:
                raise ValueError("contacts must have shape [M,5]")
            if contact_mask.dtype != np.bool_ or contact_mask.shape != contacts.shape[:1]:
                raise ValueError("contact_mask must be boolean and align with contacts")
            if not contact_mask.any():
                raise ValueError("each sample must contain at least one active contact")
            active_contacts = contacts[contact_mask]
            if not np.isfinite(active_contacts).all():
                raise ValueError("active contacts must be finite")
            if np.any(active_contacts[:, 2] <= 0) or np.any(active_contacts[:, 3] <= 0):
                raise ValueError("active contact radii and conductances must be positive")
            if np.any((active_contacts[:, :2] < 0) | (active_contacts[:, :2] > 1)):
                raise ValueError("active contact coordinates must lie in the unit square")

            if query_xy.ndim != 2 or query_xy.shape[1] != 2 or not np.isfinite(query_xy).all():
                raise ValueError("query_xy must be finite with shape [Q,2]")
            if np.any((query_xy < 0) | (query_xy > 1)):
                raise ValueError("query coordinates must lie in the unit square")
            current_query_count = int(query_xy.shape[0])
            if current_query_count < 1 or (query_count is not None and current_query_count != query_count):
                raise ValueError("samples in a batch must have the same positive query count")
            query_count = current_query_count
            if targets.shape != (current_query_count, len(FIELD_ORDER)) or not np.isfinite(targets).all():
                raise ValueError("fields must be finite [Q,5] in canonical HET-MF field order")
            if (query_weights.shape != (current_query_count,) or not np.isfinite(query_weights).all()
                    or np.any(query_weights < 0) or query_weights.sum() <= 0):
                raise ValueError("query_weight must be finite, nonnegative [Q] with positive total mass")

            metadata = dict(sample.get("metadata", {}))
            contact_ids = self._contact_ids(sample, contacts.shape[0], contact_mask, row_index)
            state_id = str(sample.get("state_id", metadata.get("state_id", f"case-{row_index}")))
            source_query_ids = sample.get("query_ids", metadata.get("native_cell_ids"))
            if source_query_ids is None:
                query_ids = None
            else:
                query_ids = np.asarray(source_query_ids)
                if (query_ids.shape != (current_query_count,) or not np.issubdtype(query_ids.dtype, np.integer)
                        or np.any(query_ids < 0) or len(np.unique(query_ids)) != current_query_count):
                    raise ValueError("native query IDs must be unique nonnegative integers aligned with queries")
                query_ids = query_ids.astype(np.int64, copy=False)

            rows.append({
                "environment": environment,
                "contacts": active_contacts,
                "contact_ids": contact_ids,
                "query_xy": query_xy,
                "targets": targets,
                "query_weights": query_weights,
                "query_ids": query_ids,
                "state_id": state_id,
                "metadata": metadata,
            })

        if any(row["query_ids"] is None for row in rows) and not all(row["query_ids"] is None for row in rows):
            raise ValueError("native query IDs must be supplied for every row or omitted for every row")

        batch_size = len(rows)
        module_count = max(len(row["contacts"]) for row in rows)
        query_count = int(query_count or 0)
        module_centers = np.zeros((batch_size, module_count, 2), dtype=np.float32)
        module_features = np.zeros((batch_size, module_count, 3), dtype=np.float32)
        module_present = np.zeros((batch_size, module_count), dtype=np.float32)
        module_measures = np.zeros((batch_size, module_count), dtype=np.float32)
        module_lengths = np.zeros((batch_size, module_count, 2), dtype=np.float32)
        environment_coords = np.stack([row["environment"][:, :2] for row in rows]).astype(np.float32)
        environment_features = np.stack([np.log(row["environment"][:, 5:8]) for row in rows]).astype(np.float32)
        environment_measures = np.stack([row["environment"][:, 4] for row in rows]).astype(np.float32)
        environment_lengths = np.stack([row["environment"][:, 2:4] for row in rows]).astype(np.float32)
        contact_id_slots: list[tuple[str | None, ...]] = []
        for batch_index, row in enumerate(rows):
            contacts = row["contacts"]
            count = len(contacts)
            module_centers[batch_index, :count] = contacts[:, :2]
            module_features[batch_index, :count] = np.column_stack((contacts[:, 2], np.log(contacts[:, 3]), contacts[:, 4]))
            module_present[batch_index, :count] = 1.0
            module_measures[batch_index, :count] = np.pi * contacts[:, 2] ** 2
            module_lengths[batch_index, :count] = (2.0 * contacts[:, 2, None]).repeat(2, axis=1)
            contact_id_slots.append(tuple(row["contact_ids"]) + (None,) * (module_count - count))

        contact_ids = tuple(contact_id_slots)
        environment_ids = tuple(tuple(range(144)) for _ in rows)
        query_ids_values = None if rows[0]["query_ids"] is None else np.stack([row["query_ids"] for row in rows])
        tensor = lambda value, dtype=None: torch.as_tensor(value, device=self.device, dtype=dtype)
        environment_measures_tensor = tensor(environment_measures, torch.float32)
        module_measures_tensor = tensor(module_measures, torch.float32)
        module_lengths_tensor = tensor(module_lengths, torch.float32)
        environment_lengths_tensor = tensor(environment_lengths, torch.float32)
        state_ids = tuple(row["state_id"] for row in rows)
        scene_metadata = {
            "adapter_schema": CLASSIC_ADAPTER_SCHEMA,
            "coordinate_frame": COORDINATE_FRAME,
            "state_ids": state_ids,
            "contact_ids": contact_ids,
            "environment_ids": environment_ids,
            "module_measures": module_measures_tensor.detach().cpu(),
            "environment_measures": environment_measures_tensor.detach().cpu(),
            "module_characteristic_lengths": module_lengths_tensor.detach().cpu(),
            "environment_characteristic_lengths": environment_lengths_tensor.detach().cpu(),
        }

        from honf_forward_core import BatchData

        scene = BatchData(
            module_centers=tensor(module_centers, torch.float32),
            module_present=tensor(module_present, torch.float32),
            module_features=tensor(module_features, torch.float32),
            global_context=tensor(np.zeros((batch_size, 1), dtype=np.float32), torch.float32),
            query_xy=tensor(np.stack([row["query_xy"] for row in rows]), torch.float32),
            query_time=None,
            target_field=None,
            case_name=state_ids[0] if batch_size == 1 else "hetmf-batch",
            metadata=scene_metadata,
            env_coords=tensor(environment_coords, torch.float32),
            env_features=tensor(environment_features, torch.float32),
            env_weights=environment_measures_tensor,
        )
        scene_metadata["batch_data_type"] = type(scene).__name__
        targets_tensor = tensor(np.stack([row["targets"] for row in rows]), torch.float32)
        weights_tensor = tensor(np.stack([row["query_weights"] for row in rows]), torch.float32)
        return ClassicReadyBatch(
            scene=scene,
            targets=targets_tensor,
            query_weights=weights_tensor,
            module_measures=module_measures_tensor,
            environment_measures=environment_measures_tensor,
            module_characteristic_lengths=module_lengths_tensor,
            environment_characteristic_lengths=environment_lengths_tensor,
            contact_ids=contact_ids,
            environment_ids=environment_ids,
            query_ids=None if query_ids_values is None else tensor(query_ids_values, torch.long),
            state_ids=state_ids,
            metadata=tuple(row["metadata"] for row in rows),
        )

    @staticmethod
    def _contact_ids(
        sample: Mapping[str, Any],
        slot_count: int,
        mask: np.ndarray,
        row_index: int,
    ) -> tuple[str, ...]:
        metadata = sample.get("metadata", {})
        inputs = sample.get("inputs", {})
        raw_ids = metadata.get("contact_ids") if isinstance(metadata, Mapping) else None
        if raw_ids is None:
            raw_ids = inputs.get("contact_ids") if isinstance(inputs, Mapping) else None
        if raw_ids is None:
            values = tuple(f"case-{row_index}:contact-{index}" for index in range(slot_count))
        else:
            values = tuple(raw_ids)
        active_count = int(mask.sum())
        if len(values) == slot_count:
            active_values = [value for value, active in zip(values, mask) if active]
        elif len(values) == active_count:
            active_values = list(values)
        else:
            raise ValueError("contact IDs must align with contact slots or active contact records")
        result = tuple(value.decode() if isinstance(value, bytes) else str(value) for value in active_values)
        if any(not value for value in result) or len(set(result)) != len(result):
            raise ValueError("active literal contact IDs must be nonempty and unique within each case")
        return result


class HETMFClassicModel(nn.Module):
    """Prediction-only facade over the generic HONF or dense interface core."""

    def __init__(self, name: str, core: nn.Module, *, display_name: str) -> None:
        super().__init__()
        self.name = name
        self.display_name = display_name
        self.core = core

    def predict(self, batch: Any) -> torch.Tensor:
        """Return normalized five-field predictions for a ready batch or scene."""
        return self.forward(batch)

    def forward(self, batch: Any) -> torch.Tensor:
        scene = batch.scene if isinstance(batch, ClassicReadyBatch) else batch
        if not hasattr(scene, "module_centers") or not hasattr(scene, "query_xy"):
            raise TypeError("model input must be BatchData or ClassicReadyBatch")
        if self.name == SIX_HYPEREDGE:
            output = self.core(scene)
        elif self.name == DENSE_PAIRWISE:
            encoded = self.core.encode_case(scene)
            prepared = self.core.prepare(encoded, encoded.module_tokens)
            output = self.core.decode_queries(prepared, scene.query_xy, scene.query_features)
        else:  # construction is restricted to the two named standard models
            raise RuntimeError(f"unsupported HET-MF classic model: {self.name}")
        prediction = output["pred_field"]
        if prediction.ndim != 3 or prediction.shape[-1] != len(FIELD_ORDER):
            raise RuntimeError("classic cores must return normalized five-field predictions [B,Q,5]")
        return prediction


def build_classic_model(
    name: str,
    *,
    hidden_dim: int = 256,
    seed: int = 0,
    init_seed: int | None = None,
    display_name: str | None = None,
    num_hyperedges: int = 6,
    position_fourier_frequencies: int = 4,
    query_fourier_frequencies: int = 4,
    pairwise_kernel_fourier_frequencies: int = 4,
    pairwise_kernel_hidden_dim: int = 256,
    pairwise_kernel_num_layers: int = 4,
    message_hidden_dim: int = 128,
    attention_heads: int = 4,
    receiver_chunk_size: int = 128,
    relative_fourier_frequencies: int = 4,
    coarse_latent_count: int = 8,
    coarse_blocks: int = 1,
    main_latent_count: int = 16,
    main_latent_blocks: int = 2,
    local_radius_factor: float = 2.5,
    dropout: float = 0.0,
    activation_checkpointing: bool = False,
) -> HETMFClassicModel:
    """Build one standard HET-MF model from its coworker-facing name.

    Initialization is seeded on the CPU without changing the caller's random
    state or initializing CUDA. Training remains the caller's responsibility.
    """
    canonical = str(name).strip()
    display_names = {
        SIX_HYPEREDGE: SIX_HYPEREDGE_DISPLAY_NAME,
        DENSE_PAIRWISE: DENSE_PAIRWISE_DISPLAY_NAME,
    }
    if canonical not in display_names:
        raise ValueError(f"name must be one of {tuple(display_names)}")
    if display_name is not None and display_name != display_names[canonical]:
        raise ValueError("display_name must match the canonical HET-MF model name")
    if int(hidden_dim) <= 0 or not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("hidden_dim must be positive and dropout must be in [0,1)")
    if canonical == SIX_HYPEREDGE and int(num_hyperedges) != 6:
        raise ValueError("the classic organization model uses exactly six hyperedge slots")
    if int(attention_heads) <= 0 or int(hidden_dim) % int(attention_heads):
        raise ValueError("hidden_dim must be divisible by positive attention_heads")
    init_seed_value = int(seed if init_seed is None else init_seed)

    from honf_forward_core import HONFNeuralField, UnifiedForwardConfig
    from honf_forward_core.config import InterfaceFieldConfig
    from honf_forward_core.interface_fields import InterfaceFieldCore

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(init_seed_value)
        if canonical == SIX_HYPEREDGE:
            config = UnifiedForwardConfig(
                field_dim=len(FIELD_ORDER),
                forward_architecture="legacy_honf",
                domain_length_x=1.0,
                domain_length_y=1.0,
                coordinate_scale=[1.0, 1.0],
                module_radius=0.05,
                num_hyperedges=6,
                organizer_mode="fixed_projection",
                edge_capacity=0,
                initial_active_edges=6,
                minimum_active_edges=1,
                hidden_dim=int(hidden_dim),
                dropout=float(dropout),
                geometry_mode="nonperiodic",
                query_time_mode="none",
                decoder_mode="enhanced_honf_pairwise_no_global_near",
                use_hyper_value_context=False,
                query_fourier_frequencies=int(query_fourier_frequencies),
                position_fourier_frequencies=int(position_fourier_frequencies),
                use_position_fourier_for_modules=True,
                use_position_fourier_for_env=True,
                use_hyper_mechanism_encoder=False,
                pairwise_aggregation_mode="fused_query_module",
                pairwise_module_token_source="organizer_contextualized",
                pairwise_kernel_hidden_dim=int(pairwise_kernel_hidden_dim),
                pairwise_kernel_num_layers=int(pairwise_kernel_num_layers),
                pairwise_kernel_use_fourier=True,
                pairwise_kernel_fourier_frequencies=int(pairwise_kernel_fourier_frequencies),
                routing_execution="dense",
            )
            core = HONFNeuralField(config)
        else:
            config = UnifiedForwardConfig(
                field_dim=len(FIELD_ORDER),
                forward_architecture="dense_pairwise_field",
                interface_model=InterfaceFieldConfig(
                    message_hidden_dim=int(message_hidden_dim),
                    attention_heads=int(attention_heads),
                    coarse_latent_count=int(coarse_latent_count),
                    coarse_blocks=int(coarse_blocks),
                    main_latent_count=int(main_latent_count),
                    main_latent_blocks=int(main_latent_blocks),
                    local_radius_factor=float(local_radius_factor),
                    relative_fourier_frequencies=int(relative_fourier_frequencies),
                    receiver_chunk_size=int(receiver_chunk_size),
                    activation_checkpointing=bool(activation_checkpointing),
                ),
                domain_length_x=1.0,
                domain_length_y=1.0,
                coordinate_scale=[1.0, 1.0],
                module_radius=0.05,
                hidden_dim=int(hidden_dim),
                dropout=float(dropout),
                geometry_mode="nonperiodic",
                query_time_mode="none",
                query_fourier_frequencies=int(query_fourier_frequencies),
                position_fourier_frequencies=int(position_fourier_frequencies),
                use_position_fourier_for_modules=True,
                use_position_fourier_for_env=True,
            )
            core = InterfaceFieldCore(config)
    return HETMFClassicModel(canonical, core, display_name=display_names[canonical])
