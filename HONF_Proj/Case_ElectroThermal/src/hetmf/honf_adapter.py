"""Target-separated, case-owned bridge to the general HONF tensor contracts.

The five electrothermal fields use the general nonlinear field head. Applied
voltage belongs in the prescribed contact features: the core's affine thermal
control path cannot represent the quadratic heat and temperature response.
This bridge prepares batches and random initialized models; it does not fit a
model, select an organizer, or define a future training campaign.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from copy import deepcopy
from pathlib import Path
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .processing import FIELD_ORDER


MODULE_FEATURE_NAMES = ("radius", "log_total_contact_conductance", "terminal_voltage")
ENVIRONMENT_FEATURE_NAMES = ("log_sheet_conductance", "log_thermal_conductance", "log_cooling_strength")
COORDINATE_FRAME = "hetmf_unit_square_xy_v1"
ADAPTER_SCHEMA = "hetmf_honf_adapter_v1"
CONTACT_IDENTITY_SCHEMA = "hetmf_contact_literal_sha256_int63_v1"


def _literal_contact_ids(literals: Any, mask: Any) -> np.ndarray:
    """Encode literal identities, independently of record order or slot count."""
    valid = np.asarray(mask)
    values = np.asarray(literals)
    if valid.dtype != np.bool_ or values.shape != valid.shape:
        raise ValueError("literal contact IDs must align with the boolean contact mask")
    names = []
    for value in values[valid]:
        if not isinstance(value, (str, np.str_)) or not value:
            raise ValueError("active literal contact IDs must be nonempty strings")
        names.append(str(value))
    if len(names) != len(set(names)):
        raise ValueError("active literal contact IDs must be unique")
    encoded = [int.from_bytes(hashlib.sha256(f"{CONTACT_IDENTITY_SCHEMA}:{name}".encode()).digest()[:8], "big")
               & ((1 << 63) - 1) for name in names]
    if len(encoded) != len(set(encoded)):
        raise ValueError("literal contact ID hash collision")
    result = np.full(valid.shape, -1, dtype=np.int64)
    result[valid] = encoded
    return result


@dataclass(frozen=True)
class HONFReadyBatch:
    """Model inputs and supervision have separate owners and tensor members."""

    scene: Any
    query: Any
    targets: torch.Tensor
    query_weights: torch.Tensor
    metadata: tuple[Mapping[str, Any], ...]

    def to(self, device: str | torch.device) -> HONFReadyBatch:
        """Move contracts after CPU worker collation, preserving opaque keys.

        DataLoader workers should use a CPU adapter. The main process moves the
        returned batch to its model device. Select the spawn worker context if
        the parent has already initialized Torch threads.
        """
        def move_contract(contract):
            if not is_dataclass(contract):
                raise TypeError("HONF scene/query contracts must be dataclasses")
            changes = {field.name: getattr(contract, field.name).to(device)
                       for field in fields(contract) if torch.is_tensor(getattr(contract, field.name))}
            return replace(contract, **changes)
        return HONFReadyBatch(move_contract(self.scene), move_contract(self.query),
                              self.targets.to(device), self.query_weights.to(device), self.metadata)


class ElectroThermalHONFAdapter:
    """Translate physical HETMFDataset inputs to SceneBatch and QueryBatch.

    Input columns follow training_data.ENVIRONMENT_COLUMNS and CONTACT_COLUMNS.
    The log transforms below are fixed physical transforms; they use no target
    statistics or split labels. Targets pass through unchanged, so a caller
    using standardized targets must apply its saved TRAIN-only inverse transform
    to predicted fields before reporting physical metrics.
    """

    def __init__(self, *, device: str | torch.device = "cpu", local_radius_factor: float = 3.0):
        self.device = torch.device(device)
        if not np.isfinite(local_radius_factor) or local_radius_factor <= 0:
            raise ValueError("local_radius_factor must be finite and positive")
        self.local_radius_factor = float(local_radius_factor)

    @staticmethod
    def model_config(*, context_dim: int = 32, num_heads: int = 4, init_seed: int = 0) -> dict[str, Any]:
        """Explicit general-core baseline configuration, with no affine controls."""
        return {
            "schema": "content_hypergraph_core_v2", "architecture": "A2-D",
            "module_input_dim": len(MODULE_FEATURE_NAMES),
            "environment_input_dim": len(ENVIRONMENT_FEATURE_NAMES),
            "coordinate_dim": 2, "query_input_dim": 0, "operating_input_dim": 0,
            "output_dim": len(FIELD_ORDER), "context_dim": context_dim,
            "num_heads": num_heads, "context_depth": 2, "num_hyperedges": 0,
            "thermal_response_rank": 0, "local_hidden_dim": context_dim,
            "dropout": 0.0, "init_seed": init_seed,
        }

    def build_model(self, **kwargs: Any):
        """Import the real general HONF core lazily and instantiate on this device."""
        from honf_forward_core.content_hypergraph.model_v2 import ContentHypergraphCoreV2

        config = self.model_config(**kwargs)
        config.pop("schema")
        return ContentHypergraphCoreV2(**config).to(self.device)

    def make_scene(self, inputs: Sequence[Mapping[str, Any]], *, scene_keys: Sequence[str] | None = None):
        """Accept physical records and optional integer IDs; no targets are read.

        ``collate`` preserves the archive's literal source IDs automatically.
        Direct callers that reorder records must carry matching ``contact_ids``;
        otherwise the input slots define their correspondence IDs.
        """
        from honf_forward_core.content_hypergraph.contracts import SceneBatch

        if not inputs:
            raise ValueError("scene batch cannot be empty")
        rows = []
        for item in inputs:
            environment = np.asarray(item["environment_records"], dtype=np.float64)
            contacts = np.asarray(item["contacts"], dtype=np.float64)
            mask = np.asarray(item["contact_mask"])
            if environment.shape != (144, 8) or not np.isfinite(environment).all():
                raise ValueError("environment_records must be finite [144,8]")
            if contacts.ndim != 2 or contacts.shape[1] != 5 or mask.shape != contacts.shape[:1]:
                raise ValueError("contacts/contact_mask must have shapes [M,5]/[M]")
            if mask.dtype != np.bool_ or not mask.any():
                raise ValueError("contact_mask must be boolean with at least one active contact")
            active = contacts[mask]
            if not np.isfinite(active).all() or np.any(active[:, 2:4] <= 0):
                raise ValueError("active contacts must be finite with positive radius and conductance")
            if np.any((active[:, :2] < 0) | (active[:, :2] > 1)):
                raise ValueError("active contact coordinates must lie in the unit square")
            if np.any(environment[:, 2:] <= 0) or np.any((environment[:, :2] < 0) | (environment[:, :2] > 1)):
                raise ValueError("environment spacings, measures and coefficients must be positive")
            ids = np.asarray(item.get("contact_ids", np.arange(len(contacts))))
            if not np.issubdtype(ids.dtype, np.integer):
                raise ValueError("integer contact IDs must use an integer dtype")
            ids = ids.astype(np.int64)
            if ids.shape != mask.shape or np.any(ids[mask] < 0) or len(np.unique(ids[mask])) != int(mask.sum()):
                raise ValueError("active integer contact IDs must be unique and nonnegative")
            rows.append((environment, active, ids[mask]))
        batch, modules, environments = len(rows), max(len(row[1]) for row in rows), 144
        features = np.zeros((batch, modules, 3), dtype=np.float32)
        coordinates = np.zeros((batch, modules, 2), dtype=np.float32)
        mask = np.zeros((batch, modules), dtype=bool)
        measures = np.zeros((batch, modules), dtype=np.float32)
        ids = np.full((batch, modules), -1, dtype=np.int64)
        lengths = np.ones((batch, modules, 2), dtype=np.float32)
        radius = np.ones((batch, modules), dtype=np.float32)
        for index, (_, contacts, source_ids) in enumerate(rows):
            count = len(contacts)
            features[index, :count] = np.column_stack((contacts[:, 2], np.log(contacts[:, 3]), contacts[:, 4]))
            coordinates[index, :count] = contacts[:, :2]
            mask[index, :count] = True
            measures[index, :count] = np.pi * contacts[:, 2] ** 2
            ids[index, :count] = source_ids
            lengths[index, :count] = 2 * contacts[:, 2, None]
            radius[index, :count] = self.local_radius_factor * contacts[:, 2]
        records = np.stack([row[0] for row in rows])
        tensor = lambda value, dtype=None: torch.tensor(value, device=self.device, dtype=dtype)
        if scene_keys is not None and len(scene_keys) != batch:
            raise ValueError("scene_keys must match batch length")
        scene = SceneBatch(
            module_features=tensor(features), environment_features=tensor(np.log(records[..., 5:8]), torch.float32),
            module_coordinates=tensor(coordinates), environment_coordinates=tensor(records[..., :2], torch.float32),
            module_mask=tensor(mask, torch.bool), environment_mask=torch.ones((batch, environments), device=self.device, dtype=torch.bool),
            module_measure=tensor(measures), environment_measure=tensor(records[..., 4], torch.float32),
            module_ids=tensor(ids, torch.long), environment_ids=tensor(np.tile(np.arange(environments), (batch, 1)), torch.long),
            local_radius=tensor(radius), coordinate_origin=torch.zeros((batch, 2), device=self.device),
            coordinate_scale=torch.ones((batch, 2), device=self.device),
            module_measure_reference=torch.ones(batch, device=self.device),
            environment_measure_reference=torch.full((batch,), 1 / 144, device=self.device),
            module_characteristic_lengths=tensor(lengths),
            environment_characteristic_lengths=tensor(records[..., 2:4], torch.float32),
            scene_key=None if scene_keys is None else tuple(scene_keys),
            normalizer_key=(ADAPTER_SCHEMA, "fixed_log_inputs_v1", self.local_radius_factor),
            coordinate_frame_key=COORDINATE_FRAME,
        )
        scene.validate(coordinate_dim=2, module_input_dim=3, environment_input_dim=3, operating_input_dim=0)
        return scene

    def make_queries(self, scene: Any, coordinates: Any, *, query_ids: Any = None):
        """Construct arbitrary finite unit-square queries [B,Q,2]."""
        from honf_forward_core.content_hypergraph.contracts import QueryBatch

        xy = torch.tensor(np.asarray(coordinates), device=self.device, dtype=torch.float32)
        if xy.ndim != 3 or xy.shape[0] != scene.batch_size or xy.shape[-1] != 2:
            raise ValueError("query coordinates must have shape [B,Q,2]")
        if not torch.isfinite(xy).all() or ((xy < 0) | (xy > 1)).any():
            raise ValueError("queries must be finite and inside the unit square")
        ids = None if query_ids is None else torch.tensor(np.asarray(query_ids), device=self.device, dtype=torch.long)
        query = QueryBatch(coordinates=xy, query_mask=torch.ones(xy.shape[:2], device=self.device, dtype=torch.bool),
                           query_ids=ids, scene_key=scene.scene_key, coordinate_frame_key=COORDINATE_FRAME)
        query.validate(batch_size=scene.batch_size, coordinate_dim=2, query_input_dim=0)
        return query

    def collate(self, samples: Sequence[Mapping[str, Any]]) -> HONFReadyBatch:
        """Collate equal-query-count samples with dynamic contact padding."""
        if not samples:
            raise ValueError("sample batch cannot be empty")
        metadata = tuple(item.get("metadata", {}) for item in samples)
        physical_inputs = []
        for item, meta in zip(samples, metadata):
            inputs = dict(item["inputs"])
            # Literal source identity is the only metadata used for token IDs.
            # IDs preserve correspondence and are never encoded as features.
            if "contact_ids" not in inputs and "contact_ids" in meta:
                inputs["contact_ids"] = _literal_contact_ids(meta["contact_ids"], inputs["contact_mask"])
            physical_inputs.append(inputs)
        scene = self.make_scene(physical_inputs,
                                scene_keys=[str(item.get("state_id", meta.get("state_id", index)))
                                            for index, (item, meta) in enumerate(zip(samples, metadata))])
        native_ids = [item.get("query_ids", meta.get("native_cell_ids")) for item, meta in zip(samples, metadata)]
        query = self.make_queries(scene, np.stack([item["query_xy"] for item in samples]),
                                  query_ids=np.stack(native_ids) if all(value is not None for value in native_ids) else None)
        targets = torch.tensor(np.stack([item["fields"] for item in samples]), device=self.device, dtype=torch.float32)
        weights = torch.tensor(np.stack([item["query_weight"] for item in samples]), device=self.device, dtype=torch.float32)
        if targets.shape != (*query.coordinates.shape[:2], len(FIELD_ORDER)) or not torch.isfinite(targets).all():
            raise ValueError("targets must be finite [B,Q,5] in canonical field order")
        if weights.shape != targets.shape[:2] or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("query weights must be finite nonnegative [B,Q]")
        if (weights.sum(dim=1) <= 0).any():
            raise ValueError("each case must have positive query measure")
        return HONFReadyBatch(scene, query, targets, weights, metadata)


def diagnose_honf_dataset(path: str | Path, *, queries: int = 17, normalize: bool = True) -> dict[str, Any]:
    """Exercise a real archive through the general CPU core without fitting.

    Select one M2 TRAIN case and one M16 extrapolation case using input masks
    only. This is an interface diagnostic, including target access to the two
    declared cases; it is neither validation accuracy nor an optimizer update.
    Full payload verification belongs to export/verify_training_h5; the loader
    performs its structural archive checks here.
    """
    import h5py
    from .io import sha256_file
    from .training_data import HETMFDataset, verify_training_h5
    from honf_forward_core.content_hypergraph import model_v2

    if type(queries) is not int or queries < 2:
        raise ValueError("smoke queries must be an integer >=2")
    path = Path(path).resolve()
    archive = verify_training_h5(path, verify_payload=False)
    counts_to_splits = ((2, "train"), (16, "panel_extrapolation"))
    samples = []
    with h5py.File(path, "r") as h:
        masks = h["inputs/contact_mask"][:]
        for count, split in counts_to_splits:
            indices = h[f"splits/{split}"][:]
            # Bounded input-only scan: no target values select the smoke cases.
            matches = np.flatnonzero(masks[indices].sum(axis=1) == count)
            if not len(matches):
                raise ValueError(f"archive needs an M{count} {split} case for this diagnostic")
            dataset = HETMFDataset(path, split=split, query_count=queries, seed=0, normalize=normalize)
            try:
                sample = dataset[int(matches[0])]
                samples.append(sample)
            finally:
                dataset.close()
        normalization_fingerprint = str(h["normalization"].attrs["train_membership_sha256"])
    adapter = ElectroThermalHONFAdapter(device="cpu")
    cuda_before = torch.cuda.is_initialized()
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = adapter.build_model(context_dim=16, num_heads=2).eval()
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        batch = adapter.collate(samples)
        prediction = model.decode(model.prepare(batch.scene), batch.query).nonlinear_field
        loss = ((prediction - batch.targets).square().mean(dim=-1) * batch.query_weights).sum(dim=1).mean()
        if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
            raise RuntimeError("nonfinite CPU HONF output/loss")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise RuntimeError("missing or nonfinite CPU HONF gradients")
        if not any(value.abs().sum() > 0 for value in gradients):
            raise RuntimeError("all CPU HONF gradients are zero")
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
        changed = deepcopy(samples)
        for sample in changed:
            for name in ("contacts", "contact_mask"):
                sample["inputs"][name] = sample["inputs"][name][::-1].copy()
            sample["metadata"]["contact_ids"] = tuple(reversed(sample["metadata"]["contact_ids"]))
        with torch.no_grad():
            # Compare invariants under the same autograd execution mode. Some
            # CPU attention kernels use a different reduction in no_grad mode.
            reference = model.decode(model.prepare(batch.scene), batch.query).nonlinear_field
            torch.testing.assert_close(prediction, reference, atol=2e-6, rtol=2e-5)
            permutation_batch = adapter.collate(changed)
            for row in range(len(samples)):
                original_ids = batch.scene.module_ids[row, batch.scene.module_mask[row]]
                permuted_ids = permutation_batch.scene.module_ids[row, permutation_batch.scene.module_mask[row]]
                torch.testing.assert_close(permuted_ids, original_ids.flip(0), atol=0, rtol=0)
            permutation = model.decode(model.prepare(permutation_batch.scene), permutation_batch.query).nonlinear_field
            torch.testing.assert_close(reference, permutation, atol=2e-6, rtol=2e-5)
            organization = model.prepare(batch.scene)
            middle = queries // 2
            chunks = [model.decode(organization, adapter.make_queries(batch.scene, batch.query.coordinates[:, a:b].numpy())).nonlinear_field
                      for a, b in ((0, middle), (middle, queries))]
            chunked = torch.cat(chunks, dim=1)
            torch.testing.assert_close(reference, chunked, atol=2e-6, rtol=2e-5)
            single = adapter.collate(samples[:1])
            single_prediction = model.decode(model.prepare(single.scene), single.query).nonlinear_field
            torch.testing.assert_close(reference[:1], single_prediction, atol=2e-6, rtol=2e-5)
            changed_labels = deepcopy(samples)
            for sample in changed_labels:
                sample["fields"] = sample["fields"] + 123
                sample["metadata"].update(split="label_changed", solver_residual=123, seed=-999)
            label_batch = adapter.collate(changed_labels)
            label_prediction = model.decode(model.prepare(label_batch.scene), label_batch.query).nonlinear_field
            torch.testing.assert_close(reference, label_prediction, atol=0, rtol=0)
        return {
            "schema": "hetmf_honf_cpu_smoke_v1", "dataset": str(path),
            "archive_structure": archive, "queries_per_case": queries,
            "case_ids": [sample["metadata"]["state_id"] for sample in samples],
            "case_splits": [split for _, split in counts_to_splits],
            "contact_counts": batch.scene.module_mask.sum(dim=1).tolist(),
            "model_config": model.get_config(), "adapter_schema": ADAPTER_SCHEMA,
            "adapter_source_sha256": sha256_file(Path(__file__)),
            "contact_identity_schema": CONTACT_IDENTITY_SCHEMA,
            "general_model_source_sha256": sha256_file(Path(model_v2.__file__)),
            "module_features": list(MODULE_FEATURE_NAMES), "environment_features": list(ENVIRONMENT_FEATURE_NAMES),
            "field_order": list(FIELD_ORDER), "prediction_shape": list(prediction.shape),
            "target_transform": "saved TRAIN-only population z-score" if normalize else "physical dimensionless fields",
            "normalization_train_membership_sha256": normalization_fingerprint,
            "diagnostic_loss": float(loss.detach()), "finite_gradients": True,
            "parameter_tensors_with_gradients": len(gradients), "optimizer_steps": 0,
            "parameters_unchanged": True, "full_contact_permutation_passed": True,
            "stable_literal_contact_ids_passed": True,
            "query_chunking_passed": True, "mixed_padding_passed": True, "target_label_exclusion_passed": True,
            "maximum_permutation_difference": float((reference - permutation).abs().max()),
            "maximum_chunking_difference": float((reference - chunked).abs().max()),
            "cuda_initialized_before": cuda_before, "cuda_initialized_after": torch.cuda.is_initialized(),
            "physical_accuracy_measured": False, "trained_organizer_or_inverse": False,
        }
    finally:
        torch.set_num_threads(old_threads)
