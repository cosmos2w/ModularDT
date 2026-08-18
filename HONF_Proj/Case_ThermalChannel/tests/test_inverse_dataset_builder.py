"""Synthetic data-ABI tests without invoking the expensive forward model.

Physical designs ``D`` and contexts ``c`` are paired with fake verified
compact targets ``G`` and exact request values ``R``. The test treats those
targets as realized ``G_hat`` solely to exercise splitting and storage.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np

from honf_forward_core.evaluation.hypergraph_plan import extract_hypergraph_plan
from honf_inverse_core.contracts import FunctionalValue, PhysicalDesign
from channelthermal.inverse.compact_plan import extract_compact_plan
from channelthermal.inverse.context import parse_context
from channelthermal.inverse.dataset_builder import (
    CaseBuildRecord,
    assign_inverse_splits,
    build_inverse_dataset_from_records,
    case_id_sha256,
)
from channelthermal.inverse.dataset_io import (
    TOPOLOGY_SET_DATASET_SCHEMA_NAME,
    InverseH5Dataset,
    validate_inverse_hdf5,
)
from channelthermal.inverse.vocabulary import NONREGIONAL_REQUEST_TYPES


def _context(offset: float = 0.0):
    return parse_context(
        {
            "schema_name": "thermalchannel_inverse_context",
            "schema_version": 1,
            "re": 100.0 + offset,
            "u_in": 1.0,
            "nu": 1.0e-5,
            "solid_alpha": 1.0e-5,
            "fluid_alpha": 2.0e-5,
            "solid_k": 10.0,
            "fluid_k": 0.6,
            "module_radius": 0.1,
            "domain_length_x": 4.0,
            "domain_length_y": 2.0,
        }
    )


def _record(case_id: str, source_split: str, source_index: int, offset: float) -> CaseBuildRecord:
    context = _context(offset)
    design = PhysicalDesign(
        module_centers=np.asarray([[0.6, 0.6], [2.0, 1.0], [3.3, 1.4]], dtype=np.float32),
        module_present=np.ones(3, dtype=np.float32),
        heat_powers=np.asarray([10.0 + offset, 20.0, 30.0], dtype=np.float32),
    )
    A_mh = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]], dtype=np.float32)
    A_eh = np.asarray([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.1, 0.1, 0.8]], dtype=np.float32)
    env_coords = np.asarray([[0.5, 0.5], [1.5, 0.5], [2.5, 1.5], [3.5, 1.5]], dtype=np.float32)
    module_mass = A_mh.sum(0) / A_mh.sum()
    env_mass = A_eh.sum(0) / A_eh.sum()
    organizer = {
        "A_mh": A_mh,
        "A_eh": A_eh,
        "hyper_source_coords": (A_mh / A_mh.sum(0, keepdims=True)).T @ design.module_centers,
        "hyper_region_coords": (A_eh / A_eh.sum(0, keepdims=True)).T @ env_coords,
        "hyper_module_mass": module_mass,
        "hyper_env_mass": env_mass,
        "hyper_strength": np.sqrt(module_mass * env_mass + 1.0e-6),
        "env_coords": env_coords,
    }
    full = extract_hypergraph_plan(organizer, design.module_present, domain_length_x=4.0, domain_length_y=2.0)
    compact = extract_compact_plan(full, design, context)
    functionals = {
        name: FunctionalValue(name, 20.0 + offset + index, True, 20, "unit")
        for index, name in enumerate(NONREGIONAL_REQUEST_TYPES)
    }

    def regional(name: str, region: tuple[float, float, float, float]) -> FunctionalValue:
        value = 15.0 + offset + sum(region) + (2.0 if name.endswith("max") else 0.0)
        return FunctionalValue(name, value, True, 64, "temperature", {"region": region})

    return CaseBuildRecord(
        case_id=case_id,
        source_split=source_split,
        source_index=source_index,
        design=design,
        context=context,
        compact_plan=compact,
        full_plan=full,
        nonregional_functionals=functionals,
        regional_evaluator=regional,
    )


def test_split_before_augmentation_and_case_major_hdf5_round_trip(tmp_path: Path) -> None:
    records = [
        _record("train-a", "train", 0, 0.0),
        _record("train-b", "train", 1, 1.0),
        _record("train-c", "train", 2, 2.0),
        _record("test-outlier", "test", 3, 1000.0),
    ]
    destination = tmp_path / "inverse_v1.h5"
    build_inverse_dataset_from_records(records, destination, variants_per_case=5, seed=77)
    summary = validate_inverse_hdf5(destination)
    assert summary["num_cases"] == 4
    assert summary["variants_per_case"] == 5
    assert sum(summary["split_counts"].values()) == 4
    dataset = InverseH5Dataset(destination, split="all")
    assert len(dataset) == 20
    sample = dataset[0]
    assert sample["request"]["active_mask"].sum() in {2, 3, 4}
    assert sample["plan"].shape == (3, 12)
    assert sample["layout"].shape == (3, 3)
    dataset.close()
    streamed = InverseH5Dataset(destination, split="all", preload=False)
    streamed_sample = streamed[0]
    np.testing.assert_array_equal(sample["layout"], streamed_sample["layout"])
    np.testing.assert_array_equal(
        sample["request"]["active_mask"], streamed_sample["request"]["active_mask"]
    )
    streamed.close()

    with h5py.File(destination, "r") as h5:
        splits = [value.decode() if isinstance(value, bytes) else str(value) for value in h5["inverse_split"][...]]
        assert splits[-1] == "test"
        train_values = [20.0 + index for index, split in enumerate(splits[:3]) if split == "train"]
        assert float(h5["normalization/functional_mean"][0]) == np.mean(train_values)
        seeds = h5["variant_seed"][...]
        assert len(np.unique(seeds)) == seeds.size
        assert h5["requests/active_mask"].shape == (4, 5, 4)
        assert np.all(h5["geometry/valid"][...] == 1)

    repeat = tmp_path / "repeat.h5"
    build_inverse_dataset_from_records(records, repeat, variants_per_case=5, seed=77)
    with h5py.File(destination, "r") as first, h5py.File(repeat, "r") as second:
        np.testing.assert_array_equal(first["variant_seed"][...], second["variant_seed"][...])
        np.testing.assert_array_equal(first["requests/type_id"][...], second["requests/type_id"][...])


def test_split_hashes_are_stable_and_disjoint() -> None:
    ids = ["a", "b", "c", "d"]
    first = assign_inverse_splits(ids, ["train", "train", "train", "test"], split_seed=9)
    second = assign_inverse_splits(list(reversed(ids)), ["test", "train", "train", "train"], split_seed=9)
    assert dict(zip(ids, first)) == dict(zip(reversed(ids), second))
    train_ids = [case_id for case_id, split in zip(ids, first) if split == "train"]
    assert case_id_sha256(train_ids) == case_id_sha256(list(reversed(train_ids)))


def test_topology_set_dataset_is_schema_and_checkpoint_bound(tmp_path: Path) -> None:
    records = [
        _record("train-a", "train", 0, 0.0),
        _record("train-b", "train", 1, 1.0),
        _record("test-a", "test", 2, 2.0),
    ]
    compact_path = tmp_path / "compact.h5"
    build_inverse_dataset_from_records(records, compact_path, variants_per_case=2, seed=91)
    topology_path = tmp_path / "topology.h5"
    shutil.copy2(compact_path, topology_path)
    with h5py.File(topology_path, "r+") as h5:
        tokens_raw = h5["plan/compact_raw"][...]
        tokens_normalized = h5["plan/compact_normalized"][...]
        del h5["plan"]
        topology = h5.create_group("topology")
        topology.create_dataset("tokens_raw", data=tokens_raw)
        topology.create_dataset("tokens_normalized", data=tokens_normalized)
        topology.create_dataset("active_mask", data=(tokens_raw[..., 0] > 0.5).astype(np.uint8))
        topology.create_dataset(
            "relations",
            data=np.zeros((tokens_raw.shape[0], tokens_raw.shape[1], tokens_raw.shape[1], 9), dtype=np.float32),
        )
        h5.attrs["schema_name"] = TOPOLOGY_SET_DATASET_SCHEMA_NAME
        h5.attrs["topology_signature_schema_name"] = "honf_topology_signature"
        h5.attrs["topology_signature_schema_version"] = 3
        h5.attrs["forward_topology_checkpoint_sha256"] = "c" * 64

    summary = validate_inverse_hdf5(topology_path)
    assert summary["plan_token_mode"] == "exchangeable_set"
    assert summary["topology_schema_version"] == 3
    assert summary["forward_topology_checkpoint_sha256"] == "c" * 64
    dataset = InverseH5Dataset(topology_path, split="all")
    assert dataset.plan_token_mode == "exchangeable_set"
    assert dataset[0]["plan"].shape == (3, 12)
    dataset.close()

    with h5py.File(topology_path, "r+") as h5:
        h5.attrs["forward_topology_checkpoint_sha256"] = "not-a-digest"
    with np.testing.assert_raises_regex(ValueError, "checkpoint SHA-256"):
        validate_inverse_hdf5(topology_path)
