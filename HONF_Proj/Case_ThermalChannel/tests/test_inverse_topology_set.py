from __future__ import annotations

import numpy as np

from channelthermal.inverse.topology_set import (
    extract_topology_set,
    topology_set_dataset_arrays,
)
from honf_forward_core.evaluation.topology_signature import extract_topology_signature
from honf_inverse_core.contracts import NamedContext, PhysicalDesign


def _signature() -> dict[str, np.ndarray]:
    module_incidence = np.asarray(
        [[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    environment_incidence = np.asarray(
        [[0.7, 0.3, 0.0], [0.2, 0.8, 0.0], [0.4, 0.6, 0.0]], dtype=np.float32
    )
    return extract_topology_signature(
        {
            "A_mh": module_incidence,
            "A_eh": environment_incidence,
            "edge_active_mask": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
            "edge_quality": np.asarray([0.8, 0.7, 0.0], dtype=np.float32),
            "hyper_source_coords": np.asarray([[1.0, 0.5], [3.0, 1.5], [0.0, 0.0]], dtype=np.float32),
            "hyper_region_coords": np.asarray([[2.0, 0.7], [4.0, 1.4], [0.0, 0.0]], dtype=np.float32),
            "hyper_source_scale": np.asarray([[0.2, 0.2], [0.3, 0.2], [0.0, 0.0]], dtype=np.float32),
            "hyper_region_scale": np.asarray([[0.6, 0.4], [0.8, 0.5], [0.0, 0.0]], dtype=np.float32),
            "hyper_module_mass": np.asarray([0.45, 0.55, 0.0], dtype=np.float32),
            "hyper_env_mass": np.asarray([0.5, 0.5, 0.0], dtype=np.float32),
        },
        np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        field_names=("temperature",),
        domain_length_x=6.0,
        domain_length_y=3.0,
        case_id="topology-set-test",
        forward_checkpoint_sha256="b" * 64,
        canonicalize=False,
    )


def test_topology_signature_converts_to_unordered_inverse_set() -> None:
    design = PhysicalDesign(
        module_centers=np.asarray([[1.0, 0.5], [3.0, 1.5], [0.0, 0.0]], dtype=np.float32),
        module_present=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        heat_powers=np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
    )
    context = NamedContext(
        feature_names=("domain_length_x", "domain_length_y"),
        vector=np.asarray([6.0, 3.0], dtype=np.float32),
        schema_name="test_context",
    )
    plan = extract_topology_set(_signature(), design, context)

    assert plan.raw.shape == (3, 12)
    assert np.array_equal(plan.raw[:, 0], plan.active_mask)
    assert np.all(plan.raw[plan.active_mask < 0.5] == 0.0)
    assert np.isclose(plan.raw[:, 10].sum(), 1.0)
    assert np.isclose(plan.raw[:, 11].sum(), 1.0)
    arrays = topology_set_dataset_arrays([plan, plan])
    assert arrays["tokens_normalized"].shape == (2, 3, 12)
    assert arrays["relations"].shape[:3] == (2, 3, 3)
