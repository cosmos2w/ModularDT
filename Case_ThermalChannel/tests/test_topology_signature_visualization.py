from __future__ import annotations

import numpy as np

from channelthermal.evaluation_tools.topology_signature_visualization import (
    render_topology_signature_diagnostics,
)
from honf_forward_core.evaluation.topology_signature import extract_topology_signature


def test_all_topology_signature_views_are_written(tmp_path) -> None:
    generator = np.random.default_rng(251)
    x, y = np.meshgrid(np.linspace(0.0, 4.0, 5), np.linspace(0.0, 2.0, 4))
    module_present = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    module_incidence = generator.random((3, 4), dtype=np.float32)
    module_incidence /= module_incidence.sum(axis=-1, keepdims=True)
    environment_incidence = generator.random((6, 4), dtype=np.float32)
    environment_incidence /= environment_incidence.sum(axis=-1, keepdims=True)
    active = np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    routes = generator.random((x.size, 4), dtype=np.float32) * active
    routes /= routes.sum(axis=-1, keepdims=True)
    edge_fields = generator.normal(size=(x.size, 4, 2)).astype(np.float32) * routes[:, :, None]
    signature = extract_topology_signature(
        {
            "A_mh": module_incidence,
            "A_eh": environment_incidence,
            "edge_active_mask": active,
            "edge_quality": np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float32),
            "hyper_source_coords": np.asarray(
                [[0.7, 0.7], [1.8, 1.3], [3.0, 0.9], [0.0, 0.0]], dtype=np.float32
            ),
            "hyper_region_coords": np.asarray(
                [[1.4, 0.8], [2.7, 1.2], [3.7, 1.0], [0.0, 0.0]], dtype=np.float32
            ),
            "hyper_source_scale": np.full((4, 2), 0.2, dtype=np.float32),
            "hyper_region_scale": np.full((4, 2), 0.5, dtype=np.float32),
        },
        module_present,
        decoder_outputs={
            "query_hyper_attention": routes,
            "pred_field_by_edge": edge_fields,
        },
        reference_query_xy=np.stack((x.reshape(-1), y.reshape(-1)), axis=-1),
        field_names=("u", "temperature"),
        domain_length_x=4.0,
        domain_length_y=2.0,
    )
    sample = {
        "x_grid": x,
        "y_grid": y,
        "structure": {
            "module_centers": np.asarray([[0.7, 0.7], [1.8, 1.3], [0.0, 0.0]], dtype=np.float32),
            "module_present": module_present,
            "domain_length_x": np.asarray([4.0], dtype=np.float32),
            "domain_length_y": np.asarray([2.0], dtype=np.float32),
        },
    }

    outputs = render_topology_signature_diagnostics(
        tmp_path,
        sample,
        signature,
        edge_fields=edge_fields,
        field_names=("u", "temperature"),
    )

    assert set(outputs) == {
        "topology_active_regions",
        "topology_memberships",
        "topology_overlap_graph",
        "topology_field_contributions",
    }
    assert all((tmp_path / name).is_file() for name in (
        "topology_active_regions.png",
        "topology_memberships.png",
        "topology_overlap_graph.png",
        "topology_field_contributions.png",
    ))
