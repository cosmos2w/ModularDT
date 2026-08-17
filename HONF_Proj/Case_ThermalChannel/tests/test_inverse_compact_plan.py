from __future__ import annotations

import numpy as np
import pytest

from honf_forward_core.evaluation.hypergraph_plan import extract_hypergraph_plan
from honf_inverse_core.contracts import PhysicalDesign
from channelthermal.inverse.compact_plan import (
    COMPACT_PLAN_FEATURE_NAMES,
    denormalize_compact_plan,
    extract_compact_plan,
    normalize_compact_plan,
    validate_compact_plan,
)
from channelthermal.inverse.context import parse_context


def _context():
    return parse_context(
        {
            "schema_name": "thermalchannel_inverse_context",
            "schema_version": 1,
            "re": 100.0,
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


def _full_and_design():
    design = PhysicalDesign(
        module_centers=np.asarray([[0.5, 0.5], [2.0, 1.0], [3.5, 1.5]], dtype=np.float32),
        module_present=np.ones(3, dtype=np.float32),
        heat_powers=np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
    )
    A_mh = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]], dtype=np.float32)
    A_eh = np.asarray([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.1, 0.1, 0.8]], dtype=np.float32)
    env_coords = np.asarray([[0.5, 0.5], [1.5, 0.5], [2.5, 1.5], [3.5, 1.5]], dtype=np.float32)
    module_mass = A_mh.sum(axis=0) / A_mh.sum()
    env_mass = A_eh.sum(axis=0) / A_eh.sum()
    source = (A_mh / A_mh.sum(axis=0, keepdims=True)).T @ design.module_centers
    region = (A_eh / A_eh.sum(axis=0, keepdims=True)).T @ env_coords
    strength = np.sqrt(module_mass * env_mass + 1.0e-6)
    organizer = {
        "A_mh": A_mh,
        "A_eh": A_eh,
        "hyper_source_coords": source,
        "hyper_region_coords": region,
        "hyper_module_mass": module_mass,
        "hyper_env_mass": env_mass,
        "hyper_strength": strength,
        "active_hyperedge_mask": (strength > 0.05).astype(np.float32),
        "env_coords": env_coords,
    }
    full = extract_hypergraph_plan(
        organizer,
        design.module_present,
        domain_length_x=4.0,
        domain_length_y=2.0,
    )
    return full, design


def test_compact_plan_extract_and_normalization_round_trip() -> None:
    full, design = _full_and_design()
    compact = extract_compact_plan(full, design, _context())
    assert compact.raw.shape == (3, 12)
    assert compact.feature_names == COMPACT_PLAN_FEATURE_NAMES
    assert np.sum(compact.raw[:, 10]) == pytest.approx(1.0)
    assert np.sum(compact.raw[:, 11]) == pytest.approx(1.0)
    assert np.allclose(
        denormalize_compact_plan(normalize_compact_plan(compact.raw, _context()), _context()),
        compact.raw,
        atol=1.0e-6,
    )
    validate_compact_plan(compact.raw, _context())


def test_compact_plan_rejects_activity_strength_disagreement() -> None:
    full, design = _full_and_design()
    raw = extract_compact_plan(full, design, _context()).raw.copy()
    raw[0, 0] = 1.0 - raw[0, 0]
    with pytest.raises(ValueError, match="strength > 0.05"):
        validate_compact_plan(raw, _context())


def test_compact_excludes_dense_generated_variables() -> None:
    full, design = _full_and_design()
    compact = extract_compact_plan(full, design, _context())
    assert "A_mh" not in compact.feature_names
    assert "A_eh" not in compact.feature_names
    assert "hyper_state" not in compact.feature_names
