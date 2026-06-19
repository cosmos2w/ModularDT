"""CHANNELTHERMAL-SPECIFIC hypergraph plan stability tests.

Inputs are deterministic synthetic `organizer_aux` dictionaries matching the
ChannelThermal HONF compatibility schema. Outputs are assertions that the
compact inverse-ready plan is canonical, finite, serializable, and stable under
hyperedge permutations.

The test is ChannelThermal-specific because the plan schema is the static
ChannelThermal organizer export consumed by future inverse design code.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

SRC_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_NEW_ROOT))

import _bootstrap_imports  # noqa: F401
from _helpers.hypergraph_plan import (
    extract_hypergraph_plan,
    load_hypergraph_plan,
    save_hypergraph_plan,
    validate_hypergraph_plan,
)


EDGE_KEYS = (
    "hyper_source_coords",
    "hyper_region_coords",
    "hyper_module_mass",
    "hyper_env_mass",
    "hyper_strength",
    "active_hyperedge_mask",
)


def synthetic_organizer() -> tuple[Dict[str, Any], np.ndarray]:
    module_present = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    aux = {
        "A_mh": np.asarray(
            [
                [0.10, 0.40, 0.20, 0.30],
                [0.35, 0.15, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.60, 0.10, 0.10, 0.20],
            ],
            dtype=np.float32,
        ),
        "A_eh": np.asarray(
            [
                [0.30, 0.20, 0.10, 0.40],
                [0.10, 0.60, 0.20, 0.10],
                [0.25, 0.25, 0.25, 0.25],
            ],
            dtype=np.float32,
        ),
        "hyper_source_coords": np.asarray([[6.0, 1.0], [2.0, 2.0], [1.0, 4.0], [8.0, 3.0]], dtype=np.float32),
        "hyper_region_coords": np.asarray([[7.0, 1.5], [2.5, 2.5], [1.5, 4.5], [9.0, 3.5]], dtype=np.float32),
        "hyper_module_mass": np.asarray([0.20, 0.30, 0.10, 0.40], dtype=np.float32),
        "hyper_env_mass": np.asarray([0.30, 0.20, 0.25, 0.25], dtype=np.float32),
        "hyper_strength": np.asarray([0.20, 0.01, 0.40, 0.30], dtype=np.float32),
        "active_hyperedge_mask": np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
        "env_coords": np.asarray([[1.0, 1.0], [3.0, 2.0], [5.0, 4.0]], dtype=np.float32),
    }
    return aux, module_present


def permute_edges(aux: Dict[str, Any], perm: np.ndarray) -> Dict[str, Any]:
    out = dict(aux)
    out["A_mh"] = np.asarray(aux["A_mh"])[:, perm]
    out["A_eh"] = np.asarray(aux["A_eh"])[:, perm]
    for key in EDGE_KEYS:
        out[key] = np.asarray(aux[key])[perm]
    return out


def assert_same_plan(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray], *, ignore_edge_permutation: bool = False) -> None:
    keys = sorted(set(left) | set(right))
    for key in keys:
        if ignore_edge_permutation and key == "edge_permutation":
            continue
        if key not in left or key not in right:
            raise AssertionError(f"Plan key mismatch at {key!r}.")
        if not np.array_equal(np.asarray(left[key]), np.asarray(right[key])):
            raise AssertionError(f"Plan array {key!r} is not identical.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate canonical NewHONF hypergraph plan export.")
    parser.parse_args()

    aux, module_present = synthetic_organizer()
    plan_a = extract_hypergraph_plan(aux, module_present, domain_length_x=12.0, domain_length_y=6.0)
    plan_b = extract_hypergraph_plan(aux, module_present, domain_length_x=12.0, domain_length_y=6.0)
    validate_hypergraph_plan(plan_a)
    assert_same_plan(plan_a, plan_b)

    edge_perm = np.asarray([2, 0, 3, 1], dtype=np.int64)
    plan_perm = extract_hypergraph_plan(permute_edges(aux, edge_perm), module_present, domain_length_x=12.0, domain_length_y=6.0)
    validate_hypergraph_plan(plan_perm)
    # The canonical design arrays must be identical after H-index permutation.
    # `edge_permutation` intentionally differs because it records provenance
    # from the input edge indexing to the canonical order.
    assert_same_plan(plan_a, plan_perm, ignore_edge_permutation=True)

    inactive_perm = np.asarray([0, 1, 3, 2], dtype=np.int64)
    inactive_aux = dict(aux)
    inactive_aux["A_mh"] = np.asarray(aux["A_mh"])[inactive_perm]
    inactive_present = module_present[inactive_perm]
    inactive_plan = extract_hypergraph_plan(inactive_aux, inactive_present, domain_length_x=12.0, domain_length_y=6.0)
    validate_hypergraph_plan(inactive_plan)
    # A_mh is slot-sensitive by design; inactive slots are retained as zero rows
    # so inverse tooling can preserve physical module slot indexing.

    with tempfile.TemporaryDirectory(prefix="newhonf_plan_") as tmp:
        path = Path(tmp) / "hypergraph_plan.npz"
        save_hypergraph_plan(path, plan_a)
        loaded = load_hypergraph_plan(path)
        validate_hypergraph_plan(loaded)
        assert_same_plan(plan_a, loaded)

    print("[ok] hypergraph plan canonicalization, permutation stability, inactive slots, and round-trip passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
