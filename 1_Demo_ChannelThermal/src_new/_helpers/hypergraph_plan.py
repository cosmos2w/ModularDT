"""CHANNELTHERMAL-SPECIFIC compact hypergraph plan export.

Inputs are the legacy-compatible `organizer_aux` dictionary returned by the
ChannelThermal HONF wrapper plus the module-present mask for a case. Outputs
are small NumPy arrays describing the static organization: module/environment
assignments, source/region coordinates, masses, strengths, and active masks.

This is ChannelThermal-specific because it defines the inverse-design handoff
schema used by this demo. The dense learned `hyper_state`, query-dependent
`alpha_qk`, and raw module tokens are intentionally excluded: `alpha_qk` is
recomputed by the forward decoder for each query, and raw module tokens are
recomputed from any generated physical design.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


def _to_numpy(value: Any, *, detach: bool) -> np.ndarray:
    if torch.is_tensor(value):
        tensor = value.detach() if detach else value
        return tensor.cpu().numpy()
    return np.asarray(value)


def _squeeze_batch(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def extract_hypergraph_plan(
    organizer_aux: Dict[str, Any],
    module_present: Any,
    *,
    detach: bool = True,
) -> Dict[str, np.ndarray]:
    """Extract static organizer variables suitable for inverse-design seeding."""

    plan: Dict[str, np.ndarray] = {}
    for key in (
        "A_mh",
        "A_eh",
        "hyper_source_coords",
        "hyper_region_coords",
        "hyper_module_mass",
        "hyper_env_mass",
        "hyper_strength",
        "active_hyperedge_mask",
    ):
        if key in organizer_aux:
            plan[key] = _squeeze_batch(_to_numpy(organizer_aux[key], detach=detach)).astype(np.float32, copy=False)

    if "hyper_region_coords" not in plan and "hyper_thermal_region_coords" in organizer_aux:
        plan["hyper_region_coords"] = _squeeze_batch(
            _to_numpy(organizer_aux["hyper_thermal_region_coords"], detach=detach)
        ).astype(np.float32, copy=False)

    present = _squeeze_batch(_to_numpy(module_present, detach=detach)).astype(np.float32, copy=False)
    plan["module_present"] = present
    if "A_mh" in plan and plan["A_mh"].ndim == 2:
        plan["A_mh"] = plan["A_mh"] * present[:, None]

    if "hyper_strength" in plan and "active_hyperedge_mask" not in plan:
        plan["active_hyperedge_mask"] = (plan["hyper_strength"] > 0.05).astype(np.float32)
    if "active_hyperedge_mask" in plan:
        plan["active_hyperedge_mask"] = (plan["active_hyperedge_mask"] > 0.5).astype(np.float32)
    return plan

