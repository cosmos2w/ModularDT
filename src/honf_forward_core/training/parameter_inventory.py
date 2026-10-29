"""Count actual materialized wrapper parameters by scientific component."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def _component(name: str) -> str:
    if name.startswith("local_coupling.local_surrogate."):
        return "Stage_A"
    if name.startswith("core.backend.organizer."):
        return "organizer"
    if name.startswith(("core.backend.control_gain.", "core.backend.control_score.")):
        return "interaction_control"
    if name.startswith(("core.module_feature_encoder.", "core.module_position_encoder.", "core.env_encoder.", "core.global_encoder.")):
        return "source_encoders"
    if name.startswith(("core.backend.mm_message.", "core.backend.me_message.", "core.backend.em_message.",
                        "core.backend.module_update.", "core.backend.env_update.")):
        return "fine_MM_ME_EM"
    if name.startswith(("core.backend.query_module_", "core.backend.env_query.", "core.backend.env_geometry_bias.",
                        "core.backend.env_attention.", "core.common.field_head.", "core.common.context_norm.",
                        "core.common.global_background.")):
        return "fine_QM_QE_and_field_head"
    if name.startswith(("core.common.coarse_", "core.common.local_")):
        return "Dense_coarse_local"
    if name.startswith(("local_coupling.", "fallback_heads.")):
        return "Thermal_coupling_and_fallback_heads"
    return "other"


def materialized_parameter_inventory(model: torch.nn.Module) -> dict[str, Any]:
    """Count distinct parameters, retaining names and trainable/frozen status.

    Parameters allocated but unused by the current task remain counted. This
    describes model capacity and freeze policy, not active gradients or cost.
    Unmaterialized lazy parameters raise rather than receiving inferred sizes.
    """

    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "trainable_scalars": 0, "frozen_scalars": 0,
        "trainable_tensors": 0, "frozen_tensors": 0, "parameters": [],
    })
    for name, parameter in model.named_parameters():
        if isinstance(parameter, torch.nn.parameter.UninitializedParameter):
            raise TypeError(f"Parameter inventory requires native materialization: {name}")
        category = _component(name)
        status = "trainable" if parameter.requires_grad else "frozen"
        groups[category][status + "_scalars"] += parameter.numel()
        groups[category][status + "_tensors"] += 1
        groups[category]["parameters"].append({"name": name, "shape": list(parameter.shape),
                                             "scalars": parameter.numel(), "status": status})
    totals = {key: sum(group[key] for group in groups.values())
              for key in ("trainable_scalars", "frozen_scalars", "trainable_tensors", "frozen_tensors")}
    return {"scope": "actual distinct materialized wrapper parameters; allocated capacity includes task-unused parameters",
            "totals": totals, "components": dict(groups)}


__all__ = ["materialized_parameter_inventory"]
