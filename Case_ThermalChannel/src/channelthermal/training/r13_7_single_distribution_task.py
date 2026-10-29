"""Maintained native Thermal provider for the R13-7 U reader."""
from __future__ import annotations

from ..member_routing import ThermalSingleDistributionMemberRoutingAdapter
from .r13_6_member_routing_task import (
    ThermalR136MemberRoutingTask,
    _build_thermal_member_routing_task,
)


CONFIG_SECTION_SCHEMA = "thermal_r13_7_member_routing_v1"
SOURCE_KIND = "scratch_r13_7_single_distribution_member_routing"
CORE_SCHEMA = "interaction_reused_single_distribution_member_routing_v1"
ARM = "R13-7-U-ME-SingleDistributionJoint95"
WORKFLOW = "interaction_reused_single_distribution_member_routing_training_v1"
PROVIDER_SCHEMA = "thermal_r13_7_single_distribution_member_routing_provider_v1"
TRAINING_STATE_SCHEMA = "thermal_r13_7_single_distribution_member_routing_training_state_v1"

_VARIANTS = {
    CORE_SCHEMA: {
        "adapter": ThermalSingleDistributionMemberRoutingAdapter,
        "arm": ARM,
        "workflow": WORKFLOW,
        "provider_schema": PROVIDER_SCHEMA,
        "training_state_schema": TRAINING_STATE_SCHEMA,
        "candidate": "U",
    },
}


class ThermalSingleDistributionMemberRoutingTask(ThermalR136MemberRoutingTask):
    """Reuse R13-6's native objective, sampling, provider, and paired dispatch."""

    VARIANTS = _VARIANTS
    EXPECTED_SOURCE_KIND = SOURCE_KIND
    LINEAGE = "R13-7 U"


def build_thermal_single_distribution_member_routing_task(config, *, arm=None, device="cpu"):
    """Build U's strict adapter with the maintained paired native provider."""
    return _build_thermal_member_routing_task(
        config, arm=arm, device=device,
        task_type=ThermalSingleDistributionMemberRoutingTask,
        variants=_VARIANTS,
        section_schema=CONFIG_SECTION_SCHEMA,
        source_kind=SOURCE_KIND,
        lineage="R13-7 U",
    )


__all__ = [
    "ARM", "CONFIG_SECTION_SCHEMA", "CORE_SCHEMA", "PROVIDER_SCHEMA", "SOURCE_KIND",
    "TRAINING_STATE_SCHEMA", "WORKFLOW", "ThermalSingleDistributionMemberRoutingTask",
    "build_thermal_single_distribution_member_routing_task",
]
