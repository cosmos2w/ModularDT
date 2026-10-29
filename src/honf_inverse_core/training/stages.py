"""Explicit freeze/mixing policy for the four inverse training stages."""

from __future__ import annotations

from typing import Any

from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner


TRAINING_STAGES = (
    "stage_plan",
    "stage_layout_teacher_plan",
    "stage_layout_mixed_plan",
    "stage_joint_consistency",
)


def _requires(module: Any, enabled: bool) -> None:
    if module is not None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)


def configure_stage(designer: HierarchicalInverseDesigner, stage: str) -> dict[str, bool]:
    """Apply stage ownership and return the resulting trainable-module policy."""

    if stage not in TRAINING_STAGES:
        raise ValueError(f"Unknown inverse training stage: {stage!r}")
    policy = {
        # Stage two learns the layout decoder against the representation fixed
        # by stage one's best plan model. Moving the shared encoder while the
        # plan flow is frozen would silently invalidate q(G|R,c). Stage three
        # deliberately unfreezes the whole unguided hierarchy again.
        "request_encoder": stage in {"stage_plan", "stage_layout_mixed_plan"},
        "plan_flow": stage in {"stage_plan", "stage_layout_mixed_plan"},
        "layout_flow": stage in {"stage_layout_teacher_plan", "stage_layout_mixed_plan", "stage_joint_consistency"},
        "corrector": stage == "stage_joint_consistency" and designer.corrector is not None,
    }
    _requires(designer.request_encoder, policy["request_encoder"])
    _requires(designer.plan_flow, policy["plan_flow"])
    _requires(designer.layout_flow, policy["layout_flow"])
    _requires(designer.corrector, policy["corrector"])
    return policy


def generated_plan_probability(epoch: int, total_epochs: int, *, maximum: float = 0.5) -> float:
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")
    return float(maximum) * min(max((int(epoch) + 1) / float(total_epochs), 0.0), 1.0)


__all__ = ["TRAINING_STAGES", "configure_stage", "generated_plan_probability"]
