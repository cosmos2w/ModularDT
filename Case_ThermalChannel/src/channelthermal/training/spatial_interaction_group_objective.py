"""Head-only regression for paired spatial-group subdivision advantages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def conditional_utility_huber(core, features: torch.Tensor, label: torch.Tensor | float):
    """Return detached label/prediction and Huber gradients for the utility head.

    The feature vector and signed normalized advantage are detached before the
    auxiliary objective. Its coefficient is exactly one; macro case-count
    reduction belongs only to the paired physical losses.
    """
    if not callable(getattr(core, "predict_utility", None)):
        raise TypeError("Spatial interaction-group core must expose predict_utility(features).")
    head = getattr(core, "utility_head", None)
    if not isinstance(head, torch.nn.Module):
        raise TypeError("Spatial interaction-group core must expose its utility_head module.")
    detached_features = torch.as_tensor(features).detach()
    detached_label = torch.as_tensor(label, device=detached_features.device,
                                     dtype=detached_features.dtype).detach().reshape(())
    prediction = core.predict_utility(detached_features).reshape(())
    loss = F.huber_loss(prediction, detached_label, delta=1.0, reduction="sum")
    parameters = tuple(parameter for parameter in head.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("Spatial conditional-utility head has no trainable parameters.")
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    return prediction.detach(), loss.detach(), tuple(
        (parameter, gradient.detach()) for parameter, gradient in zip(parameters, gradients, strict=True)
    )


def merge_named_gradients(named_parameters: Mapping[str, Any], gradients, *, scale: float = 1.0) -> None:
    """Add a named gradient mapping to the existing macro accumulator once."""
    if not torch.isfinite(torch.as_tensor(scale)):
        raise ValueError("Gradient scale must be finite.")
    for name, gradient in gradients.items():
        parameter = named_parameters[name]
        contribution = gradient.detach().to(parameter) * float(scale)
        if parameter.grad is None:
            parameter.grad = contribution.clone()
        else:
            parameter.grad.add_(contribution)


__all__ = ["conditional_utility_huber", "merge_named_gradients"]
