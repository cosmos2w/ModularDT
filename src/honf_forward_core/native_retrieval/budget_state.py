"""Serialized phase/noise schedule and device-resident native-work dual state."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn


class BudgetSchedule:
    """Absolute-epoch Round8 schedule; values are inputs to a recipe, not learned state."""

    schema = "native_budget_schedule_v2"
    dual_update_every = 10

    @staticmethod
    def state_at_epoch(epoch: int, optimizer_macro_step: int | None = None) -> dict[str, Any]:
        age = int(epoch)
        if age < 500:
            raise ValueError("Budgeted native schedule is bound to the Round7 e500 parent.")
        if age == 500:
            phase = "all_open_boundary"
            target = 1.0
            temperature = 1.0
            noise_scale = 1.0
            deterministic = True
            exact_expectation = False
        elif age <= 700:
            phase = "stochastic_dense_router"
            progress = (age - 501) / (700 - 501)
            target = 0.98 + progress * (0.85 - 0.98)
            temperature = 1.0 + progress * (0.5 - 1.0)
            noise_scale = 1.0
            deterministic = False
            exact_expectation = True
        elif age <= 1500:
            phase = "hard_selected_with_dense_router_steps"
            progress = (age - 701) / (1500 - 701)
            target = 0.85 + progress * (0.60 - 0.85)
            temperature = 0.5
            noise_scale = max(0.0, 1.0 - progress)
            deterministic = age >= 1500
            exact_expectation = False
        else:
            phase = "deterministic_hard"
            target = 0.60
            temperature = 0.5
            noise_scale = 0.0
            deterministic = True
            exact_expectation = False
        if age == 501:
            target, temperature = 0.98, 1.0
        if age == 700:
            target, temperature = 0.85, 0.5
        if age == 701:
            target = 0.85
        if age == 1500:
            target, noise_scale = 0.60, 0.0
        if age >= 1501:
            target = 0.60
        if optimizer_macro_step is None:
            dense_router_step: bool | None = None
        elif age <= 700:
            dense_router_step = True
        else:
            dense_router_step = (int(optimizer_macro_step) % 4) == 0
        return {
            "schema": BudgetSchedule.schema,
            "absolute_epoch": age,
            "phase": phase,
            "target_removable_retention": float(target),
            "rho": float(target),
            "temperature": float(temperature),
            "noise_scale": float(noise_scale),
            "deterministic": bool(deterministic),
            "dense_router_step": dense_router_step,
            "stochastic_expectation_exact": bool(exact_expectation),
            "dual_update_every": BudgetSchedule.dual_update_every,
            "budget_controller_source": "unit_noise_expectation" if exact_expectation else "observed_hard_work",
        }


_PHASE_CODES = {
    "uninitialized": 0,
    "all_open_boundary": 1,
    "stochastic_dense_router": 2,
    "hard_selected_with_dense_router_steps": 3,
    "deterministic_hard": 4,
}


class BudgetDualState(nn.Module):
    """Device-resident dual ascent state for a scalar equal-case residual.

    The caller forms the case native-fraction target from actual case floors and
    removable-row counts, then passes its residual to `penalty_residual` and
    `observe_residual`. This avoids treating removable retention as an all-row
    fraction when module counts/protected floors differ across cases.
    """

    def __init__(
        self,
        lambda0: float,
        *,
        update_every: int = 10,
        rng_identity: str = "round8_native_router_v2",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not math.isfinite(float(lambda0)) or lambda0 < 0.0:
            raise ValueError("lambda0 must be finite and nonnegative.")
        if update_every < 1:
            raise ValueError("update_every must be positive.")
        self.update_every = int(update_every)
        self.rng_identity = str(rng_identity)
        value = torch.tensor(float(lambda0), dtype=dtype)
        self.register_buffer("lambda0", value.clone())
        self.register_buffer("dual_lambda", value.clone())
        self.register_buffer("zeta", value.clone())
        self.register_buffer("lambda_max", 10.0 * value.clone())
        self.register_buffer("eta_lambda", value.clone())
        self.register_buffer("target_removable_retention", torch.tensor(0.98, dtype=dtype))
        self.register_buffer("temperature", torch.tensor(1.0, dtype=dtype))
        self.register_buffer("noise_scale", torch.tensor(1.0, dtype=dtype))
        self.register_buffer("phase_code", torch.tensor(_PHASE_CODES["uninitialized"], dtype=torch.long))
        self.register_buffer("macro_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("window_residual_sum", torch.zeros((), dtype=dtype))
        self.register_buffer("window_step_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("dual_update_count", torch.zeros((), dtype=torch.long))

    def get_extra_state(self) -> dict[str, str]:
        return {"rng_identity": self.rng_identity, "schema": BudgetSchedule.schema}

    def set_extra_state(self, state: dict[str, str]) -> None:
        if state.get("schema") != BudgetSchedule.schema:
            raise ValueError("Budget dual checkpoint schedule schema is incompatible.")
        self.rng_identity = state["rng_identity"]

    @property
    def phase(self) -> str:
        code = int(self.phase_code.detach().cpu())
        return next((name for name, value in _PHASE_CODES.items() if value == code), "unknown")

    @torch.no_grad()
    def apply_schedule(self, state: dict[str, Any], *, rng_identity: str | None = None) -> None:
        if state.get("schema") != BudgetSchedule.schema:
            raise ValueError("Budget schedule state has an incompatible schema.")
        phase = state["phase"]
        if phase not in _PHASE_CODES:
            raise ValueError(f"Unknown budget phase {phase!r}.")
        self.target_removable_retention.fill_(float(state["target_removable_retention"]))
        self.temperature.fill_(float(state["temperature"]))
        self.noise_scale.fill_(float(state["noise_scale"]))
        self.phase_code.fill_(_PHASE_CODES[phase])
        if rng_identity is not None:
            self.rng_identity = str(rng_identity)

    @torch.no_grad()
    def set_calibrated_lambda0(self, value: float | Tensor) -> None:
        calibrated = torch.as_tensor(value, dtype=self.lambda0.dtype, device=self.lambda0.device)
        if calibrated.numel() != 1 or not torch.isfinite(calibrated).all() or bool(calibrated < 0.0):
            raise ValueError("Calibrated lambda0 must be one finite nonnegative scalar.")
        self.lambda0.copy_(calibrated)
        self.dual_lambda.copy_(calibrated)
        self.zeta.copy_(calibrated)
        self.lambda_max.copy_(10.0 * calibrated)
        self.eta_lambda.copy_(calibrated)

    def penalty_residual(self, residual: Tensor) -> Tensor:
        """Return lambda*[C-target]+ + zeta/2*[C-target]+^2."""
        positive = torch.relu(residual)
        return self.dual_lambda.to(positive) * positive + 0.5 * self.zeta.to(positive) * positive.square()

    def penalty(self, case_native_fraction: Tensor, target_native_fraction: Tensor) -> Tensor:
        """Convenience wrapper when the caller has the case-specific native target."""
        return self.penalty_residual(case_native_fraction - target_native_fraction)

    @torch.no_grad()
    def observe_residual(self, residual: Tensor, macro_step: int) -> bool:
        """Accumulate device scalars and update the dual every complete window."""
        if residual.numel() != 1 or not torch.isfinite(residual).all():
            raise ValueError("Budget residual must be one finite scalar per optimizer macro-step.")
        step = int(macro_step)
        if step < 1:
            raise ValueError("macro_step is a one-based optimizer macro-step index.")
        value = residual.detach().to(device=self.window_residual_sum.device, dtype=self.window_residual_sum.dtype)
        self.window_residual_sum.add_(value)
        self.window_step_count.add_(1)
        self.macro_step.fill_(step)
        if step % self.update_every != 0:
            return False
        full_window = self.window_step_count == self.update_every
        if full_window.device.type == "cuda":
            torch._assert_async(full_window, "Dual update cadence reached without a complete residual window.")
        elif not bool(full_window):
            raise ValueError("Dual update cadence reached without a complete residual window.")
        mean_residual = self.window_residual_sum / self.window_step_count.to(self.window_residual_sum.dtype)
        candidate = self.dual_lambda + self.eta_lambda * mean_residual
        next_lambda = torch.minimum(torch.maximum(candidate, torch.zeros_like(candidate)), self.lambda_max)
        self.dual_lambda.copy_(next_lambda)
        self.window_residual_sum.zero_()
        self.window_step_count.zero_()
        self.dual_update_count.add_(1)
        return True

    @torch.no_grad()
    def observe_work(
        self,
        case_native_fraction: Tensor,
        target_native_fraction: Tensor,
        macro_step: int,
    ) -> bool:
        return self.observe_residual(case_native_fraction - target_native_fraction, macro_step)


__all__ = ["BudgetDualState", "BudgetSchedule"]
