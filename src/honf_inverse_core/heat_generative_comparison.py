"""Paired frozen heat heads and immediately persisted individual draw evidence.

Public tasks never contain clean allocations or held observations. Training
supervision and evaluation references are separate caller-owned tensors. This
module launches no physical solver and establishes no organizer qualification.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import torch

from .models.centered_simplex_velocity import (
    CenteredSimplexVelocityDiffusion,
    HeatSimplexCondition,
    decode_heat_simplex,
    encode_heat_simplex,
    heat_provider_proxy,
)
from .models.frozen_packet_diffusion import dense_access


@dataclass(frozen=True)
class PublicHeatTask:
    case_id: str
    partition: str
    condition: HeatSimplexCondition
    provider: object
    predictor: object
    observed_reference: torch.Tensor | None = None  # public physical observations


def validate_review_stage(start, stop):
    if {0: 200, 200: 750, 750: 1500}.get(start) != stop:
        raise ValueError("Review stages must stop at 200, then resume to 750, then resume to 1500")


def save_heat_payload(payload, path):
    """Publish only a complete draw/checkpoint; preserve partial failure evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def observation_intervention(condition, mode):
    """Keep public locations fixed; remove or permute only observation values."""
    if mode == "original":
        return condition
    if mode not in {"removed", "shuffled"}:
        raise ValueError("Observation intervention must be original, removed or shuffled")
    features = condition.sensor_features.clone()
    available = (torch.ones_like(features, dtype=torch.bool) if condition.sensor_channel_valid is None
                 else condition.sensor_channel_valid.clone())
    if mode == "removed":
        features[..., 2:] = 0
        available[..., 2:] = False
    else:
        for batch in range(features.shape[0]):
            ids = torch.nonzero(condition.sensor_valid[batch], as_tuple=False).flatten()
            if ids.numel() > 1:
                features[batch, ids, 2:] = condition.sensor_features[batch, ids.roll(1), 2:]
                available[batch, ids, 2:] = available[batch, ids.roll(1), 2:].clone()
    return replace(condition, sensor_features=features, sensor_channel_valid=available)


def link_policy(task: PublicHeatTask, policy: str):
    """Uniform public-total baseline or current projected noisy candidate.

    The candidate policy may change both frozen embeddings and links with the
    noisy proxy. Paired training arms share the single call's exact tensors.
    Sampling trajectories subsequently differ and are labelled as such.
    """
    if policy == "candidate_projected":
        return task.provider
    if policy != "uniform_public_total":
        raise ValueError("Unknown frozen embedding/link policy")
    condition = task.condition
    with torch.no_grad():
        uniform = heat_provider_proxy(torch.zeros_like(condition.known_state), condition)
        links = CenteredSimplexVelocityDiffusion._detach_links(task.provider(uniform, condition))
    return lambda _candidate, _condition: links


class PairedHeatHeads:
    """Canonical equal initialization, two optimizers and one task/noise stream."""

    def __init__(self, denoiser, *, steps=20, learning_rate=3e-4, seed=0):
        self.graph = CenteredSimplexVelocityDiffusion(copy.deepcopy(denoiser), steps=steps)
        self.full = CenteredSimplexVelocityDiffusion(copy.deepcopy(denoiser), steps=steps)
        self.optimizers = {name: torch.optim.AdamW(model.parameters(), lr=learning_rate)
                           for name, model in self.models.items()}
        self.generator = torch.Generator(device=next(denoiser.parameters()).device).manual_seed(seed)
        self.update = 0
        self.history = []

    @property
    def models(self):
        return {"graph": self.graph, "full": self.full}

    def step(self, task: PublicHeatTask, target_heat, *, provider=None):
        if task.partition != "train":
            raise ValueError("Inverse updates and conditioning checks require train-only tasks")
        if self.update >= 1500:
            raise ValueError("The bounded comparison permits at most 1500 updates per arm")
        condition = task.condition
        clean = encode_heat_simplex(target_heat, condition.total_heat, condition.module_valid)
        time = torch.rand((clean.shape[0],), device=clean.device, generator=self.generator)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=self.generator)
        noisy, _ = self.graph.noisy_state_and_velocity_target(clean, condition, time, noise)
        # Both learned arms receive identical embeddings and typed link tensors.
        links = (self.graph.prepare_candidate_links(noisy, condition, provider or task.provider)
                 if bool((condition.free_dimensions > 0).any()) else None)
        row = {"update": self.update + 1, "case_id": task.case_id, "time": time.cpu().tolist()}
        for name, model in self.models.items():
            optimizer = self.optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(clean, condition, links=links, dense=name == "full", time=time, noise=noise)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {name} heat velocity loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite {name} heat head gradient")
            optimizer.step()
            row[name + "_loss"] = float(loss.detach())
        self.update += 1
        self.history.append(row)
        return row

    def training_conditioner_check(self, task, target_heat, *, provider=None):
        """Fixed train-only probe: held references never affect updates/checks."""
        if task.partition != "train":
            raise ValueError("Conditioner check must use a training record")
        condition = task.condition
        clean = encode_heat_simplex(target_heat, condition.total_heat, condition.module_valid)
        if not bool((condition.free_dimensions > 0).any()):
            return {"case_id": task.case_id, "partition": "train", "update": self.update,
                    "loss": {name: {mode: 0. for mode in ("original", "removed", "shuffled")} for name in self.models},
                    "same_weight_graph_full_velocity_rms": 0., "observation_value_gradient_norm": 0.,
                    "deterministic_zero_free_dimension": True}
        time = clean.new_full((clean.shape[0],), .5)
        noise = torch.arange(clean.numel(), device=clean.device, dtype=clean.dtype).reshape_as(clean) / max(clean.numel(), 1)
        noisy, _ = self.graph.noisy_state_and_velocity_target(clean, condition, time, noise)
        links = self.graph.prepare_candidate_links(noisy, condition, provider or task.provider)
        result = {"case_id": task.case_id, "partition": "train", "update": self.update, "loss": {}}
        with torch.no_grad():
            for name, model in self.models.items():
                result["loss"][name] = {
                    mode: float(model.training_loss(clean, observation_intervention(condition, mode),
                         links=links, dense=name == "full", time=time, noise=noise))
                    for mode in ("original", "removed", "shuffled")}
            graph = self.graph._velocity(noisy, time, condition, links)
            full = self.graph._velocity(noisy, time, condition, dense_access(links, condition))
            result["same_weight_graph_full_velocity_rms"] = float((graph-full).square().mean().sqrt())
        # Numerical value-channel gradient measures actual conditioner use.
        values = condition.sensor_features.detach().clone().requires_grad_()
        predicted = self.graph._velocity(noisy, time, replace(condition, sensor_features=values), links)
        derivative = torch.autograd.grad(predicted.square().sum(), values)[0]
        result["observation_value_gradient_norm"] = float(derivative[..., 2:].norm())
        return result

    def checkpoint(self):
        return {"checkpoint_format": "paired_fixed_total_heat_heads_v1", "update": self.update,
                "steps": self.graph.steps, "prediction_type": self.graph.prediction_type,
                "state_parameterization": self.graph.state_parameterization,
                "models": {name: model.state_dict() for name, model in self.models.items()},
                "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
                "generator_state": self.generator.get_state(), "history": self.history}

    def restore(self, state):
        if state.get("checkpoint_format") != "paired_fixed_total_heat_heads_v1" or not 0 <= state["update"] <= 1500:
            raise ValueError("Wrong paired heat checkpoint identity or update boundary")
        if (state.get("steps") != self.graph.steps or state.get("prediction_type") != self.graph.prediction_type
                or state.get("state_parameterization") != self.graph.state_parameterization):
            raise ValueError("Paired checkpoint heat parameterization or diffusion schedule changed")
        for name, model in self.models.items():
            model.load_state_dict(state["models"][name], strict=True)
            self.optimizers[name].load_state_dict(state["optimizers"][name])
        # Generator state is a CPU byte tensor, including for CUDA generators.
        self.generator.set_state(state["generator_state"].cpu())
        self.update, self.history = state["update"], list(state["history"])


@torch.no_grad()
def save_heat_draw(model, task: PublicHeatTask, held_reference, *, initial_noise,
                   path, dense=False, observation_mode="original", provider=None):
    """Save every completed draw, including raw/projection and physical sensors.

    Predictions use feasible projected allocations only. Raw negative heat is
    retained numerically rather than passed to the physical surrogate.
    """
    condition = observation_intervention(task.condition, observation_mode)
    start = perf_counter()
    elapsed, access_trail = [], []
    def record_progress(*_args):
        if condition.known_state.is_cuda:
            torch.cuda.synchronize(condition.known_state.device)
        elapsed.append(perf_counter()-start)
    def record_access(index, time, links):
        record = {"step": index, "time": time}
        for name in ("module_source", "sensor_source", "module_environment", "environment_module",
                     "sensor_environment", "module_coordinates", "sensor_coordinates"):
            value = getattr(links, name)
            if value is not None:
                record[name] = value.detach().cpu()
        access_trail.append(record)
    trail = model.sample(condition, provider or task.provider, dense=dense, initial_noise=initial_noise,
                         progress_callback=record_progress,
                         access_callback=record_access)
    states = torch.stack(trail.states)
    raw = torch.stack([decode_heat_simplex(state, condition.total_heat, condition.module_valid, project=False)
                       for state in trail.states])
    projected = torch.stack([decode_heat_simplex(state, condition.total_heat, condition.module_valid, project=True)
                             for state in trail.states])
    predictions = [task.predictor(heat) for heat in projected]
    observed = torch.stack([value["observed"] for value in predictions])
    held = torch.stack([value["held"] for value in predictions])
    if condition.known_state.is_cuda:
        torch.cuda.synchronize(condition.known_state.device)
    # Public observations remain the original physical target for interventions.
    reference = (task.condition.sensor_features[..., 2:3] if task.observed_reference is None
                 else task.observed_reference)
    payload = {"case_id": task.case_id, "partition": task.partition,
        "observation_mode": observation_mode, "access_mode": "full" if dense else "graph",
        "states": states.cpu(), "state_timesteps": trail.state_timesteps,
        "clean_estimates": torch.stack(trail.clean_estimates).cpu(),
        "clean_estimate_timesteps": trail.clean_estimate_timesteps,
        "raw_heat": raw.cpu(), "projected_heat": projected.cpu(),
        "projection_mask": (~torch.isclose(raw, projected, atol=1e-6, rtol=1e-6)).cpu(),
        "observed_predictions": observed.cpu(), "held_predictions": held.cpu(),
        "observed_reference": reference.cpu(), "held_reference": held_reference.cpu(),
        "observed_residual": (observed-reference).cpu(), "held_residual": (held-held_reference).cpu(),
        "observed_rmse": (observed-reference).square().flatten(1).mean(-1).sqrt().cpu(),
        "held_rmse": (held-held_reference).square().flatten(1).mean(-1).sqrt().cpu(),
        "sampling_elapsed_seconds": elapsed, "total_elapsed_seconds": perf_counter()-start,
        "typed_access_trail": access_trail,
        "free_dimensions": trail.free_dimensions.cpu(), "organizer_calls": trail.organizer_calls,
        "public_total_heat": condition.total_heat.cpu(), "initial_noise": initial_noise.cpu(),
        "projected_nonnegative": bool((projected >= 0).all()),
        "projected_total_preserved": bool(torch.allclose(projected.sum(2), condition.total_heat.expand(projected.shape[0], -1, -1), atol=1e-5, rtol=1e-5)),
        "units": {"heat": "adapter physical benchmark heat", "sensors": "adapter physical benchmark temperature", "time": "normalized diffusion time"},
        "timing_scope": "synchronized sampling includes frozen candidate provider and access recording; total additionally includes feasible physical sensor trail predictions",
        "evidence_limit": "frozen surrogate predictions; no independent physical solve; raw heat is not physically evaluated"}
    for name in ("peaks", "pressure"):
        if all(name in value for value in predictions):
            payload[name] = torch.stack([value[name] for value in predictions]).cpu()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    save_heat_payload(payload, path)
    return payload


__all__ = ["PairedHeatHeads", "PublicHeatTask", "link_policy", "observation_intervention", "save_heat_draw", "save_heat_payload", "validate_review_stage"]
