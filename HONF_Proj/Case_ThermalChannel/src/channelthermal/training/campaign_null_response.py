"""Opt-in, benchmark-supported fixed-geometry heat-null training exposure.

Only selected training inputs define transfers and queries. No new reference
solution, perturbed temperature label, or additional optimizer step is used.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.response_control.contracts import DesignInput, RoleQuery
from channelthermal.response_control.native import DifferentiableThermalOperator
from channelthermal.response_control.thermal import pressure_drop_from_field

from .campaign import HYPERGRAPH_ARCHITECTURES
from .campaign_work import CampaignForwardWork, merge_forward_work

NULL_CHANNELS = ("u", "v", "p", "omega")


def fixed_total_transfer(heat, present, *, seed: int, fraction: float = .1, max_heat: float | None = None):
    """Input-selected positive donor/receiver pair, nonnegative fixed total."""
    values = np.asarray(heat, dtype=np.float64)
    active = np.flatnonzero((np.asarray(present) > .5) & (values > 0))
    if not 0 < fraction <= .25 or not np.isfinite(values).all():
        raise ValueError("Heat-null transfers require finite heat and fraction in (0,.25].")
    if active.size < 2:
        return None
    rng = np.random.default_rng(seed)
    pairs = [(int(i), int(j)) for i in active for j in active if i != j]
    for position in rng.permutation(len(pairs)):
        donor, receiver = pairs[int(position)]
        amplitude = fraction * min(values[donor], values[receiver])
        if max_heat is not None:
            amplitude = min(amplitude, max(0., float(max_heat) - values[receiver]))
        if amplitude <= 1e-8:
            continue
        direction = np.zeros_like(values)
        direction[donor], direction[receiver] = -amplitude, amplitude
        return direction, {"donor": donor, "receiver": receiver, "amplitude": amplitude,
                           "fraction": fraction, "total_heat": float(values.sum())}
    return None


def heat_null_loss(before, after, channel_names, train_scales):
    """Four native-unit increments, normalized by fixed nonzero train scales."""
    indices = [tuple(channel_names).index(name) for name in NULL_CHANNELS]
    scale = torch.as_tensor(train_scales, dtype=before.dtype, device=before.device).flatten()[indices]
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("Heat-null scales must be finite positive training channel scales.")
    return ((after[:, indices] - before[:, indices]) / scale).square().mean()


def null_role_queries(sample, field_names, *, device, fluid_count=256):
    """Geometry-only fixed fluid panel including maintained pressure sections."""
    structure = sample["structure"]
    positions = np.asarray(structure["module_centers"])
    present = np.asarray(structure["module_present"]) > .5
    material = np.asarray(structure["material_params"]).reshape(-1)
    lx = float(np.asarray(structure["domain_length_x"]).reshape(-1)[0])
    ly = float(np.asarray(structure["domain_length_y"]).reshape(-1)[0])
    radius = float(material[5])
    xs = (np.arange(64) + .5) / 64 * lx
    ys = (np.arange(32) + .5) / 32 * ly
    xx, yy = np.meshgrid(xs, ys)
    coords = np.stack((xx.ravel(), yy.ravel()), -1)
    valid = np.ones(coords.shape[0], dtype=bool)
    for center in positions[present]:
        valid &= np.linalg.norm(coords - center, axis=-1) > radius
    available = np.flatnonzero(valid)
    bands = [available[coords[available, 0] <= .08 * lx],
             available[coords[available, 0] >= .92 * lx]]
    if any(len(band) < 8 for band in bands) or len(available) < fluid_count:
        raise ValueError("Input geometry does not support the declared heat-null fluid panel.")
    fixed = np.concatenate([band[np.linspace(0, len(band) - 1, 8).round().astype(int)] for band in bands])
    rest = available[~np.isin(available, fixed)]
    chosen = np.concatenate((fixed, rest[np.linspace(0, len(rest) - 1, fluid_count - len(fixed)).round().astype(int)]))
    if len(np.unique(chosen)) != fluid_count:
        raise ValueError("Heat-null fluid panel must contain distinct geometry rows.")
    ports = np.asarray(sample["interface_condition"])[present, :, :3]
    slots = np.flatnonzero(present)
    # Sixteen fixed interior material points; no solved mask or temperature.
    angles = np.arange(16) * (2 * np.pi / 16)
    solid_xy = np.stack((.5 * np.cos(angles), .5 * np.sin(angles)), -1)
    return {
        "fluid_fields": RoleQuery("fluid_fields", torch.as_tensor(coords[chosen], dtype=torch.float32, device=device),
            tuple(field_names), tuple("benchmark_units" for _ in field_names), None, "eulerian"),
        "interface": RoleQuery("interface", torch.as_tensor(ports.reshape(-1, 3), dtype=torch.float32, device=device),
            ("T_surface", "q_normal"), ("benchmark_temperature", "benchmark_flux_proxy"),
            tuple(int(slot) for slot in slots for _ in range(ports.shape[1])), "interface_material_angle"),
        "solid_temperature": RoleQuery("solid_temperature", torch.as_tensor(np.tile(solid_xy, (len(slots), 1)), dtype=torch.float32, device=device),
            ("temperature",), ("benchmark_temperature",),
            tuple(int(slot) for slot in slots for _ in range(16)), "solid_material_normalized_xy"),
    }


class NativeHeatNullResponse:
    """Two eligible train cases per epoch; one fixed train-only calibration."""

    def __init__(self, model, dataset, dataset_config, settings):
        self.model, self.dataset_config = model, dataset_config
        self.settings = dict(settings)
        if self.settings.get("benchmark_verified") is not True:
            raise ValueError("Heat-null auxiliary requires recorded source/stored-control verification.")
        if self.settings.get("cases_per_epoch", 2) != 2 or self.settings.get("fluid_queries", 256) != 256:
            raise ValueError("This bounded heat-null protocol uses two cases and Q256.")
        self.dataset = GlobalChannelThermalDataset(dataset.path, split="train", points_per_case=256,
            normalize_inputs=False, normalize_targets=False, random_point_sampling=False,
            normalizer=dataset.normalizer, case_ids=dataset.selected_case_ids)
        self.stats = dataset.normalizer.stats
        self.eligible = []
        maxima = []
        for index, case_id in enumerate(self.dataset.selected_case_ids):
            group = self.dataset.h5["cases"][case_id]
            heat, present = np.asarray(group["heat_powers"]), np.asarray(group["module_present"]) > .5
            maxima.extend(heat[present].tolist())
            if int((present & (heat > 0)).sum()) >= 2:
                self.eligible.append(index)
        self.max_heat = float(max(maxima))
        if len(self.eligible) < 2:
            raise ValueError("Heat-null training requires two eligible selected-training cases.")

    def selected_indices(self, epoch):
        output = []
        for offset in range(2 * (epoch - 101), 2 * (epoch - 101) + 2):
            cycle, position = divmod(offset, len(self.eligible))
            order = np.random.default_rng(np.random.SeedSequence([0, cycle])).permutation(self.eligible)
            output.append(int(order[position]))
        return output

    def __call__(self, epoch, native_loss, accumulation_weight):
        started = time.perf_counter()
        losses, descriptions, forward_records = [], [], {}
        queries_charged = 0
        device = native_loss.device
        hypergraph = self.model.config.core_honf.forward_architecture in HYPERGRAPH_ARCHITECTURES
        for index in self.selected_indices(epoch):
            sample = self.dataset[index]
            structure = sample["structure"]
            transfer = fixed_total_transfer(structure["heat_powers"], structure["module_present"],
                seed=epoch * 1000 + index, fraction=float(self.settings.get("fraction", .1)), max_heat=self.max_heat)
            if transfer is None:
                continue
            direction, description = transfer
            description["case_id"] = self.dataset.selected_case_ids[index]
            material = np.asarray(structure["material_params"]).reshape(-1)
            context = {name: float(material[i]) for i, name in enumerate(
                ("nu", "solid_alpha", "fluid_alpha", "solid_k", "fluid_k", "module_radius"))}
            context.update({name: float(np.asarray(structure[name]).reshape(-1)[0]) for name in
                ("re", "u_in", "domain_length_x", "domain_length_y")})
            design = DesignInput(torch.as_tensor(structure["module_centers"], device=device),
                torch.as_tensor(structure["heat_powers"], device=device),
                torch.as_tensor(structure["module_present"] > .5, device=device))
            trial = DesignInput(design.module_positions, design.module_heating + torch.as_tensor(direction, dtype=design.module_heating.dtype, device=device), design.module_present)
            queries = null_role_queries(sample, self.model.config.channelthermal.field_names, device=device)
            operator = DifferentiableThermalOperator(self.model, sample, dataset_config=self.dataset_config,
                normalization_stats=self.stats, query_batch_size=256, organizer_shadow=hypergraph)
            with CampaignForwardWork(self.model.core) as measured:
                before = operator(design, context, queries).role_values["fluid_fields"]
                after = operator(trial, context, queries).role_values["fluid_fields"]
            merge_forward_work(forward_records, measured.records)
            losses.append(heat_null_loss(before, after, queries["fluid_fields"].channel_names, self.stats["field_std_by_channel"]))
            description["pressure_drop_increment"] = float((pressure_drop_from_field(after, queries["fluid_fields"], trial, context)
                - pressure_drop_from_field(before, queries["fluid_fields"], design, context)).detach())
            descriptions.append(description)
            queries_charged += 2 * sum(query.query_features.shape[0] for query in queries.values())
        if not losses:
            raise ValueError("No eligible bounded training heat transfer was realized.")
        auxiliary = torch.stack(losses).mean()
        state = getattr(self.model, "campaign_training_state", {})
        if "heat_null_coefficient" not in state:
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            native_grad = torch.autograd.grad(native_loss * accumulation_weight, parameters, retain_graph=True, allow_unused=True)
            null_grad = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
            native_norm = sum(float(((torch.zeros_like(p) if g is None else g.detach())
                + (torch.zeros_like(p) if p.grad is None else p.grad.detach())).square().sum())
                for p, g in zip(parameters, native_grad)) ** .5
            null_norm = sum(float(g.detach().square().sum()) for g in null_grad if g is not None) ** .5
            coefficient = min(.1, .1 * native_norm / null_norm) if native_norm > 1e-10 and null_norm > 1e-10 else 0.
            state.update(heat_null_coefficient=coefficient, heat_null_native_gradient_norm=native_norm,
                heat_null_gradient_norm=null_norm, heat_null_calibration_epoch=epoch,
                heat_null_calibration_scope="one training-only gradient calibration; at most ten percent")
            self.model.campaign_training_state = state
        coefficient = float(state["heat_null_coefficient"])
        return coefficient * auxiliary / accumulation_weight, {
            "heat_null_loss": float(auxiliary.detach()), "heat_null_coefficient": coefficient,
            "heat_null_cases": descriptions, "heat_null_eligible_train_cases": len(self.eligible),
            "heat_null_wrapper_calls": len(losses) * (4 if hypergraph else 2),
            "heat_null_primary_fluid_queries": len(losses) * 2 * 256,
            "heat_null_role_queries": queries_charged, "heat_null_seconds": time.perf_counter() - started,
            "heat_null_forward_work": forward_records,
            "heat_null_scope": "selected train inputs; null u/v/p/omega only; temperature unsupervised; no extra optimizer boundaries",
        }
