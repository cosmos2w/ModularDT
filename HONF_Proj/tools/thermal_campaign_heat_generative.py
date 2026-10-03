#!/usr/bin/env python3
"""Bounded, staged paired heat heads through an explicitly qualified organizer.

No training runs on import. Invoke only after the organizer milestone review.
The supplied total is public; individual heat targets and held sensors are
stored separately from public tasks. No physical solver is invoked.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from thermal_campaign_evaluate import screen_indices
from thermal_campaign_heat_inference import native_heat_predictor, sensor_panel

from honf_inverse_core.heat_generative_comparison import (
    PairedHeatHeads,
    PublicHeatTask,
    link_policy,
    save_heat_draw,
    save_heat_payload,
    validate_review_stage,
)
from honf_inverse_core.models.centered_simplex_velocity import HeatSimplexCondition
from honf_inverse_core.models.frozen_packet_diffusion import PacketLinks, SpatialConditionalPacketDenoiser


def public_module_features(sample, total):
    """Known geometry/material/context and public total, never slot heat."""
    structure = sample["structure"]
    coordinates = np.asarray(structure["module_centers"], dtype=np.float32)
    material = np.asarray(structure["material_params"], dtype=np.float32).reshape(-1)
    context = np.asarray([float(np.asarray(structure[name]).reshape(-1)[0])
                          for name in ("re", "u_in", "domain_length_x", "domain_length_y")], dtype=np.float32)
    common = np.concatenate((material, context, np.asarray([total], dtype=np.float32)))
    return np.concatenate((coordinates, np.broadcast_to(common, (len(coordinates), len(common)))), -1)


def public_template(sample):
    """Strip hidden allocations, solved interfaces and all physical targets."""
    structure = sample["structure"]
    names = ("module_centers", "module_present", "material_params", "re", "u_in",
             "domain_length_x", "domain_length_y")
    return {"structure": {name: np.asarray(structure[name]).copy() for name in names},
            # The first three columns are public receiver geometry, not solved values.
            "interface_condition": np.asarray(sample["interface_condition"])[..., :3].copy(),
            "module_internal_query_points": np.asarray(sample["module_internal_query_points"]).copy()}


@dataclass(frozen=True)
class TrainOnlyNormalization:
    module_mean: np.ndarray
    module_std: np.ndarray
    sensor_mean: np.ndarray
    sensor_std: np.ndarray
    case_ids: tuple[str, ...]

    @classmethod
    def fit(cls, dataset):
        if len(dataset) != 600 or dataset.split != "train":
            raise ValueError("Scientific inverse normalization requires exactly the 600 training records")
        modules, sensors, ids = [], [], []
        for index in range(len(dataset)):
            sample = dataset[index]
            active = np.asarray(sample["structure"]["module_present"]) > .5
            total = float(np.asarray(sample["structure"]["heat_powers"]).sum())
            modules.append(public_module_features(sample, total)[active])
            coordinates, grid_rows, _, observed, _ = sensor_panel(sample)
            values = np.asarray(sample["steady_field"]).reshape(-1, len(dataset.channel_order))[
                grid_rows[observed], list(dataset.channel_order).index("temperature")]
            sensors.append(np.concatenate((coordinates[observed], values[:, None]), -1))
            ids.append(str(sample["case_id"]))
        module, sensor = np.concatenate(modules), np.concatenate(sensors)
        return cls(module.mean(0), module.std(0).clip(1e-6), sensor.mean(0), sensor.std(0).clip(1e-6), tuple(ids))

    def metadata(self):
        return {"module_mean": self.module_mean.tolist(), "module_std": self.module_std.tolist(),
                "sensor_mean": self.sensor_mean.tolist(), "sensor_std": self.sensor_std.tolist(),
                "case_ids": self.case_ids, "partition": "train", "records": len(self.case_ids)}


def typed_packet_links(model, prepared, coordinates):
    """Common final-phase export with all five physical typed link matrices."""
    native = prepared.prepared
    state, encoded = native.backend_state, native.encoded
    exported = model.core.export_typed_hypergraph(native)
    module_valid, env_valid = exported["source_valid"]["M"], exported["source_valid"]["E"]
    modules = module_valid.shape[-1]
    actual = state["hypergraph_accesses"]
    mm = actual["MM"].weight * (module_valid[:, :, None] & module_valid[:, None, :]
          & ~torch.eye(modules, device=module_valid.device, dtype=torch.bool)[None])
    sensor = coordinates[None]
    access = exported["receiver_access"]
    return PacketLinks(mm, access(sensor, "QM").weight * module_valid[:, None],
        state["module_tokens"], actual["ME"].weight * module_valid[:, :, None] * env_valid[:, None],
        actual["EM"].weight * env_valid[:, :, None] * module_valid[:, None],
        access(sensor, "QE").weight * env_valid[:, None], state["env_tokens"], env_valid,
        encoded.module_centers, sensor)


def build_public_task(model, checkpoint, sample, partition, normalization, channel_order):
    """Construct the conditioner before exposing clean allocation supervision."""
    device = next(model.parameters()).device
    coordinates, grid_rows, _, observed, held = sensor_panel(sample)
    # Public sum is extracted once from this stored benchmark's supplied budget.
    total = float(np.asarray(sample["structure"]["heat_powers"]).sum())
    active = torch.as_tensor(np.asarray(sample["structure"]["module_present"]) > .5, device=device)[None]
    fields = np.asarray(sample["steady_field"]).reshape(-1, len(channel_order))
    observed_values = fields[grid_rows[observed], list(channel_order).index("temperature")]
    sensor_features = np.concatenate((coordinates[observed], observed_values[:, None]), -1)
    modules = (public_module_features(sample, total)-normalization.module_mean)/normalization.module_std
    sensor_features = (sensor_features-normalization.sensor_mean)/normalization.sensor_std
    condition = HeatSimplexCondition(torch.zeros((*active.shape, 1), device=device), active, active,
        torch.as_tensor(modules, dtype=torch.float32, device=device)[None],
        torch.as_tensor(sensor_features, dtype=torch.float32, device=device)[None],
        torch.ones((1, len(observed)), dtype=torch.bool, device=device),
        total_heat=torch.tensor([[total]], dtype=torch.float32, device=device))
    physical_predictor, _ = native_heat_predictor(model, checkpoint, public_template(sample), coordinates, observed, held)

    def predictor(heat):
        prediction = physical_predictor(heat[0, :, 0])
        return {name: value[None, ..., None] if name in {"observed", "held"} else value[None]
                for name, value in prediction.items() if torch.is_tensor(value)}

    def provider(heat, public_condition):
        # Only projected caller-owned candidate heat and public task reach the adapter.
        if public_condition.total_heat is None or not torch.allclose(heat.sum(1), public_condition.total_heat):
            raise ValueError("Candidate provider must preserve the public total")
        captured = []
        hook = model.register_forward_hook(lambda _model, _args, output: captured.append(output.get("prepared_state")))
        try:
            with torch.no_grad():
                physical_predictor(heat[0, :, 0])
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0] is None:
            raise RuntimeError("Native model did not expose one final prepared candidate")
        xy = torch.as_tensor(coordinates[observed], device=device)
        return typed_packet_links(model, captured[0], xy)

    task = PublicHeatTask(str(sample["case_id"]), partition, condition, provider, predictor,
                         torch.as_tensor(observed_values, device=device)[None, :, None])
    # These tensors are returned separately and never captured by providers.
    target = torch.as_tensor(sample["structure"]["heat_powers"], device=device, dtype=torch.float32)[None, :, None]
    held_reference = torch.as_tensor(fields[grid_rows[held], list(channel_order).index("temperature")],
                                     device=device)[None, :, None]
    return task, target, held_reference


def eligible_training_probe(dataset):
    """Input-only selection excludes zero-free-dimension/public-zero-budget tasks."""
    if dataset.split != "train":
        raise ValueError("Conditioner probe selection is training-only")
    for index in range(len(dataset)):
        sample = dataset[index]
        if (np.count_nonzero(np.asarray(sample["structure"]["module_present"]) > .5) >= 2
                and float(np.asarray(sample["structure"]["heat_powers"]).sum()) > 0):
            return sample
    raise ValueError("No training record has M>=2 and positive public total for a meaningful conditioner probe")


def run_comparison(args):
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.evaluation.loading import load_model

    from honf_runtime.compat import load_trusted_checkpoint

    output = args.output_dir.resolve()
    if not any(output.is_relative_to(PROJECT_ROOT / root) for root in ("diagnostics", "Trained_Results")):
        raise ValueError("Generative evidence must remain in ignored local output paths")
    if (output / "identity.json").exists() and not args.resume:
        raise ValueError("Existing paired-head evidence requires its explicit saved checkpoint continuation")
    model, checkpoint = load_model(args.checkpoint, torch.device(args.device))
    model.eval().requires_grad_(False)
    if not hasattr(model.core.backend, "organizer"):
        raise ValueError("A qualified typed organizer checkpoint is required")
    train = GlobalChannelThermalDataset(args.dataset, split="train", points_per_case=1,
                                      random_point_sampling=False, include_grid=True)
    dev = GlobalChannelThermalDataset(args.dataset, split="test", points_per_case=1,
                                    random_point_sampling=False, include_grid=True)
    normalization = TrainOnlyNormalization.fit(train)
    probe_sample = eligible_training_probe(train)
    first, first_target, _ = build_public_task(model, checkpoint, probe_sample, "train", normalization, train.channel_order)
    with torch.no_grad():
        links = first.provider(first.condition.module_valid[..., None]*first.condition.total_heat[:, None]
                               / first.condition.module_valid.sum(-1)[:, None, None], first.condition)
    with torch.random.fork_rng(devices=[torch.device(args.device)] if torch.device(args.device).type == "cuda" else []):
        torch.manual_seed(args.seed)
        denoiser = SpatialConditionalPacketDenoiser(design_dim=1, module_dim=first.condition.module_features.shape[-1],
            sensor_dim=3, embedding_dim=links.module_embeddings.shape[-1], coordinate_dim=2,
            hidden_dim=args.hidden_dim, layers=2).to(args.device)
    heads = PairedHeatHeads(denoiser, steps=args.steps, learning_rate=args.learning_rate, seed=args.seed)
    task_stream = torch.Generator().manual_seed(args.seed+1)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"forward_checkpoint": str(args.checkpoint.resolve()), "forward_epoch": checkpoint.get("epoch"),
        "forward_run_identity": checkpoint.get("train_config", {}).get("run"),
        "forward_model_config": checkpoint.get("model_config"),
        "dataset": str(args.dataset.resolve()), "seed": args.seed, "hidden_dim": args.hidden_dim,
        "steps": args.steps, "learning_rate": args.learning_rate, "embedding_policy": args.embedding_policy,
        "normalization": normalization.metadata(), "qualification": args.qualified_organizer,
        "pairing_scope": "one current noisy projected candidate and frozen links/embeddings per paired training update; independent reverse trajectories use same initial noise but subsequently differing candidates",
        "baseline_scope": "uniform_public_total fixes both links and embeddings from uniform public budget; candidate_projected rebuilds both from the current noisy feasible proxy",
        "condition_inputs": "known geometry, material/context, supplied total and six named observations; hidden individual heat is a separate training target only",
        "reference_limit": "exposed development records, frozen surrogate; no new physical solve",
        "alternative_observation_limit": "distinct stored development tasks are used; no separately matched same-geometry alternative physical target is inferred or fabricated"}
    identity["intervention_scope"] = "96 original draws: 12 tasks x4 noises x2 heads; three labelled same-weight/observation controls on4 eligible tasks x1 noise,12 extra draws"
    identity["training_probe_selection"] = {"case_id": first.case_id, "criteria": "known M>=2 and positive public total; no hidden individual allocation criterion"}
    if args.resume:
        saved = load_trusted_checkpoint(args.resume, map_location=args.device)
        if saved["identity"] != identity:
            raise ValueError("Resume changed the frozen checkpoint, data, normalization or head policy")
        heads.restore(saved["heads"]); task_stream.set_state(saved["task_stream"])
    if args.evaluate_only:
        if not args.resume:
            raise ValueError("Evaluation recovery requires an exact paired-head checkpoint")
        evaluate_draws(heads, model, checkpoint, dev, normalization, output / f"review_{heads.update:04d}", args)
        return output
    validate_review_stage(heads.update, args.stop_update)
    (output / "identity.json").write_text(json.dumps(identity, indent=2)+"\n")
    check = heads.training_conditioner_check(first, first_target, provider=link_policy(first, args.embedding_policy))
    initial_check = saved["initial_conditioner"] if args.resume else check
    (output / f"train_conditioner_before_{heads.update:04d}.json").write_text(json.dumps(check, indent=2)+"\n")
    while heads.update < args.stop_update:
        index = int(torch.randint(len(train), (), generator=task_stream))
        task, target, _ = build_public_task(model, checkpoint, train[index], "train", normalization, train.channel_order)
        row = heads.step(task, target, provider=link_policy(task, args.embedding_policy))
        with (output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        if heads.update in {200, 750, args.stop_update}:
            # Review checkpoints are recoverable before potentially costly evaluation.
            save_heat_payload({"identity": identity, "heads": heads.checkpoint(), "task_stream": task_stream.get_state(),
                        "initial_conditioner": initial_check},
                       output / f"update_{heads.update:04d}.pt")
            check = heads.training_conditioner_check(first, first_target, provider=link_policy(first, args.embedding_policy))
            check["fixed_train_probe_original_loss_change_since_initial"] = {
                name: check["loss"][name]["original"]-initial_check["loss"][name]["original"]
                for name in heads.models}
            (output / f"train_conditioner_{heads.update:04d}.json").write_text(json.dumps(check, indent=2)+"\n")
            evaluate_draws(heads, model, checkpoint, dev, normalization, output / f"review_{heads.update:04d}", args)
            print(f"review {heads.update}: complete paired updates and individual draw files", flush=True)
    return output


def evaluate_draws(heads, model, checkpoint, dataset, normalization, output, args):
    control_tasks = []
    for index in screen_indices(dataset, 12):
        task, reference_heat, held = build_public_task(model, checkpoint, dataset[index], "development", normalization, dataset.channel_order)
        directory = output / task.case_id
        directory.mkdir(parents=True, exist_ok=True)
        reference_path = directory / "evaluation_reference.pt"
        if not reference_path.exists():
            save_heat_payload({"case_id": task.case_id, "reference_heat": reference_heat.cpu(), "held": held.cpu(),
                        "observation": task.observed_reference.cpu(), "public_total": task.condition.total_heat.cpu(),
                        "reference_heat_scope": "evaluation only; absent from public task/provider/embeddings"}, reference_path)
        provider = link_policy(task, args.embedding_policy)
        controls = len(control_tasks) < 4 and int(task.condition.free_dimensions[0]) > 0 and float(task.condition.total_heat[0, 0]) > 0
        if controls:
            control_tasks.append(task.case_id)
        generator = torch.Generator(device=task.condition.known_state.device).manual_seed(args.seed+100000+int(index))
        for draw in range(4):
            noise = torch.randn(task.condition.known_state.shape, device=task.condition.known_state.device, generator=generator)
            variants = [("graph", heads.graph, False, "original"), ("full", heads.full, True, "original")]
            if controls and draw == 0:
                variants += [("graph_weight_full_access", heads.graph, True, "original"),
                    ("graph_observations_removed", heads.graph, False, "removed"),
                    ("graph_observations_shuffled", heads.graph, False, "shuffled")]
            for name, head, dense, observation in variants:
                path = output / task.case_id / f"draw_{draw:02d}_{name}.pt"
                if path.exists():
                    if not args.evaluate_only:
                        raise ValueError(f"Preserve existing individual draw evidence: {path}")
                    from honf_runtime.compat import load_trusted_checkpoint
                    saved = load_trusted_checkpoint(path, map_location="cpu")
                    if (saved["case_id"] != task.case_id or not torch.equal(saved["initial_noise"], noise.cpu())
                            or "held_predictions" not in saved or len(saved["state_timesteps"]) != args.steps+1):
                        raise ValueError(f"Existing individual draw identity/integrity mismatch: {path}")
                    continue
                save_heat_draw(head, task, held, initial_noise=noise, path=path, dense=dense,
                               observation_mode=observation, provider=provider)
    (output / "draw_scope.json").write_text(json.dumps({"original_tasks": 12, "noises_per_task": 4,
        "original_heads": ["graph", "full"], "intervention_tasks": control_tasks, "intervention_noises_per_task": 1,
        "interventions": ["graph_weight_full_access", "graph_observations_removed", "graph_observations_shuffled"]}, indent=2)+"\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qualified-organizer", required=True, help="Reviewed finalist qualification lineage label")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-only", action="store_true", help="Recover a saved review, preserving each completed draw")
    parser.add_argument("--stop-update", type=int, choices=(200, 750, 1500), default=200)
    parser.add_argument("--embedding-policy", choices=("candidate_projected", "uniform_public_total"), default="candidate_projected")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 2 <= args.steps <= 100 or args.hidden_dim < 2 or args.learning_rate <= 0:
        parser.error("Steps must be 2..100, width >=2, and learning rate positive")
    return args


if __name__ == "__main__":
    print(run_comparison(parse_args()))
