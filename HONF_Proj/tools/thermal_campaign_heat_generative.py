#!/usr/bin/env python3
"""Bounded, staged paired heat heads through an explicitly qualified organizer.

No training runs on import. Invoke only after the organizer milestone review.
The supplied total is public; individual heat targets and held sensors are
stored separately from public tasks. No physical solver is invoked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from thermal_campaign_evaluate import screen_indices
from thermal_campaign_heat_inference import (
    atomic_json,
    native_heat_predictor,
    sensor_panel,
    snapshot_forward_state,
    verify_frozen_forward,
)

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
    def fit(cls, dataset, *, expected_case_ids=None):
        expected = None if expected_case_ids is None else tuple(str(value) for value in expected_case_ids)
        if expected is not None and (not expected or len(set(expected)) != len(expected)):
            raise ValueError("Inverse normalization requires distinct explicit training case IDs")
        if dataset.split != "train" or len(dataset) != (600 if expected is None else len(expected)):
            scope = "exactly the 600 training records" if expected is None else "the explicit selected training records"
            raise ValueError(f"Scientific inverse normalization requires {scope}")
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
        if len(set(ids)) != len(ids) or (expected is not None and set(ids) != set(expected)):
            raise ValueError("Inverse normalization records differ from the explicit training case IDs")
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


def build_public_observation_task(model, checkpoint, sample, partition, normalization, *,
                                  case_id, total, coordinates, observed, held, observed_values):
    """Build from public observations only; held values/clean heat are separate."""
    device = next(model.parameters()).device
    coordinates = np.array(coordinates, dtype=np.float32, copy=True)
    observed, held = np.array(observed, dtype=np.int64, copy=True), np.array(held, dtype=np.int64, copy=True)
    observed_values = np.array(observed_values, dtype=np.float32, copy=True).reshape(-1)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2
            or not np.isfinite(coordinates).all() or not np.isfinite(observed_values).all()
            or not np.isfinite(total) or total < 0):
        raise ValueError("Public observation coordinates, values and heat budget must be finite and valid")
    if (observed.shape != (6,) or held.shape != (6,) or observed_values.shape != (6,)
            or len(np.unique(np.concatenate((observed, held)))) != 12
            or min(observed.min(), held.min()) < 0
            or max(observed.max(), held.max()) >= len(coordinates) - 2):
        raise ValueError("Public task requires six distinct observed/held sensors and separate pressure endpoints")
    active = torch.as_tensor(np.asarray(sample["structure"]["module_present"]) > .5, device=device)[None]
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

    return PublicHeatTask(str(case_id), partition, condition, provider, predictor,
                          torch.as_tensor(observed_values, device=device)[None, :, None])


def build_public_task(model, checkpoint, sample, partition, normalization, channel_order):
    """Extract public observations, then return clean supervision separately."""
    device = next(model.parameters()).device
    coordinates, grid_rows, _, observed, held = sensor_panel(sample)
    total = float(np.asarray(sample["structure"]["heat_powers"]).sum())
    fields = np.asarray(sample["steady_field"]).reshape(-1, len(channel_order))
    temperature = list(channel_order).index("temperature")
    task = build_public_observation_task(model, checkpoint, public_template(sample), partition, normalization,
        case_id=sample["case_id"], total=total, coordinates=coordinates, observed=observed, held=held,
        observed_values=fields[grid_rows[observed], temperature])
    target = torch.as_tensor(sample["structure"]["heat_powers"], device=device, dtype=torch.float32)[None, :, None]
    held_reference = torch.as_tensor(fields[grid_rows[held], temperature],
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


def resolve_evidence_output(path):
    """Retain ignored workspace placement through a data-backed symlink."""
    requested = Path(os.path.abspath(Path(path).expanduser()))
    roots = tuple(PROJECT_ROOT / name for name in ("diagnostics", "Trained_Results"))
    if not any(requested.is_relative_to(root) for root in roots):
        raise ValueError("Generative evidence must remain in ignored local output paths")
    resolved = requested.resolve()
    if resolved.is_relative_to(PROJECT_ROOT) and not any(resolved.is_relative_to(root.resolve()) for root in roots):
        raise ValueError("Generative evidence symlink resolves into a non-output project path")
    return resolved


def record_frozen_review(model, snapshot, output, *, update, draw_scope):
    """Verify this invocation without certifying recovered draws retroactively."""
    result = {**verify_frozen_forward(model, snapshot), "head_update": update,
        **draw_scope,
        "verification_scope": "current invocation forward parameters/persistent buffers; reused saved draws are not retrospectively certified"}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "forward_freeze_verification.json", result)
    return result


def load_alternative_observation_panel(path):
    """Four retained, independently recorded pairs; no fitted review inputs."""
    from thermal_campaign_atlas_observations import load_final_review_pair

    from honf_runtime.paths import resolve_path

    panel = json.loads(Path(path).read_text())
    sources = [resolve_path(source) for source in panel["stencils"]]
    if len(sources) != 4 or len(set(sources)) != 4:
        raise ValueError("Alternative observation panel requires four distinct stored families")
    pairs = [load_final_review_pair(source) for source in sources]
    if ({len(pair[0][0].source_module_ids) for pair in pairs} != {3, 5, 7, 10}
            or len({pair[0][0].metadata["family_id"] for pair in pairs}) != 4):
        raise ValueError("Alternative observation panel must span distinct M3/5/7/10 families")
    return pairs


def build_atlas_observation_task(model, checkpoint, public, normalization):
    """The public archive object contains no clean heat or held values."""
    return build_public_observation_task(model, checkpoint, public.public_sample,
        "final_review_previously_exposed", normalization, case_id=public.case_id,
        total=public.public_total_heat, coordinates=public.sensor_coordinates,
        observed=public.observed_rows, held=public.held_rows, observed_values=public.observed_temperatures)


def load_paired_resume(path):
    """Keep saved RNG bytes on CPU; model/optimizer restore handles devices."""
    from honf_runtime.compat import load_trusted_checkpoint

    return load_trusted_checkpoint(path, map_location="cpu")


def build_generative_datasets(checkpoint, dataset_path):
    """Use checkpoint-bound cohorts and normalization while retaining physical values."""
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from thermal_development import evaluation_dataset_kwargs, resolve_evaluation_manifest

    config = checkpoint.get("train_config", {}).get("dataset", {})
    manifest = resolve_evaluation_manifest(config, dataset_path)
    stats = {key: np.asarray(value, dtype=np.float32)
             for key, value in checkpoint.get("global_normalization_stats", {}).items()}
    if manifest is not None and not stats:
        raise ValueError("Development inverse heads require saved selected-training global normalization stats")
    normalizer = H5Normalizer(stats) if stats else None
    common = {"points_per_case": 1, "random_point_sampling": False, "include_grid": True,
              "normalize_inputs": False, "normalize_targets": False, "normalizer": normalizer}
    train = GlobalChannelThermalDataset(dataset_path, split="train", **common,
                                      **evaluation_dataset_kwargs(manifest, "train"))
    dev = GlobalChannelThermalDataset(dataset_path, split="test", **common,
                                    **evaluation_dataset_kwargs(manifest, "test"))
    return train, dev, manifest


def run_comparison(args):
    from channelthermal.evaluation.loading import load_model

    output = resolve_evidence_output(args.output_dir)
    if (output / "identity.json").exists() and not args.resume:
        raise ValueError("Existing paired-head evidence requires its explicit saved checkpoint continuation")
    model, checkpoint = load_model(args.checkpoint, torch.device(args.device))
    model.eval().requires_grad_(False)
    frozen_snapshot = snapshot_forward_state(model)
    verify_frozen_forward(model, frozen_snapshot)
    if not hasattr(model.core.backend, "organizer"):
        raise ValueError("A qualified typed organizer checkpoint is required")
    train, dev, manifest = build_generative_datasets(checkpoint, args.dataset)
    alternative_pairs = load_alternative_observation_panel(args.alternative_observation_panel)
    expected_ids = None if manifest is None else tuple(manifest["partitions"]["train"]["case_ids"])
    normalization = TrainOnlyNormalization.fit(train, expected_case_ids=expected_ids)
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
        "alternative_observation_panel": str(args.alternative_observation_panel.resolve()),
        "alternative_observation_pairs": [[dict(public.metadata) for public, _hidden in pair]
                                          for pair in alternative_pairs],
        "alternative_observation_limit": "four auxiliary archived baseline/heat-transfer-plus pairs preserve geometry/material/context/public total within pair; original twelve native tasks remain separate; all records previously exposed"}
    if manifest is not None:
        identity["development_subset"] = {"manifest_sha256": manifest["manifest_sha256"],
            "train_case_ids": list(expected_ids),
            "test_case_ids": list(manifest["partitions"]["test"]["case_ids"]),
            "review_case_ids": [str(dev[index]["case_id"]) for index in screen_indices(dev, 12)],
            "scope": "checkpoint-bound development training and native review cohort; archived alternative pairs are separate"}
    identity["intervention_scope"] = "96 original draws: 12 tasks x4 noises x2 heads; three labelled same-weight/observation controls on4 eligible tasks x1 noise,12 extra draws; four valid-observation pairs x2 physical targets x2 heads x1 common noise,16 auxiliary draws"
    identity["training_probe_selection"] = {"case_id": first.case_id, "criteria": "known M>=2 and positive public total; no hidden individual allocation criterion"}
    if args.resume:
        saved = load_paired_resume(args.resume)
        if saved["identity"] != identity:
            raise ValueError("Resume changed the frozen checkpoint, data, normalization or head policy")
        heads.restore(saved["heads"]); task_stream.set_state(saved["task_stream"])
    if args.evaluate_only:
        if not args.resume:
            raise ValueError("Evaluation recovery requires an exact paired-head checkpoint")
        review = output / f"review_{heads.update:04d}"
        scope = evaluate_draws(heads, model, checkpoint, dev, normalization, review, args,
                               alternative_pairs=alternative_pairs)
        record_frozen_review(model, frozen_snapshot, review, update=heads.update, draw_scope=scope)
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
            verify_frozen_forward(model, frozen_snapshot)
            # Review checkpoints are recoverable before potentially costly evaluation.
            save_heat_payload({"identity": identity, "heads": heads.checkpoint(), "task_stream": task_stream.get_state(),
                        "initial_conditioner": initial_check},
                       output / f"update_{heads.update:04d}.pt")
            check = heads.training_conditioner_check(first, first_target, provider=link_policy(first, args.embedding_policy))
            check["fixed_train_probe_original_loss_change_since_initial"] = {
                name: check["loss"][name]["original"]-initial_check["loss"][name]["original"]
                for name in heads.models}
            (output / f"train_conditioner_{heads.update:04d}.json").write_text(json.dumps(check, indent=2)+"\n")
            review = output / f"review_{heads.update:04d}"
            scope = evaluate_draws(heads, model, checkpoint, dev, normalization, review, args,
                                   alternative_pairs=alternative_pairs)
            record_frozen_review(model, frozen_snapshot, review, update=heads.update, draw_scope=scope)
            print(f"review {heads.update}: complete paired updates and individual draw files", flush=True)
    return output


def public_input_evidence(sample):
    """Physical inputs read by the native adapter, without solved supervision."""
    template = public_template(sample)
    return {"structure": {name: torch.tensor(value) for name, value in template["structure"].items()},
            "interface_condition": torch.tensor(template["interface_condition"]),
            "module_internal_query_points": torch.tensor(template["module_internal_query_points"])}


def same_reference(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype
                and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping) and left.keys() == right.keys()
                and all(same_reference(left[name], right[name]) for name in left))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)) and len(left) == len(right)
                and all(same_reference(a, b) for a, b in zip(left, right)))
    return left == right


def save_or_validate_reference(payload, path):
    """Never reuse trials against a changed physical task at the same path."""
    if path.exists():
        from honf_runtime.compat import load_trusted_checkpoint
        if not same_reference(load_trusted_checkpoint(path, map_location="cpu"), payload):
            raise ValueError(f"Saved evaluation reference changed physical inputs or supervision: {path}")
    else:
        if any(path.parent.glob("draw_*.pt")):
            raise ValueError(f"Saved draws lack their original evaluation reference: {path}")
        save_heat_payload(payload, path)


def save_or_reuse_draw(head, task, held, *, noise, path, dense, observation, provider, args):
    """Recover a complete draw only when its public target and noise agree."""
    if path.exists():
        if not args.evaluate_only:
            raise ValueError(f"Preserve existing individual draw evidence: {path}")
        from honf_runtime.compat import load_trusted_checkpoint
        saved = load_trusted_checkpoint(path, map_location="cpu")
        if (saved["case_id"] != task.case_id or saved["partition"] != task.partition
                or saved["observation_mode"] != observation
                or saved["access_mode"] != ("full" if dense else "graph")
                or not torch.equal(saved["initial_noise"], noise.cpu())
                or not torch.equal(saved["observed_reference"], task.observed_reference.cpu())
                or not torch.equal(saved["held_reference"], held.cpu())
                or not torch.equal(saved["public_total_heat"], task.condition.total_heat.cpu())
                or "held_predictions" not in saved or len(saved["state_timesteps"]) != args.steps+1):
            raise ValueError(f"Existing individual draw identity/integrity mismatch: {path}")
        return False
    save_heat_draw(head, task, held, initial_noise=noise, path=path, dense=dense,
                   observation_mode=observation, provider=provider)
    return True


def evaluate_alternative_draws(heads, model, checkpoint, normalization, output, args, pairs):
    """Same noise across two recorded physical targets and both paired heads."""
    new_draws = reused_draws = 0
    for index, pair in enumerate(pairs):
        baseline = pair[0][0]
        task = build_atlas_observation_task(model, checkpoint, baseline, normalization)
        generator = torch.Generator(device=task.condition.known_state.device).manual_seed(args.seed+200000+index)
        noise = torch.randn(task.condition.known_state.shape, device=task.condition.known_state.device,
                            generator=generator)
        for public, hidden in pair:
            task = build_atlas_observation_task(model, checkpoint, public, normalization)
            directory = output / "valid_observation_pairs" / public.metadata["family_id"] / public.metadata["variant"]
            directory.mkdir(parents=True, exist_ok=True)
            reference_path = directory / "evaluation_reference.pt"
            held = torch.tensor(hidden.held_temperatures, dtype=torch.float32,
                                device=task.condition.known_state.device)[None, :, None]
            save_or_validate_reference({"case_id": public.case_id, "source": dict(public.metadata),
                    "public_geometry_context": public_input_evidence(public.public_sample),
                    "source_module_ids": public.source_module_ids, "source_id_to_slot": dict(public.source_id_to_slot),
                    "sensor_names": public.sensor_names, "sensor_query_ids": public.sensor_query_ids,
                    "sensor_coordinates": torch.tensor(public.sensor_coordinates),
                    "observed_rows": torch.tensor(public.observed_rows), "held_rows": torch.tensor(public.held_rows),
                    "reference_heat": torch.tensor(hidden.heat), "held": held.cpu(),
                    "observation": task.observed_reference.cpu(), "public_total": task.condition.total_heat.cpu(),
                    "temperature_unit": public.temperature_unit,
                    "reference_heat_scope": "evaluation only; absent from public task/provider/embeddings"}, reference_path)
            provider = link_policy(task, args.embedding_policy)
            for name, head, dense in (("graph", heads.graph, False), ("full", heads.full, True)):
                created = save_or_reuse_draw(head, task, held, noise=noise,
                    path=directory / f"draw_00_{name}.pt", dense=dense, observation="original", provider=provider, args=args)
                new_draws += int(created)
                reused_draws += int(not created)
    return {"new_completed_draws": new_draws, "reused_saved_draws": reused_draws}


def evaluate_draws(heads, model, checkpoint, dataset, normalization, output, args, *, alternative_pairs):
    control_tasks = []
    new_draws = reused_draws = 0
    for index in screen_indices(dataset, 12):
        sample = dataset[index]
        task, reference_heat, held = build_public_task(model, checkpoint, sample, "development", normalization, dataset.channel_order)
        directory = output / task.case_id
        directory.mkdir(parents=True, exist_ok=True)
        reference_path = directory / "evaluation_reference.pt"
        coordinates, grid_rows, sensor_names, observed_rows, held_rows = sensor_panel(sample)
        save_or_validate_reference({"case_id": task.case_id, "reference_heat": reference_heat.cpu(), "held": held.cpu(),
                        "public_geometry_context": public_input_evidence(sample),
                        "sensor_coordinates": torch.tensor(coordinates), "sensor_grid_rows": torch.tensor(grid_rows),
                        "sensor_names": sensor_names, "observed_rows": torch.tensor(observed_rows),
                        "held_rows": torch.tensor(held_rows),
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
                created = save_or_reuse_draw(head, task, held, noise=noise, path=path,
                    dense=dense, observation=observation, provider=provider, args=args)
                new_draws += int(created)
                reused_draws += int(not created)
    alternative_scope = evaluate_alternative_draws(heads, model, checkpoint, normalization, output, args,
                                                   alternative_pairs)
    (output / "draw_scope.json").write_text(json.dumps({"original_tasks": 12, "noises_per_task": 4,
        "original_heads": ["graph", "full"], "intervention_tasks": control_tasks, "intervention_noises_per_task": 1,
        "interventions": ["graph_weight_full_access", "graph_observations_removed", "graph_observations_shuffled"],
        "valid_alternative_pairs": len(alternative_pairs), "targets_per_pair": 2,
        "alternative_noises_per_pair": 1, "alternative_heads": ["graph", "full"],
        "alternative_noise_scope": "identical initial noise within each baseline/plus pair and both heads",
        "alternative_partition": "stored final_review, previously exposed; auxiliary to original twelve native tasks",
        "alternative_draw_scope": alternative_scope}, indent=2)+"\n")
    return {"new_completed_draws": new_draws+alternative_scope["new_completed_draws"],
            "reused_saved_draws": reused_draws+alternative_scope["reused_saved_draws"]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qualified-organizer", required=True, help="Reviewed finalist qualification lineage label")
    parser.add_argument("--alternative-observation-panel", type=Path,
        default=PROJECT_ROOT / "src/config_core/evaluation/thermal_campaign_alternative_observation_panel.json")
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
