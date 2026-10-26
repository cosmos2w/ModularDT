"""R13-4 setup and exact recovery through the maintained research runner."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from . import contextual_frontier_training as maintained
from . import research_training as research
from .compat import load_trusted_checkpoint
from .guided_member_routing_child import WORKFLOW
from .run_layout import RunLayout, resolve_checkpoint
from .unified_training import _atomic_json


MANIFEST_SCHEMA = "honf_member_routing_run_manifest_v3"


def load_config(path):
    path = Path(path).expanduser().resolve(strict=True)
    config = json.loads(path.read_text())
    section = config.get("member_routing", {})
    if (config.get("schema") != research.RESEARCH_SCHEMA
            or section.get("schema") != "thermal_member_routing_r13_4_v1"
            or section.get("model", {}).get("schema") != "interaction_reused_member_routing_v3"
            or section.get("arm") not in ("R13-4-Full", "R13-4-Guide")):
        raise ValueError("Expected a versioned R13-4 relation-guided configuration.")
    from honf_forward_core.interaction_reused_member_routing.guided import GuidedMemberRoutingCore
    GuidedMemberRoutingCore.validate_config(section["model"])
    training = config["training"]
    if (training["membership"] not in ("fixed25_train150_dev22", "full_train600")
            or training["seed"] != 0 or training["primary_queries"] != 1024
            or training["effective_batch_size"] != 48
            or training["microbatch_cases"] != 24
            or training["latest_every"] != 1 or training["curve_every"] != 10
            or config["optimizer"]["schedule"] != "constant_config_lr"
            or config["objective"].get("response_fit") is not False):
        raise ValueError("R13-4 retains the native sampling, objective, optimizer and output conventions.")
    for key in ("data_path", "manifest_path", "base_recipe_path"):
        value = config["dataset"].get(key)
        if value is not None:
            config["dataset"][key] = str((path.parent / value).resolve(strict=True))
    return config, path


def training_identity(config):
    # Preserve the ordinary identity inventory, with the new capability named
    # explicitly rather than relabelling a historical v2 checkpoint.
    value = maintained._run_identity(config, None)
    value.update(schema="honf_interaction_reused_member_routing_training_v3",
                 workflow=WORKFLOW, model_family="relation_guided_original_member_joint_field_v3")
    return value


def _build(config, device):
    from channelthermal.training.guided_member_routing_task import build_thermal_guided_member_routing_task
    return build_thermal_guided_member_routing_task(
        config, arm=config["member_routing"]["arm"], device=device)


def _write_manifest(output, config, config_path, storage, *, resumed, execution_budget=None):
    RunLayout(output).ensure()
    _atomic_json(output / "run_manifest.json", {
        "schema": MANIFEST_SCHEMA, "artifact_layout_version": 1,
        "arm": config["member_routing"]["arm"], "config": config,
        "config_path": str(config_path), "resumed": resumed, "storage": storage,
        **({"execution_budget": execution_budget} if execution_budget is not None else {}),
    })


def train(config_path, *, output, device="cuda:1", stop_after_epoch=None):
    config, config_path = load_config(config_path)
    stop = int(stop_after_epoch or config["training"]["first_segment_epochs"])
    parent = config["initialization"].get("source_checkpoint")
    origin = int(config["initialization"]["source_parent_age"])
    if not origin < stop <= config["training"]["local_horizon"]:
        raise ValueError("The stop must advance the declared parent within the execution horizon.")
    output = Path(output).expanduser().resolve()
    root = Path(config["storage"]["output_root"]).resolve()
    if not output.is_relative_to(root) or (output.exists() and any(output.iterdir())):
        raise ValueError("A new branch needs a fresh directory under its declared DATA2 output root.")
    bindings = maintained._ensure_workspace_storage_bindings()
    research.verify_storage(config, output=output, estimated_bytes=0, peak_temporary_bytes=0)
    selected, device_info = research.resolve_device(device, permitted_devices=(device,))
    model, provider = _build(config, selected)
    estimated, peak = maintained._storage_estimate(
        model, completed_epoch=origin, stop_after=stop, config=config, fresh_namespace=True)
    storage = maintained._prepare_storage(config, output, estimated_bytes=estimated,
                                          peak_temporary_bytes=peak, bindings=bindings)
    _write_manifest(output, config, config_path, storage, resumed=False)
    print(json.dumps({"event": "guided_member_access_prepared", "run": str(output),
                      "parent_epoch": origin, "stop_epoch": stop, "device": device_info}), flush=True)
    source = None if parent is None else Path(parent).resolve(strict=True)
    declaration = None if source is None else {"source_checkpoint": str(source), "source_epoch": origin}
    return maintained._fit(
        model, provider, config, output, device=selected, stop_after=stop,
        training_identity=training_identity(config), branch_from_checkpoint=source,
        relation_guided_transfer=declaration)


def resume(run, *, device="cuda:1", stop_after_epoch, extend_horizon=False,
           checkpoint_epochs=None, curve_every=None, wall_clock_budget_seconds=None):
    run = Path(run).expanduser().resolve(strict=True)
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Use the matching maintained recovery command for this run's capability.")
    config = copy.deepcopy(manifest["config"])
    checkpoint = resolve_checkpoint(run, "latest")
    payload = load_trusted_checkpoint(checkpoint, map_location="cpu")
    stop = int(stop_after_epoch)
    if stop <= payload["epoch"]:
        raise ValueError("Resume must advance the latest completed epoch.")
    saved_engine = payload["experiment_identity"]["engine_config"]
    saved_horizon = int(saved_engine["total_epochs"])
    extension = stop > saved_horizon
    if extension and not extend_horizon:
        raise ValueError("An explicit horizon extension is required beyond the saved execution horizon.")
    if (checkpoint_epochs is not None or curve_every is not None) and not extension:
        raise ValueError("Output changes require an explicit execution-horizon extension.")
    if checkpoint_epochs is None:
        checkpoint_epochs = (list(range(600 if config["initialization"]["source_parent_age"] else 100,
                                       stop + 1, 100)) if extension
                             else list(saved_engine["checkpoint_epochs"]))
    if (not isinstance(checkpoint_epochs, (list, tuple)) or not checkpoint_epochs
            or any(type(value) is not int or not 0 < value <= max(saved_horizon, stop)
                   for value in checkpoint_epochs)
            or list(checkpoint_epochs) != sorted(set(checkpoint_epochs))):
        raise ValueError("Checkpoint epochs must be sorted, unique, positive epochs within the execution horizon.")
    if curve_every is None:
        curve_every = saved_engine.get("curve_every", 10)
    if type(curve_every) is not int or curve_every < 1:
        raise ValueError("Curve frequency must be a positive epoch interval.")
    config["training"]["checkpoint_epochs"] = list(checkpoint_epochs)
    config["training"]["curve_every"] = curve_every
    execution_budget = manifest.get("execution_budget")
    if wall_clock_budget_seconds is not None:
        if (type(wall_clock_budget_seconds) is not int
                or wall_clock_budget_seconds < config["training"]["wall_clock_budget_seconds"]):
            raise ValueError("An explicit cumulative execution budget cannot reduce the saved recipe allowance.")
        execution_budget = {"cumulative_wall_clock_budget_seconds": wall_clock_budget_seconds,
                            "source_epoch": int(payload["epoch"]),
                            "optimizer_and_recipe_schedule_unchanged": True}
    selected, _ = research.resolve_device(device, permitted_devices=(device,))
    maintained._ensure_workspace_storage_bindings()
    model, provider = _build(config, selected)
    estimated, peak = maintained._storage_estimate(
        model, completed_epoch=payload["epoch"], stop_after=stop, config=config,
        checkpoint_epochs=checkpoint_epochs)
    storage = maintained._prepare_storage(config, run, estimated_bytes=estimated, peak_temporary_bytes=peak)
    _write_manifest(run, config, Path(manifest["config_path"]), storage, resumed=True,
                    execution_budget=execution_budget)
    return maintained._fit(
        model, provider, config, run, device=selected, stop_after=stop,
        resume_checkpoint=checkpoint, training_identity=training_identity(config),
        engine_horizon=max(saved_horizon, stop), engine_checkpoint_epochs=checkpoint_epochs,
        wall_clock_budget_seconds=(execution_budget["cumulative_wall_clock_budget_seconds"]
                                   if execution_budget is not None else None),
        allow_horizon_extension=extension)


def load_checkpoint(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    payload = load_trusted_checkpoint(checkpoint, map_location="cpu")
    if payload.get("workflow") != WORKFLOW:
        raise ValueError("Not an R13-4 relation-guided checkpoint.")
    manifest = json.loads((checkpoint.parents[1] / "run_manifest.json").read_text())
    model, provider = _build(manifest["config"], torch.device(device))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    provider.load_training_state_dict(payload["provider_training_state"])
    if provider.identity_payload() != payload["experiment_identity"]["provider_identity"]:
        raise ValueError("Reconstructed native provider differs from the saved guided capability.")
    model.eval()
    return model, provider, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("train")
    start.add_argument("--config", required=True)
    start.add_argument("--output", required=True)
    start.add_argument("--device", default="cuda:1")
    start.add_argument("--stop-after-epoch", type=int)
    recovery = commands.add_parser("resume")
    recovery.add_argument("--run", required=True)
    recovery.add_argument("--device", default="cuda:1")
    recovery.add_argument("--stop-after-epoch", type=int, required=True)
    recovery.add_argument("--extend-horizon", action="store_true")
    recovery.add_argument("--checkpoint-epochs", nargs="+", type=int)
    recovery.add_argument("--curve-every", type=int, help="Redraw saved curves every this many epochs.")
    recovery.add_argument("--wall-clock-budget-seconds", type=int,
                          help="Explicit cumulative execution allowance; scientific schedules stay unchanged.")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--device", default="cuda:1")
    evaluation.add_argument("--output", required=True)
    args = vars(parser.parse_args(argv));command = args.pop("command")
    if command == "train":
        result = train(args.pop("config"), **args)
    elif command == "resume":
        result = resume(**args)
    else:
        from .guided_member_routing_evaluation import evaluate
        result = evaluate(**args)
    print(json.dumps(result, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
