#!/usr/bin/env python3
"""Prepare one manually selected Thermal 5000-epoch launch without training.

The default is read-only validation. --prepare writes one profile and, for
continuation, a new RunStore workspace containing an unchanged parent copy.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.plugin import ThermalChannelPlugin
from channelthermal.training.campaign import validate_campaign, validate_campaign_resume
from channelthermal.workflows.train_forward import build_model_config

from honf_runtime.case_protocol import WorkflowRequest
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.config_loader import load_config_bundle
from honf_runtime.paths import PROJECT_ROOT
from honf_runtime.run_store import RunStore, atomic_write_json

PORTFOLIO_ARCHITECTURES = {
    "B-native": "dense_pairwise_field", "B-fine": "three_term_full_access_honf",
    "H-tree": "adaptive_receiver_hypergraph_honf", "H-overlap": "overlap_control_hypergraph_honf",
    "H-local": "local_overlap_hypergraph_honf",
}


def build_profile(core: dict, *, run_id: str, output_root: Path, fresh: bool, seed: int = 0) -> dict:
    """Extend only the selected e1000 profile's duration and run placement."""

    if core.get("workflow") != "forward" or core.get("model_family") != "honf_forward":
        raise ValueError("Manual campaign support requires a forward HONF profile.")
    if int(core["training"]["epochs"]) != 1000:
        raise ValueError("Select a reviewed e1000 source profile; finalist selection remains manual.")
    if seed != 0 or int(core["training"].get("seed", -1)) != 0:
        raise ValueError("This matched campaign supports fresh seed0 only; nonzero replicas are a separate policy.")
    if fresh and core["training"].get("init_checkpoint_path"):
        raise ValueError("Fresh initialization cannot inherit a global weight-initialization checkpoint.")
    profile = copy.deepcopy(core)
    profile["training"]["epochs"] = 5000
    campaign = profile["training"]["campaign"]
    if (campaign.get("schedule_total_epochs") != 5000 or not campaign.get("require_full_epoch")
            or not campaign.get("matched_fresh_initialization")):
        raise ValueError("Retain the full-epoch matched campaign and its absolute 5000-epoch horizon.")
    if PORTFOLIO_ARCHITECTURES.get(campaign.get("arm")) != profile["model"]["core_honf"]["forward_architecture"]:
        raise ValueError("Select one unchanged arm from the finite five-arm campaign.")
    if campaign.get("physical_loss_policy_version") != 2 or campaign.get("native_loss_denominators_start_epoch") != 101:
        raise ValueError("The selected profile must retain physical policy2 from epoch101.")
    validate_campaign(profile)
    if not str(run_id).isdigit():
        raise ValueError("Choose an explicit numeric run ID.")
    profile["run"].update(id=str(run_id).zfill(4), output_root=str(output_root.resolve()),
                           name=f"thermal_{campaign['arm'].lower()}_{'fresh_seed0' if fresh else 'continue1000'}_e5000")
    profile["_note"] = ("Manual selected-arm 5000 recipe; no automatic launch or finalist selection. "
        + ("Fresh seed0 initialization, policy1 through100 and policy2 from101; not an independent-seed replicate."
           if fresh else "Exact e1000 optimizer/RNG/calibration continuation in a new workspace; parent directory preserved."))
    return profile


def resolved_native_model_config(native_config: dict) -> dict:
    """Resolve native auto dimensions from Q1 metadata without building a model."""

    settings = native_config["dataset"]
    dataset = GlobalChannelThermalDataset(settings["packed_h5_path"], split=settings.get("train_split", "train"),
        points_per_case=1, normalize_inputs=bool(settings.get("normalize_inputs", False)),
        normalize_targets=bool(settings.get("normalize_targets", False)), random_point_sampling=False,
        seed=int(native_config["training"]["seed"]), require_converged=bool(settings.get("require_converged", False)))
    model_config = build_model_config(native_config, dataset)
    if model_config.channelthermal.internal_prediction_mode == "auto":
        model_config.channelthermal.internal_prediction_mode = (
            "local_surrogate" if model_config.channelthermal.use_local_surrogate else "global_head")
    return model_config.to_dict()


def validate_parent_checkpoint(checkpoint: dict, native_config: dict) -> dict:
    """Validate full continuation state without creating a model or optimizer."""

    if int(checkpoint.get("epoch", -1)) != 1000:
        raise ValueError("Continuation requires the exact completed epoch1000 checkpoint.")
    if not checkpoint.get("model_state_dict") or not checkpoint.get("model_config"):
        raise ValueError("Continuation requires full saved model weights and configuration.")
    validate_campaign_resume(checkpoint, native_config)
    if set(checkpoint.get("rng_state") or {}) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Continuation requires all four saved RNG streams.")
    if any(value is None for value in checkpoint["rng_state"].values()):
        raise ValueError("Continuation requires non-null saved RNG streams.")
    optimizer = checkpoint["optimizer_state_dict"]
    if not optimizer.get("state") or not optimizer.get("param_groups"):
        raise ValueError("Continuation requires full optimizer state and parameter groups.")
    saved = checkpoint["train_config"]
    if saved.get("case") != native_config.get("case"):
        raise ValueError("Continuation changed case identity.")
    runtime = {"epochs", "Run_ID", "run_name", "device"}
    if ({key: value for key, value in saved["training"].items() if key not in runtime}
            != {key: value for key, value in native_config["training"].items() if key not in runtime}):
        raise ValueError("Continuation changed training policy beyond duration or run placement.")
    if ChannelThermalHONFConfig.from_dict(checkpoint["model_config"]).to_dict() != resolved_native_model_config(native_config):
        raise ValueError("Continuation changed the native model configuration or Stage-A binding.")
    return {"checkpoint_epoch": 1000, "next_epoch": 1001, "stop_epoch": 5000,
            "schedule_total_epochs": 5000, "physical_loss_policy_version": 2,
            "active_optimizer_states": len(optimizer["state"]), "RNG_streams": sorted(checkpoint["rng_state"]),
            "calibration_preserved": True}


def launch_commands(profile_path: Path, *, physical_gpu: int, checkpoint: Path | None = None) -> dict[str, str]:
    if physical_gpu not in (1, 2):
        raise ValueError("Choose physical GPU1 or GPU2 explicitly.")
    paths = [PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src"]
    prefix = [f"CUDA_VISIBLE_DEVICES={physical_gpu}", f"PYTHONPATH={shlex.quote(':'.join(map(str, paths)))}"]
    args = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config", str(profile_path), "--device", "cuda:0"]
    if checkpoint is not None:
        args += ["--resume-checkpoint", str(checkpoint)]
    command = " ".join(prefix) + " " + shlex.join(args)
    return {"validate_command": command + " --dry-run", "launch_command": command + " --yes"}


def prepare_launch(profile_path: Path, *, run_id: str, output_root: Path, physical_gpu: int,
                   fresh: bool, parent_checkpoint: Path | None = None, prepare: bool = False, seed: int = 0) -> dict:
    """Validate one selected recipe; materialize artifacts only when requested."""

    if fresh == (parent_checkpoint is not None):
        raise ValueError("Choose fresh initialization or an exact e1000 parent, exclusively.")
    profile_path = profile_path.expanduser().resolve()
    source = load_config_bundle(profile_path)
    if source.effective["case"]["id"] != "ThermalChannel":
        raise ValueError("This launcher supports the native Thermal campaign only.")
    profile = build_profile(source.core, run_id=run_id, output_root=output_root, fresh=fresh, seed=seed)
    profile["case"]["config"] = str(source.case_source)
    effective = copy.deepcopy(source.effective)
    effective["training"]["epochs"] = 5000
    effective["run"] = copy.deepcopy(profile["run"])
    effective["Run_ID"] = profile["run"]["id"]
    provisional = replace(source, core=profile, effective=effective)
    plugin = ThermalChannelPlugin()
    plugin.validate_config(provisional)
    request = WorkflowRequest(workflow="forward", device="cuda:0", run_id=profile["run"]["id"],
                              run_name=profile["run"]["name"], epochs=5000)
    facts = dict(plugin.inspect_launch(provisional, request))
    store = RunStore(output_root)
    proposal = store.propose(case_id="ThermalChannel", workflow="forward", model_family="honf_forward",
                             run_id=request.run_id, run_name=request.run_name)
    existing_roots = {proposal.path.parent, PROJECT_ROOT / "Trained_Results/ThermalChannel/HONF_Forward_Runs"}
    if fresh and (profile["run"]["id"] == source.core["run"]["id"]
                  or any(list(root.glob(f"Run_{request.run_id}_*")) for root in existing_roots)):
        raise ValueError("Fresh initialization requires an unused run ID.")
    parent = None
    lineage = {"mode": "fresh_seed0" if fresh else "exact1000_continuation", "source_profile": str(profile_path.resolve()),
               "physical_gpu": physical_gpu, "logical_device": "cuda:0", "training_launched": False}
    if parent_checkpoint is not None:
        parent_checkpoint = parent_checkpoint.expanduser().resolve()
        checkpoint = load_trusted_checkpoint(parent_checkpoint, map_location="cpu")
        native = plugin._forward_config(provisional, request, proposal.path)
        lineage.update(validate_parent_checkpoint(checkpoint, native))
        lineage["parent_checkpoint"] = str(parent_checkpoint)
        parent = proposal.path / "continuation_parent_epoch_1000_model.pt"
    generated = output_root.resolve() / "launch_profiles" / f"{request.run_id}_{request.run_name}.json"
    commands = launch_commands(generated, physical_gpu=physical_gpu, checkpoint=parent)
    if prepare:
        generated.parent.mkdir(parents=True, exist_ok=True)
        with generated.open("x", encoding="utf-8") as stream:
            json.dump(profile, stream, indent=2)
            stream.write("\n")
        if parent is not None:
            bundle = load_config_bundle(generated)
            run_dir = store.create(proposal, bundle, launch_facts=facts)
            shutil.copy2(parent_checkpoint, parent)
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["continuation"] = lineage
            atomic_write_json(manifest_path, manifest)
    return {"prepared": prepare, "profile": str(generated), "new_workspace": str(proposal.path) if parent is not None else "created by ordinary trainer at launch",
            "lineage": lineage, "launch_facts": facts, **commands}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="One manually selected reviewed e1000 core profile.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="Explicit data-backed RunStore root.")
    parser.add_argument("--physical-gpu", type=int, choices=(1, 2), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    write = parser.add_mutually_exclusive_group()
    write.add_argument("--prepare", action="store_true", help="Write one profile and optional parent-copy workspace; never train.")
    write.add_argument("--dry-run", action="store_true", help="Default: validate and print without writes.")
    args = parser.parse_args()
    print(json.dumps(prepare_launch(args.profile, run_id=args.run_id, output_root=args.output_root,
        physical_gpu=args.physical_gpu, fresh=args.fresh, parent_checkpoint=args.parent_checkpoint,
        prepare=args.prepare, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
