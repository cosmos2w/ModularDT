#!/usr/bin/env python3
"""Prepare two explicit e500 development children; never launch or touch the parent."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    sys.path.insert(0, str(source_root))

from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.model import ChannelThermalHONFModel
from channelthermal.training.checkpoints import _file_sha256
from channelthermal.training.lean_attachment import PARENT_SHA256, attach_lean_checkpoint, validate_lean_parent
from thermal_development import validate_generated_output

from honf_runtime.case_protocol import WorkflowRequest
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.config_loader import load_config_bundle
from honf_runtime.registry import load_case_plugin
from honf_runtime.run_store import RunStore, atomic_write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--first-run-id", type=int, default=3601)
    parser.add_argument("--arm", choices=("G-fast", "Tensor-H", "both"), default="both")
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    parent_path = Path(args.parent_checkpoint).expanduser().resolve()
    if parent_path.name != "epoch_0500_model.pt" or not parent_path.parent.name.startswith("Run_3402_"):
        raise ValueError("The declared source is exact Run3402 epoch_0500_model.pt.")
    output_root = validate_generated_output(args.output_root)
    source_sha256 = _file_sha256(parent_path)
    if source_sha256 != PARENT_SHA256:
        raise ValueError("Run3402 e500 content hash does not match the inspected declared parent.")
    parent = load_trusted_checkpoint(parent_path, map_location="cpu")
    validate_lean_parent(parent)
    source_config = ChannelThermalHONFConfig.from_dict(parent["model_config"])
    source_model = ChannelThermalHONFModel(source_config)
    source_model.load_state_dict(parent["model_state_dict"], strict=True)
    source_profile = json.loads((parent_path.parent / "configs/core_source.json").read_text())
    prepared_path = output_root / "prepared_children.json"
    results = json.loads(prepared_path.read_text()) if prepared_path.exists() else {}
    for offset, arm in enumerate(("G-fast", "Tensor-H")):
        if args.arm not in (arm, "both"):
            continue
        if arm in results:
            raise FileExistsError(f"Child already prepared: {arm}")
        options = {"global_fast_reader": True}
        if arm == "Tensor-H":
            options.update(tensor_source_residual=True, residual_parent_epoch=500)
        profile = copy.deepcopy(source_profile)
        profile["profile_name"] = f"thermal_lean25_{arm.lower()}_e500_attachment_v1"
        profile["model"]["core_honf"]["interface_model"]["hypergraph_options"].update(options)
        profile["training"]["campaign"] = copy.deepcopy(parent["train_config"]["training"]["campaign"])
        profile["training"]["epochs"] = 600
        profile["case"]["dataset"]["development_manifest_sha256"] = parent["train_config"]["dataset"]["development_manifest_sha256"]
        profile["checkpointing"]["save_epoch_milestones"] = [600, 700, 800, 900, 1000]
        profile["run"].update(id=str(args.first_run_id + offset), name=profile["profile_name"], output_root=str(output_root))
        preparation = output_root / "preparation"
        preparation.mkdir(parents=True, exist_ok=True)
        profile_path = preparation / f"{arm.lower()}_additional100.json"
        if profile_path.exists():
            raise FileExistsError(f"Refusing to overwrite preparation: {profile_path}")
        atomic_write_json(profile_path, profile)
        bundle = load_config_bundle(profile_path)
        plugin = load_case_plugin(bundle.case["plugin"])
        plugin.validate_config(bundle)
        request = WorkflowRequest(workflow="forward", device="cpu")
        launch_facts = plugin.inspect_launch(bundle, request)
        store = RunStore(output_root)
        proposal = store.propose(case_id="ThermalChannel", workflow="forward", model_family="honf_forward",
                                 run_id=bundle.effective["Run_ID"], run_name=profile["run"]["name"])
        canonical = plugin._forward_config(bundle, request, proposal.path)
        from channelthermal.workflows.train_forward import resolve_development_training
        resolve_development_training(canonical, effective_epochs=600)
        for section in ("dataset", "loss"):
            if canonical[section] != parent["train_config"][section]:
                raise ValueError(f"Attachment profile changed parent {section}.")
        target_payload = copy.deepcopy(parent["model_config"])
        target_payload["core_honf"]["interface_model"].setdefault("hypergraph_options", {}).update(options)
        target_config = ChannelThermalHONFConfig.from_dict(target_payload)
        target_model = ChannelThermalHONFModel(target_config)
        child, receipt = attach_lean_checkpoint(parent, source_model=source_model, target_model=target_model,
                                               target_config=target_config, source_sha256=source_sha256,
                                               source_path=parent_path)
        run_dir = store.create(proposal, bundle, launch_facts=launch_facts)
        child_path = run_dir / "attached_parent_e500.pt"
        torch.save(child, child_path)
        atomic_write_json(run_dir / "attachment_receipt.json", receipt)
        RunStore.update_status(run_dir, "prepared", parent_checkpoint=str(parent_path), parent_epoch=500,
                               matched_continuation_arm=arm, additional_epochs_completed=0)
        results[arm] = {"run_dir": str(run_dir), "config": str(profile_path), "checkpoint": str(child_path),
                        "command": f"CUDA_VISIBLE_DEVICES={1 + offset} python train.py --config {shlex.quote(str(profile_path))} "
                                   f"--resume-checkpoint {shlex.quote(str(child_path))} --device cuda:0 --yes"}
        del target_model, child
    atomic_write_json(prepared_path, results)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
