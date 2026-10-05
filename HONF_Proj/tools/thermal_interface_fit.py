#!/usr/bin/env python3
"""Prepare matched, frozen G-fast e1000 query-interface fits; never launch training."""

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
from channelthermal.training.interface_fit import (
    FIT_KIND,
    FIT_PARAMETER_PREFIX,
    attach_interface_fit_checkpoint,
    validate_interface_fit_parent,
)
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
    parser.add_argument("--first-run-id", type=int, default=3701)
    parser.add_argument("--arm", choices=("H-add", "H-joint", "both"), default="both")
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    parent_path = Path(args.parent_checkpoint).expanduser().resolve()
    if parent_path.name not in {"epoch_1000_model.pt", "best_by_field_mse_model.pt"} or not parent_path.parent.name.startswith("Run_3601_"):
        raise ValueError("The declared frozen source is Run3601 G-fast selected/exact e1000.")
    parent = load_trusted_checkpoint(parent_path, map_location="cpu")
    validate_interface_fit_parent(parent)
    source_sha256 = _file_sha256(parent_path)
    output_root = validate_generated_output(args.output_root)
    source_profile = json.loads((parent_path.parent / "configs/core_source.json").read_text())
    results_path = output_root / "prepared_interface_fits.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for offset, arm in enumerate(("H-add", "H-joint")):
        if args.arm not in (arm, "both"):
            continue
        if arm in results:
            raise FileExistsError(f"Fit already prepared: {arm}")
        torch.manual_seed(0)
        options = {"global_fast_reader": True, "tensor_query_interaction": True,
                   "query_interaction_mode": "add" if arm == "H-add" else "joint",
                   "query_distance_alpha": .25, "query_distance_length": .25,
                   "query_background_zero_bias": True, "query_interface_parent_epoch": 1000}
        profile = copy.deepcopy(source_profile)
        profile["profile_name"] = f"thermal_receiver25_{arm.lower()}_frozen_gfast_v1"
        profile["model"]["core_honf"]["interface_model"]["hypergraph_options"].update(options)
        campaign = copy.deepcopy(parent["train_config"]["training"]["campaign"])
        campaign.update(name="thermal_receiver_interaction25_v1", arm=arm, matched_fresh_initialization=False,
                        parent={"checkpoint": str(parent_path), "epoch": 1000, "kind": FIT_KIND},
                        interface_fit={"backbone_epoch": 1000, "parameter_prefix": FIT_PARAMETER_PREFIX})
        profile["training"].update(epochs=100, campaign=campaign, seed=0, learning_rate=3e-4, weight_decay=1e-5)
        profile["case"]["dataset"]["development_manifest_sha256"] = parent["train_config"]["dataset"]["development_manifest_sha256"]
        profile["checkpointing"]["save_epoch_milestones"] = list(range(100, 1001, 100))
        profile["run"].update(id=str(args.first_run_id + offset), name=profile["profile_name"], output_root=str(output_root))
        profile["_note"] = "Fit only the new QM/QE interface; Run3601 G-fast e1000 physical parameters/buffers and normalizers remain frozen. Epoch is interface-fit age, not another backbone optimizer epoch."
        preparation = output_root / "preparation"
        preparation.mkdir(parents=True, exist_ok=True)
        profile_path = preparation / f"{arm.lower()}_fit100.json"
        if profile_path.exists():
            raise FileExistsError(f"Refusing to overwrite fit preparation: {profile_path}")
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
        resolve_development_training(canonical, effective_epochs=100)
        payload = copy.deepcopy(parent["model_config"])
        payload["core_honf"]["interface_model"].setdefault("hypergraph_options", {}).update(options)
        target_config = ChannelThermalHONFConfig.from_dict(payload)
        target = ChannelThermalHONFModel(target_config)
        child, receipt = attach_interface_fit_checkpoint(parent, target_model=target, target_config=target_config,
                                                          fit_config=canonical, source_path=parent_path,
                                                          source_sha256=source_sha256)
        run_dir = store.create(proposal, bundle, launch_facts=launch_facts)
        child_path = run_dir / "attached_frozen_backbone_fit0000.pt"
        torch.save(child, child_path)
        atomic_write_json(run_dir / "interface_fit_attachment.json", receipt)
        RunStore.update_status(run_dir, "prepared", parent_checkpoint=str(parent_path),
                               frozen_backbone_epoch=1000, interface_fit_epoch=0, interface_fit_arm=arm)
        results[arm] = {"run_dir": str(run_dir), "config": str(profile_path), "checkpoint": str(child_path),
                        "command": f"CUDA_VISIBLE_DEVICES={1 + offset} python train.py --config {shlex.quote(str(profile_path))} "
                                   f"--resume-checkpoint {shlex.quote(str(child_path))} --device cuda:0 --yes"}
        del target, child
    atomic_write_json(results_path, results)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
