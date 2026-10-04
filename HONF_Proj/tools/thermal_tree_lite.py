#!/usr/bin/env python3
"""Prepare one exact-e200 Tree child; never launch or change parent state."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from channelthermal.plugin import ThermalChannelPlugin
from channelthermal.training.campaign import validate_campaign, validate_campaign_resume
from channelthermal.training.checkpoints import _file_sha256, _validate_resume_checkpoint
from channelthermal.workflows.train_forward import (
    build_model_config,
    build_training_datasets,
    resolve_development_training,
    resolved_config_payload,
    validate_development_resume,
)
from thermal_development import validate_generated_output

from honf_runtime.case_protocol import WorkflowRequest
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.config_loader import load_config_bundle
from honf_runtime.run_store import RunStore, atomic_write_json


def child_profile(core: dict, *, parent_checkpoint: Path, run_id: str, output_root: Path) -> dict:
    """Retain the physical model/objective and amend only the organizer derivative."""
    profile = copy.deepcopy(core)
    if (profile.get("workflow") != "forward" or profile.get("model_family") != "honf_forward"
            or profile["model"]["core_honf"]["forward_architecture"] != "faithful_receiver_hypergraph_honf"
            or profile["training"]["epochs"] != 200):
        raise ValueError("Select the reviewed Tree-F e200 source profile.")
    if not str(run_id).isdigit() or str(run_id).zfill(4) == str(profile["run"]["id"]).zfill(4):
        raise ValueError("The gradient-policy child requires a different numeric run ID.")
    profile["training"]["epochs"] = 500
    profile["training"]["campaign"].update(organizer_gradient_policy="local_context_shadow_v1",
        parent={"checkpoint": str(parent_checkpoint.resolve()), "epoch": 200,
                "from": "whole_wrapper_shadow_v1", "to": "local_context_shadow_v1"})
    profile["profile_name"] = "thermal_tree_lite25_local_context_shadow_v1"
    profile["run"].update(id=str(run_id).zfill(4), output_root=str(output_root.resolve()),
                          name="thermal_tree_lite25_local_context_shadow_v1")
    profile["_note"] = ("Exact Tree-F e200 optimizer/RNG/normalizer continuation to500 in a child workspace; "
        "e201 changes only organizer surrogate derivative to local_context_shadow_v1. "
        "Hard physical operator/objective, selected150/22, seed0 and absolute1000 schedule preserved. "
        "Preparation does not launch training or authorize e1000.")
    return profile


def prepare_child(profile_path: Path, parent_checkpoint: Path, *, run_id: str,
                  output_root: Path, prepare: bool = False) -> dict:
    """Use existing RunStore and trusted checkpoint guards before copying a parent."""
    output_root = validate_generated_output(output_root)
    source = load_config_bundle(profile_path)
    parent_checkpoint = parent_checkpoint.expanduser().resolve()
    checkpoint = load_trusted_checkpoint(parent_checkpoint, map_location="cpu")
    if checkpoint.get("epoch") != 200:
        raise ValueError("Only an exact retained e200 parent can seed Tree-Lite.")
    profile = child_profile(source.core, parent_checkpoint=parent_checkpoint,
                            run_id=run_id, output_root=output_root)
    effective = copy.deepcopy(source.effective)
    effective["training"] = copy.deepcopy(profile["training"])
    effective["run"] = copy.deepcopy(profile["run"])
    effective["Run_ID"] = profile["run"]["id"]
    bundle = replace(source, core=profile, effective=effective)
    plugin = ThermalChannelPlugin()
    plugin.validate_config(bundle)
    request = WorkflowRequest(workflow="forward", device="cuda:0", epochs=500,
                              run_id=profile["run"]["id"], run_name=profile["run"]["name"])
    facts = dict(plugin.inspect_launch(bundle, request))
    store = RunStore(output_root)
    proposal = store.propose(case_id="ThermalChannel", workflow="forward", model_family="honf_forward",
                             run_id=request.run_id, run_name=request.run_name)
    native = plugin._forward_config(bundle, request, proposal.path)
    development = resolve_development_training(native, effective_epochs=500)
    if development is None:
        raise ValueError("Tree-Lite requires the bound fixed development population.")
    train, val = build_training_datasets(native, development)
    config = build_model_config(native, train)
    config.channelthermal.internal_prediction_mode = "local_surrogate"
    native = resolved_config_payload(native, config, native["dataset"],
                                    config.channelthermal.local_surrogate_checkpoint_path)
    validate_campaign(native)
    amendment = validate_campaign_resume(checkpoint, native)
    validate_development_resume(checkpoint, native["dataset"], normalizer=train.normalizer)
    _validate_resume_checkpoint(checkpoint, model=SimpleNamespace(local_surrogate_attached=True),
        model_config=config, dataset=train, dataset_config=native["dataset"], campaign_amendment=amendment)
    if (checkpoint.get("selection_state") != {"epoch": 200, "total_epochs": 1000}
            or set(checkpoint.get("rng_state") or {}) != {"python", "numpy", "torch", "cuda"}
            or not checkpoint["optimizer_state_dict"].get("state")
            or not checkpoint["optimizer_state_dict"].get("param_groups")):
        raise ValueError("Tree-Lite requires unchanged selection horizon and complete optimizer/RNG state.")
    lineage = {"mode": "exact_e200_organizer_gradient_child", "parent_checkpoint": str(parent_checkpoint),
               "parent_checkpoint_sha256": _file_sha256(parent_checkpoint),
               "checkpoint_epoch": 200, "next_epoch": 201, "stop_epoch": 500,
               "schedule_total_epochs": 1000, "train_cases": len(train), "val_cases": len(val),
               "normalization_preserved": True, "optimizer_rng_preserved": True,
               "amendment": amendment, "training_launched": False}
    generated = output_root / "launch_profiles" / f"{request.run_id}_tree_lite_e500.json"
    copied_parent = proposal.path / "continuation_parent_epoch_200_model.pt"
    if prepare:
        generated.parent.mkdir(parents=True, exist_ok=True)
        with generated.open("x", encoding="utf-8") as stream:
            json.dump(profile, stream, indent=2)
            stream.write("\n")
        prepared_bundle = load_config_bundle(generated)
        run_dir = store.create(proposal, prepared_bundle, launch_facts=facts)
        shutil.copy2(parent_checkpoint, copied_parent)
        copied_sha256 = _file_sha256(copied_parent)
        if copied_sha256 != lineage["parent_checkpoint_sha256"]:
            raise ValueError("Copied continuation parent differs from the declared literal source checkpoint.")
        lineage["copied_checkpoint_sha256"] = copied_sha256
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["continuation"] = lineage
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(run_dir / "continuation_lineage.json", lineage)
    return {"prepared": prepare, "profile": str(generated), "child_run": str(proposal.path),
            "resume_checkpoint": str(copied_parent), "lineage": lineage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_child(args.profile, args.parent_checkpoint, run_id=args.run_id,
                                 output_root=args.output_root, prepare=args.prepare), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
