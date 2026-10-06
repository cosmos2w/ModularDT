#!/usr/bin/env python3
"""Prepare the matched fit100 physical refinements; never launch or solve."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.local_surrogate.model import LocalModuleConfig, LocalModuleSurrogate
from channelthermal.model import ChannelThermalHONFModel
from channelthermal.training.checkpoints import _file_sha256
from channelthermal.training.refinement_policy import (
    POLICY_DECLARATION,
    REFINEMENT_KIND,
    apply_refinement_policy,
    attach_refinement_checkpoint,
    validate_refinement_campaign,
    validate_refinement_parent,
)
from channelthermal.training.response_refinement import (
    response_refinement_declaration,
    validate_response_refinement_declaration,
)
from thermal_development import validate_generated_output

from honf_runtime.case_protocol import WorkflowRequest
from honf_runtime.compat import load_trusted_checkpoint, strip_module_prefix
from honf_runtime.config_loader import load_config_bundle
from honf_runtime.registry import load_case_plugin
from honf_runtime.run_store import RunStore, atomic_write_json

DEFAULT_FITS = Path("/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/continuation/prepared_interface_fits.json")
DEFAULT_ATLAS = PROJECT_ROOT / "diagnostics/generated/interactions/physical_response_atlas_20260926/families"
TEMPLATE_ROOT = PROJECT_ROOT / "src/config_core/forward/thermal_response_refinement"


def build_refinement_profile(parent, *, source_profile, parent_path, arm, run_id, output_root,
                             atlas_directory=DEFAULT_ATLAS, calibration=None, epochs=100):
    """Inherit the exact primary experiment and bind only explicit refinement changes."""
    if arm not in ("H-add", "H-joint") or epochs not in (100, 500, 1000):
        raise ValueError("Refinement has only H-add/H-joint and reviewed100/500/1000 stops.")
    if not 0 <= int(run_id) <= 9999:
        raise ValueError("Refinement Run_ID must fit four digits.")
    template = json.loads((TEMPLATE_ROOT / f"{arm.lower()}.json").read_text())
    if template["training"]["campaign"]["forward_refinement"] != POLICY_DECLARATION:
        raise ValueError("Refinement template policy differs from its maintained implementation.")
    profile = copy.deepcopy(source_profile)
    profile["profile_name"] = template["profile_name"]
    payload = copy.deepcopy(parent["model_config"])
    options = payload["core_honf"]["interface_model"].setdefault("hypergraph_options", {})
    if options.get("query_interaction_mode") != {"H-add": "add", "H-joint": "joint"}[arm]:
        raise ValueError("Refinement must inherit its own matched arm's fit100.")
    options["query_admission_mode"] = "soft"
    resolved_model = ChannelThermalHONFConfig.from_dict(payload).to_dict()
    # Case-specific coupling belongs to the inherited case profile. Core
    # profile schemas accept only the reusable core declaration here.
    profile["model"] = {"core_honf": resolved_model["core_honf"]}
    campaign = copy.deepcopy(parent["train_config"]["training"]["campaign"])
    campaign.pop("interface_fit", None)
    campaign.pop("heat_null_response", None)
    campaign.update(copy.deepcopy(template["training"]["campaign"]))
    campaign["parent"] = {"kind": REFINEMENT_KIND, "epoch": 100, "checkpoint": str(Path(parent_path).resolve())}
    campaign.update(response_refinement_declaration(atlas_directory))
    campaign["response_refinement"]["calibration"] = copy.deepcopy(calibration)
    campaign["response_stencils"] = []  # The separately identified addendum owns all new response exposure.
    validate_response_refinement_declaration(campaign)
    profile["training"] = copy.deepcopy(parent["train_config"]["training"])
    # Native checkpoints include these runtime fields; core profiles declare
    # their identity under run instead of duplicating it in training.
    for key in ("Run_ID", "run_name"):
        profile["training"].pop(key, None)
    profile["training"].update(copy.deepcopy(template["training"]))
    profile["training"].update(campaign=campaign, epochs=epochs)
    profile["case"].setdefault("dataset", {}).update(
        development_manifest=parent["train_config"]["dataset"]["development_manifest"],
        development_manifest_sha256=parent["train_config"]["dataset"]["development_manifest_sha256"])
    profile["checkpointing"].update(copy.deepcopy(template["checkpointing"]))
    profile["run"].update(id=f"{int(run_id):04d}", name=template["run"]["name"], output_root=str(output_root))
    profile["_note"] = template["_note"].replace("Unbound override template. ", "")
    validate_refinement_parent(parent, {"model": profile["model"]})
    return profile


def construct_refinement_model(parent, *, device="cpu"):
    """Restore embedded Stage-A and all normalization before strict parent copying."""
    payload = copy.deepcopy(parent["model_config"])
    payload["core_honf"]["interface_model"].setdefault("hypergraph_options", {})["query_admission_mode"] = "soft"
    config = ChannelThermalHONFConfig.from_dict(payload)
    validate_refinement_parent(parent, {"model": config.to_dict()})
    model = ChannelThermalHONFModel(config, attach_local_from_checkpoint=False)
    if config.channelthermal.use_local_surrogate:
        local_payload = parent.get("local_model_config")
        if not isinstance(local_payload, dict) or parent.get("local_surrogate_frozen") is not True:
            raise ValueError("Refinement requires the parent's embedded frozen Stage-A identity.")
        local_model = LocalModuleSurrogate(LocalModuleConfig.from_dict(local_payload))
        model.local_coupling.set_local_surrogate(local_model, freeze=True,
            normalization_config=copy.deepcopy(parent.get("local_normalization_config", {})),
            normalization_stats=copy.deepcopy(parent.get("local_normalization_stats", {})))
        model.local_coupling.local_surrogate_checkpoint_path = parent.get(
            "local_checkpoint_provenance", parent.get("local_surrogate_checkpoint_path"))
    normalization = parent.get("global_normalization_config", parent["train_config"]["dataset"])
    model.set_global_target_normalization(copy.deepcopy(parent.get("global_normalization_stats", {})),
        normalize_targets=bool(normalization.get("normalize_targets", False)))
    model.load_state_dict(strip_module_prefix(parent["model_state_dict"]), strict=True)
    apply_refinement_policy(model)
    model.set_training_progress(epoch=0, total_epochs=1000)
    model.to(device)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-fits", default=str(DEFAULT_FITS))
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--atlas-directory", default=str(DEFAULT_ATLAS))
    parser.add_argument("--first-run-id", type=int, default=3801)
    parser.add_argument("--arm", choices=("H-add", "H-joint", "both"), default="both")
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    calibration = json.loads(Path(args.calibration_json).expanduser().read_text())
    if calibration is None:
        raise ValueError("Preparation requires the pooled positive train-only calibration, never inherited frozen-fit coefficients.")
    declaration = response_refinement_declaration(args.atlas_directory)
    declaration["response_refinement"]["calibration"] = calibration
    validate_response_refinement_declaration(declaration)
    fits = json.loads(Path(args.prepared_fits).expanduser().read_text())
    output_root = validate_generated_output(args.output_root)
    results_path = output_root / "prepared_response_refinements.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for offset, arm in enumerate(("H-add", "H-joint")):
        if args.arm not in (arm, "both"):
            continue
        if arm in results:
            raise FileExistsError(f"Refinement already prepared: {arm}")
        parent_path = Path(fits[arm]["run_dir"]).resolve() / "epoch_0100_model.pt"
        parent = load_trusted_checkpoint(parent_path, map_location="cpu")
        source_profile = json.loads((parent_path.parent / "configs/core_source.json").read_text())
        profile = build_refinement_profile(parent, source_profile=source_profile, parent_path=parent_path,
            arm=arm, run_id=args.first_run_id + offset, output_root=output_root,
            atlas_directory=args.atlas_directory, calibration=calibration)
        preparation = output_root / "preparation"
        profile_path = preparation / f"{arm.lower()}_refinement100.json"
        if profile_path.exists():
            raise FileExistsError(f"Refusing to overwrite refinement profile: {profile_path}")
        preparation.mkdir(parents=True, exist_ok=True)
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
        from channelthermal.workflows.train_forward import (
            build_model_config,
            build_training_datasets,
            resolve_auto_internal_mode,
            resolve_development_training,
        )
        development = resolve_development_training(canonical, effective_epochs=100)
        target = construct_refinement_model(parent)
        train_dataset, val_dataset = build_training_datasets(canonical, development)
        resolved_config = build_model_config(canonical, train_dataset)
        resolve_auto_internal_mode(resolved_config, target)
        if resolved_config.to_dict() != target.config.to_dict():
            raise ValueError("Inherited case profile resolves to different physical/coupling settings from the exact fit100 parent.")
        canonical["model"] = target.config.to_dict()
        validate_refinement_campaign(canonical["training"]["campaign"], canonical)
        child, receipt = attach_refinement_checkpoint(parent, target_model=target, target_config=target.config,
            refinement_config=canonical, source_path=parent_path, source_sha256=_file_sha256(parent_path))
        run_dir = store.create(proposal, bundle, launch_facts=launch_facts)
        child_path = run_dir / "attached_refinement0000.pt"
        torch.save(child, child_path)
        atomic_write_json(run_dir / "forward_refinement_attachment.json", receipt)
        RunStore.update_status(run_dir, "prepared", parent_checkpoint=str(parent_path),
            frozen_backbone_epoch=1000, interface_prefit_epoch=100, refinement_epoch=0, refinement_arm=arm)
        results[arm] = {"run_dir": str(run_dir), "config": str(profile_path), "checkpoint": str(child_path),
            "parent_checkpoint": str(parent_path), "source_checkpoint_sha256": receipt["source_checkpoint_sha256"],
            "command": f"CUDA_VISIBLE_DEVICES={1 + offset} python train.py --config {shlex.quote(str(profile_path))} "
                       f"--resume-checkpoint {shlex.quote(str(child_path))} --device cuda:0 --yes"}
        atomic_write_json(results_path, results)
        del target, child, train_dataset, val_dataset
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
