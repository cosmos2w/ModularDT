#!/usr/bin/env python3
"""Materialize explicit-arm Thermal development profiles without launching work.

Tracked profiles are unbound templates. One validated external quarter manifest
supplies identical train/test membership to every arm; native preparation rejects
templates until this tool binds the manifest fingerprint.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for source_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "Case_ThermalChannel/src", PROJECT_ROOT / "tools"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from thermal_campaign import ARMS, portfolio_profiles

DEFAULT_MANIFEST = "/data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json"
DEFAULT_RUN_ROOT = "/data/wanglz/ModularDT/thermal_development/HONF_Forward_Runs"
FAITHFULNESS_ARMS = {
    "Tree-L": "adaptive_receiver_hypergraph_honf",
    "Tree-F": "faithful_receiver_hypergraph_honf",
    "Pair-F": "direct_pairwise_control_honf",
    "Dense-new": "dense_pairwise_field",
}
MATURATION_ARMS = {"Fine-F": "three_term_full_access_honf"}


def validate_generated_output(value) -> Path:
    """Keep generated evidence outside Git or in its established ignored roots."""
    output = Path(value).expanduser().resolve()
    allowed = (PROJECT_ROOT / "diagnostics", PROJECT_ROOT / "Trained_Results")
    if output.is_relative_to(REPOSITORY_ROOT) and not any(output.is_relative_to(root) for root in allowed):
        raise ValueError("Generated output must live outside tracked repository paths or under ignored diagnostics/Trained_Results")
    return output


def registry_dataset_path(profile: dict) -> Path:
    """Resolve the native catalogue placement using JSON metadata only."""
    from channelthermal.resources import DatasetRegistry

    from honf_runtime.paths import resolve_path

    case = json.loads(resolve_path(profile["case"]["config"]).read_text())
    registry = DatasetRegistry.from_case_config(case["dataset"])
    return registry.resolve(profile["case"]["dataset_id"]).path.resolve()


def development_profiles(*, first_run_id: int = 3101, stage: int = 100,
                         manifest: dict | None = None,
                         manifest_path: str = DEFAULT_MANIFEST,
                         run_output_root: str = DEFAULT_RUN_ROOT,
                         microbatch_size: int = 8) -> dict[str, dict]:
    """Compose maintained templates or bind an already validated shared manifest."""
    if stage not in (100, 500, 1000):
        raise ValueError("Development stops are 100, 500 or 1000; later stages require explicit review.")
    if not 0 <= first_run_id <= 9995 or microbatch_size < 1:
        raise ValueError("Run IDs must fit four digits and microbatch size must be positive.")
    profiles = portfolio_profiles(first_run_id=first_run_id, stage=stage, microbatch_size=microbatch_size)
    train_ids = None
    if manifest is not None:
        from channelthermal.data.development_split import development_case_ids

        train_ids = set(development_case_ids(manifest, "train"))
    for arm, config in profiles.items():
        config["profile_name"] = f"thermal_development25_{arm.lower()}_v1"
        config["case"]["dataset"] = {
            "development_manifest": manifest_path,
            "development_manifest_sha256": None if manifest is None else manifest["manifest_sha256"],
        }
        config["training"]["plot_every_epochs"] = 100
        campaign = config["training"]["campaign"]
        campaign.update(name="thermal_development25_v1", schedule_total_epochs=1000)
        available = campaign["response_stencils"]
        campaign["response_stencils"] = ([] if train_ids is None else [path for path in available
            if Path(path).name.removeprefix("train_").removesuffix("_responses.npz") in train_ids][:4])
        if stage > 100 and not campaign["response_stencils"]:
            raise ValueError("A development continuation requires available selected-train response stencils; "
                             "do not silently drop response exposure or add excluded cases.")
        # The field alias is the sole quality selector; other metrics remain recorded.
        config["checkpointing"].update(save_best=False, save_best_field_mse=True,
            save_best_temperature_mse=False, save_best_predicted=False,
            save_best_every_epochs=100, save_latest_every_epochs=100,
            save_epoch_milestones=list(range(100, 1001, 100)))
        config["run"]["name"] = f"thermal_development25_{arm.lower()}_v1"
        config["run"]["output_root"] = run_output_root
        config["_note"] = (
            "Fixed shared 25% train/test development membership, bound before normalization. "
            "Full selected-case epochs, seed 0, native effective batch 48/Q1024; no batch caps. "
            "Independent 1000-epoch development horizon; review 100 before explicit 500/1000 continuation. "
            "Latest, field-best, retained milestones and plots every 100; no automatic portfolio launch. "
            "Response exposure uses at most 4 established atlas anchors within selected training IDs. "
            "Materialize this template with thermal_development.py before native preparation. "
            "Historical formal profiles/manual 5000 retain their separate full-data schedules."
        )
    return profiles


def resolve_evaluation_manifest(dataset_config: dict, dataset_path, *, scope: str = "auto",
                                manifest_path: Path | None = None) -> dict | None:
    """Preserve historical full scope; default a bound development checkpoint to its subset."""
    if scope not in ("auto", "development", "formal-full"):
        raise ValueError("Evaluation scope must be auto, development or formal-full.")
    if scope == "formal-full":
        if manifest_path is not None:
            raise ValueError("Formal-full evaluation cannot also request a development manifest.")
        return None
    from channelthermal.data.development_split import load_development_manifest, resolve_development_manifest

    if manifest_path is not None:
        manifest = load_development_manifest(manifest_path, dataset_path)
        saved = resolve_development_manifest(dataset_config, dataset_path, allow_embedded=True)
        if saved is not None and manifest["manifest_sha256"] != saved["manifest_sha256"]:
            raise ValueError("Evaluation manifest differs from the checkpoint development membership.")
    else:
        manifest = resolve_development_manifest(dataset_config, dataset_path, allow_embedded=True)
    if scope == "development" and manifest is None:
        raise ValueError("Development evaluation requires an explicit or checkpoint-bound manifest.")
    return manifest


def faithfulness_profiles(*, first_run_id: int = 3200, stage: int = 100,
                          manifest: dict | None = None,
                          manifest_path: str = DEFAULT_MANIFEST,
                          run_output_root: str = DEFAULT_RUN_ROOT,
                          microbatch_size: int = 8) -> dict[str, dict]:
    """Prepare the three named fresh controls and explicitly budgeted Dense.

    Tree-L has only its initial 100 screen. Tree-F/Pair-F accept saved review
    ages through 500 under the absolute 1000 schedule. Dense-new is the user's
    separately authorized 1000-epoch subset reference, with no warm start.
    This function prepares recipes; it never authorizes or launches extension.
    """
    if stage not in (100, 200, 300, 400, 500, 1000):
        raise ValueError("Faithfulness review ages are 100,200,300,400,500; Dense may use 1000.")
    ordinary = development_profiles(first_run_id=first_run_id, stage=100,
        manifest=manifest, manifest_path=manifest_path, run_output_root=run_output_root,
        microbatch_size=microbatch_size)
    profiles = {}
    for offset, (arm, architecture) in enumerate(FAITHFULNESS_ARMS.items()):
        if (arm == "Tree-L" and stage != 100) or (arm != "Dense-new" and stage > 500):
            continue
        config = copy.deepcopy(ordinary["B-native" if arm == "Dense-new" else "H-tree"])
        config["profile_name"] = f"thermal_faithfulness25_{arm.lower()}_v1"
        config["model"]["core_honf"]["forward_architecture"] = architecture
        config["training"]["epochs"] = stage
        campaign = config["training"]["campaign"]
        campaign.update(name="thermal_faithfulness25_v1", arm=arm)
        if stage > 100:
            campaign.update(physical_loss_policy_version=2, native_loss_denominators_start_epoch=101)
        if arm in {"Tree-F", "Pair-F"}:
            # Domain-owned semantics stay in the case profile. Training
            # preparation fits its selected-train-only scale before model
            # construction and records it in resolved run configuration.
            config["case"]["config"] = "project://Case_ThermalChannel/configs/case_source_local.json"
        if arm == "Pair-F":
            config["model"]["core_honf"]["interface_model"]["hypergraph_options"] = {"pair_control_width": 80}
        if arm == "Tree-F":
            config["model"]["core_honf"]["interface_model"].setdefault("hypergraph_options", {})[
                "structural_measure_policy_version"] = 1
        config["run"].update(id=f"{first_run_id + offset:04d}", name=f"thermal_faithfulness25_{arm.lower()}_v1")
        config["_note"] = (
            f"{arm}: fresh seed0 fixed25_v1 150/22 control; train-only normalization, "
            "Q1024 FP32 effective batch48, common B-fine physical initializer. "
            "Absolute development schedule1000; checkpoints/field selection/plots every100. "
            "Tree-L stops100; Tree-F/Pair-F stop100 then require documented review before paired "
            "100-epoch continuation through at most500. Dense-new separately authorized1000. "
            "No full-data resume, formal launch, new solves or Wind training."
        )
        profiles[arm] = config
    return profiles


def evaluation_dataset_kwargs(manifest: dict | None, split: str) -> dict:
    """Apply the same exact membership to normalized and raw native datasets."""
    if manifest is None:
        return {}
    from channelthermal.data.development_split import development_case_ids

    return {"case_ids": development_case_ids(manifest, split)}


def maturation_profiles(*, first_run_id: int = 3301, stage: int = 100,
                        manifest: dict | None = None, manifest_path: str = DEFAULT_MANIFEST,
                        run_output_root: str = DEFAULT_RUN_ROOT,
                        microbatch_size: int = 8) -> dict[str, dict]:
    """Prepare the single fresh source-local backbone sufficiency control.

    The profile carries policy2 from preparation, but its denominator/atlas/null
    exposure still activates only at e101. Preparation does not authorize a
    fresh arm's e100 review or any conditional e1000 continuation.
    """
    if stage not in (100, 500, 1000):
        raise ValueError("Fine-F preparation stops are 100,500 or a separately budgeted1000.")
    config = copy.deepcopy(development_profiles(first_run_id=first_run_id, stage=stage,
        manifest=manifest, manifest_path=manifest_path, run_output_root=run_output_root,
        microbatch_size=microbatch_size)["B-fine"])
    config["case"]["config"] = "project://Case_ThermalChannel/configs/case_source_local.json"
    config["profile_name"] = "thermal_tree_lite25_fine-f_v1"
    config["run"].update(id=f"{first_run_id:04d}", name="thermal_tree_lite25_fine-f_v1")
    config["training"]["campaign"].update(name="thermal_tree_lite25_v1", arm="Fine-F",
        structural_weight=0.0, physical_loss_policy_version=2, native_loss_denominators_start_epoch=101,
        heat_null_response={"benchmark_verified": True, "cases_per_epoch": 2,
                            "fluid_queries": 256, "fraction": .1})
    config["_note"] = ("Fresh common B-fine seed0 physical initializer, source_local_v3, fixed25_v1; "
        "full-access fine backbone without Tree/Pair heads or structural objective. Fewer parameters; "
        "same absolute1000 schedule and e101 denominator/selected-train atlas/tiny heat-null exposure. "
        "Review e100 before continuing through500. No full-source or formal training.")
    return {"Fine-F": config}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", nargs="+", choices=(*ARMS, *FAITHFULNESS_ARMS, *MATURATION_ARMS), required=True,
                        help="Only these explicitly named arms are prepared; no trainer is launched.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--first-run-id", required=True, type=int)
    parser.add_argument("--stage", type=int, choices=(100, 200, 300, 400, 500, 1000), default=100)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--run-output-root", type=Path, default=Path(DEFAULT_RUN_ROOT))
    args = parser.parse_args(argv)
    if len(set(args.arm)) != len(args.arm):
        parser.error("Arm names must not repeat.")
    return args


def main(argv=None) -> int:
    from channelthermal.data.development_split import development_case_ids, load_development_manifest

    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    manifest = load_development_manifest(manifest_path, dataset_path)
    faithful = any(arm in FAITHFULNESS_ARMS for arm in args.arm)
    maturation = any(arm in MATURATION_ARMS for arm in args.arm)
    if ((faithful or maturation) and any(arm in ARMS for arm in args.arm)) or (faithful and maturation):
        raise ValueError("Prepare legacy, faithfulness and maturation arm names in separate explicit invocations")
    generator = maturation_profiles if maturation else faithfulness_profiles if faithful else development_profiles
    profiles = generator(first_run_id=args.first_run_id, stage=args.stage,
        manifest=manifest, manifest_path=str(manifest_path), microbatch_size=args.microbatch_size,
        run_output_root=str(validate_generated_output(args.run_output_root)))
    for arm in args.arm:
        if arm not in profiles:
            raise ValueError(f"{arm} is outside its authorized preparation age at epoch{args.stage}")
        if dataset_path != registry_dataset_path(profiles[arm]):
            raise ValueError("Supplied dataset differs from the native registry path; update the established catalogue location before preparing profiles")
    output = validate_generated_output(args.output_dir)
    paths = [output / f"{arm.lower()}_e{args.stage}.json" for arm in args.arm]
    if any(path.exists() for path in [*paths, output / "preparation_index.json"]):
        raise FileExistsError("Development preparation already exists; preserve the previous profiles/index.")
    output.mkdir(parents=True, exist_ok=True)
    index = {"scope": "Development profile preparation only; no run workspace/model/trainer created",
        "manifest": str(manifest_path), "manifest_sha256": manifest["manifest_sha256"],
        "dataset": str(dataset_path),
        "train_cases": len(development_case_ids(manifest, "train")),
        "test_cases": len(development_case_ids(manifest, "test")), "stage": args.stage,
        "schedule_total_epochs": 1000, "profiles": {}}
    for arm in args.arm:
        config = copy.deepcopy(profiles[arm])
        path = output / f"{arm.lower()}_e{args.stage}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        index["profiles"][arm] = {"path": str(path), "run_id": config["run"]["id"],
            "run_output_root": config["run"]["output_root"],
            "response_stencils": config["training"]["campaign"]["response_stencils"],
            "response_case_ids": [Path(p).name.removeprefix("train_").removesuffix("_responses.npz")
                for p in config["training"]["campaign"]["response_stencils"]],
            "response_case_count": len(config["training"]["campaign"]["response_stencils"]),
            "response_diversity": "Available selected-train atlas anchors only; no forced case inclusion"}
        suffix = "" if args.stage == 100 or arm == "Dense-new" else " --resume-checkpoint PATH_TO_EXACT_DEVELOPMENT_PARENT"
        print(f"{arm}: python train.py --config {shlex.quote(str(path))} --device cpu{suffix} --dry-run")
    (output / "preparation_index.json").write_text(json.dumps(index, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
