"""Case-local HET-MF command line interface."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .config import load_config
from .io import sha256_file, write_json_atomic
from .resources import gpu2_status, link_dataset, require_budget
from .study_design import build_formal_plan, build_pilot_plan, hash_plan


CASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = CASE_DIR / "configs" / "pilot_v1.yaml"


def _plan(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("plan_hash") != hash_plan(data):
        raise RuntimeError("input plan hash mismatch")
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hetmf")
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    planning = commands.add_parser("plan")
    planning.add_argument("--profile", choices=("pilot", "compact", "standard", "expanded"), required=True)
    planning.add_argument("--output", type=Path, required=True)
    planning.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pilot = commands.add_parser("pilot")
    pilot.add_argument("--stage", choices=("representatives", "qualify", "qualify384", "interactions", "interaction_refine", "throughput", "all"), required=True)
    pilot.add_argument("--plan", type=Path, required=True)
    pilot.add_argument("--root", type=Path, required=True)
    pilot.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pilot.add_argument("--backend", choices=("cpu", "gpu_fp64"), default="cpu")
    pilot.add_argument("--batch-size", type=int, default=1)
    validate = commands.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--full", action="store_true")
    pack = commands.add_parser("pack")
    pack.add_argument("--root", type=Path)
    pack.add_argument("--release", type=Path)
    pack.add_argument("--plan", type=Path)
    pack.add_argument("--approval", type=Path)
    pack.add_argument("--verify-source-hashes", action="store_true")
    pack.add_argument("--scope", choices=("pilot",), default="pilot")
    viz = commands.add_parser("visualize")
    viz.add_argument("--root", type=Path, required=True)
    viz.add_argument("--gallery", choices=("representatives", "quartets", "thermal_null", "numerical_board", "responses"), default="representatives")
    forecast = commands.add_parser("forecast")
    forecast.add_argument("--pilot", type=Path, required=True)
    forecast.add_argument("--profiles", nargs="+", choices=("compact", "standard", "expanded"),
                          default=("compact", "standard", "expanded"))
    gpu_benchmark = commands.add_parser("gpu-benchmark")
    gpu_benchmark.add_argument("--pilot", type=Path, required=True)
    gpu_benchmark.add_argument("--allow-concurrent-gpu2", action="store_true")
    links = commands.add_parser("link")
    links.add_argument("--root", type=Path, required=True)
    links.add_argument("--case-dir", type=Path, default=CASE_DIR)
    links.add_argument("--replace-owned-links", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--root", type=Path, required=True)
    formal = commands.add_parser("generate")
    formal.add_argument("--plan", type=Path, required=True)
    formal.add_argument("--root", type=Path)
    formal.add_argument("--dry-run", action="store_true")
    formal.add_argument("--approval", type=Path)
    formal.add_argument("--yes", action="store_true")
    formal.add_argument("--device", default="cuda:0")
    formal.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "preflight":
        config = load_config(args.config)
        r = config["resources"]
        budget = require_budget(retained_bytes=int(r["retained_budget_gib"] * 2**30),
                                temporary_bytes=int(r["temporary_budget_gib"] * 2**30),
                                minimum_free_bytes=int(r["minimum_free_disk_gib"] * 2**30))
        print(json.dumps({"storage": budget, "gpu2": gpu2_status()}, indent=2))
    elif args.command == "plan":
        from .resources import require_owned_root
        config = load_config(args.config)
        output = Path(args.output)
        require_owned_root(output.parent)
        data = build_pilot_plan(config["rng"]["root_seed"]) if args.profile == "pilot" else build_formal_plan(args.profile, config["rng"]["root_seed"])
        if output.exists() and _plan(output) != data:
            raise RuntimeError("refusing to change frozen input plan")
        write_json_atomic(output, data)
        print(json.dumps({"path": str(output), "plan_hash": data["plan_hash"], "counts": data["counts"]}, indent=2))
    elif args.command == "pilot":
        from .pipeline import run_pilot_stage
        config = load_config(args.config)
        plan = _plan(args.plan)
        if args.batch_size < 1 or args.batch_size > 32:
            raise ValueError("pilot batch size must be 1..32")
        stages = ("representatives", "qualify", "interactions", "throughput") if args.stage == "all" else (args.stage,)
        for stage in stages:
            result = run_pilot_stage(args.root, plan, config, stage, backend=args.backend,
                                     batch_size=args.batch_size)
            print(json.dumps(result, indent=2))
            if not result["complete"]:
                raise RuntimeError(f"pilot stage incomplete: {stage}")
    elif args.command == "validate":
        from .validation import audit_persisted
        from .io import validate_ledger_case, validate_raw_case
        from .pipeline import _solver_hash
        root = args.root
        plan = _plan(root / "inputs" / "pilot_plan.json")
        config = load_config(DEFAULT_CONFIG)
        lookup = {s["state_id"]: s for s in plan["scenes"]}
        cases = sorted((root / "raw").glob("*_n*.h5"))
        results = []
        for path in cases:
            name, suffix = path.stem.rsplit("_n", 1)
            scene, grid = lookup[name], int(suffix)
            backend = "cpu"
            for candidate in ("cpu", "gpu_fp64"):
                try:
                    validate_raw_case(path, scene, _solver_hash(config, grid, candidate))
                    backend = candidate
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(f"case does not match known solver hash: {path}")
            validate_ledger_case(root, path)
            results.append({"state_id": name, "grid": grid, "backend": backend,
                            **audit_persisted(root, scene, grid)})
        output = {"checked": len(results), "accepted": sum(r["accepted"] for r in results),
                  "failed": [r for r in results if not r["accepted"]]}
        write_json_atomic(root / "diagnostics" / "post_storage_validation.json", output)
        print(json.dumps(output, indent=2))
        if output["failed"]:
            raise RuntimeError("post-storage validation failed")
    elif args.command == "pack":
        if args.release is not None:
            if args.root is not None or args.plan is None or args.approval is None or not args.verify_source_hashes:
                raise ValueError("formal packing requires --release --plan --approval --verify-source-hashes")
            from .formal import pack_formal_release
            print(json.dumps(pack_formal_release(_plan(args.plan), json.loads(args.approval.read_text()),
                                                 root=args.release), indent=2))
            return
        if args.root is None:
            raise ValueError("pilot packing requires --root")
        from .processing import pack_case
        started = time.perf_counter()
        records = []
        for path in sorted((args.root / "raw").glob("*_n*.h5")):
            state_id, n = path.stem.rsplit("_n", 1)
            if int(n) == 96:
                records.append(pack_case(args.root, state_id, int(n)))
        summary = {"processed": len(records), "bytes": sum(x["bytes"] for x in records),
                   "elapsed_seconds": time.perf_counter() - started,
                   "new_files": sum(not x["reused"] for x in records)}
        write_json_atomic(args.root / "diagnostics" / "packing_summary.json", summary)
        print(json.dumps(summary, indent=2))
    elif args.command == "visualize":
        from .resources import require_owned_root
        owned = require_owned_root(args.root)
        (owned / "temporary" / "mplconfig").mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(owned / "temporary" / "mplconfig")
        os.environ["TMPDIR"] = str(owned / "temporary")
        from .visualization import (visualize_case, visualize_numerical_board,
                                    visualize_physical_responses, visualize_quartet,
                                    visualize_thermal_null)
        plan = _plan(args.root / "inputs" / "pilot_plan.json")
        lookup = {s["state_id"]: s for s in plan["scenes"]}
        if args.gallery == "representatives":
            records = [visualize_case(args.root, lookup[sid]) for sid in plan["representative_ids"]]
        elif args.gallery == "quartets":
            records = [visualize_quartet(args.root, [lookup[sid] for sid in q["endpoint_state_ids"]],
                                         quartet_id=q["quartet_id"])
                       for q in plan["quartets"]]
        elif args.gallery == "thermal_null":
            records = [visualize_thermal_null(args.root, plan)]
        elif args.gallery == "responses":
            records = [visualize_physical_responses(args.root)]
        else:
            records = [visualize_numerical_board(args.root,
                                                json.loads((args.root / "diagnostics" / "pilot_summary.json").read_text()),
                                                json.loads((args.root / "diagnostics" / "forecasts_cpu.json").read_text()))]
        print(json.dumps({"figures": records}, indent=2))
    elif args.command == "forecast":
        from .forecasting import forecast_profiles
        print(json.dumps(forecast_profiles(args.pilot, args.profiles), indent=2))
    elif args.command == "gpu-benchmark":
        if not args.allow_concurrent_gpu2:
            raise PermissionError("shared GPU2 pilot requires explicit --allow-concurrent-gpu2")
        from .gpu_benchmark import run_gpu_pilot
        print(json.dumps(run_gpu_pilot(args.pilot, _plan(args.pilot / "inputs" / "pilot_plan.json")), indent=2))
    elif args.command == "link":
        print(json.dumps(link_dataset(args.root, args.case_dir, replace_owned_links=args.replace_owned_links), indent=2))
    elif args.command == "status":
        root = args.root
        print(json.dumps({"root": str(root), "raw_cases": len(list((root / "raw").glob("*.h5"))),
                          "processed_cases": len(list((root / "processed").glob("*.h5"))),
                          "figures": len(list((root / "figures").glob("*.pdf"))),
                          "plan_sha256": sha256_file(root / "inputs" / "pilot_plan.json") if (root / "inputs" / "pilot_plan.json").exists() else None}, indent=2))
    elif args.command == "generate":
        from .formal import dry_run_formal, run_formal
        plan = _plan(args.plan)
        if args.dry_run:
            if args.approval or args.yes:
                raise ValueError("dry-run accepts no approval/execution switch")
            result = dry_run_formal(plan, root=args.root)
        else:
            if args.approval is None or args.root is None:
                raise PermissionError("formal generation requires --approval and --root")
            result = run_formal(plan, json.loads(args.approval.read_text()), root=args.root,
                                yes=args.yes, device=args.device, resume=args.resume)
        print(json.dumps(result, indent=2))
