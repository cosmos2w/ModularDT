"""Profile projections from complete observed pilot costs and stored bytes."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .io import write_json_atomic


PROFILE_COUNTS = {"compact": (2304, 120, 240), "standard": (4608, 240, 480),
                  "expanded": (9216, 480, 960)}


def _gib(value: float) -> float:
    return value / 2**30


def forecast_profiles(root: Path, profiles: tuple[str, ...] | list[str] = ("compact", "standard", "expanded")) -> dict:
    root = Path(root)
    attempts = [json.loads(line) for line in (root / "diagnostics" / "attempts.jsonl").read_text().splitlines()]
    panel = [row for row in attempts if row.get("stage") == "throughput" and row.get("status") == "accepted"
             and row.get("backend") == "cpu" and row.get("grid") == 96]
    if len(panel) != 256 or len({row["state_id"] for row in panel}) != 256:
        raise RuntimeError("complete 256-state CPU throughput panel required for forecast")
    complete_seconds = np.asarray([row["complete_seconds_amortized"] for row in panel])
    if np.any(~np.isfinite(complete_seconds)) or np.any(complete_seconds <= 0):
        raise RuntimeError("invalid measured host timings")
    raw_bytes = np.asarray([(root / "raw" / f"{row['state_id']}_n96.h5").stat().st_size
                            for row in panel], dtype=np.float64)
    processed_bytes = np.asarray([(root / "processed" / f"{row['state_id']}_n96.h5").stat().st_size
                                  for row in panel], dtype=np.float64)
    packing = json.loads((root / "diagnostics" / "packing_summary.json").read_text())
    pilot_pack_seconds_per_case = packing["elapsed_seconds"] / packing["processed"]
    shard_qa = json.loads((root / "diagnostics" / "formal_shard_mechanical_qa.json").read_text())
    shard_pack_seconds_per_case = shard_qa["first_seconds"] / shard_qa["first"]["states"]
    raw_shard_bytes_per_case = shard_qa["first"]["raw_bytes"] / shard_qa["first"]["states"]
    processed_shard_bytes_per_case = shard_qa["first"]["processed_bytes"] / shard_qa["first"]["states"]
    gpu = json.loads((root / "diagnostics" / "gpu2_pilot_timing_summary.json").read_text())
    if gpu["by_batch_size"]["32"]["accepted_evaluations"] != 768:
        raise RuntimeError("three complete 256-state B32 GPU timing repeats required")
    gpu_rows = [json.loads(line) for line in (root / "diagnostics" / "gpu2_pilot_timing.jsonl").read_text().splitlines()]
    b32 = np.asarray([r["elapsed_seconds"] / r["count"] for r in gpu_rows
                      if r.get("phase") == "timing" and r.get("batch_size") == 32 and r.get("status") == "accepted"])
    host_overhead = np.asarray([row["complete_seconds_amortized"] - row["solve_seconds"] for row in panel])
    if np.any(host_overhead <= 0):
        raise RuntimeError("invalid measured non-solver host overhead")
    gpu_case_mean = float(b32.mean() + host_overhead.mean())
    gpu_case_p90 = float(np.quantile(b32, 0.9) + np.quantile(host_overhead, 0.9))
    figures = list((root / "figures").glob("*.pdf")) + list((root / "figures").glob("*.png"))
    selected_figure_bytes = sum(path.stat().st_size for path in figures)
    qa_files = [path for path in (root / "raw").glob("*_n*.h5") if not path.stem.endswith("_n96")]
    qa_bytes_measured = sum(path.stat().st_size for path in qa_files)
    qa_seconds_measured = sum(row.get("complete_seconds_amortized", 0.0) for row in attempts
                              if row.get("stage") in ("qualify", "qualify384", "interaction_refine")
                              and row.get("status") == "accepted")
    plotting = json.loads((root / "diagnostics" / "initial_visualization_timings.json").read_text())
    plot_seconds_per_representative = plotting["representative_wall_seconds"] / plotting["representative_pages"]
    plot_seconds_per_quartet = plotting["quartet_wall_seconds"] / plotting["quartet_pages"]
    case_mean = float(complete_seconds.mean())
    case_p90 = float(np.quantile(complete_seconds, 0.9))
    pilot_plan = json.loads((root / "inputs" / "pilot_plan.json").read_text())
    pilot_lookup = {scene["state_id"]: scene for scene in pilot_plan["scenes"]}
    cpu_strata = defaultdict(list)
    for row in panel:
        scene = pilot_lookup[row["state_id"]]
        cpu_strata[(scene["environment"]["family"], len(scene["contacts"]))].append(row["complete_seconds_amortized"])
    cpu_family = defaultdict(list)
    for (family, _), values in cpu_strata.items():
        cpu_family[family].extend(values)
    results = {}
    for profile in profiles:
        if profile not in PROFILE_COUNTS:
            raise ValueError(profile)
        plan_path = root / "inputs" / f"formal_{profile}_plan.json"
        plan = json.loads(plan_path.read_text())
        states = len(plan["states"])
        _, env_roots_train, layout_roots_train = PROFILE_COUNTS[profile]
        if states != PROFILE_COUNTS[profile][0] or plan["counts"]["total_states"] != states:
            raise RuntimeError(f"formal {profile} plan/count mismatch")
        challenges = Counter(row.get("challenge", "default") for row in plan["states"])
        contact_counts = Counter(len(plan["layouts"][row["layout_root_id"]]["contacts"])
                                 for row in plan["states"])
        weighted_cpu = []
        missing_strata = Counter()
        for row in plan["states"]:
            family = plan["environments"][row["environment_root_id"]]["family"]
            m = len(plan["layouts"][row["layout_root_id"]]["contacts"])
            values = cpu_strata.get((family, m))
            if values is None:
                missing_strata[f"{family}:m{m}"] += 1
                values = cpu_family.get(family, complete_seconds)
            weighted_cpu.append(float(np.mean(values)))
        weighted_case_mean_cpu = float(np.mean(weighted_cpu))
        gallery_pages = {"compact": 24, "standard": 36, "expanded": 48}[profile]
        quartets = {"compact": 32, "standard": 64, "expanded": 128}[profile]
        selected_figure_count = gallery_pages + quartets
        gallery_bytes = selected_figure_count * selected_figure_bytes / max(len(figures), 1)
        plotting_seconds = gallery_pages * plot_seconds_per_representative + quartets * plot_seconds_per_quartet
        qa_bytes = qa_bytes_measured
        raw_source = states * float(raw_bytes.mean())
        # The approved packer retains per-case raw sources and also creates raw
        # shards, so budget both copies until a separately reviewed retention
        # change. This is conservative against measured per-case HDF5 overhead.
        raw_shards = states * raw_shard_bytes_per_case
        raw = raw_source + raw_shards
        processed = states * processed_shard_bytes_per_case
        pack = states * shard_pack_seconds_per_case
        # Five-percent retry/time reserve is prospective; no CPU numerical retry
        # occurred in the 256 accepted throughput states.
        mean_seconds = states * weighted_case_mean_cpu * 1.05 + pack + plotting_seconds + qa_seconds_measured
        conservative_seconds = states * case_p90 * 1.1 + pack + plotting_seconds + qa_seconds_measured
        gpu_mean_seconds = states * gpu_case_mean * 1.05 + pack + plotting_seconds + qa_seconds_measured
        gpu_conservative_seconds = states * gpu_case_p90 * 1.1 + pack + plotting_seconds + qa_seconds_measured
        results[profile] = {
            "plan_hash": plan["plan_hash"], "states": states,
            "environment_roots_total": len(plan["environments"]),
            "layout_roots_total": len(plan["layouts"]),
            "train_environment_roots": env_roots_train,
            "train_layout_roots": layout_roots_train, "native_grid": 96,
            "electrical_systems_base": states, "thermal_systems_base": states,
            "challenge_counts": dict(challenges), "contact_count_counts": {str(k): v for k, v in contact_counts.items()},
            "cpu_unmeasured_strata": dict(missing_strata),
            "raw_dtype": "float64", "processed_dtype": "float32",
            "mean_total_hours_cpu": mean_seconds / 3600,
            "conservative_total_hours_cpu": conservative_seconds / 3600,
            "mean_total_hours_gpu2_shared": gpu_mean_seconds / 3600,
            "conservative_total_hours_gpu2_shared": gpu_conservative_seconds / 3600,
            "raw_gib": _gib(raw), "raw_source_gib": _gib(raw_source),
            "raw_shards_gib": _gib(raw_shards), "processed_gib": _gib(processed),
            "qa_gib": _gib(qa_bytes), "selected_figures_gib": _gib(gallery_bytes),
            "peak_temporary_gib": _gib(64 * (max(float(raw_bytes.max()), raw_shard_bytes_per_case) +
                                          max(float(processed_bytes.max()), processed_shard_bytes_per_case))),
            "selected_plot_seconds": plotting_seconds,
            "qa_seconds_from_pilot": qa_seconds_measured,
            "formal_shard_pack_time_status": "four-case verified pilot mechanical shard timing extrapolated",
            "measured_case_seconds_mean": case_mean,
            "measured_case_seconds_p90": case_p90,
            "measured_cpu_stratified_case_seconds_mean": weighted_case_mean_cpu,
            "measured_gpu2_shared_b32_plus_host_overhead_seconds_mean": gpu_case_mean,
            "measured_gpu2_shared_b32_plus_host_overhead_seconds_p90": gpu_case_p90,
            "measured_pilot_pack_seconds_per_case": pilot_pack_seconds_per_case,
            "measured_formal_shard_pack_seconds_per_case": shard_pack_seconds_per_case,
            "incremental_60_train_environments_720_states_hours_cpu":
                (720 * (weighted_case_mean_cpu * 1.05 + shard_pack_seconds_per_case)) / 3600,
            "incremental_60_train_environments_720_states_hours_gpu2_shared":
                (720 * (gpu_case_mean * 1.05 + shard_pack_seconds_per_case)) / 3600,
            "status": "cpu_complete_and_gpu2_shared_measured; formal_extrapolation",
        }
    summary = {
        "source": "256 accepted full CPU FP64 native-96 throughput states, per-case atomic HDF5 LZF writes and post-storage audits",
        "timing_scope": "CPU complete wall includes preparation, sparse direct solve, diagnostics, raw write, post-storage audit; GPU2 shared B32 wall includes preparation, transfer, coupled PCG, CPU finalization and in-memory validation, plus measured CPU non-solver host overhead as a conservative write/audit proxy; four-case formal shard pack and selected plotting are added separately",
        "measured_panel_states": 256,
        "measured_raw_bytes_per_case_mean": float(raw_bytes.mean()),
        "measured_processed_bytes_per_case_mean": float(processed_bytes.mean()),
        "measured_p50_case_seconds": float(np.median(complete_seconds)),
        "measured_p90_case_seconds": case_p90,
        "measured_gpu2_shared_b32_case_seconds": float(b32.mean()),
        "measured_cpu_non_solver_host_overhead_seconds": float(host_overhead.mean()),
        "measured_gpu2_peak_reserved_bytes": gpu["by_batch_size"]["32"]["peak_reserved_bytes"],
        "assumptions": ["GPU2 was shared with an unrelated 100%-utilization job; these timings do not predict exclusive-device speed", "GPU timing-only replay did not include raw write; the measured CPU non-solver host overhead is added as a proxy and may double-count some validation", "mean includes prospective 5% retry/time reserve", "conservative uses observed p90 case time plus 10% reserve", "formal raw storage counts retained per-case sources plus a measured second raw-shard copy", "QA time and QA bytes use the measured fixed pilot refinement set", "selected plot time uses measured initial page and quartet rendering rates", "formal shard pack uses a measured four-case pilot subset rate", "all formal rows are assumed native 96; approved 192/384 challenge groups require separate added cost", "M10/M16 and challenge strata lack matched throughput evidence; CPU family fallback and GPU aggregate rate are extrapolations", "temporary estimate covers simultaneous one 64-case raw and processed temporary shard using measured case/shard upper sizes"],
        "profiles": results,
    }
    write_json_atomic(root / "diagnostics" / "forecasts_cpu.json", summary)
    return summary
