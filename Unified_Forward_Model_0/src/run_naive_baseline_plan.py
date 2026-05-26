"""Run a small plan of external naive ChannelThermal baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from diagnostics import plot_ablation_summary
from train_naive_baseline import load_json, resolve_path, train_one_run
from train_unified import jsonable, write_json


def main() -> None:
    args = parse_args()
    base_config = load_json(args.config)
    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = resolve_path(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, list):
        raise ValueError("Naive baseline plan must be a list of run rows.")

    summaries: List[Dict[str, Any]] = []
    for row in plan:
        payload = merge_plan_row(base_config, row)
        if args.device is not None:
            payload.setdefault("training", {})["device"] = args.device
        summary = train_one_run(payload, run_name=str(row.get("name", "naive_baseline")), max_steps=args.max_steps)
        summaries.append({"name": str(row.get("name", "naive_baseline")), **summary})

    root = resolve_path(base_config.get("output", {}).get("saved_root", "results/naive_baselines"))
    root.mkdir(parents=True, exist_ok=True)
    write_summary_csv(root / "naive_baseline_summary.csv", summaries)
    write_json(root / "naive_baseline_summary.json", {"runs": summaries})
    plot_ablation_summary(summaries, root / "naive_baseline_summary.png", metric_key="best_val_field_mse_physical")
    print(json.dumps(jsonable({"runs": summaries}), indent=2, sort_keys=True))


def merge_plan_row(base_config: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.loads(json.dumps(base_config))
    payload.setdefault("model", {}).update(dict(row.get("model_overrides", {}) or {}))
    payload.setdefault("training", {}).update(dict(row.get("training_overrides", {}) or {}))
    return payload


def write_summary_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    rows = [flatten_scalars(row) for row in summaries]
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_scalars(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_scalars(value, prefix=f"{name}."))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/naive_baseline_channelthermal_template.json")
    parser.add_argument("--plan", default="configs/naive_baseline_plan.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
