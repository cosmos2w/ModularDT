"""Audit the Run-1501 query rule on frozen Run-1409 source organization.

The command loads an explicit occupancy-adaptive Run-1409 checkpoint and
keeps its prepared Dense/source/group state fixed.  It decodes the same query
subset twice: first with the checkpoint's native query route, then with only
the proposed prototype-anchored RMS keys, current-centre geometry bias, and
masked sparsemax.  No model parameter, checkpoint, or managed run is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _proposed_route(
    router: Any,
    encoded: Any,
    state: Any,
    receivers: Any,
    receiver_features: Any = None,
    **_: Any,
) -> Any:
    """Run the proposed query rule against an already prepared source state."""

    import torch
    from honf_forward_core.interface_fields.group_control_router import GroupQueryRoute
    from honf_forward_core.interface_fields.routing_index.sparse_projection import (
        masked_sparsemax,
    )

    if receiver_features is None:
        query_features = router.query_fourier(receivers / router._scale(encoded))
    else:
        expected_width = int(
            (router.spatial_dim if router.query_fourier.include_input else 0)
            + 2 * router.spatial_dim * router.query_fourier.num_frequencies
        )
        query_features = (
            receiver_features
            if int(receiver_features.shape[-1]) == expected_width
            else router.query_fourier(receivers / router._scale(encoded))
        )
    query_input = torch.cat(
        [
            query_features,
            state.global_control[:, None, :].expand(-1, receivers.shape[1], -1),
        ],
        dim=-1,
    )
    query_control = router.query_projection(query_input)
    prototypes = router.group_codes.to(
        device=state.group_control.device, dtype=state.group_control.dtype
    )
    keys = prototypes[None, :, :] + router.query_group_projection(
        state.group_control
    )

    def rms(values: Any) -> Any:
        return values / torch.sqrt(
            values.square().mean(dim=-1, keepdim=True) + 1.0e-6
        )

    scaled_query = rms(query_control)
    scaled_keys = rms(keys)
    logits = torch.einsum("bqd,bkd->bqk", scaled_query, scaled_keys) / math.sqrt(
        float(router.control_dim)
    )
    module_has_mass = state.module_mass > 0.0
    environment_has_mass = state.environment_mass > 0.0
    module_centres = torch.where(
        module_has_mass[..., None],
        state.module_centres,
        state.environment_centres,
    )
    environment_centres = torch.where(
        environment_has_mass[..., None],
        state.environment_centres,
        state.module_centres,
    )
    geometry = -0.5 * (
        torch.linalg.vector_norm(
            receivers[:, :, None, :] - module_centres[:, None, :, :], dim=-1
        )
        + torch.linalg.vector_norm(
            receivers[:, :, None, :] - environment_centres[:, None, :, :],
            dim=-1,
        )
    ) / router._geometry_scale(encoded)[:, None, None]
    logits = logits + geometry
    valid = (state.packed_valid & state.phase_occupied)[:, None, :].expand_as(
        logits
    )
    assignment = masked_sparsemax(logits, valid)
    return GroupQueryRoute(
        query_control=query_control,
        assignment=assignment,
        logits=logits,
        query_keys=scaled_keys,
    )


def _support(query: np.ndarray, source: np.ndarray, active: np.ndarray) -> dict[str, float]:
    q = query > 0.0
    s = source > 0.0
    logical = int(np.sum(q[:, None, :] & s[None, :, :]))
    unique = int(np.sum(np.any(q[:, None, :] & s[None, :, :], axis=-1)))
    dense = int(query.shape[0] * np.sum(active))
    return {
        "logical_paths": logical,
        "unique_pairs": unique,
        "dense_pairs": dense,
        "support_ratio": float(unique / dense) if dense else 0.0,
        "logical_multiplicity": float(logical / unique) if unique else 0.0,
    }


def _render(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histogram: Counter[int] = Counter()
    for row in rows:
        histogram.update({int(key): int(value) for key, value in row["kq_histogram"].items()})
    degrees = sorted(histogram)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].bar(degrees, [histogram[value] for value in degrees], color="#35618f")
    axes[0, 0].set(title="Frozen Run 1409 → proposed Run 1501 query Kq", xlabel="positive groups", ylabel="queries")
    axes[0, 1].scatter(
        [row["module_support_ratio"] for row in rows],
        [row["environment_support_ratio"] for row in rows],
        c=[row["kq_mean"] for row in rows],
        cmap="viridis",
        s=24,
    )
    axes[0, 1].axhline(0.9, color="crimson", linestyle="--", linewidth=1)
    axes[0, 1].set(title="Unique physical support", xlabel="R_M support", ylabel="R_E support")
    axes[1, 0].hist([row["prediction_relative_rms_change"] for row in rows], bins=18, color="#cf7c34")
    axes[1, 0].set(title="Prediction change from query rule only", xlabel="relative RMS change", ylabel="cases")
    axes[1, 1].scatter(
        [row["kq_mean"] for row in rows],
        [row["environment_support_ratio"] for row in rows],
        color="#2f7d63",
        s=24,
    )
    axes[1, 1].set(title="Query degree vs environment support", xlabel="mean Kq", ylabel="R_E support")
    figure.suptitle("Run 1501 frozen query-rule diagnostic (source organization fixed)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "Case_ThermalChannel" / "src",
        PROJECT_ROOT / "tools" / "diagnostics",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.evaluation.loading import load_model
    from channelthermal.evaluation.prepared import predict_case
    from run_run1409_occupancy_population import (
        _dataset_for_checkpoint,
        _query_sample,
    )

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = torch.device("cpu")
    model, checkpoint = load_model(checkpoint_path, device)
    architecture = str(model.config.core_honf.forward_architecture)
    if architecture != "occupancy_adaptive_group_control_honf":
        raise RuntimeError(
            f"frozen diagnostic requires occupancy Run 1409, got {architecture!r}"
        )
    dataset, dataset_path, _ = _dataset_for_checkpoint(
        checkpoint,
        dataset_path=args.dataset,
        split=args.split,
        GlobalChannelThermalDataset=GlobalChannelThermalDataset,
        H5Normalizer=H5Normalizer,
    )
    available = [str(value) for value in dataset.selected_case_ids]
    case_ids = [str(value) for value in args.case_id] if args.case_id else available
    if int(args.expected_cases) > 0 and len(case_ids) != int(args.expected_cases):
        raise RuntimeError(
            f"expected {args.expected_cases} cases, selected {len(case_ids)}"
        )
    index_by_case = {case_id: index for index, case_id in enumerate(available)}
    rows: list[dict[str, Any]] = []
    pooled_delta_square = 0.0
    pooled_parent_square = 0.0
    router = model.core.backend.router
    original_route = router.route_queries
    for order, case_id in enumerate(case_ids):
        sample = dataset[index_by_case[case_id]]
        selected, query_xy = _query_sample(sample, int(args.query_count))
        parent = predict_case(
            model,
            selected,
            device,
            query_batch_size=int(args.query_batch_size),
            local_port_condition_mode="predicted",
            mixed_teacher_ratio=0.0,
            return_routing_maps=True,
            return_prepared_state=True,
        )
        prepared_case = parent["_prepared_state"]
        prepared = prepared_case.prepared
        controls = prepared.backend_state["group_control_state"]
        query_tensor = torch.from_numpy(query_xy).unsqueeze(0)
        try:
            router.route_queries = MethodType(_proposed_route, router)
            with torch.no_grad():
                proposed = model.decode_prepared(
                    prepared_case,
                    query_tensor,
                    return_routing_maps=True,
                    receiver_chunk_size=int(args.query_batch_size),
                )
        finally:
            router.route_queries = original_route
        alpha = proposed["group_control_query_routing"].detach().cpu().numpy()[0]
        module_assignment = controls.module_membership.detach().cpu().numpy()[0]
        environment_assignment = controls.environment_membership.detach().cpu().numpy()[0]
        module_active = controls.module_measure.detach().cpu().numpy()[0] > 0.0
        environment_active = controls.environment_measure.detach().cpu().numpy()[0] > 0.0
        module_support = _support(alpha, module_assignment, module_active)
        environment_support = _support(alpha, environment_assignment, environment_active)
        kq = np.sum(alpha > 0.0, axis=-1)
        effective = 1.0 / np.maximum(np.sum(np.square(alpha), axis=-1), 1.0e-12)
        parent_field = np.asarray(parent["pred_field_grid"], dtype=np.float64).reshape(
            -1, int(model.config.field_dim)
        )
        proposed_field = proposed["pred_field"].detach().cpu().numpy()[0].astype(
            np.float64
        )
        delta_square = float(np.sum(np.square(proposed_field - parent_field)))
        parent_square = float(np.sum(np.square(parent_field)))
        pooled_delta_square += delta_square
        pooled_parent_square += parent_square
        row = {
            "case_id": case_id,
            "query_count": int(alpha.shape[0]),
            "kq_mean": float(np.mean(kq)),
            "kq_median": float(np.median(kq)),
            "kq_p95": float(np.quantile(kq, 0.95)),
            "kq_min": int(np.min(kq)),
            "kq_max": int(np.max(kq)),
            "effective_query_groups_mean": float(np.mean(effective)),
            "module_support_ratio": module_support["support_ratio"],
            "environment_support_ratio": environment_support["support_ratio"],
            "module_logical_paths": int(module_support["logical_paths"]),
            "module_unique_pairs": int(module_support["unique_pairs"]),
            "module_dense_pairs": int(module_support["dense_pairs"]),
            "environment_logical_paths": int(environment_support["logical_paths"]),
            "environment_unique_pairs": int(environment_support["unique_pairs"]),
            "environment_dense_pairs": int(environment_support["dense_pairs"]),
            "prediction_relative_rms_change": float(
                math.sqrt(delta_square / max(parent_square, 1.0e-30))
            ),
            "kq_histogram": {
                str(key): int(value)
                for key, value in sorted(Counter(kq.tolist()).items())
            },
        }
        rows.append(row)
        print(
            f"[run1501-frozen-query] {order + 1}/{len(case_ids)} case={case_id} "
            f"Kq={row['kq_mean']:.3f} RE={row['environment_support_ratio']:.3f}",
            flush=True,
        )
        del parent, proposed, prepared_case, prepared, controls

    histogram: Counter[int] = Counter()
    for row in rows:
        histogram.update(
            {int(key): int(value) for key, value in row["kq_histogram"].items()}
        )
    total_queries = int(sum(row["query_count"] for row in rows))
    summary = {
        "schema_version": 1,
        "task": "run1501_frozen_query_routing_diagnostic",
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": "explicit latest K=12 occupancy Run 1409 epoch 50",
        "architecture": architecture,
        "dataset": str(dataset_path),
        "split": str(args.split),
        "case_count": len(rows),
        "query_count_per_case": int(args.query_count),
        "total_recorded_queries": total_queries,
        "source_organization": "frozen prepared Run 1409 P2 state",
        "changed_rule_only": "prototype-anchored RMS keys + current-centre bias + masked sparsemax",
        "temperature_sweep": False,
        "managed_run_allocated": False,
        "checkpoint_written": False,
        "kq": _summary([row["kq_mean"] for row in rows]),
        "effective_query_groups": _summary(
            [row["effective_query_groups_mean"] for row in rows]
        ),
        "kq_histogram": {
            str(key): int(value) for key, value in sorted(histogram.items())
        },
        "kq_fraction": {
            str(key): float(value / total_queries)
            for key, value in sorted(histogram.items())
        },
        "module_support_ratio": _summary(
            [row["module_support_ratio"] for row in rows]
        ),
        "environment_support_ratio": _summary(
            [row["environment_support_ratio"] for row in rows]
        ),
        "prediction_relative_rms_change": _summary(
            [row["prediction_relative_rms_change"] for row in rows]
        ),
        "prediction_pooled_relative_rms_change": float(
            math.sqrt(pooled_delta_square / max(pooled_parent_square, 1.0e-30))
        ),
        "source_overlap_bottleneck_flag": bool(
            np.mean([row["environment_support_ratio"] for row in rows]) >= 0.9
        ),
        "interpretation_limit": (
            "Frozen-query prediction change is a controlled route perturbation, not retrained accuracy. "
            "Logical support is not executed-row or speed evidence."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "cases.json", rows)
    fields = [key for key in rows[0] if key != "kq_histogram"]
    with (output_dir / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    _render(rows, output_dir / "frozen_query_diagnostic.png")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--expected-cases", type=int, default=90)
    parser.add_argument("--query-count", type=int, default=1024)
    parser.add_argument("--query-batch-size", type=int, default=1024)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(build_parser().parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run"]
