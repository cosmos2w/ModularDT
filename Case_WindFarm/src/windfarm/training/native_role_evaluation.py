"""Physical five-role diagnostics, independent of the TRAIN objective.

These calculations preserve the qualified Round3 native-board definitions.
They neither fit role scales nor change the two-region training/monitor law.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from honf_runtime.unified_training import SamplingKey

from .content_hypergraph_task import ROLE_NAMES, WindContentHypergraphTask


def make_native_role_panel_batch(
    provider: WindContentHypergraphTask,
    row: int,
    *,
    split: str,
    batch_index: int = 0,
) -> Any:
    """Use the original fixed Q8192 five-role sampler for a permitted row.

    The TRAIN fixture's historical stage tag is only a query-stream reference;
    it does not instantiate a historical decoder or any selection machinery.
    """
    if not isinstance(provider, WindContentHypergraphTask):
        raise TypeError("Native role panels require a bound native Wind provider.")
    if split not in ("validation", "train", "train-fixture"):
        raise ValueError("Native role panels support validation or TRAIN fixtures only.")
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("Native panel batch_index must be a nonnegative integer.")
    validation = split == "validation"
    stage = "native_endpoint" if validation else "source_read_train_fixture"
    key = SamplingKey(
        int(provider.sampling_seed), 0, 0, batch_index, stage, "shared",
        sampling_version=provider.sampling_version, dataset_id=provider.sampling_dataset_id,
    )
    return WindContentHypergraphTask._make_batch(
        provider, (int(row),), key, validation=validation, stream_validation=validation,
        query_count=8192, track_samples=False,
    )


def _metric_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Role metric summaries require nonempty finite row values.")
    worst_i = int(np.argmax(array))
    return {
        "mean": float(np.mean(array, dtype=np.float64)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "worst": float(array[worst_i]),
        "worst_row_position": worst_i,
        "count": int(array.size),
    }


def physical_role_metrics(
    metadata: Mapping[str, Any],
    prediction_mps: np.ndarray,
    target_mps: np.ndarray,
    role_ids: np.ndarray,
) -> dict[str, Any]:
    """Casewise component and vector RMSE in m/s, without role scales."""
    prediction = np.asarray(prediction_mps, dtype=np.float64)
    target = np.asarray(target_mps, dtype=np.float64)
    roles = np.asarray(role_ids, dtype=np.int64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("Wind predictions and targets must align as [Q,3] physical m/s arrays.")
    if roles.shape != (prediction.shape[0],) or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Wind role IDs and finite prediction/target arrays must align as [Q].")
    result: dict[str, Any] = {**dict(metadata), "roles": {}}
    for role_index, role in enumerate(ROLE_NAMES):
        selected = roles == role_index
        if not selected.any():
            raise ValueError(f"Native panel has no queries for role {role!r}.")
        delta = prediction[selected] - target[selected]
        component_rmse = np.sqrt(np.mean(np.square(delta), axis=0, dtype=np.float64))
        vector_rmse = float(np.sqrt(np.mean(np.sum(np.square(delta), axis=-1), dtype=np.float64)))
        result["roles"][role] = {
            "component_rmse_mps": component_rmse.astype(float).tolist(),
            "vector_rmse_mps": vector_rmse,
            "query_count": int(selected.sum()),
        }
    return result


def _summarize_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"row_count": len(rows)}
    if not rows:
        return result
    roles = tuple(rows[0]["roles"])
    result["roles"] = {}
    for role in roles:
        vectors = np.asarray([row["roles"][role]["vector_rmse_mps"] for row in rows], dtype=np.float64)
        components = np.asarray([row["roles"][role]["component_rmse_mps"] for row in rows], dtype=np.float64)
        result["roles"][role] = {
            "vector_rmse_mps": _metric_summary(vectors),
            "component_rmse_mps": {
                component: _metric_summary(components[:, index]) for index, component in enumerate(("Ux", "Uy", "Uz"))
            },
        }
    return result


def summarize_physical_role_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep case tails and group correlated direction rows by layout."""
    if not rows:
        raise ValueError("A native role summary requires at least one case row.")
    layout_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    module_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        layout_groups[str(int(row["layout_index"]))].append(row)
        module_groups[str(int(row["n_turbines"]))].append(row)
    return {
        "physical_metric_scope": "raw m/s RMSE; no five-role training scales or objective weights",
        "overall": _summarize_group(rows),
        "by_layout_index": {key: _summarize_group(value) for key, value in sorted(layout_groups.items())},
        "by_active_module_count_M": {
            key: _summarize_group(value) for key, value in sorted(module_groups.items(), key=lambda item: int(item[0]))
        },
    }
