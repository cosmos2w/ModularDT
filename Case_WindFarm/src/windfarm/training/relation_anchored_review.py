"""Native Wind field review and fixed-support surrogate layout diagnostics."""
from __future__ import annotations

import numpy as np
import torch

from .native_role_evaluation import physical_role_metrics, summarize_physical_role_metrics


def review_native_batches(model, provider, batches, *, height_edges_D):
    """Summarize supplied TRAIN/DEV native batches, with sampled vertical profiles.

    Height bins summarize the supplied role-query mixture, not a plane integral.
    The caller chooses and labels the split, checkpoint, and fixed panel.
    """
    edges = np.asarray(height_edges_D, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("Height edges must be strictly increasing.")
    records = []
    count = np.zeros(len(edges)-1, dtype=np.int64)
    pred_sum = np.zeros((len(count), 3), dtype=np.float64)
    true_sum = np.zeros_like(pred_sum)
    squared_error = np.zeros_like(pred_sum)
    outside_height_bins = 0
    work = {}
    with torch.no_grad():
        for batch in batches:
            pred, auxiliary = provider.predict_native(model, provider.make_scene(batch.scene_inputs),
                batch.receivers, execution_mode="relation_anchored", phase="validation")
            for name, value in provider.work_counts(batch, pred, auxiliary).items():
                work[name] = work.get(name, 0) + value
            prediction = pred.main_mps.detach().cpu().numpy().astype(np.float64)
            target = np.asarray(batch.targets.velocity_mps, dtype=np.float64)
            coords = np.asarray(batch.receivers.coordinates_D, dtype=np.float64)
            roles = np.asarray(batch.receivers.role_ids)
            for i, metadata in enumerate(batch.targets.case_metadata):
                record = physical_role_metrics(metadata, prediction[i], target[i], roles[i])
                error = prediction[i]-target[i]
                record["roles"]["all_sampled_queries"] = {
                    "component_rmse_mps": np.sqrt(np.square(error).mean(axis=0)).tolist(),
                    "vector_rmse_mps": float(np.sqrt(np.square(error).sum(axis=-1).mean())),
                    "query_count": int(len(error))}
                records.append(record)
                z = coords[i,:,2]
                bins = np.searchsorted(edges, z, side="right") - 1
                bins[z == edges[-1]] = len(count)-1
                outside_height_bins += int(((bins < 0) | (bins >= len(count))).sum())
                for j in range(len(count)):
                    selected = bins == j
                    if selected.any():
                        count[j] += int(selected.sum())
                        pred_sum[j] += prediction[i,selected].sum(axis=0)
                        true_sum[j] += target[i,selected].sum(axis=0)
                        squared_error[j] += np.square(prediction[i,selected]-target[i,selected]).sum(axis=0)
    if not records:
        raise ValueError("A Wind native review needs at least one batch.")
    profiles = []
    for j, n in enumerate(count):
        profiles.append({"height_interval_D": [float(edges[j]), float(edges[j+1])], "query_count": int(n),
            "prediction_mean_mps": (pred_sum[j]/n).tolist() if n else None,
            "reference_mean_mps": (true_sum[j]/n).tolist() if n else None,
            "component_rmse_mps": np.sqrt(squared_error[j]/n).tolist() if n else None})
    return {"physical_metrics": summarize_physical_role_metrics(records), "rows": records,
            "vertical_profile": {"scope": "sampled native role-query mixture", "bins": profiles,
                                 "outside_height_bins": outside_height_bins},
            "work_counts": work}


def layout_sensitivity(model, case, coordinates, horizontal_direction, *, steps=(1e-2, 3e-3, 1e-3)):
    """Compare decoded velocity AD and central differences for fixed-support layout motion."""
    parameter = next(model.parameters())
    coords = torch.as_tensor(coordinates, dtype=parameter.dtype, device=parameter.device)
    centers = torch.as_tensor(np.array(case.module_centers, copy=True), dtype=parameter.dtype,
                              device=parameter.device).requires_grad_(True)
    direction = torch.as_tensor(horizontal_direction, dtype=parameter.dtype, device=parameter.device)
    if direction.shape != centers.shape or not torch.all(direction[:, 2] == 0):
        raise ValueError("Layout direction must match module centers and have zero vertical component.")
    def decoded(layout):
        return model(case, coords, layout)[0]
    value, tangent = torch.autograd.functional.jvp(decoded, centers, direction)
    comparisons = []
    for step in steps:
        h = float(step)
        if h <= 0:
            raise ValueError("Finite-difference steps must be positive.")
        with torch.no_grad():
            fd = (decoded(centers + h*direction) - decoded(centers - h*direction))/(2*h)
        diff = fd-tangent
        comparisons.append({"step_D": h, "ad_rms_mps_per_D": float(tangent.square().mean().sqrt()),
            "fd_rms_mps_per_D": float(fd.square().mean().sqrt()),
            "difference_rms_mps_per_D": float(diff.square().mean().sqrt()),
            "max_absolute_difference_mps_per_D": float(diff.abs().max())})
    return {"scope": "trained-surrogate sensitivity on fixed support; no physical perturbation reference",
            "query_count": int(value.shape[1]), "finite_difference": comparisons}


def cover_boundary_response(model, case, coordinates_minus, coordinates_plus):
    """Measure decoded velocity change at caller-selected paired cover-boundary probes."""
    parameter = next(model.parameters())
    left = torch.as_tensor(coordinates_minus, dtype=parameter.dtype, device=parameter.device)
    right = torch.as_tensor(coordinates_plus, dtype=parameter.dtype, device=parameter.device)
    if left.shape != right.shape or left.ndim != 2 or left.shape[-1] != 3:
        raise ValueError("Boundary probe coordinates must be paired [Q,3] arrays.")
    with torch.no_grad():
        a = model(case, left)[0]; b = model(case, right)[0]
    difference = b-a
    return {"scope": "complete decoded trained-surrogate output; paired fixed-support probes",
            "query_count": int(left.shape[0]), "coordinate_separation_D": (right-left).norm(dim=-1).cpu().tolist(),
            "velocity_change_mps": difference.squeeze(0).cpu().tolist(),
            "max_absolute_velocity_change_mps": float(difference.abs().max())}
