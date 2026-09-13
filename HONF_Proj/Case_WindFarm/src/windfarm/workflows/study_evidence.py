"""Post-training native, geometry and cost evidence through root evaluation."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from honf_runtime.run_store import atomic_write_json

from ..data import WindFarmNativeDataset, WindFarmNativeView, case_batch, collate_windfarm
from ..geometry import support_weights
from ..normalization import VerticalProfileBaseline
from ..study_cost import disposable_update, measure, synchronize
from ..study_diagnostics import geometry_diagnostics
from ..study_spatial import native_coordinates, native_plane, render_native_plane, stream_native_errors
from .evaluate_forward import _compact_metadata, _load_split, evaluate_rows, load_checkpoint
from .train_forward import _as_device_batch


def select_geometry_rows(view: WindFarmNativeView, rows: np.ndarray) -> list[int]:
    """Choose validation cases by support volume and count, without fields."""
    ordered = sorted((view.run(int(row)) for row in rows), key=lambda case: (case.support.volume_D3, case.case))
    chosen = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    if max(case.n_turbines for case in chosen) < 26:
        maximum = sorted(ordered, key=lambda case: (-case.n_turbines, case.case))[0]
        if maximum.index not in {case.index for case in chosen}:
            chosen.append(maximum)
    return [case.index for case in chosen]


def _predictor(model: Any, case: Any, prepared: Any, device: torch.device, chunk: int = 1024):
    @torch.no_grad()
    def predict(coords_D: np.ndarray) -> np.ndarray:
        query = case.geometry_for_queries(coords_D)
        coordinates = torch.from_numpy(query["query_xy"][None]).to(device)
        features = torch.from_numpy(query["query_features"][None]).to(device)
        result = model.predict_physical(prepared, coordinates, features, receiver_chunk_size=chunk)
        return result[0].cpu().numpy()
    return predict


def _vertical_profile(case: Any, predict: Any, baseline: Any, output: Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = case.run
    ix = int(np.argmin(abs(run.x_m)))
    iy = int(np.argmin(abs(run.y_m)))
    flat = ix + run.nx * (iy + run.ny * np.arange(run.nz))
    coordinates = native_coordinates(run, flat, 80.)
    target = np.array(run.U[flat], copy=True)
    prediction = predict(coordinates)
    background = baseline.predict(coordinates[:, 2])
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained", sharey=True)
    for channel, (ax, name) in enumerate(zip(axes, ("Ux", "Uy", "Uz"))):
        for values, label, style in ((target, "Reference", "-"), (prediction, "Prediction", "--"), (background, "Training profile", ":")):
            ax.plot(values[:, channel], coordinates[:, 2], style, label=label)
        ax.set(xlabel=f"{name} [m/s]", ylabel="z / D")
        ax.grid(alpha=.2)
    axes[-1].legend()
    fig.suptitle(f"{case.case}: native x={run.x_m[ix]:.3f} m, y={run.y_m[iy]:.3f} m")
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return {"x_m": float(run.x_m[ix]), "y_m": float(run.y_m[iy]), "z_D": coordinates[:, 2].tolist(),
            "target_mps": target.tolist(), "prediction_mps": prediction.tolist(), "background_mps": background.tolist(),
            "prediction_channel_span_mps": np.ptp(prediction, axis=0).tolist(),
            "reference_channel_span_mps": np.ptp(target, axis=0).tolist()}


def _routing_summary(model: Any, case: Any, prepared: Any, device: torch.device, figure: Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plane = native_plane(case.run, "z", 70.)
    chosen = np.linspace(0, len(plane["coords_D"]) - 1, 512, dtype=np.int64)
    coords = plane["coords_D"][chosen]
    batch = case_batch(case, coords).to(device)
    with torch.no_grad():
        output = model.decode(prepared, batch.query_xy, batch.query_features,
                              receiver_chunk_size=128, return_routing_maps=True)
    summary: dict[str, Any] = {"receiver_count": 512, "architecture": model.architecture,
                               "native_slice_z_m": plane["actual_m"]}
    fig = plt.figure(figsize=(15, 6), layout="constrained")
    spatial = fig.add_subplot(1, 2, 2, projection="3d")
    spatial.set(xlabel="x / D", ylabel="y / D", zlabel="z / D")
    spatial.set_box_aspect(case.support.extent_D)
    if model.architecture == "legacy_honf":
        for key in ("hyper_source_coords", "hyper_region_coords", "A_mh"):
            summary[key] = prepared.encoded[key].detach().cpu().tolist()
        attention = output.get("query_hyper_attention")
        if attention is not None:
            values = attention.detach()
            summary["mean_query_edge_attention"] = values.mean(dim=1).cpu().tolist()
            summary["query_coords_D"] = coords.tolist()
            summary["dominant_edge"] = values.argmax(dim=-1).cpu().tolist()
            ax = fig.add_subplot(1, 2, 1)
            ax.scatter(coords[:, 0], coords[:, 1], c=values.argmax(dim=-1)[0].cpu(), cmap="tab10", s=10, vmin=0, vmax=9)
            ax.scatter(case.module_centers[:, 0], case.module_centers[:, 1], marker="x", c="black", s=22)
            ax.set(xlabel="x / D", ylabel="y / D", title=f"Dominant latent edge on z={plane['actual_m']:.2f} m slice")
            ax.set_aspect("equal", adjustable="box")
            source = prepared.encoded["hyper_source_coords"][0].detach().cpu().numpy()
            region = prepared.encoded["hyper_region_coords"][0].detach().cpu().numpy()
            for edge, (a, b) in enumerate(zip(source, region)):
                color = plt.get_cmap("tab10")(edge)
                spatial.scatter(*a, c=[color], marker="o", s=55)
                spatial.scatter(*b, c=[color], marker="^", s=55)
                spatial.plot(*np.stack((a, b)).T, c=color, alpha=.7)
                ax.scatter([], [], c=[color], s=16, label=f"Edge {edge}")
            ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=6, fontsize=8)
            spatial.set_title("3-D source (circle) / region (triangle) centroids", fontsize=11)
    else:
        attention = output.get("dense_environment_attention")
        if attention is not None:
            summary["mean_environment_attention"] = attention.detach().mean(dim=(1, 2)).cpu().tolist()
            mean_attention = attention.detach().mean(dim=(1, 2))[0].cpu().numpy()
            env = case.env_coords
            artist = spatial.scatter(env[:, 0], env[:, 1], env[:, 2], c=mean_attention, s=12, cmap="viridis")
            fig.colorbar(artist, ax=spatial, shrink=.6, pad=.1, orientation="horizontal",
                         label="Mean environment attention")
            spatial.set_title("Environment attention averaged over heads/receivers", fontsize=11)
        for key in ("dense_module_context_norm", "main_context_norm", "coarse_context_norm", "local_context_norm"):
            if key in output:
                summary[key] = float(output[key].detach().mean())
        ax = fig.add_subplot(1, 2, 1)
        names = [key for key in ("dense_module_context_norm", "main_context_norm", "coarse_context_norm", "local_context_norm") if key in summary]
        ax.barh(names, [summary[key] for key in names])
        ax.set(xlabel="Mean context norm", title="Executed dense contribution summaries")
    spatial.scatter(*case.module_centers.T, c="black", marker="x", s=12)
    fig.suptitle(f"{case.case}: bounded routing diagnostic; no physical edge labels", fontsize=12)
    fig.savefig(figure, dpi=150, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)
    summary["figure"] = str(figure)
    return summary


def _native(model: Any, view: Any, rows: list[int], normalizer: Any, baseline: Any,
            device: torch.device, output: Path) -> dict[str, Any]:
    results = []
    for row in rows:
        case = view.run(row)
        initial = case_batch(case, case.module_centers).to(device)
        synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            prepared = model.prepare_case(initial)
        synchronize(device)
        prepare_seconds = time.perf_counter() - start
        predict = _predictor(model, case, prepared, device)
        weights = tuple(axis / 80. for axis in support_weights(case.x_m, case.y_m, case.z_m))
        start = time.perf_counter()
        metrics = stream_native_errors(case.run, predict, baseline.predict, weights, case.module_centers,
                                       normalizer.safe_std * normalizer.u_ref_mps)
        elapsed = time.perf_counter() - start
        case_dir = output / case.case
        case_dir.mkdir(parents=True, exist_ok=True)
        planes = []
        for axis, coordinate in (("z", 70.), ("y", 0.), ("x", 0.)):
            plane = native_plane(case.run, axis, coordinate)
            prediction = predict(plane["coords_D"])
            figure = case_dir / f"native_{axis}_slice.png"
            render_native_plane(plane, prediction, case.module_centers,
                                f"{case.case}, M={case.n_turbines}, direction={case.wind_direction_deg:g}°", figure)
            planes.append({"fixed_axis": axis, "requested_m": coordinate, "actual_m": plane["actual_m"], "figure": str(figure)})
        record = {"row": row, "case": case.case, "shape_nxyz": list(case.shape_nxyz),
                  "support_lower_D": case.support.lower_D.tolist(), "support_upper_D": case.support.upper_D.tolist(),
                  "n_turbines": case.n_turbines, "native_cells": case.run.cell_count,
                  "prepare_seconds": prepare_seconds, "stream_metrics_seconds": elapsed,
                  "metrics": metrics, "planes": planes,
                  "vertical_profile": _vertical_profile(case, predict, baseline, case_dir / "vertical_profile.png"),
                  "routing": _routing_summary(model, case, prepared, device, case_dir / "routing.png")}
        atomic_write_json(case_dir / "native_evidence.json", record)
        results.append(record)
        print(f"[windfarm native] {case.case}: {case.run.cell_count} cells, {elapsed:.1f}s", flush=True)
        del prepared, predict, initial
    return {"cases": results, "measure": "native centre-support quadrature; D3 weights", "prediction_volume_saved": False}


def _timing(model: Any, view: Any, rows: list[int], normalizer: Any,
            training_rows: np.ndarray, device: torch.device) -> dict[str, Any]:
    results = []
    model.eval()
    for row in rows[:3]:
        case = view.run(row)
        record: dict[str, Any] = {"row": row, "case": case.case}
        with torch.no_grad():
            for count in (8192, 65536):
                def sample_batch():
                    sampled_case = view.run(row)
                    sample = sampled_case.sample_queries(count, np.random.default_rng(np.random.SeedSequence([42, row, count])))
                    return case_batch(sampled_case, sample.coords_D)
                record[f"cpu_sampling_{count}"] = measure(sample_batch, torch.device("cpu"))
                cpu_batch = sample_batch()
                record[f"transfer_{count}"] = measure(lambda: cpu_batch.to(device), device)
                batch = cpu_batch.to(device)
                record[f"prepare_{count}"] = measure(lambda: model.prepare_case(batch), device)
                prepared = model.prepare_case(batch)
                record[f"prepared_read_{count}"] = measure(
                    lambda: model.decode(prepared, batch.query_xy, batch.query_features, receiver_chunk_size=1024),
                    device, warmups=3 if count == 8192 else 1, repeats=10 if count == 8192 else 3)
                del prepared, batch, cpu_batch
            if row in (rows[0], rows[2]):
                batch = case_batch(case, case.module_centers).to(device)
                def native_inference():
                    prepared = model.prepare_case(batch)
                    predict = _predictor(model, case, prepared, device)
                    for start in range(0, case.run.cell_count, 8192):
                        flat = np.arange(start, min(start + 8192, case.run.cell_count))
                        predict(native_coordinates(case.run, flat, 80.))
                record["complete_native_inference"] = measure(native_inference, device, warmups=0, repeats=1)
        results.append(record)
    dataset = WindFarmNativeDataset(view, training_rows[:8], normalizer=normalizer, queries_per_case=1024,
                                   seed=42, fixed_sampling=True)
    batch = _as_device_batch(collate_windfarm([dataset[i] for i in range(8)]), device)
    step = disposable_update(model, batch)
    return {"cases": results, "training_step": step,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "native_inference_scope": "prepare once plus all native coordinates, transfer, decode and CPU output transfer; outputs discarded",
            "routing_maps": False, "inference_receiver_chunk": 1024}


def run_study(*, checkpoint: str | Path, volume_path: str | Path, derived_view: str | Path,
              device: str | torch.device, output_dir: str | Path, mode: str,
              compact_path: str | Path | None = None) -> int:
    """Execute one requested evidence stage using an existing checkpoint."""
    device = torch.device(device)
    view = WindFarmNativeView(
        volume_path,
        compact_metadata=None if compact_path is None else _compact_metadata(compact_path),
    )
    split = _load_split(view, Path(derived_view))
    rows = select_geometry_rows(view, split.validation)
    materialization_case = view.run(int(split.train[0]))
    batch = case_batch(materialization_case, materialization_case.module_centers)
    model, payload = load_checkpoint(checkpoint, device=device, materialization_batch=batch)
    normalizer = model.velocity_transform
    baseline = VerticalProfileBaseline.from_dict(payload["vertical_profile_baseline"])
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if mode == "native":
        result = _native(model, view, rows, normalizer, baseline, device, output)
    elif mode == "diagnostics":
        batches = []
        for row in rows[:3]:
            case = view.run(row)
            sample = case.sample_queries(512, np.random.default_rng(np.random.SeedSequence([42, row, 802])))
            batches.append(case_batch(case, sample.coords_D, normalizer=normalizer,
                                      velocity_mps=sample.velocity_mps).to(device))
        result = {"geometry": geometry_diagnostics(model, batches),
                  "independent_sample": evaluate_rows(model, view, rows[:3], normalizer=normalizer,
                                                      q_volume=32768, q_band=8192, seed=314159,
                                                      baseline=baseline, receiver_chunk_size=1024)}
    elif mode == "timing":
        result = _timing(model, view, rows, normalizer, split.train, device)
    else:
        raise ValueError(f"Unknown WindFarm evidence mode: {mode}")
    result.update({"checkpoint": str(Path(checkpoint).resolve()), "epoch": int(payload["epoch"]),
                   "mode": mode, "selected_validation_rows": rows, "device": str(device)})
    atomic_write_json(output / f"{mode}_evidence.json", result)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0
