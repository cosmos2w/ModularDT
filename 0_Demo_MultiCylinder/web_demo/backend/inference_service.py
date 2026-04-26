from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from cache import find_cached_job, new_job_id, register_cache, request_hash
from design_validation import model_limits, validate_design
from deterministic_service import DeterministicService
from generative_service import GenerativeService, GenerativeUnavailableError
from hypergraph_service import build_hypergraph
from kpi_service import compute_kpis
from model_registry import ModelRegistry
from render_service import render_frames
from schemas import DesignRequest
from settings import settings


def _model_dump(model) -> Dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _request_with_phase_bins(request: DesignRequest, phase_bins: int) -> DesignRequest:
    if int(request.phase_bins) == int(phase_bins):
        return request
    if hasattr(request, "model_copy"):
        return request.model_copy(update={"phase_bins": int(phase_bins)})
    return request.copy(update={"phase_bins": int(phase_bins)})


class InferenceService:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.det_service = DeterministicService()
        self.gen_service = GenerativeService()

    def _build_structure(self, request: DesignRequest, model_cfg: Dict, device):
        import torch

        max_num = int(model_cfg["max_num_cylinders"])
        centers = torch.zeros((1, max_num, 2), dtype=torch.float32, device=device)
        mask = torch.zeros((1, max_num), dtype=torch.float32, device=device)
        for idx, cyl in enumerate(request.cylinders[:max_num]):
            centers[0, idx, 0] = float(cyl.x)
            centers[0, idx, 1] = float(cyl.y)
            mask[0, idx] = 1.0
        return {
            "re_values": torch.tensor([[float(request.re)]], dtype=torch.float32, device=device),
            "num_cylinders": torch.tensor([[float(len(request.cylinders))]], dtype=torch.float32, device=device),
            "centers": centers,
            "cyl_mask": mask,
        }

    def _make_grid(self, model_cfg: Dict, nx: int, ny: int, device):
        import torch

        lx = float(model_cfg["domain_length_x"])
        ly = float(model_cfg["domain_length_y"])
        x = (torch.arange(nx, dtype=torch.float32, device=device) + 0.5) / float(nx) * lx
        y = (torch.arange(ny, dtype=torch.float32, device=device) + 0.5) / float(ny) * ly
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return xx, yy

    def _generative_initial_latent(self, artifact, tau_value: float, num_samples: int, seed: int | None, noise_mode: str, dtype):
        import math
        import torch

        flow = artifact.flow
        latent_h = flow.ae.H_pad // (2 ** flow.ae.n_levels)
        latent_w = flow.ae.W_pad // (2 ** flow.ae.n_levels)
        shape_one = (1, flow.ae.latent_ch, latent_h, latent_w)
        base_seed = 1234 if seed is None else int(seed)
        mode = str(noise_mode or "independent").lower()
        if mode == "independent":
            return None

        latents = []
        for sample_idx in range(int(num_samples)):
            gen_a = torch.Generator(device=artifact.device).manual_seed(base_seed + 100003 * sample_idx)
            a = torch.randn(shape_one, generator=gen_a, device=artifact.device, dtype=dtype)
            if mode == "harmonic":
                gen_b = torch.Generator(device=artifact.device).manual_seed(base_seed + 100003 * sample_idx + 7919)
                b = torch.randn(shape_one, generator=gen_b, device=artifact.device, dtype=dtype)
                angle = 2.0 * math.pi * float(tau_value)
                z = math.cos(angle) * a + math.sin(angle) * b
            elif mode == "shared":
                z = a
            else:
                return None
            latents.append(z)
        return torch.cat(latents, dim=0)

    def _infer_generative(self, request: DesignRequest, entry, req_hash: str, validation) -> Dict:
        import torch
        from model_gen import build_dense_condition_grid, build_global_condition_vector, denormalize_grid

        artifact = self.gen_service.load(entry)
        model_cfg = dict(artifact.deterministic_model_config)
        limits = model_limits({"model": model_cfg})
        model_cfg.setdefault("max_num_cylinders", limits["max_num_cylinders"])
        model_cfg.setdefault("domain_length_x", limits["domain_length_x"])
        model_cfg.setdefault("domain_length_y", limits["domain_length_y"])
        model_cfg.setdefault("re_scale", limits["re_scale"])

        validation_config = entry.load_config_json()
        validation_config = dict(validation_config)
        validation_config["model"] = model_cfg
        model_validation = validate_design(request, validation_config, entry.raw)
        if not model_validation.valid:
            raise ValueError("Invalid design: " + "; ".join(model_validation.warnings))

        effective_request = _request_with_phase_bins(request, validation.effective_phase_bins)

        job_id = new_job_id()
        job_dir = settings.cache_dir / job_id
        frames_dir = job_dir / "frames"
        job_dir.mkdir(parents=True, exist_ok=True)

        structure = self._build_structure(effective_request, model_cfg, artifact.device)
        x_grid, y_grid = self._make_grid(model_cfg, effective_request.resolution_nx, effective_request.resolution_ny, artifact.device)
        query_batch_size = int(artifact.model_config.get("generation", {}).get("det_query_batch_size", 32768))
        include_field = bool(artifact.model_config.get("stage2", {}).get("conditioning", {}).get("include_pred_field", True))
        n_samples = int(request.generative.num_samples)
        n_steps = int(request.generative.n_steps)
        ode_solver = str(artifact.model_config.get("stage2", {}).get("sampling", {}).get("ode_solver", "euler"))

        generated_cycle: List[np.ndarray] = []
        det_field_cycle: List[np.ndarray] = []
        det_mean_cycle: List[np.ndarray] = []
        det_residual_cycle: List[np.ndarray] = []
        aux = None

        ema_context = artifact.ema.average_parameters(artifact.flow.velocity_net) if artifact.ema is not None else torch.no_grad()
        with torch.no_grad(), ema_context:
            for phase_idx in range(effective_request.phase_bins):
                tau_value = (phase_idx + 0.5) / float(effective_request.phase_bins)
                det_out = artifact.deterministic_model.reconstruct_full_grid(
                    structure,
                    x_grid,
                    y_grid,
                    torch.tensor(tau_value, dtype=torch.float32, device=artifact.device),
                    query_batch_size=query_batch_size,
                )
                det_field = det_out["pred_field"].permute(0, 3, 1, 2).contiguous()
                det_mean = det_out["pred_mean"].permute(0, 3, 1, 2).contiguous()
                det_residual = det_out["pred_residual"].permute(0, 3, 1, 2).contiguous()
                tau = torch.tensor([[tau_value]], dtype=torch.float32, device=artifact.device)
                x_b = x_grid.unsqueeze(0)
                y_b = y_grid.unsqueeze(0)
                cond_grid = build_dense_condition_grid(
                    det_mean=det_mean,
                    det_residual=det_residual,
                    det_field=det_field,
                    x_grid=x_b,
                    y_grid=y_b,
                    tau=tau,
                    re_values=structure["re_values"],
                    stats=artifact.stats.to(artifact.device, dtype=det_mean.dtype),
                    domain_length_x=float(model_cfg["domain_length_x"]),
                    domain_length_y=float(model_cfg["domain_length_y"]),
                    re_scale=float(model_cfg.get("re_scale", 200.0)),
                    include_field=include_field,
                )
                global_cond = build_global_condition_vector(det_out, structure)
                cond_grid = cond_grid.repeat(n_samples, 1, 1, 1)
                global_cond = global_cond.repeat(n_samples, 1)
                det_mean_rep = det_mean.repeat(n_samples, 1, 1, 1)
                initial_latent = self._generative_initial_latent(
                    artifact,
                    tau_value,
                    n_samples,
                    effective_request.generative.seed,
                    effective_request.generative.noise_mode,
                    cond_grid.dtype,
                )
                seed = None if initial_latent is not None else (
                    None if effective_request.generative.seed is None else int(effective_request.generative.seed) + phase_idx * 100003
                )
                gen_res_norm = artifact.flow.sample(
                    cond_grid,
                    global_cond,
                    n_steps=n_steps,
                    ode_solver=ode_solver,
                    seed=seed,
                    initial_latent=initial_latent,
                )
                gen_res = denormalize_grid(gen_res_norm, artifact.stats.to(artifact.device, dtype=gen_res_norm.dtype))
                generated_cycle.append((det_mean_rep + gen_res).detach().cpu().numpy().astype(np.float32))
                det_field_cycle.append(det_field[0].detach().cpu().numpy().astype(np.float32))
                det_mean_cycle.append(det_mean[0].detach().cpu().numpy().astype(np.float32))
                det_residual_cycle.append(det_residual[0].detach().cpu().numpy().astype(np.float32))
                if aux is None:
                    aux = {key: value for key, value in det_out.items() if key not in {"pred_field", "pred_mean", "pred_residual"}}

        generated_samples = np.stack(generated_cycle, axis=1)
        field_arr = generated_samples.mean(axis=0)
        sample_std_arr = generated_samples.std(axis=0)
        det_field_arr = np.stack(det_field_cycle, axis=0)
        mean_arr = np.stack(det_mean_cycle, axis=0)
        residual_arr = np.stack(det_residual_cycle, axis=0)
        np.savez_compressed(
            job_dir / "fields.npz",
            pred_field=field_arr,
            pred_sample_std=sample_std_arr,
            generated_samples=generated_samples,
            deterministic_field=det_field_arr,
            pred_mean=mean_arr,
            pred_residual=residual_arr,
        )

        cylinders = [{"x": cyl.x, "y": cyl.y} for cyl in effective_request.cylinders]
        kpis = compute_kpis(field_arr) if effective_request.return_kpis else None
        hypergraph = None
        if effective_request.return_hypergraph and aux is not None:
            hypergraph = build_hypergraph(
                aux,
                cylinders,
                float(model_cfg["domain_length_x"]),
                float(model_cfg["domain_length_y"]),
            )
        render_meta = render_frames(
            field_arr,
            frames_dir,
            cylinders,
            float(model_cfg["domain_length_x"]),
            float(model_cfg["domain_length_y"]),
            display_smoothing=effective_request.display_smoothing,
            display_scale=effective_request.display_scale,
            render_interpolation=effective_request.render_interpolation,
        )
        frame_urls = {
            field_name: [
                f"/api/jobs/{job_id}/frames/{field_name}/{idx:03d}"
                for idx in range(effective_request.phase_bins)
            ]
            for field_name in render_meta["fields"]
        }
        rendering_meta = {
            "raw_resolution": render_meta["raw_resolution"],
            "display_resolution": render_meta["display_resolution"],
            "display_smoothing": render_meta["display_smoothing"],
            "render_interpolation": render_meta["render_interpolation"],
            "kpi_source": "raw_model_grid",
            "note": "Display frames are presentation renders; raw model arrays are unchanged.",
        }
        result = {
            "job_id": job_id,
            "status": "complete",
            "request_hash": req_hash,
            "model": entry.to_public_dict(),
            "validation": _model_dump(validation),
            "domain": {
                "length_x": float(model_cfg["domain_length_x"]),
                "length_y": float(model_cfg["domain_length_y"]),
                "resolution_nx": effective_request.resolution_nx,
                "resolution_ny": effective_request.resolution_ny,
                "phase_bins": effective_request.phase_bins,
                "requested_phase_bins": request.phase_bins,
                "effective_phase_bins": effective_request.phase_bins,
                "max_phase_bins": validation.max_phase_bins,
            },
            "fields": ["u", "v", "p", "omega"],
            "frame_urls": frame_urls,
            "render": render_meta,
            "rendering": rendering_meta,
            "kpis": kpis,
            "hypergraph": hypergraph,
            "generative": {
                "num_samples": n_samples,
                "n_steps": n_steps,
                "ode_solver": ode_solver,
                "noise_mode": effective_request.generative.noise_mode,
                "rendered_statistic": "ensemble_mean",
                "export_contains": ["generated_samples", "pred_sample_std", "deterministic_field"],
            },
            "export_npz_url": f"/api/jobs/{job_id}/export.npz",
        }
        with (job_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        register_cache(req_hash, job_id)
        return {"job_id": job_id, "status": "complete", "result_url": f"/api/jobs/{job_id}/result"}

    def infer(self, request: DesignRequest) -> Dict:
        entry = self.registry.get_entry(request.model_id)
        if request.mode != entry.mode:
            raise ValueError(f"Request mode {request.mode!r} does not match model mode {entry.mode!r}.")

        config = entry.load_config_json()
        validation = validate_design(request, config, entry.raw)
        if not validation.valid:
            raise ValueError("Invalid design: " + "; ".join(validation.warnings))
        effective_request = _request_with_phase_bins(request, validation.effective_phase_bins)

        req_hash = request_hash(request)
        cached_job = find_cached_job(req_hash)
        if cached_job:
            return {"job_id": cached_job, "status": "cached", "result_url": f"/api/jobs/{cached_job}/result"}

        if request.mode == "generative":
            if not entry.enabled or entry.stage != 2:
                raise GenerativeUnavailableError("Generative stage-2 model is not available yet.")
            return self._infer_generative(request, entry, req_hash, validation)

        artifact = self.det_service.load(entry)
        model_cfg = dict(artifact.model_config)
        limits = model_limits(config)
        model_cfg.setdefault("max_num_cylinders", limits["max_num_cylinders"])
        model_cfg.setdefault("domain_length_x", limits["domain_length_x"])
        model_cfg.setdefault("domain_length_y", limits["domain_length_y"])

        job_id = new_job_id()
        job_dir = settings.cache_dir / job_id
        frames_dir = job_dir / "frames"
        job_dir.mkdir(parents=True, exist_ok=True)

        field_cycle: List[np.ndarray] = []
        mean_cycle: List[np.ndarray] = []
        residual_cycle: List[np.ndarray] = []
        aux = None

        import torch

        structure = self._build_structure(effective_request, model_cfg, artifact.device)
        x_grid, y_grid = self._make_grid(model_cfg, effective_request.resolution_nx, effective_request.resolution_ny, artifact.device)
        query_batch_size = int(config.get("validation", {}).get("query_batch_size", 16384))

        with torch.no_grad():
            for phase_idx in range(effective_request.phase_bins):
                tau_value = (phase_idx + 0.5) / float(effective_request.phase_bins)
                out = artifact.model.reconstruct_full_grid(
                    structure,
                    x_grid,
                    y_grid,
                    torch.tensor(tau_value, dtype=torch.float32, device=artifact.device),
                    query_batch_size=query_batch_size,
                )
                field_cycle.append(out["pred_field"][0].detach().cpu().numpy().astype(np.float32))
                mean_cycle.append(out["pred_mean"][0].detach().cpu().numpy().astype(np.float32))
                residual_cycle.append(out["pred_residual"][0].detach().cpu().numpy().astype(np.float32))
                if aux is None:
                    aux = {key: value for key, value in out.items() if key not in {"pred_field", "pred_mean", "pred_residual"}}

        field_arr = np.stack(field_cycle, axis=0)
        mean_arr = np.stack(mean_cycle, axis=0)
        residual_arr = np.stack(residual_cycle, axis=0)
        np.savez_compressed(job_dir / "fields.npz", pred_field=field_arr, pred_mean=mean_arr, pred_residual=residual_arr)

        cylinders = [{"x": cyl.x, "y": cyl.y} for cyl in effective_request.cylinders]
        kpis = compute_kpis(field_arr) if effective_request.return_kpis else None
        hypergraph = None
        if effective_request.return_hypergraph and aux is not None:
            hypergraph = build_hypergraph(
                aux,
                cylinders,
                float(model_cfg["domain_length_x"]),
                float(model_cfg["domain_length_y"]),
            )
        render_meta = render_frames(
            field_arr,
            frames_dir,
            cylinders,
            float(model_cfg["domain_length_x"]),
            float(model_cfg["domain_length_y"]),
            display_smoothing=effective_request.display_smoothing,
            display_scale=effective_request.display_scale,
            render_interpolation=effective_request.render_interpolation,
        )

        frame_urls = {
            field_name: [
                f"/api/jobs/{job_id}/frames/{field_name}/{idx:03d}"
                for idx in range(effective_request.phase_bins)
            ]
            for field_name in render_meta["fields"]
        }
        rendering_meta = {
            "raw_resolution": render_meta["raw_resolution"],
            "display_resolution": render_meta["display_resolution"],
            "display_smoothing": render_meta["display_smoothing"],
            "render_interpolation": render_meta["render_interpolation"],
            "kpi_source": "raw_model_grid",
            "note": "Display frames are presentation renders; raw model arrays are unchanged.",
        }
        result = {
            "job_id": job_id,
            "status": "complete",
            "request_hash": req_hash,
            "model": entry.to_public_dict(),
            "validation": _model_dump(validation),
            "domain": {
                "length_x": float(model_cfg["domain_length_x"]),
                "length_y": float(model_cfg["domain_length_y"]),
                "resolution_nx": effective_request.resolution_nx,
                "resolution_ny": effective_request.resolution_ny,
                "phase_bins": effective_request.phase_bins,
                "requested_phase_bins": request.phase_bins,
                "effective_phase_bins": effective_request.phase_bins,
                "max_phase_bins": validation.max_phase_bins,
            },
            "fields": ["u", "v", "p", "omega"],
            "frame_urls": frame_urls,
            "render": render_meta,
            "rendering": rendering_meta,
            "kpis": kpis,
            "hypergraph": hypergraph,
            "export_npz_url": f"/api/jobs/{job_id}/export.npz",
        }
        with (job_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        register_cache(req_hash, job_id)
        return {"job_id": job_id, "status": "complete", "result_url": f"/api/jobs/{job_id}/result"}

    def result_path(self, job_id: str) -> Path:
        return settings.cache_dir / job_id / "result.json"
