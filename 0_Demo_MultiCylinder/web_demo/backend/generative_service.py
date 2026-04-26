from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from model_registry import ModelEntry
from settings import settings


class GenerativeUnavailableError(RuntimeError):
    pass


@dataclass
class GenerativeArtifact:
    flow: Any
    ema: Any
    stats: Any
    checkpoint: Dict[str, Any]
    model_config: Dict[str, Any]
    deterministic_model: Any
    deterministic_model_config: Dict[str, Any]
    deterministic_checkpoint_path: Path
    config: Dict[str, Any]
    checkpoint_path: Path
    device: Any


def _select_device():
    import torch

    requested = settings.device.lower().strip()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _safe_torch_load(path: Path, map_location: Any):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_path(path_like: str | Path, *, base: Path = settings.demo_root) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _build_ae_from_checkpoint(ckpt: Dict[str, Any]):
    from model_gen import ConvResidualAE

    cfg = ckpt.get("ae_config", {})
    return ConvResidualAE(
        n_fields=int(ckpt.get("n_fields", 4)),
        base_ch=int(cfg.get("base_ch", ckpt.get("config", {}).get("stage1", {}).get("architecture", {}).get("base_ch", 48))),
        latent_ch=int(cfg.get("latent_ch", ckpt.get("config", {}).get("stage1", {}).get("architecture", {}).get("latent_ch", 96))),
        n_levels=int(cfg.get("n_levels", ckpt.get("config", {}).get("stage1", {}).get("architecture", {}).get("n_levels", 3))),
        num_res_blocks=int(cfg.get("num_res_blocks", ckpt.get("config", {}).get("stage1", {}).get("architecture", {}).get("num_res_blocks", 1))),
        num_y=int(ckpt["num_y"]),
        num_x=int(ckpt["num_x"]),
    )


def _extract_deterministic_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("model_state_dict", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    raise KeyError("Could not find deterministic model state_dict in checkpoint.")


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        return {str(key)[7:]: value for key, value in state_dict.items()}
    return state_dict


def _resolve_deterministic_checkpoint(ckpt: Dict[str, Any], cfg: Dict[str, Any]) -> Path:
    raw_path = ckpt.get("deterministic_checkpoint_path") or cfg.get("deterministic_model", {}).get("checkpoint_path")
    if not raw_path or str(raw_path).lower() == "auto":
        raise KeyError(
            "Stage-2 checkpoint does not contain a concrete deterministic_checkpoint_path. "
            "Retrain or edit the manifest/config to point at the frozen deterministic checkpoint."
        )
    path = _resolve_path(str(raw_path))
    if not path.exists():
        raise FileNotFoundError(f"Deterministic checkpoint not found: {path}")
    return path


def _load_deterministic_model(checkpoint_path: Path, cfg: Dict[str, Any], device: Any):
    from model import build_model_from_config

    ckpt = _safe_torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Deterministic checkpoint payload is not a dictionary: {checkpoint_path}")
    if isinstance(ckpt.get("model_config"), dict):
        model_cfg = ckpt["model_config"]
    elif isinstance(ckpt.get("config"), dict) and isinstance(ckpt["config"].get("model"), dict):
        model_cfg = ckpt["config"]["model"]
    else:
        model_cfg = cfg.get("deterministic_model", {}).get("model", {})
    if not model_cfg:
        raise ValueError("Could not find deterministic model config in checkpoint or generative config.")
    model = build_model_from_config(model_cfg)
    model.load_state_dict(_strip_module_prefix(_extract_deterministic_state_dict(ckpt)))
    model.to(device).eval().requires_grad_(False)
    return model, model_cfg


class GenerativeService:
    def load(self, entry: ModelEntry) -> GenerativeArtifact:
        if entry.mode != "generative":
            raise ValueError(f"Model {entry.id!r} is not generative.")
        if not entry.enabled or entry.stage != 2:
            raise GenerativeUnavailableError("Generative stage-2 model is not available yet.")
        if entry.missing_files:
            raise FileNotFoundError("Model entry has missing files: " + ", ".join(entry.missing_files))

        with entry.lock:
            if entry.loaded_artifact is not None:
                return entry.loaded_artifact

            import torch
            from model_gen import GridStats, LatentEMA, LatentRectifiedFlow, LatentVelocityUNet

            device = _select_device()
            ckpt = _safe_torch_load(entry.checkpoint_path, map_location=device)
            if not isinstance(ckpt, dict):
                raise ValueError(f"Generative checkpoint payload is not a dictionary: {entry.checkpoint_path}")
            if int(ckpt.get("stage", 0)) != 2:
                raise ValueError(f"Expected a stage-2 generative checkpoint, got stage={ckpt.get('stage')}.")

            cfg = ckpt["config"]
            stats = GridStats(mean=ckpt["stats"]["mean"].to(device), std=ckpt["stats"]["std"].to(device))
            ae = _build_ae_from_checkpoint(ckpt).to(device)
            ae.load_state_dict(ckpt["ae_state_dict"])
            ae.eval().requires_grad_(False)

            arch = cfg["stage2"]["architecture"]
            velocity = LatentVelocityUNet(
                latent_ch=ae.latent_ch,
                cond_ch=int(ckpt["cond_ch"]),
                global_cond_dim=int(ckpt["global_cond_dim"]),
                base_ch=int(arch.get("base_ch", ckpt.get("fm_base_ch", 192))),
                ch_mult=tuple(arch.get("ch_mult", [1, 2])),
                num_res_blocks=int(arch.get("num_res_blocks", 2)),
                num_heads=int(arch.get("num_heads", 4)),
                dropout=float(arch.get("dropout", 0.0)),
            ).to(device)
            flow = LatentRectifiedFlow(
                ae=ae,
                velocity_net=velocity,
                cond_downsample_mode=arch.get("cond_downsample_mode", "area"),
            ).to(device)
            flow.velocity_net.load_state_dict(ckpt["velocity_state_dict"])
            ema = None
            if ckpt.get("ema_state_dict") is not None:
                ema = LatentEMA(flow.velocity_net, decay=float(arch.get("ema_decay", 0.999)))
                ema.load_state_dict(ckpt["ema_state_dict"])
            flow.eval()

            deterministic_checkpoint_path = _resolve_deterministic_checkpoint(ckpt, cfg)
            det_model, det_model_cfg = _load_deterministic_model(deterministic_checkpoint_path, cfg, device)
            entry.loaded_artifact = GenerativeArtifact(
                flow=flow,
                ema=ema,
                stats=stats,
                checkpoint=ckpt,
                model_config=cfg,
                deterministic_model=det_model,
                deterministic_model_config=det_model_cfg,
                deterministic_checkpoint_path=deterministic_checkpoint_path,
                config=entry.load_config_json(),
                checkpoint_path=entry.checkpoint_path,
                device=device,
            )
            return entry.loaded_artifact

    def preload(self, entry: ModelEntry) -> None:
        self.load(entry)
