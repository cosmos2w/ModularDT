"""Checkpoint resolution, strict reconstruction, and frozen overrides."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.local_surrogate.model import LocalModuleConfig, LocalModuleSurrogate
from channelthermal.model import ChannelThermalHONFModel
from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_runtime.compat import (
    load_trusted_checkpoint,
    recursive_to_device,
    resolve_demo_path,
    strip_module_prefix,
)


def checkpoint_file_name(selector: str) -> str:
    """Perform the checkpoint file name operation used by this module."""

    cleaned = str(selector).strip().lower()
    if cleaned in {"best_predicted", "predicted", "autonomous"}:
        return "best_predicted_model.pt"
    if cleaned in {"best", "best_total"}:
        return "best_model.pt"
    if cleaned in {"best_by_field_mse", "field"}:
        return "best_by_field_mse_model.pt"
    if cleaned in {"best_by_temperature_mse", "temperature"}:
        return "best_by_temperature_mse_model.pt"
    if cleaned in {"latest", "lastest"}:
        return "latest_model.pt"
    raise ValueError(f"Unknown checkpoint selector: {selector}")


def normalize_run_id(value: str) -> str:
    """Normalize run id."""

    raw = str(value).strip()
    if not raw.isdigit():
        raise ValueError(f"Run_ID must be numeric, e.g. 0002; got {raw!r}.")
    return f"{int(raw):04d}"


def latest_run_dir(saved_root: Path, run_id: str) -> Path:
    """Perform the latest run dir operation used by this module."""

    normalized = normalize_run_id(run_id)
    patterns = (f"Run_{normalized}_*", f"{normalized}_*", f"{normalized}*")
    matches = sorted({path for pattern in patterns for path in saved_root.glob(pattern) if path.is_dir()})
    if not matches:
        raise FileNotFoundError(f"No saved HONF-CL runs found under {saved_root} with Run_ID={normalized!r}.")
    if len(matches) > 1:
        candidates = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Run_ID={normalized!r} is ambiguous under {saved_root}; pass an explicit checkpoint path:\n  "
            f"{candidates}"
        )
    def sort_key(path: Path) -> tuple[int, str, float, str]:
        """Perform the sort key operation used by this module."""

        match = re.search(rf"Run_{normalized}_(\d{{8}}_\d{{6}})", path.name)
        # Prefer the global timestamped run naming scheme over older
        # compatibility names, then choose the newest timestamp/mtime.
        return (1 if match else 0, match.group(1) if match else "", path.stat().st_mtime, path.name)

    return sorted(matches, key=sort_key)[-1]


def resolve_checkpoint_arg(args: argparse.Namespace) -> Path:
    """Resolve checkpoint arg."""

    selector = str(args.checkpoint)
    if args.run_id:
        run_dir = latest_run_dir(resolve_demo_path(args.saved_root), args.run_id)
        candidate = (run_dir / checkpoint_file_name(selector)).resolve()
        if (
            not candidate.exists()
            and selector.lower() in {"best_predicted", "predicted", "autonomous"}
            and bool(args.allow_checkpoint_fallback)
        ):
            fallback = (run_dir / "best_model.pt").resolve()
            print(f"[warning] {candidate.name} not found; falling back to {fallback.name}.")
            return fallback
        return candidate
    candidate = resolve_demo_path(selector)
    if candidate.suffix == ".pt" or candidate.exists():
        return candidate
    raise ValueError("--Run_ID is required when --checkpoint is a named selector.")


def numpy_to_batched_tensor(value: Any) -> Any:
    """Perform the numpy to batched tensor operation used by this module."""

    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).unsqueeze(0)
    if isinstance(value, dict):
        return {key: numpy_to_batched_tensor(item) for key, item in value.items()}
    return value


def make_batch(sample: Dict[str, Any], query_xy: np.ndarray, device: torch.device) -> Dict[str, Any]:
    """Create batch."""

    payload = {key: value for key, value in sample.items() if key not in {"x_grid", "y_grid", "steady_field", "rms_field", "case_id"}}
    payload["query_xy"] = query_xy.astype(np.float32)
    return recursive_to_device(numpy_to_batched_tensor(payload), device)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ChannelThermalHONFModel, Dict[str, Any]]:
    """Load model."""

    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    validate_checkpoint_identity(
        checkpoint,
        case_id="ThermalChannel",
        model_family="honf_forward",
        workflow="forward",
    )
    model_config = ChannelThermalHONFConfig.from_dict(checkpoint.get("model_config", {}))
    model = ChannelThermalHONFModel(model_config, attach_local_from_checkpoint=False)
    if bool(model_config.channelthermal.use_local_surrogate):
        local_config_payload = checkpoint.get("local_model_config")
        if isinstance(local_config_payload, dict):
            local_model = LocalModuleSurrogate(LocalModuleConfig.from_dict(local_config_payload))
            model.local_coupling.set_local_surrogate(
                local_model,
                freeze=bool(checkpoint.get("local_surrogate_frozen", model_config.channelthermal.freeze_local_surrogate)),
                normalization_config=checkpoint.get("local_normalization_config", {}),
                normalization_stats=checkpoint.get("local_normalization_stats", {}),
            )
            model.local_coupling.local_surrogate_checkpoint_path = checkpoint.get("local_checkpoint_provenance", checkpoint.get("local_surrogate_checkpoint_path"))
        elif model_config.channelthermal.local_surrogate_checkpoint_path:
            print("[warning] checkpoint lacks embedded local_model_config; falling back to external local checkpoint path.")
            model.local_coupling.attach_from_checkpoint(
                model_config.channelthermal.local_surrogate_checkpoint_path,
                freeze=bool(model_config.channelthermal.freeze_local_surrogate),
                map_location="cpu",
            )
    model = model.to(device)
    global_norm_cfg = checkpoint.get("global_normalization_config", checkpoint.get("train_config", {}).get("dataset", {}))
    model.set_global_target_normalization(checkpoint.get("global_normalization_stats", {}), normalize_targets=bool(global_norm_cfg.get("normalize_targets", False)))
    state = strip_module_prefix(checkpoint["model_state_dict"])
    # Every maintained forward family is checkpoint-compatible.  Evaluation
    # must therefore use the same strict contract as training resume; adding a
    # permissive non-critical bucket would hide accidental parameter moves.
    model.load_state_dict(state, strict=True)
    selection_state = checkpoint.get("selection_state")
    configured_epochs = checkpoint.get("train_config", {}).get("training", {}).get("epochs")
    if isinstance(selection_state, dict) and selection_state.get("epoch") is not None:
        total_epochs = selection_state.get("total_epochs", configured_epochs)
        model.set_training_progress(
            epoch=int(selection_state["epoch"]),
            total_epochs=None if total_epochs is None else int(total_epochs),
        )
    else:
        checkpoint_epoch = checkpoint.get("epoch", checkpoint.get("current_epoch"))
        # Historical checkpoints restore their recorded epoch. A raw weights
        # checkpoint with no epoch receives an explicit final inference phase.
        default_selection_epoch = model.selection_state().get("epoch")
        inference_epoch = (
            int(checkpoint_epoch)
            if checkpoint_epoch is not None
            else int(
                model_config.core_honf.selection_warmup_epochs
                if default_selection_epoch is None
                else default_selection_epoch
            )
        )
        model.set_training_progress(
            epoch=inference_epoch,
            total_epochs=None if configured_epochs is None else int(configured_epochs),
        )
    model.eval()
    return model, checkpoint


def apply_frozen_forward_overrides(
    model: ChannelThermalHONFModel,
    *,
    mechanism_latent_residual_scale: Optional[float] = None,
    query_locality_mode: Optional[str] = None,
    query_locality_strength: Optional[float] = None,
) -> Dict[str, Any]:
    """Apply evaluation-only decoder settings without changing parameters.

    Stage-6 frozen screening deliberately reuses a checkpoint's weights while
    varying only arithmetic already represented by configuration.  The
    descriptor-first encoder caches its residual scale as a plain Python
    scalar, so it is updated alongside every shared core-config reference.
    Missing overrides leave checkpoint-owned behavior exactly unchanged.
    """

    if mechanism_latent_residual_scale is not None and not (
        0.0 <= float(mechanism_latent_residual_scale) <= 1.0
    ):
        raise ValueError("mechanism_latent_residual_scale must be in [0, 1].")
    locality_modes = {
        "none",
        "compact_kernel",
        "bounded_gaussian",
        "gaussian_bounded",
        "inherit_environment",
    }
    if query_locality_mode is not None and query_locality_mode not in locality_modes:
        raise ValueError(f"Unsupported query_locality_mode={query_locality_mode!r}.")
    if query_locality_strength is not None and float(query_locality_strength) < 0.0:
        raise ValueError("query_locality_strength must be nonnegative.")

    state_keys_before = tuple(model.state_dict())
    candidates = (
        model.config.core_honf,
        model.core.config,
        model.core.decoder.config,
        model.core.decoder.pairwise_kernel.config,
        model.core.organizer.config,
    )
    seen: set[int] = set()
    configs = []
    for candidate in candidates:
        if candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            configs.append(candidate)

    if mechanism_latent_residual_scale is not None:
        value = float(mechanism_latent_residual_scale)
        for config in configs:
            config.mechanism_latent_residual_scale = value
        mechanism_encoder = model.core.decoder.mechanism_encoder
        if mechanism_encoder is None or not hasattr(mechanism_encoder, "content_scale"):
            raise RuntimeError("Frozen mechanism-scale override requires descriptor-first encoding.")
        mechanism_encoder.content_scale = value
    if query_locality_mode is not None:
        for config in configs:
            config.query_locality_mode = str(query_locality_mode)
    if query_locality_strength is not None:
        value = float(query_locality_strength)
        for config in configs:
            config.query_locality_strength = value

    state_keys_after = tuple(model.state_dict())
    if state_keys_before != state_keys_after:
        raise RuntimeError("Frozen evaluation override changed state_dict structure.")

    core = model.config.core_honf
    effective_query_strength = (
        float(core.environment_locality_strength)
        if core.query_locality_strength is None
        else float(core.query_locality_strength)
    )
    return {
        "mechanism_latent_residual_scale": float(core.mechanism_latent_residual_scale),
        "query_locality_mode": str(core.query_locality_mode),
        "query_locality_strength": (
            None if core.query_locality_strength is None else float(core.query_locality_strength)
        ),
        "effective_query_locality_strength": effective_query_strength,
        "state_dict_key_count": len(state_keys_before),
        "state_dict_structure_unchanged": state_keys_before == state_keys_after,
    }
