"""Native sampled WindFarm velocity training through the generic run store."""

from __future__ import annotations

import copy
import csv
import json
import math
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from honf_forward_core.config import UnifiedForwardConfig
from honf_runtime.checkpoints import validate_checkpoint_identity
from honf_runtime.compat import load_trusted_checkpoint, select_device, set_seed
from honf_runtime.paths import resolve_path
from torch.utils.data import DataLoader

from ..data import (
    COMPACT_GEOMETRY_KEYS,
    WindFarmNativeDataset,
    WindFarmNativeView,
    batch_to_batch_data,
    collate_windfarm,
)
from ..geometry import ENV_TOKEN_SHAPE
from ..model import WindFarmForwardModel
from ..normalization import (
    VelocityNormalizer,
    VerticalProfileBaseline,
    fit_velocity_statistics,
    read_normalization_json,
    write_normalization_json,
)
from ..splits import GroupSplit, make_group_split, split_json, write_split_outputs


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{__import__('os').getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolved_path(value: Any) -> Path:
    """Resolve a project URI or an ordinary local path consistently."""

    return resolve_path(str(value))


def _compact_metadata(path: str | Path) -> dict[str, np.ndarray]:
    """Read only bounded geometry metadata from the compact NPZ resource.

    The full-volume reader remains mmap-backed.  This explicit path is kept
    separate so a locations map may place compact metadata and ragged fields
    in different roots without making the adapter guess or copy either one.
    """

    compact_path = _resolved_path(path)
    if not compact_path.is_file():
        raise FileNotFoundError(f"WindFarm compact metadata resource not found: {compact_path}")
    with np.load(compact_path, allow_pickle=False) as archive:
        missing = [name for name in COMPACT_GEOMETRY_KEYS if name not in archive.files]
        if missing:
            raise ValueError(f"WindFarm compact metadata is missing keys {missing}: {compact_path}")
        return {name: np.asarray(archive[name]).copy() for name in COMPACT_GEOMETRY_KEYS}


def _load_or_make_split(view: WindFarmNativeView, dataset_cfg: Mapping[str, Any]) -> GroupSplit:
    derived = _resolved_path(dataset_cfg["derived_view"])
    split_npz = derived / "split_indices.npz"
    split_meta = derived / "splits.json"
    if split_npz.is_file():
        with np.load(split_npz, allow_pickle=False) as archive:
            required = {"train", "validation", "test"}
            if set(archive.files) != required:
                raise ValueError(f"WindFarm split artifact must contain {sorted(required)}")
            split = GroupSplit(
                train=np.asarray(archive["train"], dtype=np.int64),
                validation=np.asarray(archive["validation"], dtype=np.int64),
                test=np.asarray(archive["test"], dtype=np.int64),
                metadata={},
            )
        # Verify the stored rows against the current native metadata before
        # trusting an ignored derived file.
        groups = np.asarray(view.volume.array("layout_index"))
        all_rows = np.concatenate((split.train, split.validation, split.test))
        if np.unique(all_rows).size != view.n_cases or np.any(np.sort(all_rows) != np.arange(view.n_cases)):
            raise ValueError("Stored WindFarm split indices are incomplete or duplicated.")
        memberships = [set(groups[rows].tolist()) for rows in (split.train, split.validation, split.test)]
        if memberships[0] & memberships[1] or memberships[0] & memberships[2] or memberships[1] & memberships[2]:
            raise ValueError("Stored WindFarm split indices overlap layout groups.")
        canonical = make_group_split(groups, seed=42)
        if not (
            np.array_equal(split.train, canonical.train)
            and np.array_equal(split.validation, canonical.validation)
            and np.array_equal(split.test, canonical.test)
        ):
            raise ValueError("Stored WindFarm split indices do not match the documented seed-42 split.")
        return canonical
    split = make_group_split(np.asarray(view.volume.array("layout_index")), seed=42)
    write_split_outputs(split_npz, split)
    split_json(split_meta, split)
    return split


def _load_or_fit_normalizer(
    view: WindFarmNativeView,
    split: GroupSplit,
    dataset_cfg: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[VelocityNormalizer, Any]:
    path = _resolved_path(dataset_cfg["normalization"])
    if path.is_file():
        return read_normalization_json(path)
    # The documented native sampler seed is part of the derived-data
    # contract.  It must not vary with a model's optimizer seed.
    del seed
    normalizer, profile = fit_velocity_statistics(view.volume, split.train, seed=42)
    write_normalization_json(path, normalizer, profile)
    return normalizer, profile


def _as_device_batch(raw: Mapping[str, Any], device: torch.device):
    payload = dict(raw)
    for name in (
        "module_centers", "module_present", "module_features", "global_context", "query_xy",
        "target_field", "env_coords", "env_features", "env_weights", "query_features",
    ):
        value = payload.get(name)
        if value is not None and not torch.is_tensor(value):
            payload[name] = torch.as_tensor(value)
    batch = batch_to_batch_data(payload)
    return batch.to(device)


def _weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    raw_batch: Mapping[str, Any],
    channel_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(f"WindFarm prediction/target shape mismatch: {prediction.shape} versus {target.shape}")
    squared_error = (prediction - target) ** 2
    weighted_squared_error = squared_error * channel_weights
    point_error = weighted_squared_error.mean(dim=-1)
    weights = raw_batch.get("query_loss_weight")
    if weights is None:
        point_weights = torch.full_like(point_error, 1.0 / max(int(point_error.shape[1]), 1))
    else:
        point_weights = torch.as_tensor(weights, dtype=point_error.dtype, device=point_error.device)
    per_case = (point_error * point_weights).sum(dim=1)
    return (
        per_case.mean(),
        per_case.detach(),
        weighted_squared_error.detach(),
        point_error.detach(),
    )


def _run_loader(
    model: WindFarmForwardModel,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    channel_weights: torch.Tensor,
    receiver_chunk_size: int,
    max_batches: int | None,
    gradient_clip_norm: float,
    volume_queries: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_case_loss = 0.0
    total_cases = 0
    volume_error_sum = 0.0
    volume_error_count = 0
    band_error_sum = 0.0
    band_error_count = 0
    channel_sums = np.zeros(3, dtype=np.float64)
    channel_counts = np.zeros(3, dtype=np.float64)
    gradient_norm = 0.0
    clip_scale = 1.0
    update_norm = 0.0
    data_fetch_seconds = 0.0
    host_to_device_seconds = 0.0
    prepare_seconds = 0.0
    forward_seconds = 0.0
    backward_seconds = 0.0
    wall_seconds = 0.0
    batches = 0
    iterator = iter(loader)
    while True:
        if max_batches is not None and batches >= int(max_batches):
            break
        batch_started = time.perf_counter()
        data_started = time.perf_counter()
        try:
            raw_batch = next(iterator)
        except StopIteration:
            break
        data_fetch_seconds += time.perf_counter() - data_started
        transfer_started = time.perf_counter()
        batch = _as_device_batch(raw_batch, device)
        host_to_device_seconds += time.perf_counter() - transfer_started
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        prepare_started = time.perf_counter()
        prepared = model.prepare_case(batch)
        prepare_seconds += time.perf_counter() - prepare_started
        forward_started = time.perf_counter()
        prediction = model.predict_standardized(
            prepared,
            batch.query_xy,
            query_features=batch.query_features,
            receiver_chunk_size=int(receiver_chunk_size),
        )
        forward_seconds += time.perf_counter() - forward_started
        loss, per_case, weighted_squared_error, point_error = _weighted_loss(
            prediction,
            batch.target_field,
            raw_batch,
            channel_weights,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite WindFarm loss at batch {batches}: {float(loss.detach().cpu())}")
        if optimizer is not None:
            backward_started = time.perf_counter()
            loss.backward()
            parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
            if not parameters:
                raise RuntimeError("WindFarm loss produced no parameter gradients.")
            gradient_norm_value = torch.nn.utils.clip_grad_norm_(
                parameters,
                float(gradient_clip_norm),
                error_if_nonfinite=True,
            )
            gradient_norm = float(gradient_norm_value.detach().cpu())
            clip_scale = min(1.0, float(gradient_clip_norm) / max(gradient_norm, 1.0e-12))
            tracked = next(parameter for parameter in parameters if parameter.numel() > 0)
            before = tracked.detach().clone()
            optimizer.step()
            update_norm = float(torch.linalg.vector_norm((tracked.detach() - before).reshape(-1)).cpu())
            if not math.isfinite(update_norm):
                raise FloatingPointError(f"non-finite WindFarm parameter update at batch {batches}")
            backward_seconds += time.perf_counter() - backward_started
        total_case_loss += float(per_case.sum().cpu())
        total_cases += int(per_case.numel())
        q_count = int(point_error.shape[1])
        volume_count = int(volume_queries)
        if volume_count <= 0 or volume_count >= q_count:
            raise ValueError(f"volume_queries={volume_count} must lie inside sampled Q={q_count}")
        volume_error = weighted_squared_error[:, :volume_count]
        band_error = weighted_squared_error[:, volume_count:]
        volume_error_sum += float(volume_error.sum().cpu())
        volume_error_count += int(volume_error.numel())
        band_error_sum += float(band_error.sum().cpu())
        band_error_count += int(band_error.numel())
        channel_sums += weighted_squared_error.sum(dim=(0, 1)).cpu().numpy()
        channel_counts += float(weighted_squared_error.shape[0] * weighted_squared_error.shape[1])
        batches += 1
        wall_seconds += time.perf_counter() - batch_started
        del prepared, prediction, loss, per_case, weighted_squared_error, point_error
    if not total_cases:
        raise RuntimeError("WindFarm loader produced no batches.")
    return {
        "loss": float(total_case_loss / max(total_cases, 1)),
        "volume_mse": float(volume_error_sum / max(volume_error_count, 1)),
        "band_mse": float(band_error_sum / max(band_error_count, 1)),
        "channel_0_mse": float(channel_sums[0] / max(channel_counts[0], 1.0)),
        "channel_1_mse": float(channel_sums[1] / max(channel_counts[1], 1.0)),
        "channel_2_mse": float(channel_sums[2] / max(channel_counts[2], 1.0)),
        "gradient_norm_preclip": float(gradient_norm),
        "clip_scale": float(clip_scale),
        "sampled_update_norm": float(update_norm),
        "batches": float(batches),
        "data_fetch_dispatch_seconds": float(data_fetch_seconds),
        "host_to_device_dispatch_seconds": float(host_to_device_seconds),
        "prepare_dispatch_seconds": float(prepare_seconds),
        "forward_dispatch_seconds": float(forward_seconds),
        "backward_dispatch_seconds": float(backward_seconds),
        "batch_wall_seconds": float(wall_seconds),
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    if payload.get("python") is not None:
        random.setstate(payload["python"])
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if payload.get("torch") is not None:
        torch.set_rng_state(payload["torch"].cpu())
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda"]])


def _save_checkpoint(
    path: Path,
    *,
    model: WindFarmForwardModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    best_epoch: int | None,
    config: Mapping[str, Any],
    normalizer: VelocityNormalizer,
    split: GroupSplit,
    profile: Any,
    update_count: int,
) -> None:
    payload = {
        "checkpoint_schema_version": 1,
        "case_id": "WindFarm",
        "model_family": "honf_forward",
        "workflow": "forward",
        "stage": "windfarm_native_velocity",
        "epoch": int(epoch),
        "current_epoch": int(epoch),
        "update_count": int(update_count),
        "best_metric": float(best_metric),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_config": copy.deepcopy(dict(config)),
        "dataset_id": config.get("dataset", {}).get("dataset_id", "wind_farm_volume_v1"),
        "dataset_schema": config.get("dataset", {}).get("dataset_schema"),
        "dataset_fingerprint": None,
        "channel_order": ["Ux", "Uy", "Uz"],
        "field_dim": 3,
        "feature_schemas": {
            "module_feature_names": ["rotor_radius_D", "hub_height_D"],
            "global_context_names": [
                "wd270", "wd285", "wd300", "M_over_30", "Uref_over_9",
                "support_lower_x_over_50", "support_lower_y_over_38", "support_lower_z_over_6.25",
                "support_extent_x_over_50", "support_extent_y_over_38", "support_extent_z_over_6.25",
            ],
            "query_feature_names": [
                "lower_x", "lower_y", "lower_z", "upper_x", "upper_y", "upper_z", "z_over_6.25",
            ],
        },
        "normalization": normalizer.to_dict(),
        "vertical_profile_baseline": None if profile is None else profile.to_dict(),
        "split_metadata": split.metadata,
        "split_indices": {
            "train": np.asarray(split.train, dtype=np.int64),
            "validation": np.asarray(split.validation, dtype=np.int64),
            "test": np.asarray(split.test, dtype=np.int64),
        },
        "optimizer_group_inventory": {
            "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "group_count": 1,
        },
        "rng_state": _rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _split_from_checkpoint(view: WindFarmNativeView, payload: Mapping[str, Any]) -> GroupSplit:
    """Recover and verify the exact split owned by a resume checkpoint."""

    stored = payload.get("split_indices")
    if not isinstance(stored, Mapping):
        raise TypeError("WindFarm resume checkpoint lacks split_indices metadata.")
    required = {"train", "validation", "test"}
    if set(stored) != required:
        raise ValueError(f"WindFarm resume split must contain {sorted(required)}.")
    split = GroupSplit(
        train=np.asarray(stored["train"], dtype=np.int64),
        validation=np.asarray(stored["validation"], dtype=np.int64),
        test=np.asarray(stored["test"], dtype=np.int64),
        metadata=dict(payload.get("split_metadata") or {}),
    )
    canonical = make_group_split(np.asarray(view.volume.array("layout_index")), seed=42)
    if not all(
        np.array_equal(getattr(split, name), getattr(canonical, name))
        for name in ("train", "validation", "test")
    ):
        raise ValueError("WindFarm resume split differs from the documented seed-42 layout split.")
    if not split.metadata:
        split = GroupSplit(split.train, split.validation, split.test, canonical.metadata)
    return split


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def run_from_config(config: Mapping[str, Any], request: Any, *, run_dir_override: Path | None = None) -> int:
    """Execute one normal run-store-owned WindFarm training workflow."""

    requested_cfg = copy.deepcopy(dict(config))
    resume_path_value = getattr(request, "resume_checkpoint", None)
    if getattr(request, "initialize_checkpoint", None):
        raise ValueError("WindFarm forward models always start from scratch; initialization checkpoints are not supported.")
    resume_payload: dict[str, Any] | None = None
    if resume_path_value:
        resume_payload = load_trusted_checkpoint(_resolved_path(resume_path_value), map_location="cpu")
        validate_checkpoint_identity(
            resume_payload,
            case_id="WindFarm",
            model_family="honf_forward",
            workflow="forward",
        )
        checkpoint_cfg = resume_payload.get("train_config")
        if not isinstance(checkpoint_cfg, Mapping):
            raise ValueError("WindFarm resume checkpoint lacks its resolved train_config.")
        cfg = copy.deepcopy(dict(checkpoint_cfg))
        requested_core = UnifiedForwardConfig.from_dict(
            dict(requested_cfg.get("model", {}).get("core_honf", {}))
        )
        checkpoint_core = UnifiedForwardConfig.from_dict(dict(resume_payload.get("model_config") or {}))
        if requested_core.to_dict() != checkpoint_core.to_dict():
            raise ValueError("WindFarm resume model configuration does not match the requested launch profile.")
        cfg.setdefault("model", {})["core_honf"] = checkpoint_core.to_dict()
    else:
        cfg = requested_cfg
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg.get("training", {})
    seed = int(training_cfg.get("seed", 0))
    set_seed(seed)
    device = select_device(getattr(request, "device", None) or training_cfg.get("device"))
    volume_path = _resolved_path(dataset_cfg["volume_path"])
    compact_path = _resolved_path(dataset_cfg["compact_path"])
    view = WindFarmNativeView(
        volume_path,
        compact_metadata=_compact_metadata(compact_path),
        token_shape=tuple(dataset_cfg.get("env_token_shape", ENV_TOKEN_SHAPE)),
    )
    split = _load_or_make_split(view, dataset_cfg)
    if resume_payload is not None:
        split = _split_from_checkpoint(view, resume_payload)
        normalization_payload = resume_payload.get("normalization")
        if not isinstance(normalization_payload, Mapping):
            raise ValueError("WindFarm resume checkpoint lacks its training-owned normalization.")
        normalizer = VelocityNormalizer.from_dict(dict(normalization_payload))
        profile_payload = resume_payload.get("vertical_profile_baseline")
        profile = (
            None
            if not isinstance(profile_payload, Mapping)
            else VerticalProfileBaseline.from_dict(dict(profile_payload))
        )
    else:
        normalizer, profile = _load_or_fit_normalizer(view, split, dataset_cfg, seed=42)

    train_dataset = WindFarmNativeDataset(
        view,
        split.train,
        normalizer=normalizer,
        queries_per_case=int(dataset_cfg.get("q_train", 1024)),
        volume_fraction=(
            float(cfg.get("loss", {}).get("volume_sampling_fraction", 0.75))
            if "volume_sampling_fraction" in cfg.get("loss", {})
            else float(dataset_cfg.get("q_volume", 768)) / max(float(dataset_cfg.get("q_train", 1024)), 1.0)
        ),
        seed=int(dataset_cfg.get("sample_seed", 42)),
    )
    validation_total = int(dataset_cfg.get("validation_volume_queries", 8192)) + int(dataset_cfg.get("validation_band_queries", 2048))
    validation_fraction = int(dataset_cfg.get("validation_volume_queries", 8192)) / max(validation_total, 1)
    val_dataset = WindFarmNativeDataset(
        view,
        split.validation,
        normalizer=normalizer,
        queries_per_case=validation_total,
        volume_fraction=validation_fraction,
        seed=int(dataset_cfg.get("sample_seed", 42)),
        fixed_sampling=True,
    )
    loader_kwargs = {
        "num_workers": int(dataset_cfg.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_windfarm,
    }
    train_loader = DataLoader(train_dataset, batch_size=int(dataset_cfg.get("batch_size", 8)), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=int(dataset_cfg.get("val_batch_size", 8)), shuffle=False, **loader_kwargs)

    core_payload = dict(cfg.get("model", {}).get("core_honf", {}))
    model_config = UnifiedForwardConfig.from_dict(core_payload)
    model = WindFarmForwardModel(model_config, velocity_transform=normalizer).to(device)
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(int(seed) + 104729)
    # Rebuild the loader with an explicit sampler generator.  Its seed is
    # reset per epoch below, so model construction and architecture-specific
    # RNG consumption cannot change the row order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(dataset_cfg.get("batch_size", 8)),
        shuffle=True,
        generator=train_generator,
        **loader_kwargs,
    )
    first_raw = next(iter(train_loader))
    first_batch = _as_device_batch(first_raw, device)
    model.materialize(first_batch)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-5)),
    )
    run_dir = Path(run_dir_override or cfg.get("paths", {}).get("saved_model_dir", "Trained_Results")).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(getattr(request, "epochs", None) or training_cfg.get("epochs", 500))
    cfg.setdefault("training", {})["epochs"] = epochs
    _json_write(run_dir / "config_resolved.json", cfg)
    normalization_sidecar = normalizer.to_dict()
    if profile is not None:
        normalization_sidecar["vertical_profile_baseline"] = profile.to_dict()
    _json_write(run_dir / "normalization.json", normalization_sidecar)
    max_train_batches = getattr(request, "max_train_batches", None)
    max_val_batches = getattr(request, "max_val_batches", None)
    start_epoch = 1
    best_metric = math.inf
    best_epoch: int | None = None
    update_count = 0
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        best_metric = float(resume_payload.get("best_metric", math.inf))
        best_epoch_value = resume_payload.get("best_epoch")
        best_epoch = None if best_epoch_value is None else int(best_epoch_value)
        update_count = int(resume_payload.get("update_count", 0))
        _restore_rng_state(resume_payload.get("rng_state", {}))

    channel_weights = torch.as_tensor(cfg.get("loss", {}).get("channel_weights", [1.0, 1.0, 1.0]), dtype=torch.float32, device=device)
    history: list[dict[str, Any]] = _read_history(run_dir / "metrics.csv") if resume_payload is not None else []
    receiver_chunk_size = int(core_payload.get("interface_model", {}).get("receiver_chunk_size", 128))
    gradient_clip_norm = float(training_cfg.get("gradient_clip_norm", 1.0))
    for epoch in range(start_epoch, epochs + 1):
        train_generator.manual_seed(int(seed) + 104729 + 1000003 * int(epoch))
        train_dataset.set_epoch(epoch)
        model.set_training_progress(epoch=epoch, total_epochs=epochs)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        train_metrics = _run_loader(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            channel_weights=channel_weights,
            receiver_chunk_size=receiver_chunk_size,
            max_batches=None if max_train_batches is None else int(max_train_batches),
            gradient_clip_norm=gradient_clip_norm,
            volume_queries=train_dataset.volume_queries,
        )
        train_seconds = time.perf_counter() - started
        update_count += int(train_metrics["batches"])
        val_seconds = 0.0
        val_metrics: dict[str, float] = {}
        if epoch % 5 == 0 or epoch == epochs:
            started = time.perf_counter()
            with torch.no_grad():
                val_metrics = _run_loader(
                    model,
                    val_loader,
                    device,
                    optimizer=None,
                    channel_weights=channel_weights,
                    receiver_chunk_size=receiver_chunk_size,
                    max_batches=None if max_val_batches is None else int(max_val_batches),
                    gradient_clip_norm=0.0,
                    volume_queries=val_dataset.volume_queries,
                )
            val_seconds = time.perf_counter() - started
            metric = float(val_metrics["volume_mse"])
            if metric < best_metric:
                best_metric = metric
                best_epoch = epoch
                _save_checkpoint(
                    run_dir / "best_model.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    config=cfg,
                    normalizer=normalizer,
                    split=split,
                    profile=profile,
                    update_count=update_count,
                )
                _save_checkpoint(
                    run_dir / "best_by_field_mse_model.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    config=cfg,
                    normalizer=normalizer,
                    split=split,
                    profile=profile,
                    update_count=update_count,
                )
        if epoch % int(cfg.get("checkpointing", {}).get("save_latest_every_epochs", 10)) == 0 or epoch == epochs:
            _save_checkpoint(
                run_dir / "latest_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                best_epoch=best_epoch,
                config=cfg,
                normalizer=normalizer,
                split=split,
                profile=profile,
                update_count=update_count,
            )
        milestones = {
            int(value) for value in cfg.get("checkpointing", {}).get("save_epoch_milestones", [])
        }
        if epoch in milestones:
            _save_checkpoint(
                run_dir / f"epoch_{epoch:04d}_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                best_epoch=best_epoch,
                config=cfg,
                normalizer=normalizer,
                split=split,
                profile=profile,
                update_count=update_count,
            )
        row = {
            "epoch": epoch,
            "loss_total": train_metrics["loss"],
            "train_volume_mse": train_metrics["volume_mse"],
            "train_band_mse": train_metrics["band_mse"],
            "train_channel_0_mse": train_metrics["channel_0_mse"],
            "train_channel_1_mse": train_metrics["channel_1_mse"],
            "train_channel_2_mse": train_metrics["channel_2_mse"],
            "val_volume_mse": val_metrics.get("volume_mse", ""),
            "val_band_mse": val_metrics.get("band_mse", ""),
            "val_loss_total": val_metrics.get("loss", ""),
            "gradient_norm_preclip": train_metrics["gradient_norm_preclip"],
            "clip_scale": train_metrics["clip_scale"],
            "sampled_update_norm": train_metrics["sampled_update_norm"],
            "train_wall_seconds": train_seconds,
            "val_wall_seconds": val_seconds,
            "peak_cuda_memory_mb": (float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0),
            "update_count": update_count,
            "data_fetch_dispatch_seconds": train_metrics["data_fetch_dispatch_seconds"],
            "host_to_device_dispatch_seconds": train_metrics["host_to_device_dispatch_seconds"],
            "prepare_dispatch_seconds": train_metrics["prepare_dispatch_seconds"],
            "forward_dispatch_seconds": train_metrics["forward_dispatch_seconds"],
            "backward_dispatch_seconds": train_metrics["backward_dispatch_seconds"],
            "batch_wall_seconds": train_metrics["batch_wall_seconds"],
        }
        history.append(row)
        _write_history(run_dir / "metrics.csv", history)
        print(
            f"[windfarm] epoch={epoch} loss={train_metrics['loss']:.6g} "
            f"val_volume={val_metrics.get('volume_mse', float('nan')):.6g} updates={update_count}"
        )
    _json_write(
        run_dir / "summary.json",
        {
            "best_val_volume_mse": best_metric,
            "best_epoch": best_epoch,
            "last_epoch": epochs,
            "update_count": update_count,
        },
    )
    return 0


__all__ = ["run_from_config"]
