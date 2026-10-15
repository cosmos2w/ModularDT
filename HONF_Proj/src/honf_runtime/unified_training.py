"""One case-neutral optimizer engine for matched HONF refinement campaigns.

Dataset adapters construct target-separated batches, native predictions, loss
terms, and physical metrics. This module owns visitation, accumulation,
optimizers, schedules, checkpointing, selection, and exact epoch-boundary
resume for both Thermal and Wind.
"""

from __future__ import annotations

import copy
import fcntl
import functools
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from .paths import PROJECT_ROOT
from .reproducibility import environment_snapshot, source_state_snapshot
from .run_layout import RunLayout, resolve_checkpoint


def _exclusive_training_run(function):
    """Hold one process-owned output lock across startup, fit and failure."""

    @functools.wraps(function)
    def locked(self, model, provider, output_dir, **kwargs):
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".training.lock").open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another training process owns this run directory.") from exc
            try:
                return function(self, model, provider, output, **kwargs)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return locked


@dataclass(frozen=True)
class SamplingKey:
    """Stable key for data and query streams, independent of global RNG state.

    ``legacy_packed_v1`` preserves the historical update/microbatch-keyed
    stream. ``case_epoch_v1`` keys each case/role stream by dataset, seed,
    epoch, physical identity and named stream, so changing packing does not
    change a case's native query sequence.
    """

    LEGACY_VERSION = "legacy_packed_v1"
    CASE_EPOCH_VERSION = "case_epoch_v1"
    SUPPORTED_VERSIONS = (LEGACY_VERSION, CASE_EPOCH_VERSION)

    seed: int
    epoch: int
    update_index: int
    microbatch_index: int
    stage: str
    arm: str
    sampling_version: str = LEGACY_VERSION
    dataset_id: str = ""

    def __post_init__(self) -> None:
        if self.sampling_version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported query sampling version {self.sampling_version!r}.")
        if not isinstance(self.dataset_id, str):
            raise TypeError("Query sampling dataset identity must be a string.")

    def seed_for(self, *coordinates: Any) -> int:
        """Return a stable seed under the selected historical or case stream."""

        if self.sampling_version == self.CASE_EPOCH_VERSION:
            identity = [self.sampling_version, self.dataset_id, self.seed, self.epoch, coordinates]
        else:
            identity = [self.seed, self.epoch, self.update_index, self.microbatch_index, self.stage, coordinates]

        payload = json.dumps(
            identity,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**32)

    def numpy_rng(self, *coordinates: Any, draw_start: int = 0) -> np.random.Generator:
        """Return a stream whose draws form a stable prefix across request sizes.

        With ``case_epoch_v1`` the stream is independent of arm, stage, update,
        and microbatch position. ``draw_start`` seeks to a prefix offset for
        callers that consume a stream in explicit blocks.
        """

        if type(draw_start) is not int or draw_start < 0:
            raise ValueError("Query stream draw_start must be a nonnegative integer.")
        generator = np.random.default_rng(self.seed_for(*coordinates))
        if draw_start and self.sampling_version == self.CASE_EPOCH_VERSION:
            generator.bit_generator.advance(draw_start)
        elif draw_start:
            raise ValueError("Explicit draw offsets require case_epoch_v1 sampling.")
        return generator

    def native_indices(
        self,
        population_size: int,
        count: int,
        physical_id: Any,
        role: str,
        stream: str,
        *,
        draw_start: int = 0,
        replace: bool = False,
    ) -> np.ndarray:
        """Draw native IDs from a per-case stream with prefix-stable semantics.

        Sampling without replacement uses the unique values of an iid integer
        stream in encounter order. This is exact sequential uniform sampling
        without replacement, and requesting a longer panel preserves its
        shorter panel as a prefix.
        """

        population_size, count = int(population_size), int(count)
        if population_size < 1 or count < 0 or draw_start < 0:
            raise ValueError("Native query sampling requires a nonempty population and nonnegative draw range.")
        if not replace and draw_start + count > population_size:
            raise ValueError("A without-replacement native query panel exceeds its population.")
        if count == 0:
            return np.empty(0, dtype=np.int64)
        rng = self.numpy_rng(physical_id, role, stream)
        needed = draw_start + count
        if replace:
            return rng.integers(population_size, size=needed, dtype=np.int64)[draw_start:]

        selected: list[np.ndarray] = []
        seen = np.empty(0, dtype=np.int64)
        selected_count = 0
        while selected_count < needed:
            remaining = needed - selected_count
            candidates = rng.integers(population_size, size=max(32, 2 * remaining), dtype=np.int64)
            if seen.size:
                candidates = candidates[~np.isin(candidates, seen, assume_unique=False)]
            unique, first = np.unique(candidates, return_index=True)
            ordered = unique[np.argsort(first)]
            if ordered.size:
                selected.append(ordered)
                seen = np.concatenate((seen, ordered))
                selected_count += int(ordered.size)
        return np.concatenate(selected)[:needed][draw_start:]

    def torch_generator(self, *coordinates: Any, device: torch.device | str = "cpu") -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed_for(*coordinates))
        return generator


@dataclass(frozen=True)
class TaskBatch:
    """A microbatch with an explicit inference/supervision boundary."""

    scene_inputs: Any
    receivers: Any
    targets: Any
    auxiliary: Any = None
    case_keys: tuple[Any, ...] = ()


@dataclass(frozen=True)
class LossTerm:
    """A summed differentiable numerator and its exact valid-element count."""

    numerator: torch.Tensor
    denominator: float | int | torch.Tensor
    weight: float = 1.0


@dataclass(frozen=True)
class ScheduleSpec:
    """Linear warmup, hold, then cosine decay on the absolute epoch clock."""

    peak_lr: float
    warmup_start_lr: float = 0.0
    warmup_epochs: int = 0
    hold_through_epoch: int = 1000
    total_epochs: int = 2500
    final_lr: float = 3.0e-6

    def __post_init__(self) -> None:
        if self.peak_lr <= 0 or self.final_lr <= 0 or self.warmup_start_lr < 0:
            raise ValueError("Learning rates must be finite and positive, except a zero warmup start.")
        if not all(math.isfinite(value) for value in (self.peak_lr, self.warmup_start_lr, self.final_lr)):
            raise ValueError("Learning rates must be finite.")
        if self.warmup_epochs < 0 or self.hold_through_epoch < self.warmup_epochs:
            raise ValueError("Schedule warmup/hold boundaries are inconsistent.")
        if self.total_epochs <= self.hold_through_epoch:
            raise ValueError("The schedule horizon must extend beyond its hold boundary.")
        if self.final_lr > self.peak_lr:
            raise ValueError("The final learning rate cannot exceed its peak.")

    def value(self, epoch: int) -> float:
        epoch = int(epoch)
        if epoch < 1:
            raise ValueError("Schedule epochs are one-based.")
        if self.warmup_epochs and epoch <= self.warmup_epochs:
            fraction = 1.0 if self.warmup_epochs == 1 else (epoch - 1) / (self.warmup_epochs - 1)
            return self.warmup_start_lr + fraction * (self.peak_lr - self.warmup_start_lr)
        if epoch <= self.hold_through_epoch:
            return self.peak_lr
        fraction = min(max((epoch - self.hold_through_epoch) /
                           (self.total_epochs - self.hold_through_epoch), 0.0), 1.0)
        return self.final_lr + 0.5 * (self.peak_lr - self.final_lr) * (1.0 + math.cos(math.pi * fraction))


@dataclass(frozen=True)
class OptimizerGroupSpec:
    """Named parameter selection and schedule for one AdamW parameter group."""

    name: str
    parameter_names: tuple[str, ...]
    schedule: ScheduleSpec
    weight_decay: float = 1.0e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8


@dataclass(frozen=True)
class EngineConfig:
    seed: int
    microbatch_cases: int
    effective_cases: int
    total_epochs: int = 2500
    warmup_epochs: int = 500
    open_through_epoch: int = 600
    soft_through_epoch: int = 800
    monitor_every: int = 100
    gradient_clip: float = 1.0
    deterministic_case_order: bool = True
    monitor_epochs: tuple[int, ...] = ()
    checkpoint_epochs: tuple[int, ...] | None = None
    latest_every: int | None = None
    curve_every: int | None = 10
    progress_enabled: bool = True
    sampling_version: str = SamplingKey.LEGACY_VERSION
    training_mode: str = "refinement"
    write_initial_artifacts: bool = False

    def __post_init__(self) -> None:
        if self.training_mode not in ("refinement", "joint"):
            raise ValueError("Training mode must be refinement or joint.")
        if type(self.write_initial_artifacts) is not bool:
            raise ValueError("Initial artifact control must be boolean.")
        if type(self.progress_enabled) is not bool:
            raise ValueError("Progress display control must be boolean.")
        if self.sampling_version not in SamplingKey.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported query sampling version {self.sampling_version!r}.")
        if min(self.microbatch_cases, self.effective_cases, self.total_epochs, self.monitor_every) < 1:
            raise ValueError("Batch sizes, horizon, and monitoring interval must be positive.")
        if self.effective_cases < self.microbatch_cases:
            raise ValueError("Effective batch size must be at least the microbatch size.")
        if (self.training_mode == "refinement" and not
                1 <= self.warmup_epochs <= self.open_through_epoch < self.soft_through_epoch < self.total_epochs):
            raise ValueError("Warmup/open/soft stages must be ordered inside the declared horizon.")
        if self.gradient_clip <= 0 or not math.isfinite(self.gradient_clip):
            raise ValueError("Gradient clipping must be positive and finite.")
        for name in ("monitor_epochs", "checkpoint_epochs"):
            epochs = getattr(self, name)
            if epochs is not None and (not isinstance(epochs, tuple)
                    or any(type(epoch) is not int or not 1 <= epoch <= self.total_epochs for epoch in epochs)
                    or tuple(sorted(set(epochs))) != epochs):
                raise ValueError("Declared monitoring/checkpoint epochs must be sorted unique horizon epochs.")
        for interval in (self.latest_every, self.curve_every):
            if interval is not None and (type(interval) is not int or interval < 1):
                raise ValueError("Latest-state and loss-curve intervals must be positive integers.")

    def stage_for_epoch(self, epoch: int) -> str:
        if self.training_mode == "joint":
            return "joint"
        if epoch <= self.warmup_epochs:
            return "warmup"
        if epoch <= self.open_through_epoch:
            return "open"
        if epoch <= self.soft_through_epoch:
            return "soft"
        return "hard"

    def temperature_for_epoch(self, epoch: int) -> float:
        """Anneal the smooth phase from 1.0 to 0.1 on absolute epochs."""

        if self.training_mode == "joint":
            return 1.0

        soft_start = self.open_through_epoch + 1
        if epoch <= soft_start:
            return 1.0
        if epoch >= self.soft_through_epoch:
            return 0.1
        fraction = (epoch - soft_start) / (self.soft_through_epoch - soft_start)
        return 1.0 + fraction * (0.1 - 1.0)


def _engine_config_payload(config: EngineConfig) -> dict[str, Any]:
    """Keep legacy identities unchanged when new optional output controls are unused."""
    payload = asdict(config)
    for name, default in (("monitor_epochs", ()), ("checkpoint_epochs", None),
                          ("latest_every", None), ("curve_every", 10),
                          ("progress_enabled", True),
                          ("sampling_version", SamplingKey.LEGACY_VERSION),
                          ("training_mode", "refinement"), ("write_initial_artifacts", False)):
        if payload[name] == default:
            payload.pop(name)
    # Older identities omitted both the default (None) and explicit disabled
    # curve control. Keep that serialized form stable across this output-only
    # default change.
    if payload.get("curve_every", object()) is None:
        payload.pop("curve_every")
    return payload


def _resume_identity_amendment(saved: Mapping[str, Any], current: Mapping[str, Any],
                               *, allow_microbatch_change: bool = False) -> dict[str, Any] | None:
    """Permit only an explicit microbatch amendment; all scientific bindings stay sealed."""
    if saved == current:
        return None
    message = "Checkpoint identity, engine config, selection or optimizer schedule differs."
    if not allow_microbatch_change:
        raise ValueError(message)
    previous = saved.get("engine_config", {})
    proposed = current.get("engine_config", {})
    before, after = previous.get("microbatch_cases"), proposed.get("microbatch_cases")
    if (type(before) is not int or type(after) is not int
            or not 1 <= after <= proposed.get("effective_cases", 0) or before == after):
        raise ValueError(message)
    amended = copy.deepcopy(saved)
    amended["engine_config"]["microbatch_cases"] = after
    if amended != current:
        raise ValueError(message)
    sampling_version = previous.get("sampling_version", SamplingKey.LEGACY_VERSION)
    trajectory_note = (
        "Case/epoch query identities are packing-independent; accumulation reduction order may change."
        if sampling_version == SamplingKey.CASE_EPOCH_VERSION else
        "Query seeds include microbatch index; future samples and FP32 accumulation can change."
    )
    return {"kind": "explicit_microbatch_continuation", "source_microbatch_cases": before,
            "microbatch_cases": after, "effective_cases": proposed["effective_cases"],
            "trajectory_note": trajectory_note}


def _content_pooling_source_amendment(
    saved: Mapping[str, Any], current: Mapping[str, Any], declaration: Mapping[str, Any],
    *, source_sha256: str, source_epoch: int, stop_after: int,
) -> dict[str, Any]:
    """Audit the one explicit Oct13 pooling arithmetic repair; seal all science.

    Ordinary resumes retain exact identity checks. This opt-in permits only
    audited imported-source hashes, with unchanged parameters, optimizer,
    sampling, data, physical law and schedule checked by the usual fit path.
    """
    message = "Invalid content-hypergraph numerical pooling continuation."
    recipe = saved.get("recipe", {})
    if (not isinstance(declaration, Mapping)
            or declaration.get("schema") != "content_hypergraph_pooling_numerical_repair_v1"
            or saved.get("workflow") != "content_hypergraph_joint_v1"
            or saved.get("model_family") != "honf_content_hypergraph"
            or recipe.get("dataset") != "wind" or recipe.get("architecture") != "A-H6"
            or recipe.get("protocol") != "full_train1000"
            or recipe.get("approved_stop_after") != 1000
            or declaration.get("approved_stop_after") != 1000
            or type(source_epoch) is not int or source_epoch < 100 or source_epoch >= 1000
            or source_epoch % 100 != 0 or declaration.get("source_epoch") != source_epoch
            or type(stop_after) is not int or not source_epoch < stop_after <= 1000
            or declaration.get("source_checkpoint_sha256") != source_sha256):
        raise ValueError(message)
    previous, proposed = saved.get("loaded_source"), current.get("loaded_source")
    if not isinstance(previous, Mapping) or not isinstance(proposed, Mapping):
        raise TypeError(message)
    if (declaration.get("saved_loaded_source_sha256") != previous.get("sha256")
            or declaration.get("current_loaded_source_sha256") != proposed.get("sha256")):
        raise ValueError(message)
    old_files, new_files = previous.get("files"), proposed.get("files")
    if not isinstance(old_files, Mapping) or not isinstance(new_files, Mapping):
        raise TypeError(message)
    if set(old_files) != set(new_files):
        raise ValueError(message)
    for binding in (previous, proposed):
        digest = hashlib.sha256(json.dumps(dict(binding["files"]), sort_keys=True,
                                           separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if digest != binding.get("sha256"):
            raise ValueError(message)
    allowed = {
        "src/honf_forward_core/content_hypergraph/organization.py",
        "src/honf_runtime/unified_training.py",
        "tools/content_hypergraph_train.py",
        "Case_WindFarm/src/windfarm/training/content_hypergraph_task.py",
    }
    changes = [{"path": path, "old_sha256": old_files[path], "new_sha256": new_files[path]}
               for path in sorted(old_files) if old_files[path] != new_files[path]]
    if (not changes or any(row["path"] not in allowed for row in changes)
            or "src/honf_forward_core/content_hypergraph/organization.py"
            not in {row["path"] for row in changes}
            or declaration.get("changes") != changes):
        raise ValueError(message)
    amended = copy.deepcopy(dict(saved))
    amended["loaded_source"] = copy.deepcopy(dict(proposed))
    if amended != current:
        raise ValueError(message)
    evidence_path = Path(str(declaration.get("evidence_path", "")))
    if not evidence_path.is_absolute() or not evidence_path.is_file():
        raise ValueError(message)
    evidence_bytes = evidence_path.read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != declaration.get("evidence_sha256"):
        raise ValueError(message)
    evidence = json.loads(evidence_bytes)
    checks = ("fp64_reference_pass", "extreme_finite_backward_pass", "checkpoint_forward_pass",
              "native_backward_probe_pass", "old_finite_forward_nonfinite_backward_reproduced")
    if (evidence.get("schema") != "content_hypergraph_stable_pooling_correctness_v1"
            or any(evidence.get(key) is not True for key in checks)
            or type(evidence.get("optimizer_updates")) is not int
            or evidence["optimizer_updates"] != 0):
        raise ValueError(message)
    return {"kind": "explicit_content_pooling_numerical_repair", "declaration": dict(declaration),
            "scientific_identity_unchanged": True,
            "trajectory_note": "Stable evaluation of the same typed normalized-sum equations changes FP32 arithmetic; failed and repeated work remain separately charged. No sample, weight, moment, physical-law or schedule reset."}


def _sampling_dataset_id(provider_identity: Mapping[str, Any]) -> str:
    """Build a small stable key from dataset and sealed TRAIN membership."""

    dataset = provider_identity.get("dataset", provider_identity.get("task", "unspecified"))
    native_identity = provider_identity.get("native_sampling_identity")
    if native_identity is not None:
        if (not isinstance(native_identity, Mapping)
                or native_identity.get("protocol") != "baseline_formal_v1"
                or native_identity.get("dataset") != dataset
                or not isinstance(native_identity.get("train_membership_fingerprint"), str)
                or len(native_identity["train_membership_fingerprint"]) != 64
                or any(char not in "0123456789abcdef" for char in native_identity["train_membership_fingerprint"])):
            raise ValueError("Formal native sampling identity requires a matching dataset and TRAIN fingerprint.")
        payload = json.dumps([str(dataset), native_identity["train_membership_fingerprint"]],
                             separators=(",", ":")).encode("utf-8")
        return f"{dataset}:{hashlib.sha256(payload).hexdigest()}"
    subset = next((provider_identity[name] for name in (
        "subset_manifest_sha256", "manifest_fingerprint", "manifest_sha256", "manifest",
        "train_rows_sha256", "training_case_ids_sha256", "train_case_ids_sha256",
    ) if name in provider_identity), None)
    if subset is None:
        # Test and small downstream providers may expose only case inventories.
        subset = next((provider_identity[name] for name in (
            "training_case_ids", "train_case_ids", "train_rows", "train_ids",
        ) if name in provider_identity), None)
    if subset is None:
        raise ValueError("case_epoch_v1 requires a stable dataset TRAIN-membership fingerprint.")
    payload = json.dumps([str(dataset), subset], sort_keys=True, default=str,
                         separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"{dataset}:{digest}"


def _branch_mathematical_route_amendment(
    saved: Mapping[str, Any], current: Mapping[str, Any], declaration: Mapping[str, Any],
) -> dict[str, Any]:
    """Permit one explicit gate-only child binding; never loosen exact resume.

    Models, moments, normalization, memberships, native objectives, query work,
    sampling, optimizer groups and schedules must keep their warmup bindings.
    The only permitted scientific intervention is the diagnosed compact gate.
    """
    required = {"kind", "from_gate_version", "to_gate_version", "gate_transition",
                "evidence_path", "evidence_sha256", "reason"}
    if (set(declaration) != required or declaration["kind"] != "compact_c1_gate_branch"
            or declaration["from_gate_version"] != "hard_v1"
            or declaration["to_gate_version"] != "compact_c1_v1"
            or list(declaration["gate_transition"]) != [0.35, 0.65]
            or not isinstance(declaration["reason"], str) or not declaration["reason"].strip()):
        raise ValueError("A mathematical branch requires the sealed compact C1 declaration.")
    evidence_path = Path(declaration["evidence_path"]).expanduser().resolve()
    evidence_bytes = evidence_path.read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != declaration["evidence_sha256"]:
        raise ValueError("Mathematical route diagnosis no longer matches its declared hash.")
    evidence = json.loads(evidence_bytes)
    if (evidence.get("TEST_read") is not False or evidence.get("optimizer_updates") != 0
            or evidence.get("solver_attempts") != 0
            or "TRAIN" not in str(evidence.get("partition", ""))
            or not any(row.get("consequential") is True for row in evidence.get("rows", []))):
        raise ValueError("The compact gate requires a consequential input-only TRAIN seam diagnosis.")
    amended = copy.deepcopy(dict(current))
    parent_provider = saved.get("provider_identity")
    child_provider = amended.get("provider_identity")
    if not isinstance(parent_provider, Mapping) or not isinstance(child_provider, dict):
        raise TypeError("A mathematical branch requires both sealed provider identities.")
    parent_recipe = parent_provider.get("resolved_recipe")
    child_recipe = child_provider.get("resolved_recipe")
    if isinstance(parent_recipe, Mapping) and isinstance(child_recipe, dict):
        if (parent_recipe.get("gate_version", "hard_v1") != "hard_v1"
                or child_recipe.get("gate_version") != "compact_c1_v1"
                or list(child_recipe.get("gate_transition", ())) != [0.35, 0.65]):
            raise ValueError("The child recipe must change only hard_v1 to the sealed compact C1 gate.")
        for name in ("gate_version", "gate_transition", "recipe_id", "recipe_sha256"):
            if name in parent_recipe:
                child_recipe[name] = copy.deepcopy(parent_recipe[name])
            else:
                child_recipe.pop(name, None)
        child_provider["resolved_recipe_sha256"] = parent_provider["resolved_recipe_sha256"]
        for name in ("wind_recipe_id", "wind_recipe_sha256"):
            if name in saved:
                amended[name] = saved[name]
    else:
        if (parent_provider.get("gate_version", "hard_v1") != "hard_v1"
                or child_provider.get("gate_version") != "compact_c1_v1"
                or list(child_provider.get("gate_transition", ())) != [0.35, 0.65]):
            raise ValueError("The child provider must declare the sealed compact C1 gate.")
        for name in ("gate_version", "gate_transition"):
            if name in parent_provider:
                child_provider[name] = copy.deepcopy(parent_provider[name])
            else:
                child_provider.pop(name, None)
    if amended != saved:
        raise ValueError("A mathematical route branch cannot change any other experiment binding.")
    return {**dict(declaration), "evidence_path": str(evidence_path),
            "preserved_contract": "exact warmup weights, moments, data, objective, query streams and absolute schedules"}


def _branch_joint_locality_amendment(
    saved: Mapping[str, Any], current: Mapping[str, Any], declaration: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a shape-compatible locality child without relaxing exact resume.

    Only the optional locality scalar can change. All dataset, objective,
    sampling, initialization, optimizer and absolute schedule identities must
    remain equal. The changed function resets selection, not learning age.
    """
    required = {'kind', 'source_epoch', 'review_stop', 'from_strength', 'to_strength',
                'evidence_path', 'evidence_sha256', 'reason'}
    if (set(declaration) != required
            or declaration['kind'] != 'registered_locality_prior_branch'
            or type(declaration['source_epoch']) is not int
            or type(declaration['review_stop']) is not int
            or declaration['source_epoch'] < 100
            or declaration['source_epoch'] % 100
            or declaration['review_stop'] <= declaration['source_epoch']
            or declaration['from_strength'] != 0.0
            or declaration['to_strength'] != 1.0
            or not isinstance(declaration['reason'], str) or not declaration['reason'].strip()):
        raise ValueError('Joint locality revision requires the sealed 0-to-1 branch declaration.')
    if (saved.get('engine_config', {}).get('training_mode') != 'joint'
            or current.get('engine_config', {}).get('training_mode') != 'joint'
            or saved.get('recipe', {}).get('mode') != 'J-H'
            or current.get('recipe', {}).get('mode') != 'J-H'
            or saved.get('recipe', {}).get('formal_full') is not False
            or current.get('recipe', {}).get('formal_full') is not False):
        raise ValueError('A joint locality child requires matched J-H development identities.')
    evidence_path = Path(declaration['evidence_path']).expanduser().resolve()
    evidence_bytes = evidence_path.read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != declaration['evidence_sha256']:
        raise ValueError('Joint revision evidence no longer matches its sealed hash.')
    evidence = json.loads(evidence_bytes)
    if (evidence.get('solver_attempts') != 0 or evidence.get('WindTEST_targets') != 'locked'
            or evidence.get('prefit_optimizer_updates') != 0
            or evidence.get('change') != 'registered_gaussian_attention_prior'
            or evidence.get('source_epoch') != declaration['source_epoch']
            or evidence.get('stopping_boundary') != declaration['review_stop']):
        raise ValueError('Joint revision requires a pre-fit diagnosis with fixed scope and stopping boundary.')
    amended = copy.deepcopy(dict(current))
    paths = [('recipe',), ('provider_identity', 'model_config'),
             ('provider_identity', 'standalone_model_config', 'core')]
    changed = []
    for path in paths:
        before, after = saved, amended
        for part in path:
            before = before.get(part) if isinstance(before, Mapping) else None
            after = after.get(part) if isinstance(after, Mapping) else None
        if before is None and after is None:
            continue
        if not isinstance(before, Mapping) or not isinstance(after, dict):
            raise TypeError('Joint revision model configuration binding is incomplete.')
        if (before.get('locality_prior_strength', 0.0) != declaration['from_strength']
                or after.get('locality_prior_strength', 0.0) != declaration['to_strength']):
            raise ValueError('Joint locality configurations differ from the declared scalar change.')
        if 'locality_prior_strength' in before:
            after['locality_prior_strength'] = before['locality_prior_strength']
        else:
            after.pop('locality_prior_strength', None)
        changed.append('.'.join((*path, 'locality_prior_strength')))
    if amended != saved or len(changed) != 2:
        raise ValueError('A joint core revision cannot change any other experiment binding.')
    return {**dict(declaration), 'evidence_path': str(evidence_path), 'changed_identity_paths': changed,
            'preserved_contract': 'weights, named moments, RNG, case/query streams, losses and absolute schedule',
            'selection_reset': 'only checkpoints evaluated under the new function can be selected'}


def _diagnostic_age_branch_amendment(
    saved: Mapping[str, Any], current: Mapping[str, Any], declaration: Mapping[str, Any],
    *, source_sha256: str, case_keys: Sequence[Any], stop_after: int,
) -> dict[str, Any]:
    """Seal a bounded objective diagnostic without relaxing exact resume.

    The child retains the parent's tensors, named moments, native contract and
    absolute schedules. Only an explicit diagnostic declaration and monitoring
    cadence may differ. Its shorter fitting inventory remains within the
    parent's TRAIN exposure, which is retained separately from child visits.
    """
    required = {"kind", "source_epoch", "source_checkpoint_sha256", "review_stop",
                "review_epochs", "fit_case_keys", "scientific_update_limit",
                "evidence_path", "evidence_sha256", "reason", "loss_weight_overrides"}
    if (set(declaration) != required
            or declaration["kind"] != "reference_anchored_objective_diagnostic"
            or declaration["source_checkpoint_sha256"] != source_sha256
            or type(declaration["source_epoch"]) is not int
            or type(declaration["scientific_update_limit"]) is not int
            or not 1 <= declaration["scientific_update_limit"] <= 400
            or not isinstance(declaration["reason"], str) or not declaration["reason"].strip()):
        raise ValueError("A diagnostic age branch requires its explicit hash-bound bounded declaration.")
    reviews = declaration["review_epochs"]
    if (not isinstance(reviews, (tuple, list)) or not reviews
            or any(type(epoch) is not int or epoch <= declaration["source_epoch"] for epoch in reviews)
            or list(reviews) != sorted(set(reviews))
            or declaration["review_stop"] != reviews[-1] or stop_after not in reviews):
        raise ValueError("Diagnostic stops must be predeclared review ages after the source age.")
    keys = list(case_keys)
    if (not keys or len(set(keys)) != len(keys)
            or set(keys) != set(declaration["fit_case_keys"])):
        raise ValueError("Diagnostic case visitation differs from its frozen fitting inventory.")
    parent_provider = saved.get("provider_identity", {})
    child_provider = copy.deepcopy(current.get("provider_identity", {}))
    diagnostic = child_provider.pop("diagnostic", None)
    if (child_provider != parent_provider or not isinstance(diagnostic, Mapping)
            or set(diagnostic) != {"fit_case_keys", "loss_weight_overrides"}
            or set(diagnostic.get("fit_case_keys", ())) != set(keys)):
        raise ValueError("A diagnostic child must preserve every parent provider binding outside its declaration.")
    overrides = declaration["loss_weight_overrides"]
    if (not isinstance(overrides, Mapping)
            or any(not isinstance(name, str) or not name
                   or type(value) not in (int, float)
                   or not math.isfinite(value) or value < 0
                   for name, value in overrides.items())
            or dict(overrides) != diagnostic["loss_weight_overrides"]):
        raise ValueError("Diagnostic objective overrides must match their finite, explicit declaration.")
    # Only the child run label may change outside the explicitly checked
    # provider/objective and monitoring presentation. Bind initialization,
    # data, workflow, source recipe and sampling fields without an allow-all
    # escape hatch for additional top-level controls.
    checked_fields = {"provider_identity", "engine_config", "optimizer_schedule_contract",
                      "selection_policy", "run", "run_identity"}
    parent_other = {name: value for name, value in saved.items() if name not in checked_fields}
    child_other = {name: value for name, value in current.items() if name not in checked_fields}
    if parent_other != child_other:
        raise ValueError("Diagnostic initialization, workflow, data and sampling identity must remain unchanged.")
    parent_cases = next((parent_provider[name] for name in
                         ("training_case_ids", "train_case_ids", "train_rows")
                         if name in parent_provider), None)
    if parent_cases is None and parent_provider.get("dataset") == "WindFarm":
        # Wind's immutable native view orders the three direction rows per layout.
        if parent_provider.get("directions_deg") != [270.0, 285.0, 300.0]:
            raise ValueError("Diagnostic Wind membership requires the original three-direction row binding.")
        parent_cases = [3 * int(layout) + direction
                        for layout in parent_provider.get("train_layout_indices", ())
                        for direction in range(3)]
    if parent_cases is None or not set(keys) <= set(parent_cases):
        raise ValueError("Every diagnostic fitting case must belong to the parent's declared TRAIN exposure.")
    old_engine = dict(saved.get("engine_config", {}))
    new_engine = dict(current.get("engine_config", {}))
    for name in ("monitor_every", "monitor_epochs", "checkpoint_epochs", "latest_every",
                 "curve_every", "write_initial_artifacts"):
        old_engine.pop(name, None)
        new_engine.pop(name, None)
    if (old_engine != new_engine
            or saved.get("optimizer_schedule_contract") != current.get("optimizer_schedule_contract")
            or saved.get("selection_policy") != current.get("selection_policy")):
        raise ValueError("Diagnostic batches, sampler, native stages, selection and absolute optimizer schedules must remain unchanged.")
    updates_per_epoch = math.ceil(len(keys) / int(new_engine["effective_cases"]))
    requested_updates = updates_per_epoch * (reviews[-1] - declaration["source_epoch"])
    if requested_updates > declaration["scientific_update_limit"]:
        raise ValueError("Diagnostic review horizon exceeds its explicit scientific update limit.")
    evidence_path = Path(declaration["evidence_path"]).expanduser().resolve()
    evidence_bytes = evidence_path.read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != declaration["evidence_sha256"]:
        raise ValueError("Diagnostic TRAIN evidence changed after its choice was sealed.")
    evidence = json.loads(evidence_bytes)
    if (evidence.get("selection_partition") != "TRAIN"
            or evidence.get("WindTEST_targets") != "locked"
            or evidence.get("solver_attempts") != 0):
        raise ValueError("Diagnostic selection requires TRAIN-only evidence with locked TEST and no solves.")
    allowed_overrides = evidence.get("allowed_loss_weight_overrides")
    if (not isinstance(allowed_overrides, list) or not allowed_overrides
            or not all(isinstance(item, Mapping) for item in allowed_overrides)
            or dict(overrides) not in allowed_overrides):
        raise ValueError("Diagnostic objective overrides were not selected in the sealed TRAIN evidence.")
    return {**copy.deepcopy(dict(declaration)), "evidence_path": str(evidence_path),
            "updates_per_epoch": updates_per_epoch, "planned_optimizer_updates": requested_updates,
            "parent_train_exposure_retained": True,
            "child_history_scope": "only new diagnostic visits and updates; inherited fitting exposure retained in parent identity"}


def _render_loss_curves(history: Sequence[Mapping[str, Any]], output: Path, field_metric: str,
                        loss_metadata: Mapping[str, Mapping[str, str]] | None = None) -> None:
    """Write grouped TRAIN/VALIDATION curves and preserve historical metric meaning."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    joint_run = any(row.get("phase") == "joint" for row in history)

    def objective_at(row: Mapping[str, Any]) -> Mapping[str, Any]:
        objective = row.get("validation_objective")
        if not isinstance(objective, Mapping):
            validation = row.get("validation", {})
            objective = validation.get("loss_objective") if isinstance(validation, Mapping) else None
        return objective if isinstance(objective, Mapping) else {}

    def term_group(name: str) -> int:
        if loss_metadata and name in loss_metadata:
            return {"native_prediction": 0, "physics": 1, "routing": 2}.get(
                loss_metadata[name].get("panel", "native_prediction"), 0)
        if name in {"operator_residual", "response"}:
            return 1
        if name in {"base_loss", "router_importance_loss", "expected_work"}:
            return 2
        if name == "reconstruction" or name.startswith("native_role/"):
            return 0
        return 2

    def term_label(name: str) -> str:
        if loss_metadata and name in loss_metadata:
            return loss_metadata[name].get("label", name.replace("_", " "))
        if name == "reconstruction":
            return "Field prediction loss"
        if name == "response":
            return "Response prediction loss"
        if name == "operator_residual":
            return "Heat-balance residual loss"
        if name == "base_loss":
            return "Coarse-vs-fine prediction loss"
        if name == "router_importance_loss":
            return "Router probability loss"
        if name == "expected_work":
            return "Expected selected-read penalty"
        if name.startswith("native_role/"):
            role = name.split("/", 1)[1]
            role_labels = {
                "downstream_envelope": "Downstream-region",
                "hub_slab": "Hub-height slab",
                "near_turbine": "Near-turbine hub-slab",
                "volume": "Full-volume",
                "background": "Background-region",
            }
            return f"{role_labels.get(role, role.replace('_', ' ').title())} velocity loss"
        return name.replace("_", " ")

    wind_role_regions = {
        "volume": "All sampled native-volume cells; D is the reference rotor diameter.",
        "hub_slab": "Native cells within 0.5D vertically of hub height: |z - z_hub| <= 0.5D; D is the reference rotor diameter.",
        "downstream_envelope": (
            "Geometry-defined sampling region from each active hub: 0 < dx <= 10D and "
            "sqrt(dy^2 + dz^2) <= 1.5D; D is the reference rotor diameter. This is not a physical wake label."
        ),
        "near_turbine": (
            "Native cells within 1.5D horizontal x-y distance of an active hub and within the hub-height slab; "
            "D is the reference rotor diameter."
        ),
        "background": "Complement of the downstream region and hub-height slab; D is the reference rotor diameter.",
    }

    def term_definition(name: str) -> dict[str, str]:
        if loss_metadata and name in loss_metadata:
            return dict(loss_metadata[name])
        if name == "reconstruction":
            return {
                "panel": "native_prediction",
                "label": term_label(name),
                "unit": "dimensionless weighted scaled squared-error contribution",
                "formula": "Per case: mean of the scaled fluid, surface, and material mean-square errors, plus 0.05 times scaled q-proxy mean-square error. Adaptive TRAIN may blend 75% main-path loss and 25% already-computed fine-path loss; hard VALIDATION uses the main path only.",
                "weight": "Overall provider weight 1.0; q-proxy coefficient 0.05.",
            }
        if name == "response":
            return {
                "panel": "physics",
                "label": term_label(name),
                "unit": "dimensionless weighted scaled squared-error contribution",
                "formula": "Mean-square response prediction error on eligible TRAIN response targets; no response target is available in exposed VALIDATION.",
                "weight": "Recipe response_coefficient.",
            }
        if name == "operator_residual":
            return {
                "panel": "physics",
                "label": term_label(name),
                "unit": "dimensionless weighted normalized squared-residual contribution",
                "formula": "Mean-square discrete heat-balance residual normalized by the fixed TRAIN temperature scale; stored residual supervision is TRAIN-only.",
                "weight": "Recipe operator_coefficient.",
            }
        if name == "base_loss":
            return {
                "panel": "routing",
                "label": term_label(name),
                "unit": "dimensionless weighted prediction squared-error contribution",
                "formula": "Mean-square error between coarse-path and fine-path predictions, normalized by the fixed TRAIN scale; omitted on hard VALIDATION because no extra fine reference is computed.",
                "weight": "Configured approximation coefficient 0.1 divided by the squared fixed TRAIN scale.",
            }
        if name == "router_importance_loss":
            return {
                "panel": "routing",
                "label": term_label(name),
                "unit": "dimensionless weighted probability squared-error contribution",
                "formula": "Mean-square router probability error against detached residual-importance targets; omitted on hard VALIDATION because target construction requires fine-detail values.",
                "weight": "0.01.",
            }
        if name == "expected_work":
            return {
                "panel": "routing",
                "label": term_label(name),
                "unit": "dimensionless weighted expected selected-read fraction",
                "formula": "Expected selected fine reads divided by active receiver-source pairs, multiplied by the calibrated TRAIN penalty weight.",
                "weight": "TRAIN-calibrated coefficient, capped by the provider configuration.",
            }
        if name.startswith("native_role/"):
            role = name.split("/", 1)[1]
            role_region = wind_role_regions.get(role, "fixed native role sample; see task recipe")
            return {
                "panel": "native_prediction",
                "label": term_label(name),
                "unit": "dimensionless weighted normalized velocity squared-error contribution",
                "formula": "Mean-square xyz velocity error divided by the squared fixed TRAIN scale s_role^2 (s_role is in m/s); hard VALIDATION uses the deployed main path without fine-path replay. "
                          "Adaptive TRAIN blends 75% main and 25% fine-path losses outside warmup.",
                "weight": "0.2 per native role.",
                "role_region": role_region,
            }
        return {
            "panel": "routing",
            "label": term_label(name),
            "unit": "provider-defined weighted loss contribution",
            "formula": "Provider objective term; see run configuration for its specific prediction, residual, or routing definition.",
            "weight": "Provider-defined; validation weights by epoch are listed separately.",
        }

    def value_at(row: Mapping[str, Any], name: str, source: str) -> float | None:
        if source == "train":
            value = row.get("train_losses", {}).get(name)
        else:
            value = objective_at(row).get("terms", {}).get(name)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def plot_series(
        axis: Any, epochs: Sequence[Any], values: Sequence[Any], *, label: str,
        color: Any = None, linestyle: str = "-", marker: str = ".",
    ) -> int:
        numeric = np.asarray(values, dtype=np.float64)
        invalid = ~np.isfinite(numeric) | (numeric <= 0.0)
        masked_count = int(invalid.sum())
        axis.set_yscale("log")
        if masked_count < len(values):
            axis.plot(epochs, np.ma.masked_where(invalid, numeric), linestyle=linestyle,
                      marker=marker, label=label, color=color)
        elif len(values):
            axis.set_ylim(1.0, 10.0)
        return masked_count

    names: set[str] = set()
    for row in history:
        names.update(str(name) for name in row.get("train_losses", {}))
        names.update(str(name) for name in objective_at(row).get("terms", {}))
    all_names = sorted(names)
    figure = Figure(figsize=(15.5, 8.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3, squeeze=False).ravel()
    colors = {name: f"C{index % 10}" for index, name in enumerate(all_names)}
    masked_count_total = 0

    def set_panel_title(axis: Any, heading: str, explanation: str, *, fontsize: float = 9.5) -> None:
        axis.set_title(f"{heading}\n{explanation}", fontsize=fontsize, linespacing=1.1, pad=5)

    def plot_term_group(axis: Any, group_index: int) -> None:
        nonlocal masked_count_total
        group_names = [name for name in all_names if term_group(name) == group_index]
        for name in group_names:
            color = colors[name]
            for source, series_label in (("train", "TRAIN"), ("validation", "HELD-OUT VALIDATION")):
                points = [(int(row["epoch"]), value_at(row, name, source)) for row in history
                          if type(row.get("epoch")) is int and value_at(row, name, source) is not None]
                if points:
                    masked_count_total += plot_series(
                        axis, [item[0] for item in points], [item[1] for item in points],
                        label=f"{series_label}: {term_label(name)}", color=color,
                        linestyle="-" if source == "train" else "--",
                        marker="." if source == "train" else "o",
                    )
        axis.set_xlabel("Completed epoch")
        axis.grid(alpha=0.25, which="both")
        if axis.lines:
            axis.legend(fontsize=7.0, loc="best", frameon=False)
        else:
            axis.set_yscale("log")
            axis.set_ylim(1.0, 10.0)
            empty_message = {
                0: "No prediction losses saved",
                1: "No physics or response losses saved",
                2: "No approximation or routing losses saved",
            }.get(group_index, "No losses saved for this group")
            axis.text(0.5, 0.5, empty_message, ha="center", va="center",
                      transform=axis.transAxes)

    # The objective subtotal compares exactly the terms recorded on both sides.
    total_axis = axes[0]
    comparable_rows = []
    for row in history:
        objective = objective_at(row)
        if objective.get("matched_terms") and type(row.get("epoch")) is int:
            comparable_rows.append((int(row["epoch"]), objective))
    if comparable_rows:
        for key, label, color, style, marker in (
            ("training_matched_total", "TRAIN shared-name sum", "C0", "-", "."),
            ("validation_matched_total", "HELD-OUT VALIDATION shared-name sum", "C1", "--", "o"),
        ):
            points = [(epoch, objective.get(key)) for epoch, objective in comparable_rows
                      if objective.get(key) is not None]
            if points:
                masked_count_total += plot_series(
                    total_axis, [point[0] for point in points], [point[1] for point in points],
                    label=label, color=color, linestyle=style, marker=marker,
                )
        if total_axis.lines:
            total_axis.legend(fontsize=8, loc="best", frameon=False)
    else:
        total_axis.set_yscale("log")
        total_axis.set_ylim(1.0, 10.0)
        no_validation_objective = all(not objective_at(row).get("terms") for row in history)
        total_axis.text(
            0.5, 0.5,
            ("Held-out loss was not recorded in this history."
             if no_validation_objective else "No shared held-out loss terms were saved."),
            ha="center", va="center", transform=total_axis.transAxes,
        )
    adaptive_run = any(str(row.get("arm", "")) == "adaptive_detail" for row in history)
    train_validation_path_note = (
        "TRAIN: mean of per-update means\nVALIDATION: pooled valid-element mean\n"
        "All physical heads and shared blocks train jointly\nVALIDATION: same deployed joint path"
        if joint_run else
        "TRAIN: mean of per-update means\nVALIDATION: pooled valid-element mean\n"
        "Adaptive TRAIN: 25% fine-path loss replay\nVALIDATION: hard main path"
        if adaptive_run else
        "TRAIN: mean of per-update means\nVALIDATION: pooled valid-element mean\n"
        "TRAIN: phase-specific optimization path\nVALIDATION: hard main path"
    )
    set_panel_title(
        total_axis, "Shared-name loss summaries",
        f"{train_validation_path_note}",
        fontsize=8.2,
    )
    total_axis.set_xlabel("Completed epoch")
    total_axis.set_ylabel("Shared-name weighted loss sum (dimensionless)")
    total_axis.grid(alpha=0.25, which="both")

    native_axis, physics_axis, selector_axis, routing_axis, work_axis = axes[1:]
    plot_term_group(native_axis, 0)
    wind_role_names = [name for name in all_names if name.startswith("native_role/")]
    if wind_role_names:
        set_panel_title(
            native_axis, "Wind velocity losses by sampled region",
            "Hub slab: |z-z_hub|<=0.5D; downstream: 0<dx<=10D\n"
            "Downstream yz radius<=1.5D; D = reference rotor diameter",
            fontsize=8.2,
        )
    else:
        set_panel_title(native_axis, "Prediction losses", "Prediction vs reference; scaled squared error")
    native_axis.set_ylabel("Weighted loss contribution (dimensionless)")
    plot_term_group(physics_axis, 1)
    set_panel_title(physics_axis, "Physics and response losses",
                    "Discrete stored-flow residual; response prediction")
    physics_axis.set_ylabel("Weighted loss contribution (dimensionless)")
    plot_term_group(routing_axis, 2)
    set_panel_title(
        routing_axis, "Other provider objectives" if joint_run else "Approximation and routing losses",
        "Provider-defined objectives; no detail-gate curriculum" if joint_run else
        "Coarse/fine prediction mismatch; router probability error vs residual-importance target\n"
        "Penalty on expected selected fine reads",
        fontsize=8.9,
    )
    routing_axis.set_ylabel("Weighted loss contribution (dimensionless)")

    selector_points = []
    for row in history:
        validation = row.get("validation", {})
        value = validation.get(field_metric) if isinstance(validation, Mapping) else None
        try:
            selector_points.append((int(row["epoch"]), float(value)))
        except (KeyError, TypeError, ValueError):
            continue
    if selector_points:
        selector_label = (f"EXPOSED DEV: {field_metric}" if joint_run else
                          "HELD-OUT VALIDATION: Held-out field prediction error"
                          if field_metric == "field_score" else f"HELD-OUT VALIDATION: {field_metric}")
        masked_count_total += plot_series(
            selector_axis, [point[0] for point in selector_points], [point[1] for point in selector_points],
            label=selector_label, color="C0", linestyle="-", marker="o",
        )
        selector_axis.legend(fontsize=8, loc="best", frameon=False)
    else:
        selector_axis.set_yscale("log")
        selector_axis.set_ylim(1.0, 10.0)
        selector_axis.text(0.5, 0.5, "No saved validation selector metric", ha="center", va="center",
                           transform=selector_axis.transAxes)
    selector_explanation = ("Normalized field MSE for checkpoint selection; not a loss"
                            if field_metric == "field_score"
                            else "Checkpoint selector, shown separately from loss")
    set_panel_title(selector_axis, "Exposed DEV checkpoint selector" if joint_run else
                    "Held-out field prediction error" if field_metric == "field_score"
                    else "Held-out checkpoint metric", selector_explanation)
    selector_axis.set_xlabel("Completed epoch")
    selector_axis.set_ylabel(
        "Normalized mean-square field error (dimensionless)" if field_metric == "field_score"
        else "Provider-defined metric units")
    selector_axis.grid(alpha=0.25, which="both")

    work_definitions = {
        "active_pairs": "Active receiver-source pairs processed",
        "fine_rows": "Fine detail-head rows executed",
        "selected_detail_rows": "Rows selected by the detail route",
        "cheap_rows": "Cheap-path rows evaluated",
        "gate_rows": "Router gate rows evaluated",
        "operator_rows": "Discrete thermal residual rows evaluated",
        "node_edge_rows": "Node-to-edge donor rows executed",
        "receiver_edge_rows": "Receiver-to-edge reads executed",
        "source_read_rows": "Physical source reads executed",
    }
    work_keys = sorted({key for row in history for key in row.get("work_counts", {})}
                       & set(work_definitions))
    for index, key in enumerate(work_keys):
        points = []
        for row in history:
            try:
                points.append((int(row["epoch"]), float(row["work_counts"][key])))
            except (KeyError, TypeError, ValueError):
                continue
        if points:
            masked_count_total += plot_series(
                work_axis, [point[0] for point in points], [point[1] for point in points],
                label=f"{work_definitions[key]} (TRAIN)", color=f"C{index % 10}",
            )
    if work_keys and work_axis.lines:
        work_axis.legend(fontsize=7.0, loc="best", frameon=False)
    else:
        work_axis.set_yscale("log")
        work_axis.set_ylim(1.0, 10.0)
        work_axis.text(0.5, 0.5, "No recorded work counters", ha="center", va="center",
                       transform=work_axis.transAxes)
    set_panel_title(work_axis, "Recorded TRAIN work counters",
                    "Counter sums; not measured executor time or savings")
    work_axis.set_xlabel("Completed epoch")
    work_axis.set_ylabel("Provider counter sum (rows or pairs / epoch)")
    work_axis.grid(alpha=0.25, which="both")
    if masked_count_total:
        routing_axis.text(
            0.02, 0.02,
            f"{masked_count_total} nonpositive/nonfinite values masked for log display; history unchanged.",
            ha="left", va="bottom", fontsize=8, transform=routing_axis.transAxes,
        )
    arms = sorted({str(row["arm"]) for row in history if row.get("arm")})
    case_counts = sorted({int(row["case_visits"]) for row in history if row.get("case_visits") is not None})
    scope = ""
    if len(case_counts) == 1:
        scope = f"{case_counts[0]} TRAIN cases/epoch"
    elif case_counts:
        scope = f"{case_counts[0]}–{case_counts[-1]} TRAIN cases/epoch"
    title_parts = [", ".join(arms) if arms else "training"]
    if scope:
        title_parts.append(scope)
    title_parts.append("TRAIN optimization vs exposed DEV joint inference" if joint_run
                       else "TRAIN optimization vs exposed VALIDATION hard inference")
    figure.suptitle(" | ".join(title_parts), fontsize=13)
    epochs = sorted({int(row["epoch"]) for row in history if type(row.get("epoch")) is int})
    if epochs:
        limits = (epochs[0] - 0.5, epochs[-1] + 0.5) if len(epochs) == 1 else (epochs[0], epochs[-1])
        for axis in axes:
            axis.set_xlim(*limits)

    validation_weights: dict[str, dict[str, float]] = {}
    for row in history:
        objective = objective_at(row)
        try:
            epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        for name, weight in objective.get("term_weights", {}).items():
            validation_weights.setdefault(str(name), {})[str(epoch)] = float(weight)
    metadata = {
        "figure": "loss_curves",
        "layout": "2x3 grouped overview",
        "panel_definitions": {
        "shared_name_loss_summary": "Sum same-named weighted loss terms recorded in both TRAIN and hard VALIDATION; this summary is not directly comparable because TRAIN uses a mean of macro-update means and validation pools valid elements. Adaptive TRAIN may include 25% fine-path loss replay; validation uses hard main-path loss only.",
            "native_prediction": "Visible note: prediction versus reference, scaled squared error. Validation uses the deployed hard-route main prediction and omits full-detail replay.",
            "physics": "Visible note: discrete heat-balance residual and response prediction losses. Thermal response and residual targets are TRAIN-only when absent from the exposed validation batch.",
            "field_selector": f"Visible note: normalized mean-square field error for {field_metric}; held-out selector used for checkpoint selection, shown separately from loss.",
            "routing": "Visible note: coarse-versus-fine prediction error, router probability error against residual-importance targets, and penalty on expected selected fine reads. Fine references and importance targets are unavailable on hard validation.",
            "train_work": "Visible note: TRAIN counter sums only, not measured executor time or savings. These counts do not establish sparse executor savings. Training includes full-detail replay where configured; validation work is not recorded.",
        },
        "term_definitions": {name: term_definition(name) for name in all_names},
        "validation_objective_weights_by_epoch": validation_weights,
        "recorded_value_convention": "TRAIN curves show weighted macro-update mean loss contributions. Validation curves apply the recorded provider weight after summing numerators and valid-element denominators over exposed validation batches.",
        "selector_metric_unit": "dimensionless normalized mean-square field error for field_score; otherwise provider-defined units",
        "training_loss_weight_note": "Each term definition lists its formula and fixed or calibrated provider weight. Validation objective weights by epoch are listed separately. TRAIN histories record already-weighted contributions.",
        "training_reduction": "Arithmetic mean of weighted macro-update means; each optimizer update has equal weight, including a partial final update.",
        "validation_reduction": "Pooled numerator divided by pooled valid-element denominator, then multiplied by provider term weight.",
        "prediction_path_comparison": "TRAIN follows the phase-specific optimization path and may include 25% fine-path loss replay; held-out loss uses deployed hard inference on the main prediction only.",
        "directly_comparable": False,
        "work_counter_definitions": {key: work_definitions[key] for key in work_keys},
        "validation_scope": "exposed held-out validation with hard inference; not an independent TEST result",
    }
    if joint_run:
        metadata["validation_scope"] = "fixed exposed DEV joint inference; not an independent TEST result"
        metadata["prediction_path_comparison"] = "TRAIN and DEV execute the same joint model. TRAIN averages macro-update means across updates; DEV pools valid elements at fixed review weights."
        metadata["panel_definitions"].update({
            "shared_name_loss_summary": "Same-named provider physical objectives on TRAIN and exposed DEV. Their reductions differ; this is not a directly comparable objective.",
            "native_prediction": "Provider-defined joint physical field losses against stored references, in TRAIN-fitted scales.",
            "field_selector": f"Exposed DEV {field_metric} used for checkpoint selection; not independent TEST evidence.",
            "routing": "Other provider-defined objectives; this joint family has no B/F gate or routing curriculum.",
            "train_work": "Executed primitive counter sums; counts alone do not establish sparse executor savings.",
        })
    _atomic_json(output / "loss_curves_metadata.json", metadata)
    for suffix in ("pdf", "png"):
        destination = output / f"loss_curves.{suffix}"
        temporary = output / f".loss_curves.{os.getpid()}.tmp.{suffix}"
        figure.savefig(temporary, format=suffix, dpi=140)
        os.replace(temporary, destination)
    figure.clear()

@dataclass(frozen=True)
class SelectionPolicy:
    field_metric: str = "field_score"
    response_guard_metric: str | None = None
    maximum_response_ratio: float = 1.10

    def __post_init__(self) -> None:
        if not self.field_metric:
            raise ValueError("A field metric is required for checkpoint selection.")
        if not math.isfinite(self.maximum_response_ratio) or self.maximum_response_ratio <= 0:
            raise ValueError("The response guard must be a positive finite ratio.")


class TaskProvider(Protocol):
    """Dataset-owned data, native physics, objective and metric contract."""

    def identity_payload(self) -> Mapping[str, Any]: ...

    def epoch_cases(self, epoch: int, seed: int) -> Sequence[Any]: ...

    def make_batch(self, case_keys: Sequence[Any], key: SamplingKey) -> TaskBatch: ...

    def loss_denominators(
        self, batches: Sequence[TaskBatch], phase: str, arm: str,
    ) -> Mapping[str, float]: ...

    def make_scene(self, scene_inputs: Any) -> Any: ...

    def predict_native(
        self, model: nn.Module, scene: Any, receivers: Any, execution_mode: str, phase: str,
        epoch: int, temperature: float,
    ) -> tuple[Any, Any]: ...

    def loss_terms(
        self, predictions: Any, targets: Any, phase: str, auxiliary_state: Any,
    ) -> Mapping[str, LossTerm]: ...

    def validation_batches(self) -> Iterable[TaskBatch]: ...

    def validation_metrics(self, predictions: Any, targets: Any, auxiliary_state: Any) -> Mapping[str, Any]: ...

    def reduce_native_metrics(self, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]: ...

    def validation_loss_terms(
        self, predictions: Any, targets: Any, auxiliary_state: Any, *, batch: TaskBatch, arm: str,
    ) -> Mapping[str, LossTerm]: ...

    def optimizer_groups(self, model: nn.Module, arm: str, stage: str) -> Sequence[OptimizerGroupSpec]: ...

    def work_counts(self, batch: TaskBatch, predictions: Any, auxiliary_state: Any) -> Mapping[str, int | float]: ...


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return copy.deepcopy(value)


def capture_rng_state() -> dict[str, Any]:
    """Capture all process RNGs needed for exact epoch-boundary continuation."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def named_optimizer_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Serialize AdamW moments by stable model parameter name."""

    name_by_parameter = {id(parameter): name for name, parameter in model.named_parameters()}
    state: dict[str, Any] = {}
    group_names = []
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            name = name_by_parameter.get(id(parameter))
            if name is None:
                raise ValueError("Optimizer contains a parameter not owned by the model.")
            names.append(name)
            if parameter in optimizer.state:
                state[name] = _cpu_tree(optimizer.state[parameter])
        group_names.append({"name": group["group_name"], "parameter_names": names})
    return {"state_by_name": state, "groups": group_names}


def optimizer_state_provenance(
    payload: Mapping[str, Any] | None,
    *,
    expected_groups: Sequence[OptimizerGroupSpec] | None = None,
) -> dict[str, Any]:
    """Summarize named AdamW moments without exposing or changing the state."""

    if payload is None:
        return {"source": "not_supplied", "state_parameter_count": 0, "step_counts": {}}
    states = payload.get("state_by_name", payload)
    if not isinstance(states, Mapping):
        raise TypeError("Named optimizer state must map parameter names to state mappings.")
    groups = payload.get("groups", ())
    group_membership = [
        {"name": str(group.get("name", "")), "parameter_names": list(group.get("parameter_names", ()))}
        for group in groups
        if isinstance(group, Mapping)
    ]
    step_counts: dict[str, int] = {}
    moment_summaries: dict[str, dict[str, float]] = {}
    for name, values in states.items():
        if not isinstance(values, Mapping):
            raise TypeError(f"Optimizer state for {name!r} is not a mapping.")
        step = values.get("step")
        if step is not None:
            step_value = int(step.item()) if torch.is_tensor(step) else int(step)
            step_counts[str(step_value)] = step_counts.get(str(step_value), 0) + 1
        moments = {}
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = values.get(key)
            if torch.is_tensor(value):
                moments[key] = math.sqrt(float(value.detach().double().square().sum().item()))
        if moments:
            moment_summaries[str(name)] = moments
    expected = None if expected_groups is None else [
        {"name": spec.name, "parameter_names": list(spec.parameter_names)} for spec in expected_groups
    ]
    return {
        "source": "checkpoint_named_state",
        "state_parameter_count": len(states),
        "step_counts": step_counts,
        "group_membership": group_membership,
        "expected_group_membership": expected,
        "group_membership_matches": None if expected is None else group_membership == expected,
        "moment_l2_norms_by_parameter": moment_summaries,
    }


def _load_named_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    payload: Mapping[str, Any] | None,
) -> None:
    if payload is None:
        return
    states = payload.get("state_by_name", payload)
    if not isinstance(states, Mapping):
        raise TypeError("Named optimizer seed must map parameter names to optimizer states.")
    parameters = dict(model.named_parameters())
    for name, saved_state in states.items():
        if name not in parameters:
            raise ValueError(f"Optimizer seed refers to missing model parameter {name!r}.")
        if not isinstance(saved_state, Mapping):
            raise TypeError(f"Optimizer state for {name!r} is not a mapping.")
        parameter = parameters[name]
        restored = {}
        for key, value in saved_state.items():
            if torch.is_tensor(value) and key != "step":
                restored[key] = value.to(parameter.device).clone()
            elif torch.is_tensor(value):
                restored[key] = value.detach().cpu().clone()
            else:
                restored[key] = copy.deepcopy(value)
        optimizer.state[parameter] = restored


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_clean_stop_request(output: Path, name: str) -> dict[str, Any] | None:
    request_path = output / name
    if not request_path.is_file():
        return None
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict) or not isinstance(request.get("request_id"), str):
        raise TypeError("Clean-stop requests must be JSON objects with a stable request_id.")
    if not request["request_id"]:
        raise ValueError("Clean-stop request_id must be nonempty.")
    return request


def _consume_acknowledged_clean_stop(output: Path, name: str, layout: RunLayout) -> None:
    """Remove only a leftover request whose exact identity was already acked."""

    request = _read_clean_stop_request(output, name)
    if request is None:
        return
    acknowledgement_path = layout.read_path("clean_stop_acknowledged.json")
    if not acknowledgement_path.is_file():
        return
    acknowledgement = json.loads(acknowledgement_path.read_text(encoding="utf-8"))
    if (not isinstance(acknowledgement, dict)
            or acknowledgement.get("request_id") != request["request_id"]):
        return
    request_tag = hashlib.sha256(request["request_id"].encode("utf-8")).hexdigest()[:16]
    _atomic_json(layout.write_path(f"clean_stop_consumed_{request_tag}.json"), {
        "request": request,
        "acknowledgement": acknowledgement,
        "consumed_on_resume": True,
    })
    (output / name).unlink(missing_ok=True)


def _denominator(value: float | torch.Tensor) -> float:
    result = float(value.detach().item()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Loss denominators must be finite and nonnegative.")
    return result


def _macro_update_slices(count: int, effective_cases: int) -> Iterable[slice]:
    for start in range(0, count, effective_cases):
        yield slice(start, min(start + effective_cases, count))


def _case_order_digest(case_keys: Sequence[Any]) -> str:
    encoded = json.dumps(list(case_keys), sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _named_tree_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    """Yield named leaves from the small receiver/target metadata trees."""
    if isinstance(value, Mapping):
        for name in sorted(value, key=str):
            yield from _named_tree_leaves(value[name], f"{prefix}.{name}" if prefix else str(name))
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            name = item.name
            yield from _named_tree_leaves(getattr(value, name), f"{prefix}.{name}" if prefix else name)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _named_tree_leaves(child, f"{prefix}[{index}]")
    elif torch.is_tensor(value) or isinstance(value, np.ndarray):
        yield prefix, value


def _observed_mask_digest(batches: Sequence[TaskBatch]) -> str:
    """Hash the exact first-epoch support/weight arrays without retaining them."""
    digest = hashlib.sha256()
    selectors = ("mask", "valid", "weight", "role_ids")
    for batch_index, batch in enumerate(batches):
        digest.update(json.dumps(list(batch.case_keys), sort_keys=True, default=str,
                                 separators=(",", ":")).encode("utf-8"))
        for section_name, tree in (("receivers", batch.receivers), ("targets", batch.targets)):
            for path, value in _named_tree_leaves(tree, section_name):
                leaf_name = path.rsplit(".", 1)[-1].lower()
                if not any(selector in leaf_name for selector in selectors):
                    continue
                if torch.is_tensor(value):
                    array = value.detach().to(device="cpu").contiguous().numpy()
                else:
                    array = np.ascontiguousarray(value)
                digest.update(path.encode("utf-8") + b"\0")
                digest.update(str(array.dtype).encode("ascii") + b"\0")
                digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
                digest.update(array.tobytes())
        digest.update(str(batch_index).encode("ascii") + b"\0")
    return digest.hexdigest()


def _stream_records_digest(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(list(records), sort_keys=True, default=str,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_state_sha256(model: nn.Module, *, exclude_prefixes: Sequence[str] = ()) -> str:
    """Hash a deterministic named-tensor inventory, optionally omitting declared modules."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if not torch.is_tensor(tensor):
            raise TypeError(f"Model state {name!r} is not a tensor and cannot be sealed.")
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _is_sha256_hex(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def provider_scale_binding_sha256(provider_identity: Mapping[str, Any]) -> str:
    """Hash the selected TRAIN-only transforms and loss calibration exposed by a provider."""
    task = provider_identity.get("task", provider_identity.get("dataset"))
    if task == "ThermalChannel":
        binding = {
            "normalization_stats_sha256": provider_identity.get("normalization_stats_sha256"),
            "response_scales": provider_identity.get("response_scales"),
            "loss_weights": provider_identity.get("loss_weights"),
            "auxiliary_calibration": provider_identity.get("auxiliary_calibration"),
        }
    elif task == "WindFarm":
        objective = provider_identity.get("objective")
        if not isinstance(objective, Mapping):
            raise ValueError("Wind provider identity is missing its TRAIN role calibration.")
        binding = {
            "normalizer_and_background_source": provider_identity.get("normalizer_and_background_source"),
            "component_scales_mps": objective.get("component_scales_mps"),
            "scale_rule": objective.get("scale_rule"),
            "scale_source_sha256": objective.get("scale_source_sha256"),
            "scale_calibration": objective.get("scale_calibration"),
            "scale_cache_binding": objective.get("scale_cache_binding"),
        }
    else:
        raise ValueError("Matched fresh initialization requires a supported Thermal or Wind provider identity.")
    if any(value is None for value in binding.values()):
        raise ValueError("Matched fresh provider scale binding has an unsealed TRAIN-only component.")
    return _canonical_sha256(binding)


FRESH_MATCHED_INITIALIZATION_SCHEMA = "matched_fresh_multiview_initialization_v1"
FRESH_MATCHED_ROUTING_PREFIXES = ("core.native_routing.",)
FRESH_MATCHED_NATIVE_VALIDATION_POLICY = "consolidated_route_flags_v1"


def fresh_matched_recipe_seed(recipe: Mapping[str, Any], *, verify_decision: bool = False) -> int:
    """Keep primary seeds fixed; admit only the plan's separately bound confirmation pair."""
    task = recipe.get("task")
    if task not in ("thermal", "wind"):
        raise ValueError("Fresh matched seed requires a supported dataset.")
    primary_seed = 0 if task == "thermal" else 42
    confirmation = recipe.get("conditional_confirmation")
    if "conditional_confirmation" not in recipe:
        if type(recipe.get("seed")) is not int or recipe["seed"] != primary_seed:
            raise ValueError("Fresh primary comparison must retain its fixed dataset seed.")
        return primary_seed
    required = {"schema", "task", "seed", "preferred_mode", "decision_path", "decision_sha256"}
    second_seed = 1 if task == "thermal" else 43
    if (not isinstance(confirmation, Mapping) or set(confirmation) != required
            or confirmation.get("schema") != "conditional_second_seed_multiview_v1"
            or confirmation.get("task") != task
            or type(confirmation.get("seed")) is not int or confirmation["seed"] != second_seed
            or type(recipe.get("seed")) is not int or recipe["seed"] != second_seed
            or confirmation.get("preferred_mode") not in ("P-R", "P-R2")
            or recipe.get("mode") not in ("P", confirmation["preferred_mode"])
            or not isinstance(confirmation.get("decision_path"), str)
            or not Path(confirmation["decision_path"]).is_absolute()
            or not _is_sha256_hex(confirmation.get("decision_sha256"))):
        raise ValueError("Conditional confirmation must bind one preferred-routing/P pair and its declared second seed.")
    if verify_decision:
        decision_bytes = Path(confirmation["decision_path"]).read_bytes()
        if hashlib.sha256(decision_bytes).hexdigest() != confirmation["decision_sha256"]:
            raise ValueError("Conditional confirmation decision receipt changed.")
        decision = json.loads(decision_bytes)
        endpoints = decision.get("primary_endpoints")
        if (decision.get("schema") != "conditional_second_seed_eligibility_v1"
                or decision.get("chosen_dataset") != task
                or decision.get("preferred_mode") != confirmation["preferred_mode"]
                or decision.get("second_seed") != second_seed
                or decision.get("physical_direction_pass") is not True
                or decision.get("robustness_pass") is not True
                or decision.get("mandatory_measurements_complete") is not True
                or decision.get("unresolved_implementation_failure") is not False
                or type(decision.get("q8192_cost_ratio")) not in (int, float)
                or not math.isfinite(decision["q8192_cost_ratio"])
                or not 0 < decision["q8192_cost_ratio"] <= 1.30
                or not _is_sha256_hex(decision.get("mandatory_measurements_sha256"))
                or not isinstance(endpoints, list) or len(endpoints) != 6
                or any(not isinstance(entry, Mapping) for entry in endpoints)
                or {(entry.get("task"), entry.get("mode")) for entry in endpoints}
                != {(domain, mode) for domain in ("thermal", "wind") for mode in ("P", "P-R", "P-R2")}
                or any(type(entry.get("epoch")) is not int or entry["epoch"] != 2500
                       or not _is_sha256_hex(entry.get("checkpoint_sha256")) for entry in endpoints)
                or (task == "thermal" and decision.get("wind_eligible") is not False)
                or (task == "wind" and decision.get("wind_eligible") is not True)):
            raise ValueError("Conditional confirmation requires all six endpoints and completed eligible physical/cost evidence.")
    return second_seed


def _apply_fresh_native_validation_policy(model: nn.Module) -> None:
    core = getattr(model, "core", None)
    setter = getattr(core, "set_consolidated_native_validation", None)
    if not callable(setter):
        raise TypeError("Fresh matched model core does not expose consolidated native-validation controls.")
    setter(True)
    if getattr(core, "consolidated_native_validation", None) is not True:
        raise RuntimeError("Fresh matched consolidated native-validation policy did not activate.")


def fresh_matched_initialization_receipt(
    model: nn.Module,
    provider_identity: Mapping[str, Any],
    *,
    active_mode: str,
    seed: int,
    sampling_version: str,
    deterministic_case_order: bool,
) -> dict[str, Any]:
    """Seal a P/R1/R2 common fresh start without any inherited model state."""
    if active_mode not in ("P", "P-R", "P-R2"):
        raise ValueError("Fresh matched initialization is limited to P, P-R and P-R2.")
    if provider_identity.get("fresh_matched_comparison") is not True:
        raise ValueError("Fresh initialization receipt requires an explicitly fresh matched provider identity.")
    if type(seed) is not int or sampling_version != SamplingKey.CASE_EPOCH_VERSION:
        raise ValueError("Fresh matched initialization requires its sealed seed and case_epoch_v1 streams.")
    if type(deterministic_case_order) is not bool or not deterministic_case_order:
        raise ValueError("Fresh matched initialization requires deterministic per-epoch TRAIN ordering.")
    provider_identity = dict(provider_identity)
    model_config = provider_identity.get("model_config", provider_identity.get("standalone_model_config"))
    if not isinstance(model_config, Mapping):
        raise TypeError("Fresh matched provider identity is missing its active model configuration.")
    dataset_id = _sampling_dataset_id(provider_identity)
    sampler_namespace = {
        "sampling_version": sampling_version,
        "seed": seed,
        "dataset_id": dataset_id,
        "case_order": "seed_epoch_v1_deterministic_permutation",
        "case_order_arm_independent": True,
        "query_stream_keying": "dataset_seed_epoch_named_case_role_stream",
        "query_stream_arm_independent": True,
        "query_stream_update_independent": True,
        "query_stream_microbatch_independent": True,
    }
    return {
        "schema": FRESH_MATCHED_INITIALIZATION_SCHEMA,
        "active_mode": active_mode,
        "native_validation_policy": FRESH_MATCHED_NATIVE_VALIDATION_POLICY,
        "source_checkpoint": None,
        "inherited_parameter_names": [],
        "optimizer_state_start": "fresh_zero_moments",
        "common_parameter_exclusion_prefixes": list(FRESH_MATCHED_ROUTING_PREFIXES),
        "common_initialization_sha256": model_state_sha256(
            model, exclude_prefixes=FRESH_MATCHED_ROUTING_PREFIXES),
        "initial_model_state_sha256": model_state_sha256(model),
        "active_model_config_sha256": _canonical_sha256(model_config),
        "provider_identity_sha256": _canonical_sha256(provider_identity),
        "provider_scale_binding_sha256": provider_scale_binding_sha256(provider_identity),
        "sampler_namespace": sampler_namespace,
    }


def _validate_fresh_matched_initialization(
    marker: Mapping[str, Any],
    model: nn.Module,
    provider_identity: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    sampling_version: str,
    deterministic_case_order: bool,
) -> None:
    """Check the active fresh model/provider against its immutable start receipt."""
    if marker.get("schema") != FRESH_MATCHED_INITIALIZATION_SCHEMA:
        raise ValueError("Fresh matched initialization schema is unsupported.")
    if marker.get("native_validation_policy") != FRESH_MATCHED_NATIVE_VALIDATION_POLICY:
        raise ValueError("Fresh matched initialization has an unsupported native-validation policy.")
    if marker.get("active_mode") != arm or arm not in ("P", "P-R", "P-R2"):
        raise ValueError("Fresh initialization mode differs from the active P/R1/R2 arm.")
    if (marker.get("source_checkpoint") is not None
            or marker.get("inherited_parameter_names") != []
            or marker.get("optimizer_state_start") != "fresh_zero_moments"):
        raise ValueError("Fresh matched start cannot inherit model parameters, checkpoint state or optimizer moments.")
    if marker.get("common_parameter_exclusion_prefixes") != list(FRESH_MATCHED_ROUTING_PREFIXES):
        raise ValueError("Fresh common-initialization digest must exclude only private routing tensors.")
    provider_identity = dict(provider_identity)
    model_config = provider_identity.get("model_config", provider_identity.get("standalone_model_config"))
    if not isinstance(model_config, Mapping):
        raise TypeError("Active provider has no model configuration to bind to fresh initialization.")
    current = {
        "common_initialization_sha256": model_state_sha256(
            model, exclude_prefixes=FRESH_MATCHED_ROUTING_PREFIXES),
        "initial_model_state_sha256": model_state_sha256(model),
        "active_model_config_sha256": _canonical_sha256(model_config),
        "provider_identity_sha256": _canonical_sha256(provider_identity),
        "provider_scale_binding_sha256": provider_scale_binding_sha256(provider_identity),
        "sampler_namespace": {
            "sampling_version": sampling_version,
            "seed": seed,
            "dataset_id": _sampling_dataset_id(provider_identity),
            "case_order": "seed_epoch_v1_deterministic_permutation",
            "case_order_arm_independent": True,
            "query_stream_keying": "dataset_seed_epoch_named_case_role_stream",
            "query_stream_arm_independent": True,
            "query_stream_update_independent": True,
            "query_stream_microbatch_independent": True,
        },
    }
    for name, value in current.items():
        if marker.get(name) != value:
            raise ValueError(f"Fresh initialization receipt differs from the active {name}.")
    if (type(seed) is not int or seed < 0
            or sampling_version != SamplingKey.CASE_EPOCH_VERSION
            or type(deterministic_case_order) is not bool or not deterministic_case_order):
        raise ValueError("Fresh matched runs require the declared case_epoch_v1 deterministic sampler.")
    if provider_identity.get("fresh_matched_comparison") is not True:
        raise ValueError("Fresh initialization cannot be attached to a legacy provider identity.")


def _optimizer_spec_payload(spec: OptimizerGroupSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "parameter_names": list(spec.parameter_names),
        "schedule": asdict(spec.schedule),
        "weight_decay": float(spec.weight_decay),
        "betas": list(spec.betas),
        "eps": float(spec.eps),
    }


class TrainingEngine:
    """Own the single training/evaluation loop used by all dataset adapters."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        device: torch.device | str,
        selection: SelectionPolicy | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.selection = selection or SelectionPolicy()

    def epoch_case_order(
        self, provider: TaskProvider, epoch: int,
    ) -> tuple[list[Any], str]:
        """Return the exact epoch case order and sealed case-epoch sampler namespace.

        This read-only planning method is shared with bounded measurement tools
        that need to exercise complete TRAIN update macros without launching a
        scientific fit. ``fit`` uses the same method, so cost probes cannot
        drift from its deterministic case-order or dataset-key rules.
        """
        ordered_cases = list(provider.epoch_cases(epoch, self.config.seed))
        if not ordered_cases:
            raise ValueError("Each development epoch must visit at least one selected TRAIN case.")
        if self.config.deterministic_case_order:
            rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, epoch, 0x484F4E46]))
            order = rng.permutation(len(ordered_cases)).tolist()
            ordered_cases = [ordered_cases[index] for index in order]
        dataset_id = (
            _sampling_dataset_id(provider.identity_payload())
            if self.config.sampling_version == SamplingKey.CASE_EPOCH_VERSION else ""
        )
        return ordered_cases, dataset_id

    def epoch_update_slices(self, case_count: int) -> tuple[slice, ...]:
        """Return the exact effective-case macro grouping used by ``fit``."""
        if type(case_count) is not int or case_count < 0:
            raise ValueError("Epoch case count must be a nonnegative integer.")
        return tuple(_macro_update_slices(case_count, self.config.effective_cases))

    def prepare_update_microbatches(
        self,
        provider: TaskProvider,
        *,
        arm: str,
        phase: str,
        epoch: int,
        update_index: int,
        update_cases: Sequence[Any],
        sampling_dataset_id: str,
    ) -> list[TaskBatch]:
        """Build one macro update's exact fit microbatches without optimizing."""
        microbatches: list[TaskBatch] = []
        for micro_index, start in enumerate(range(0, len(update_cases), self.config.microbatch_cases)):
            case_chunk = update_cases[start:start + self.config.microbatch_cases]
            key = SamplingKey(
                self.config.seed, epoch, update_index, micro_index, phase, arm,
                sampling_version=self.config.sampling_version,
                dataset_id=sampling_dataset_id,
            )
            batch = provider.make_batch(case_chunk, key)
            if tuple(batch.case_keys) and list(batch.case_keys) != list(case_chunk):
                raise ValueError("Provider changed the ordered case keys inside a microbatch.")
            if not tuple(batch.case_keys):
                batch = TaskBatch(batch.scene_inputs, batch.receivers, batch.targets, batch.auxiliary,
                                  tuple(case_chunk))
            microbatches.append(batch)
        return microbatches

    def _make_optimizer(
        self,
        model: nn.Module,
        provider: TaskProvider,
        arm: str,
        initial_epoch: int,
        optimizer_seed: Mapping[str, Any] | None,
        *,
        group_specs: Sequence[OptimizerGroupSpec] | None = None,
    ) -> tuple[torch.optim.Optimizer, tuple[OptimizerGroupSpec, ...]]:
        specs = tuple(group_specs or provider.optimizer_groups(
            model, arm, self.config.stage_for_epoch(initial_epoch)))
        if not specs or len({spec.name for spec in specs}) != len(specs):
            raise ValueError("Providers must declare at least one uniquely named optimizer group.")
        named = dict(model.named_parameters())
        used: set[str] = set()
        groups: list[dict[str, Any]] = []
        for spec in specs:
            if not spec.parameter_names:
                raise ValueError(f"Optimizer group {spec.name!r} has no parameters.")
            if not math.isfinite(spec.weight_decay) or spec.weight_decay < 0:
                raise ValueError(f"Optimizer group {spec.name!r} has invalid weight decay.")
            params = []
            for name in spec.parameter_names:
                if name not in named:
                    raise ValueError(f"Optimizer group {spec.name!r} references missing parameter {name!r}.")
                if name in used:
                    raise ValueError(f"Parameter {name!r} appears in more than one optimizer group.")
                if not named[name].requires_grad:
                    raise ValueError(f"Optimizer parameter {name!r} is unexpectedly frozen at engine creation.")
                params.append(named[name])
                used.add(name)
            groups.append({
                "params": params,
                "lr": spec.schedule.value(initial_epoch),
                "weight_decay": spec.weight_decay,
                "betas": spec.betas,
                "eps": spec.eps,
                "group_name": spec.name,
            })
        trainable = {name for name, parameter in named.items() if parameter.requires_grad}
        omitted = trainable - used
        if omitted:
            raise ValueError(f"Trainable model parameters lack optimizer groups: {sorted(omitted)}")
        optimizer = torch.optim.AdamW(groups)
        _load_named_optimizer_state(model, optimizer, optimizer_seed)
        return optimizer, specs

    def preflight_one_update(
        self,
        model: nn.Module,
        provider: TaskProvider,
        *,
        arm: str = "warmup",
        optimizer_seed: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Exercise one disposable warmup optimizer update without saving a run.

        The maintained CLI uses this for a CPU-only dry run. It deliberately
        delegates to the exact same batch, denominator, forward, loss,
        accumulation, clipping, and optimizer helpers as :meth:`fit`, while
        leaving no checkpoint or training history behind.
        """

        if self.config.training_mode == "refinement" and arm != "warmup":
            raise ValueError("A disposable preflight update must use the shared warmup arm.")
        model.to(self.device)
        cases = tuple(provider.epoch_cases(1, self.config.seed))
        if not cases:
            raise ValueError("The task provider has no TRAIN case for a dry run.")
        cases = cases[:1]
        phase = self.config.stage_for_epoch(1)
        provider_identity = dict(provider.identity_payload())
        dataset_id = (_sampling_dataset_id(provider_identity)
                      if self.config.sampling_version == SamplingKey.CASE_EPOCH_VERSION else "")
        key = SamplingKey(self.config.seed, 1, 0, 0, phase, arm,
                          sampling_version=self.config.sampling_version, dataset_id=dataset_id)
        batch = provider.make_batch(cases, key)
        if tuple(batch.case_keys) and list(batch.case_keys) != list(cases):
            raise ValueError("Provider changed case keys during the disposable preflight.")
        if not tuple(batch.case_keys):
            batch = TaskBatch(batch.scene_inputs, batch.receivers, batch.targets, batch.auxiliary, cases)
        on_phase_start = getattr(provider, "on_phase_start", None)
        stage_receipt = None
        if callable(on_phase_start):
            stage_receipt = on_phase_start(
                model=model, arm=arm, epoch=1, phase=phase,
                temperature=self.config.temperature_for_epoch(1))
        model.train(True)
        optimizer, specs = self._make_optimizer(
            model, provider, arm, 1, optimizer_seed)
        self._set_schedules(optimizer, specs, 1)
        losses, work, denominators, query_hash = self._run_update(
            model, provider, optimizer, (batch,), phase=phase, arm=arm,
            epoch=1, update_index=0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "status": "disposable_cpu_preflight_update" if self.device.type == "cpu" else "disposable_preflight_update",
            "arm": arm,
            "phase": phase,
            "case_keys": list(cases),
            "optimizer_updates": 1,
            "losses": losses,
            "denominators": denominators,
            "work_counts": work,
            "query_sampling_sha256": query_hash,
            "stage_receipt": stage_receipt,
        }

    @staticmethod
    def _set_schedules(optimizer: torch.optim.Optimizer, specs: Sequence[OptimizerGroupSpec], epoch: int) -> None:
        by_name = {spec.name: spec for spec in specs}
        if {group["group_name"] for group in optimizer.param_groups} != set(by_name):
            raise ValueError("Optimizer groups changed after the absolute schedule was sealed.")
        for group in optimizer.param_groups:
            group["lr"] = by_name[group["group_name"]].schedule.value(epoch)

    def diagnose_macro_update(
        self,
        model: nn.Module,
        provider: TaskProvider,
        microbatches: Sequence[TaskBatch],
        *,
        phase: str,
        arm: str,
        epoch: int,
        term_groups: Mapping[str, Sequence[str]] | None = None,
        training_mode: bool = True,
        include_gradient_tensors: bool = False,
        optimizer_state_by_name: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Measure the exact engine objective and its grouped gradients without stepping.

        This uses the provider's native samplers, target-only denominator prepass,
        predictions, and :class:`LossTerm` values just like ``_run_update``. The
        only difference is that it retains a named gradient decomposition and
        stops before clipping or calling AdamW. ``term_groups`` must partition
        the declared loss names; when omitted, each loss term is its own group.
        """

        if not microbatches:
            raise ValueError("A macro-update diagnosis requires at least one fixed microbatch.")
        planned = {
            name: _denominator(value)
            for name, value in provider.loss_denominators(microbatches, phase, arm).items()
        }
        if not planned or any(not name for name in planned):
            raise ValueError("A diagnostic update requires named loss denominators.")
        if any(value <= 0 for value in planned.values()):
            raise ValueError("Every diagnosed loss term must have a positive exact denominator.")

        specs = tuple(provider.optimizer_groups(model, arm, phase))
        if not specs or len({spec.name for spec in specs}) != len(specs):
            raise ValueError("Providers must declare uniquely named optimizer groups for diagnosis.")
        named = dict(model.named_parameters())
        group_by_parameter: dict[str, str] = {}
        for spec in specs:
            if not spec.parameter_names:
                raise ValueError(f"Optimizer group {spec.name!r} has no parameters.")
            for name in spec.parameter_names:
                if name not in named or not named[name].requires_grad or name in group_by_parameter:
                    raise ValueError(f"Invalid or duplicate diagnosed optimizer parameter {name!r}.")
                group_by_parameter[name] = spec.name
        trainable_names = {name for name, parameter in named.items() if parameter.requires_grad}
        if set(group_by_parameter) != trainable_names:
            raise ValueError("Optimizer groups must cover every trainable parameter exactly once.")

        if term_groups is None:
            groups = {name: (name,) for name in planned}
        else:
            groups = {str(name): tuple(members) for name, members in term_groups.items()}
            if not groups or any(not name or not members for name, members in groups.items()):
                raise ValueError("Gradient groups must be named and nonempty.")
        term_group: dict[str, str] = {}
        for group_name, member_names in groups.items():
            for term_name in member_names:
                if term_name not in planned:
                    raise ValueError(f"Gradient group {group_name!r} references undeclared loss {term_name!r}.")
                if term_name in term_group:
                    raise ValueError(f"Loss term {term_name!r} appears in multiple gradient groups.")
                term_group[term_name] = group_name
        if set(term_group) != set(planned):
            raise ValueError("Gradient groups must partition all declared loss terms exactly once.")

        parameters = [(name, parameter) for name, parameter in named.items() if parameter.requires_grad]
        parameter_values = [parameter for _, parameter in parameters]
        gradients: dict[str, dict[str, torch.Tensor]] = {name: {} for name in groups}
        observed = {name: 0.0 for name in planned}
        loss_values = {name: 0.0 for name in planned}
        term_numerators = {name: 0.0 for name in planned}
        term_weights: dict[str, float | None] = {name: None for name in planned}
        sampling_records: list[str] = []
        work: dict[str, float] = {}
        batch_records: list[dict[str, Any]] = []
        rng_state = capture_rng_state()
        original_training = model.training
        buffer_snapshot = {
            name: value.detach().clone() for name, value in model.named_buffers()
        } if training_mode else None
        provider_state_reader = getattr(provider, "training_state_dict", None)
        provider_state_loader = getattr(provider, "load_training_state_dict", None)
        provider_state = copy.deepcopy(provider_state_reader()) if callable(provider_state_reader) else None
        model.train(bool(training_mode))
        try:
            for batch in microbatches:
                batch_auxiliary = batch.auxiliary if isinstance(batch.auxiliary, Mapping) else {}
                sample_hash = batch_auxiliary.get("query_sampling_sha256")
                if sample_hash is not None:
                    if not isinstance(sample_hash, str) or len(sample_hash) != 64:
                        raise ValueError("Provider query-sampling receipt must be a SHA256 hex string.")
                    sampling_records.append(sample_hash)
                scene = provider.make_scene(batch.scene_inputs)
                predictions, auxiliary_state = provider.predict_native(
                    model, scene, batch.receivers, arm, phase, epoch=epoch,
                    temperature=self.config.temperature_for_epoch(epoch))
                terms = provider.loss_terms(predictions, batch.targets, phase, auxiliary_state)
                if not isinstance(terms, Mapping):
                    raise TypeError("loss_terms must return a mapping of names to LossTerm values.")
                objectives: dict[str, torch.Tensor | None] = {name: None for name in groups}
                for name, term in terms.items():
                    if name not in planned:
                        raise ValueError(f"Loss term {name!r} has no target-only macro denominator.")
                    if not torch.is_tensor(term.numerator) or term.numerator.numel() != 1:
                        raise ValueError(f"Loss numerator {name!r} must be a scalar tensor.")
                    denominator = _denominator(term.denominator)
                    if not math.isfinite(float(term.weight)):
                        raise ValueError(f"Loss weight {name!r} is not finite.")
                    if term_weights[name] is not None and not math.isclose(
                        float(term_weights[name]), float(term.weight), rel_tol=0.0, abs_tol=0.0
                    ):
                        raise ValueError(f"Loss weight for {name!r} changed across a macro-update.")
                    term_weights[name] = float(term.weight)
                    if denominator == 0:
                        if float(term.numerator.detach()) != 0.0:
                            raise ValueError(f"Loss numerator {name!r} is nonzero with an empty valid mask.")
                        continue
                    observed[name] += denominator
                    term_numerators[name] += float(term.numerator.detach())
                    weighted = term.numerator * (float(term.weight) / planned[name])
                    group_name = term_group[name]
                    objectives[group_name] = (
                        weighted if objectives[group_name] is None else objectives[group_name] + weighted
                    )
                    loss_values[name] += float(term.numerator.detach()) / planned[name] * float(term.weight)
                for name, value in provider.work_counts(batch, predictions, auxiliary_state).items():
                    work[name] = work.get(name, 0.0) + float(value)

                active_objectives = [(name, value) for name, value in objectives.items() if value is not None]
                if not active_objectives:
                    raise ValueError("A microbatch produced no differentiable loss groups.")
                for position, (group_name, objective) in enumerate(active_objectives):
                    if not bool(torch.isfinite(objective.detach())):
                        raise FloatingPointError(f"Nonfinite diagnostic objective for {group_name!r}.")
                    current = torch.autograd.grad(
                        objective,
                        parameter_values,
                        retain_graph=position + 1 < len(active_objectives),
                        allow_unused=True,
                    )
                    group_gradients = gradients[group_name]
                    for (parameter_name, _), gradient in zip(parameters, current):
                        if gradient is None:
                            continue
                        previous = group_gradients.get(parameter_name)
                        group_gradients[parameter_name] = gradient.detach().clone() if previous is None else previous + gradient.detach()

                main = getattr(predictions, "main_mps", None)
                full = getattr(predictions, "full_mps", None)
                batch_records.append({
                    "case_keys": list(batch.case_keys),
                    "query_sampling_sha256": sample_hash,
                    "all_fine_main_full_same_tensor_values": bool(
                        torch.is_tensor(main) and torch.is_tensor(full) and main.shape == full.shape
                        and torch.equal(main.detach(), full.detach())
                    ),
                })
            for name, expected in planned.items():
                actual = observed.get(name, 0.0)
                if not math.isclose(actual, expected, rel_tol=1.0e-6, abs_tol=1.0e-7):
                    raise ValueError(f"Actual denominator for {name!r} ({actual}) differs from prepass ({expected}).")
        finally:
            if buffer_snapshot is not None:
                with torch.no_grad():
                    for name, value in model.named_buffers():
                        value.copy_(buffer_snapshot[name])
            model.train(original_training)
            try:
                if provider_state is not None and callable(provider_state_loader):
                    provider_state_loader(provider_state)
            finally:
                restore_rng_state(rng_state)

        def _norm(rows: Mapping[str, torch.Tensor], selected: Sequence[str]) -> float:
            values = [rows[name] for name in selected if name in rows]
            if not values:
                return 0.0
            tensor_norms = [torch.linalg.vector_norm(value.detach(), ord=2) for value in values]
            return float(torch.linalg.vector_norm(torch.stack(tensor_norms), ord=2).item())

        combined: dict[str, torch.Tensor] = {}
        for group_gradients in gradients.values():
            for name, value in group_gradients.items():
                combined[name] = value.clone() if name not in combined else combined[name] + value
        group_parameters = {
            spec.name: tuple(spec.parameter_names)
            for spec in specs
        }
        split_names = tuple(gradients)
        total_norm = _norm(combined, tuple(group_by_parameter))
        clip_factor = min(1.0, self.config.gradient_clip / (total_norm + 1.0e-6)) if total_norm > 0 else 1.0
        clip_influence: dict[str, Any] = {}
        for excluded_name in split_names:
            without_excluded: dict[str, torch.Tensor] = {}
            for other_name, other_gradients in gradients.items():
                if other_name == excluded_name:
                    continue
                for parameter_name, value in other_gradients.items():
                    without_excluded[parameter_name] = (
                        value.clone() if parameter_name not in without_excluded
                        else without_excluded[parameter_name] + value
                    )
            without_norm = _norm(without_excluded, tuple(group_by_parameter))
            without_factor = (
                min(1.0, self.config.gradient_clip / (without_norm + 1.0e-6))
                if without_norm > 0 else 1.0
            )
            with_group_norm = _norm(gradients[excluded_name], tuple(group_by_parameter))
            clip_influence[excluded_name] = {
                "group_gradient_norm": with_group_norm,
                "global_norm_without_group": without_norm,
                "global_norm_with_group": total_norm,
                "clip_factor_without_group": without_factor,
                "actual_clip_factor_with_group": clip_factor,
                "actual_clip_factor_change": clip_factor - without_factor,
                "global_norm_change": total_norm - without_norm,
            }
        gradient_summary: dict[str, Any] = {}
        for split_name, split_gradients in gradients.items():
            per_optimizer_group = {
                group_name: _norm(split_gradients, parameter_names)
                for group_name, parameter_names in group_parameters.items()
            }
            reached = sorted(name for name in split_gradients)
            gradient_summary[split_name] = {
                "global_norm": _norm(split_gradients, tuple(group_by_parameter)),
                "optimizer_group_norms": per_optimizer_group,
                "reachable_parameter_count": len(reached),
                "reachable_parameter_names": reached,
                "nonzero_parameter_names": sorted(
                    name for name, value in split_gradients.items() if bool(torch.count_nonzero(value).item())
                ),
                "parameter_norms": {
                    name: math.sqrt(float(value.double().square().sum().item()))
                    for name, value in split_gradients.items()
                },
            }
        cosine: dict[str, dict[str, float | None]] = {}
        for first in split_names:
            cosine[first] = {}
            for second in split_names:
                first_norm = _norm(gradients[first], tuple(group_by_parameter))
                second_norm = _norm(gradients[second], tuple(group_by_parameter))
                dot = sum(
                    float((gradients[first][name].double() * gradients[second][name].double()).sum().item())
                    for name in gradients[first].keys() & gradients[second].keys()
                )
                cosine[first][second] = dot / (first_norm * second_norm) if first_norm > 0 and second_norm > 0 else None

        result = {
            "optimizer_updates": 0,
            "phase": phase,
            "arm": arm,
            "epoch_for_schedule_and_sampler": int(epoch),
            "case_keys": [key for batch in microbatches for key in batch.case_keys],
            "query_sampling_sha256": hashlib.sha256(
                json.dumps(sampling_records, separators=(",", ":")).encode("utf-8")
            ).hexdigest() if sampling_records else None,
            "term_groups": {name: list(names) for name, names in groups.items()},
            "loss_terms": {
                name: {
                    "numerator_sum": term_numerators[name],
                    "denominator": planned[name],
                    "weight": term_weights[name],
                    "weighted_objective": loss_values[name],
                }
                for name in planned
            },
            "denominators_observed": observed,
            "gradient_splits": gradient_summary,
            "gradient_cosines": cosine,
            "combined_gradient": {
                "global_norm_before_clip": total_norm,
                "clip_limit": float(self.config.gradient_clip),
                "clip_factor_formula": "min(1, clip_limit / (global_norm + 1e-6))",
                "global_clip_factor": clip_factor,
                "global_norm_after_clip": total_norm * clip_factor,
                "optimizer_group_norms_before_clip": {
                    group_name: _norm(combined, parameter_names)
                    for group_name, parameter_names in group_parameters.items()
                },
                "loss_split_clip_influence": clip_influence,
            },
            "optimizer_groups": [
                {
                    "name": spec.name,
                    "parameter_count": len(spec.parameter_names),
                    "parameter_names": list(spec.parameter_names),
                    "schedule": asdict(spec.schedule),
                    "lr_at_epoch": spec.schedule.value(epoch),
                    "weight_decay": float(spec.weight_decay),
                    "betas": list(spec.betas),
                    "eps": float(spec.eps),
                }
                for spec in specs
            ],
            "optimizer_state": optimizer_state_provenance(
                optimizer_state_by_name, expected_groups=specs),
            "work_counts": work,
            "batches": batch_records,
        }
        if include_gradient_tensors:
            result["gradient_tensors"] = {
                split_name: {name: value.detach().clone() for name, value in rows.items()}
                for split_name, rows in gradients.items()
            }
            result["combined_gradient_tensors"] = {
                name: value.detach().clone() for name, value in combined.items()
            }
        return result

    def measure_macro_loss(
        self,
        model: nn.Module,
        provider: TaskProvider,
        microbatches: Sequence[TaskBatch],
        *,
        phase: str,
        arm: str,
        epoch: int,
        training_mode: bool = True,
    ) -> dict[str, Any]:
        """Measure exact provider LossTerms on fixed batches without autograd.

        The default matches the training objective, including providers with
        training-only auxiliary terms. Set ``training_mode=False`` only for an
        explicitly eval-mode loss measurement; validation metrics use the
        provider's separate native-metrics path.
        """

        if not microbatches:
            raise ValueError("A fixed macro-loss measurement requires at least one microbatch.")
        planned = {
            name: _denominator(value)
            for name, value in provider.loss_denominators(microbatches, phase, arm).items()
        }
        if not planned or any(value <= 0 for value in planned.values()):
            raise ValueError("A fixed macro-loss measurement requires positive declared denominators.")
        observed = {name: 0.0 for name in planned}
        numerators = {name: 0.0 for name in planned}
        weights: dict[str, float | None] = {name: None for name in planned}
        weighted_losses = {name: 0.0 for name in planned}
        sampling_records: list[str] = []
        batch_records: list[dict[str, Any]] = []
        rng_state = capture_rng_state()
        original_training = model.training
        buffer_snapshot = {
            name: value.detach().clone() for name, value in model.named_buffers()
        } if training_mode else None
        provider_state_reader = getattr(provider, "training_state_dict", None)
        provider_state_loader = getattr(provider, "load_training_state_dict", None)
        provider_state = copy.deepcopy(provider_state_reader()) if callable(provider_state_reader) else None
        model.train(bool(training_mode))
        try:
            with torch.no_grad():
                for batch in microbatches:
                    batch_auxiliary = batch.auxiliary if isinstance(batch.auxiliary, Mapping) else {}
                    sample_hash = batch_auxiliary.get("query_sampling_sha256")
                    if sample_hash is not None:
                        if not isinstance(sample_hash, str) or len(sample_hash) != 64:
                            raise ValueError("Provider query-sampling receipt must be a SHA256 hex string.")
                        sampling_records.append(sample_hash)
                    scene = provider.make_scene(batch.scene_inputs)
                    predictions, auxiliary_state = provider.predict_native(
                        model, scene, batch.receivers, arm, phase, epoch=epoch,
                        temperature=self.config.temperature_for_epoch(epoch))
                    terms = provider.loss_terms(predictions, batch.targets, phase, auxiliary_state)
                    if not isinstance(terms, Mapping):
                        raise TypeError("loss_terms must return a mapping of names to LossTerm values.")
                    for name, term in terms.items():
                        if name not in planned:
                            raise ValueError(f"Loss term {name!r} has no target-only macro denominator.")
                        if not torch.is_tensor(term.numerator) or term.numerator.numel() != 1:
                            raise ValueError(f"Loss numerator {name!r} must be a scalar tensor.")
                        denominator = _denominator(term.denominator)
                        if denominator == 0:
                            if float(term.numerator) != 0.0:
                                raise ValueError(f"Loss numerator {name!r} is nonzero with an empty valid mask.")
                            continue
                        if weights[name] is not None and not math.isclose(
                            float(weights[name]), float(term.weight), rel_tol=0.0, abs_tol=0.0
                        ):
                            raise ValueError(f"Loss weight for {name!r} changed across a macro-update.")
                        weights[name] = float(term.weight)
                        observed[name] += denominator
                        numerators[name] += float(term.numerator)
                        weighted_losses[name] += float(term.numerator) / planned[name] * float(term.weight)
                    main = getattr(predictions, "main_mps", None)
                    full = getattr(predictions, "full_mps", None)
                    batch_records.append({
                        "case_keys": list(batch.case_keys),
                        "query_sampling_sha256": sample_hash,
                        "all_fine_main_full_same_tensor_values": bool(
                            torch.is_tensor(main) and torch.is_tensor(full) and main.shape == full.shape
                            and torch.equal(main, full)
                        ),
                    })
        finally:
            if buffer_snapshot is not None:
                with torch.no_grad():
                    for name, value in model.named_buffers():
                        value.copy_(buffer_snapshot[name])
            model.train(original_training)
            try:
                if provider_state is not None and callable(provider_state_loader):
                    provider_state_loader(provider_state)
            finally:
                restore_rng_state(rng_state)
        for name, expected in planned.items():
            if not math.isclose(observed[name], expected, rel_tol=1.0e-6, abs_tol=1.0e-7):
                raise ValueError(f"Actual denominator for {name!r} ({observed[name]}) differs from prepass ({expected}).")
        return {
            "optimizer_updates": 0,
            "training_mode": bool(training_mode),
            "phase": phase,
            "arm": arm,
            "epoch_for_schedule_and_sampler": int(epoch),
            "case_keys": [key for batch in microbatches for key in batch.case_keys],
            "query_sampling_sha256": hashlib.sha256(
                json.dumps(sampling_records, separators=(",", ":")).encode("utf-8")
            ).hexdigest() if sampling_records else None,
            "loss_terms": {
                name: {
                    "numerator_sum": numerators[name],
                    "denominator": planned[name],
                    "weight": weights[name],
                    "weighted_objective": weighted_losses[name],
                }
                for name in planned
            },
            "batches": batch_records,
        }

    def compare_disposable_updates(
        self,
        model: nn.Module,
        provider: TaskProvider,
        update_batches: Sequence[TaskBatch],
        probe_batches: Sequence[TaskBatch],
        *,
        phase: str,
        arm: str,
        epoch: int,
        optimizer_state_by_name: Mapping[str, Any] | None,
        interventions: Mapping[str, Mapping[str, float]],
        declared_optimizer_updates: int,
        dv_term: str | None = None,
    ) -> dict[str, Any]:
        """Apply one exact TrainingEngine update per intervention on isolated clones.

        The required explicit count is the number of cloned optimizer steps this
        call will perform. Every intervention starts from identical parameters,
        optimizer moments, batches and schedules; only listed LossTerm weights
        differ. No source model or checkpoint is mutated.
        """

        if not update_batches or not probe_batches:
            raise ValueError("A disposable update comparison requires fixed update and disjoint probe batches.")
        if not interventions or len(interventions) > 32:
            raise ValueError("A comparison must declare between one and 32 one-step interventions.")
        if type(declared_optimizer_updates) is not int or declared_optimizer_updates != len(interventions):
            raise ValueError("The explicit optimizer-update count must equal the number of cloned one-step arms.")
        if len(set(interventions)) != len(interventions):
            raise ValueError("Counterfactual arm names must be unique.")
        update_cases = {key for batch in update_batches for key in batch.case_keys}
        probe_cases = {key for batch in probe_batches for key in batch.case_keys}
        if not update_cases or not probe_cases or update_cases & probe_cases:
            raise ValueError("Counterfactual probes must use nonempty case IDs disjoint from update cases.")
        declared_terms = set(provider.loss_denominators(update_batches, phase, arm))
        for arm_name, overrides in interventions.items():
            if not arm_name or any(
                    type(value) not in (int, float)
                    or not math.isfinite(float(value))
                    or float(value) < 0
                    for value in overrides.values()):
                raise ValueError(
                    "Intervention names and LossTerm weight overrides must be finite nonnegative scalars."
                )
            if set(overrides) - declared_terms:
                raise ValueError(f"Intervention {arm_name!r} overrides undeclared terms: {sorted(set(overrides) - declared_terms)}")

        # Some existing predictors retain non-leaf tensors in transient
        # attributes during native forward. Make the pristine clone template
        # before any source-model probe can populate those caches.
        clone_template = copy.deepcopy(model).to(self.device)
        before_probe = self.measure_macro_loss(
            model, provider, probe_batches, phase=phase, arm=arm, epoch=epoch,
            training_mode=True)
        dv_gradients: dict[str, torch.Tensor] = {}
        if dv_term is not None:
            probe_denominators = provider.loss_denominators(probe_batches, phase, arm)
            if dv_term not in probe_denominators:
                raise ValueError(f"Requested d_v term {dv_term!r} is absent from the fixed probe.")
            rest = tuple(name for name in probe_denominators if name != dv_term)
            partitions = {"selected_probe_term": (dv_term,)}
            if rest:
                partitions["other_probe_terms"] = rest
            gradient_audit = self.diagnose_macro_update(
                model,
                provider,
                probe_batches,
                phase=phase,
                arm=arm,
                epoch=epoch,
                term_groups=partitions,
                training_mode=True,
                include_gradient_tensors=True,
                optimizer_state_by_name=optimizer_state_by_name,
            )
            dv_gradients = gradient_audit["gradient_tensors"]["selected_probe_term"]

        source_provider_model = getattr(provider, "model", None)
        provider_owns_source_model = source_provider_model is model
        source_group_contract = None
        if provider_owns_source_model:
            source_specs = tuple(provider.optimizer_groups(
                model, arm, self.config.stage_for_epoch(epoch)))
            source_group_contract = tuple(
                (spec.name, tuple(spec.parameter_names)) for spec in source_specs)

        results: dict[str, Any] = {}
        comparison_rng = capture_rng_state()
        provider_state_reader = getattr(provider, "training_state_dict", None)
        provider_state_loader = getattr(provider, "load_training_state_dict", None)
        starting_provider_state = copy.deepcopy(provider_state_reader()) if callable(provider_state_reader) else None
        try:
            for arm_name, overrides in interventions.items():
                restore_rng_state(comparison_rng)
                clone = copy.deepcopy(clone_template).to(self.device)
                clone.train(True)
                initial = {
                    name: value.detach().clone()
                    for name, value in clone.named_parameters()
                }
                if provider_owns_source_model:
                    # Some existing providers use their owned model throughout
                    # prediction and loss construction, including identity
                    # guards and native operator reads. Bind the disposable
                    # provider context to this clone for the entire update and
                    # probe, then restore the original binding in the outer
                    # finally block even if any provider call fails.
                    provider.model = clone
                optimizer, specs = self._make_optimizer(
                    clone, provider, arm, epoch, optimizer_state_by_name)
                if source_group_contract is not None:
                    clone_group_contract = tuple(
                        (spec.name, tuple(spec.parameter_names)) for spec in specs)
                    if clone_group_contract != source_group_contract:
                        raise ValueError(
                            "Disposable clone optimizer groups differ from the source model declaration.")
                self._set_schedules(optimizer, specs, epoch)
                update_losses, work, denominators, sampling_hash = self._run_update(
                    clone,
                    provider,
                    optimizer,
                    update_batches,
                    phase=phase,
                    arm=arm,
                    epoch=epoch,
                    update_index=0,
                    loss_weight_overrides=overrides,
                )
                deltas = {
                    name: parameter.detach() - initial[name]
                    for name, parameter in clone.named_parameters()
                }
                delta_norms: dict[str, float] = {}
                for spec in specs:
                    delta_norms[spec.name] = math.sqrt(sum(
                        float(deltas[name].double().square().sum().item())
                        for name in spec.parameter_names
                    ))
                after_probe = self.measure_macro_loss(
                    clone, provider, probe_batches, phase=phase, arm=arm, epoch=epoch,
                    training_mode=True)
                d_v = None
                if dv_term is not None:
                    d_v = sum(
                        float((dv_gradients[name].double() * deltas[name].double()).sum().item())
                        for name in dv_gradients.keys() & deltas.keys()
                    )
                results[arm_name] = {
                    "loss_weight_overrides": {name: float(value) for name, value in overrides.items()},
                    "optimizer_updates": 1,
                    "training_loss_terms": update_losses,
                    "training_denominators": denominators,
                    "training_query_sampling_sha256": sampling_hash,
                    "training_work_counts": work,
                    "optimizer_groups": [
                        {
                            "name": str(group["group_name"]),
                            "learning_rate": float(group["lr"]),
                            "weight_decay": float(group["weight_decay"]),
                            "betas": list(group["betas"]),
                            "eps": float(group["eps"]),
                            "parameter_count": len(group["params"]),
                        }
                        for group in optimizer.param_groups
                    ],
                    "parameter_delta_norms_by_group": delta_norms,
                    "probe_before": before_probe,
                    "probe_after": after_probe,
                    "probe_loss_changes": {
                        name: after_probe["loss_terms"][name]["weighted_objective"]
                        - before_probe["loss_terms"][name]["weighted_objective"]
                        for name in before_probe["loss_terms"]
                    },
                    "probe_selected_gradient_dot_update": d_v,
                    "probe_selected_gradient_term": dv_term,
                    "optimizer_state_after": optimizer_state_provenance(
                        named_optimizer_state(clone, optimizer), expected_groups=specs),
                }
                del optimizer, clone
                if starting_provider_state is not None and callable(provider_state_loader):
                    provider_state_loader(starting_provider_state)
        finally:
            try:
                if provider_owns_source_model:
                    provider.model = source_provider_model
            finally:
                try:
                    if starting_provider_state is not None and callable(provider_state_loader):
                        provider_state_loader(starting_provider_state)
                finally:
                    restore_rng_state(comparison_rng)
        return {
            "optimizer_updates": declared_optimizer_updates,
            "optimizer_state_start": "checkpoint_named_state" if optimizer_state_by_name is not None else "fresh_zero_moments",
            "update_case_keys": sorted(update_cases, key=str),
            "probe_case_keys": sorted(probe_cases, key=str),
            "arms": results,
        }

    def _run_update(
        self,
        model: nn.Module,
        provider: TaskProvider,
        optimizer: torch.optim.Optimizer,
        microbatches: Sequence[TaskBatch],
        *,
        phase: str,
        arm: str,
        epoch: int,
        update_index: int,
        loss_weight_overrides: Mapping[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], str | None]:
        planned = {
            name: _denominator(value)
            for name, value in provider.loss_denominators(microbatches, phase, arm).items()
        }
        if any(not name for name in planned):
            raise ValueError("Loss term names must be nonempty.")
        if any(value <= 0 for value in planned.values()):
            raise ValueError("A declared optimization term must have a positive macro-update denominator.")
        observed: dict[str, float] = {name: 0.0 for name in planned}
        losses: dict[str, float] = {name: 0.0 for name in planned}
        work: dict[str, float] = {}
        sampling_records: list[str] = []
        optimizer.zero_grad(set_to_none=True)
        for batch in microbatches:
            batch_auxiliary = batch.auxiliary if isinstance(batch.auxiliary, Mapping) else {}
            sample_hash = batch_auxiliary.get("query_sampling_sha256")
            if sample_hash is not None:
                if not isinstance(sample_hash, str) or len(sample_hash) != 64:
                    raise ValueError("Provider query-sampling receipt must be a SHA256 hex string.")
                sampling_records.append(sample_hash)
            scene = provider.make_scene(batch.scene_inputs)
            predictions, auxiliary_state = provider.predict_native(
                model, scene, batch.receivers, arm, phase, epoch=epoch,
                temperature=self.config.temperature_for_epoch(epoch))
            terms = provider.loss_terms(predictions, batch.targets, phase, auxiliary_state)
            if not isinstance(terms, Mapping):
                raise TypeError("loss_terms must return a mapping of names to LossTerm values.")
            objective: torch.Tensor | None = None
            for name, term in terms.items():
                if name not in planned:
                    raise ValueError(f"Loss term {name!r} has no target-only macro denominator.")
                if not torch.is_tensor(term.numerator) or term.numerator.numel() != 1:
                    raise ValueError(f"Loss numerator {name!r} must be a scalar tensor.")
                denominator = _denominator(term.denominator)
                if not math.isfinite(float(term.weight)):
                    raise ValueError(f"Loss weight {name!r} is not finite.")
                if denominator == 0:
                    if float(term.numerator.detach()) != 0.0:
                        raise ValueError(f"Loss numerator {name!r} is nonzero with an empty valid mask.")
                    continue
                observed[name] += denominator
                effective_weight = float(term.weight)
                if loss_weight_overrides is not None and name in loss_weight_overrides:
                    effective_weight = float(loss_weight_overrides[name])
                    if not math.isfinite(effective_weight):
                        raise ValueError(f"Loss override for {name!r} is not finite.")
                piece = term.numerator * (effective_weight / planned[name])
                objective = piece if objective is None else objective + piece
                losses[name] += float(term.numerator.detach()) / planned[name] * effective_weight
            for name, value in provider.work_counts(batch, predictions, auxiliary_state).items():
                work[name] = work.get(name, 0.0) + float(value)
            if objective is None:
                raise ValueError("A microbatch produced no differentiable loss terms.")
            if not bool(torch.isfinite(objective.detach())):
                raise FloatingPointError("Nonfinite optimizer objective.")
            objective.backward()
        for name, expected in planned.items():
            actual = observed.get(name, 0.0)
            if not math.isclose(actual, expected, rel_tol=1.0e-6, abs_tol=1.0e-7):
                raise ValueError(f"Actual denominator for {name!r} ({actual}) differs from prepass ({expected}).")
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]
                      if parameter.grad is not None]
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
        if not torch.isfinite(norm) or norm <= 0:
            raise FloatingPointError("Nonfinite or zero actual parameter gradient.")
        optimizer.step()
        work["gradient_norm"] = float(norm.detach())
        sampling_hash = hashlib.sha256(
            json.dumps(sampling_records, separators=(",", ":")).encode("utf-8")
        ).hexdigest() if sampling_records else None
        return losses, work, {name: observed[name] for name in observed}, sampling_hash

    @torch.no_grad()
    def _evaluate(
        self, model: nn.Module, provider: TaskProvider, arm: str, phase: str, epoch: int,
    ) -> dict[str, Any]:
        del phase
        validation_phase = "joint" if self.config.training_mode == "joint" else "hard"
        previous_mode = model.training
        rng_state = capture_rng_state()
        records: list[Mapping[str, Any]] = []
        loss_totals: dict[str, dict[str, float]] = {}
        loss_batch_counts: dict[str, int] = {}
        batch_count = 0
        model.eval()
        previous_grad_mode = torch.is_grad_enabled()
        if self.config.training_mode == "joint":
            # Native DEV metrics need the deployed values, not retained
            # backward graphs for a full receiver catalogue.
            torch.set_grad_enabled(False)
        try:
            for batch in provider.validation_batches():
                batch_count += 1
                scene = provider.make_scene(batch.scene_inputs)
                # The candidate's deployed hard route is always measured here; a
                # warmup/open or soft training phase must not leak into validation.
                predictions, auxiliary_state = provider.predict_native(
                    model, scene, batch.receivers, arm, validation_phase, epoch=epoch,
                    temperature=self.config.temperature_for_epoch(epoch))
                records.append(provider.validation_metrics(predictions, batch.targets, auxiliary_state))
                validation_loss_fn = getattr(provider, "validation_loss_terms", None)
                if callable(validation_loss_fn):
                    state_reader = getattr(provider, "training_state_dict", None)
                    state_restorer = getattr(provider, "load_training_state_dict", None)
                    provider_state = copy.deepcopy(state_reader()) if callable(state_reader) else None
                    try:
                        terms = validation_loss_fn(
                            predictions, batch.targets, auxiliary_state, batch=batch, arm=arm)
                    finally:
                        if provider_state is not None and callable(state_restorer):
                            state_restorer(provider_state)
                    if not isinstance(terms, Mapping):
                        raise TypeError("validation_loss_terms must return a mapping of names to LossTerm values.")
                    for name, term in terms.items():
                        if not isinstance(name, str) or not name:
                            raise ValueError("Validation loss term names must be nonempty strings.")
                        if not isinstance(term, LossTerm) or not torch.is_tensor(term.numerator) or term.numerator.numel() != 1:
                            raise ValueError(f"Validation loss {name!r} must be a scalar LossTerm.")
                        weight = float(term.weight)
                        denominator = _denominator(term.denominator)
                        numerator = float(term.numerator.detach())
                        if not math.isfinite(weight) or not math.isfinite(numerator):
                            raise FloatingPointError(f"Validation loss {name!r} is nonfinite.")
                        if denominator == 0:
                            if numerator != 0.0:
                                raise ValueError(f"Validation loss {name!r} has a nonzero numerator with no valid elements.")
                            continue
                        aggregate = loss_totals.setdefault(
                            name, {"numerator": 0.0, "denominator": 0.0, "weight": weight})
                        if not math.isclose(aggregate["weight"], weight, rel_tol=0.0, abs_tol=1.0e-12):
                            raise ValueError(f"Validation objective weight for {name!r} changed across batches.")
                        aggregate["numerator"] += numerator
                        aggregate["denominator"] += denominator
                        loss_batch_counts[name] = loss_batch_counts.get(name, 0) + 1
            if batch_count == 0:
                raise ValueError("The task provider returned no exposed validation batches.")
            if any(count != batch_count for count in loss_batch_counts.values()):
                raise ValueError("Each validation objective term must be present in every validation batch.")
            result = dict(provider.reduce_native_metrics(records))
            extra_metrics = getattr(provider, "extra_validation_metrics", None)
            if callable(extra_metrics):
                result.update(extra_metrics(model, arm, validation_phase))
            if loss_totals:
                result["validation_loss_terms"] = {
                    name: (values["numerator"] / values["denominator"]) * values["weight"]
                    for name, values in loss_totals.items()
                }
                result["validation_loss_term_weights"] = {
                    name: values["weight"] for name, values in loss_totals.items()
                }
                result["validation_loss_aggregation"] = (
                    "sum of numerators divided by sum of valid elements across exposed validation batches, "
                    "then multiplied by the provider objective weight"
                )
        finally:
            torch.set_grad_enabled(previous_grad_mode)
            model.train(previous_mode)
            restore_rng_state(rng_state)
        result["validation_execution_mode"] = arm
        result["validation_route_phase"] = validation_phase
        return result

    def _checkpoint_payload(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        identity: Mapping[str, Any],
        arm: str,
        epoch: int,
        history: Sequence[Mapping[str, Any]],
        best_field: float,
        best_guarded: float,
        sampler_state: Mapping[str, Any],
        group_specs: Sequence[OptimizerGroupSpec],
        provider_training_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "checkpoint_schema_version": 1,
            "workflow": identity.get("workflow", "unified_interaction_refinement"),
            "experiment_identity": _cpu_tree(identity),
            "arm": arm,
            "epoch": int(epoch),
            "current_epoch": int(epoch),
            "model_state_dict": _cpu_tree(model.state_dict()),
            "optimizer_state_by_name": named_optimizer_state(model, optimizer),
            "rng_state": _cpu_tree(capture_rng_state()),
            "sampler_state": _cpu_tree(sampler_state),
            "history": _cpu_tree(list(history)),
            "best_field_score": float(best_field),
            "best_response_guarded_score": float(best_guarded),
            "optimizer_group_names": [group["group_name"] for group in optimizer.param_groups],
            "optimizer_schedule_contract": [_optimizer_spec_payload(spec) for spec in group_specs],
            "provider_training_state": _cpu_tree(provider_training_state or {}),
        }

    @_exclusive_training_run
    def fit(
        self,
        model: nn.Module,
        provider: TaskProvider,
        output_dir: str | Path,
        *,
        identity: Mapping[str, Any],
        arm: str,
        stop_after: int,
        optimizer_seed: Mapping[str, Any] | None = None,
        resume_checkpoint: str | Path | None = None,
        branch_from_checkpoint: str | Path | None = None,
        clean_stop_name: str = "CLEAN_STOP_REQUEST.json",
        allow_microbatch_change: bool = False,
        mathematical_route_branch: Mapping[str, Any] | None = None,
        joint_core_revision: Mapping[str, Any] | None = None,
        diagnostic_age_branch: Mapping[str, Any] | None = None,
        architecture_child_initialization: Mapping[str, Any] | None = None,
        source_read_child: Mapping[str, Any] | None = None,
        source_read_child_auxiliary_fit: Callable[..., Mapping[str, Any]] | None = None,
        numerical_pooling_amendment: Mapping[str, Any] | None = None,
        output_policy_amendment: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fit one warmup or arm segment and save exact epoch-boundary state."""

        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        layout = RunLayout(output)
        layout.ensure()
        if resume_checkpoint is None:
            markers = (
                "latest_model.pt",
                "last.pt",
                "history.json",
                "fit_summary.json",
                "active_process.json",
                "progress.json",
            )
            has_epoch_checkpoints = any((output / "checkpoints").glob("epoch_*_model.pt")) or any(
                output.glob("epoch_*_model.pt")
            )
            if any(layout.read_path(name).exists() for name in markers) or has_epoch_checkpoints:
                raise ValueError("A new fit requires a fresh run directory; use exact resume for existing history.")
        else:
            requested_latest = resolve_checkpoint(output, resume_checkpoint)
            current_latest = resolve_checkpoint(output, "latest")
            if requested_latest != current_latest:
                raise ValueError("Exact resume requires this run's current latest checkpoint; historical checkpoints cannot rewind history.")
        identity = dict(identity)
        provider_identity = dict(provider.identity_payload())
        fresh_initialization = identity.get("fresh_initialization")
        fresh_provider = provider_identity.get("fresh_matched_comparison") is True
        if (fresh_initialization is not None) != fresh_provider:
            raise ValueError("Fresh matched provider identity and explicit initialization receipt must appear together.")
        if fresh_initialization is not None:
            if not isinstance(fresh_initialization, Mapping):
                raise TypeError("Fresh matched initialization receipt must be a mapping.")
            if self.config.seed != fresh_matched_recipe_seed(identity.get("recipe", {}), verify_decision=True):
                raise ValueError("Fresh matched engine seed differs from its primary/confirmation recipe.")
            if (identity.get("native_validation_policy") != FRESH_MATCHED_NATIVE_VALIDATION_POLICY
                    or identity.get("recipe", {}).get("native_validation_policy")
                    != FRESH_MATCHED_NATIVE_VALIDATION_POLICY
                    or fresh_initialization.get("native_validation_policy")
                    != FRESH_MATCHED_NATIVE_VALIDATION_POLICY):
                raise ValueError("Fresh matched recipe and identity must bind the consolidated validation policy.")
            _apply_fresh_native_validation_policy(model)
            if (branch_from_checkpoint is not None or architecture_child_initialization is not None
                    or source_read_child is not None or source_read_child_auxiliary_fit is not None
                    or diagnostic_age_branch is not None or joint_core_revision is not None
                    or mathematical_route_branch is not None or allow_microbatch_change
                    or optimizer_seed is not None):
                raise ValueError("Fresh matched arms may only start new histories or resume their own exact latest state.")
            _validate_fresh_matched_initialization(
                fresh_initialization, model, provider_identity, arm=arm,
                seed=self.config.seed, sampling_version=self.config.sampling_version,
                deterministic_case_order=self.config.deterministic_case_order)
        if "provider_identity" in identity and identity["provider_identity"] != provider_identity:
            raise ValueError("Sealed experiment provider identity differs from the active task adapter.")
        if "arm" in identity:
            raise ValueError("Matched execution arm is checkpoint metadata, not part of shared identity.")
        if resume_checkpoint is not None and branch_from_checkpoint is not None:
            raise ValueError("A run cannot be both an exact resume and a new branch.")
        if allow_microbatch_change and resume_checkpoint is None:
            raise ValueError("A microbatch amendment requires an existing run's latest checkpoint.")
        if numerical_pooling_amendment is not None and (
                resume_checkpoint is None or branch_from_checkpoint is not None
                or optimizer_seed is not None or allow_microbatch_change
                or mathematical_route_branch is not None or joint_core_revision is not None
                or diagnostic_age_branch is not None or architecture_child_initialization is not None
                or source_read_child is not None or source_read_child_auxiliary_fit is not None
                or fresh_initialization is not None or arm != "A-H6"):
            raise ValueError("A numerical pooling repair requires only the same run's exact latest-state continuation.")
        if output_policy_amendment is not None and (
                resume_checkpoint is None or branch_from_checkpoint is not None
                or optimizer_seed is not None or allow_microbatch_change
                or mathematical_route_branch is not None or joint_core_revision is not None
                or diagnostic_age_branch is not None or architecture_child_initialization is not None
                or source_read_child is not None or source_read_child_auxiliary_fit is not None
                or numerical_pooling_amendment is not None or fresh_initialization is not None
                or arm != "A2-D"):
            raise ValueError("An output-only amendment requires only this formal D run's exact latest resume.")
        if architecture_child_initialization is not None and (
                branch_from_checkpoint is None or resume_checkpoint is not None
                or optimizer_seed is not None or allow_microbatch_change
                or mathematical_route_branch is not None or joint_core_revision is not None
                or diagnostic_age_branch is not None or self.config.training_mode != "joint"
                or arm != architecture_child_initialization.get("target_mode")):
            raise ValueError("Native routing attachment requires one fresh joint architecture child of its named P parent.")
        if source_read_child is not None and (
                branch_from_checkpoint is None or resume_checkpoint is not None
                or optimizer_seed is not None or allow_microbatch_change
                or mathematical_route_branch is not None or joint_core_revision is not None
                or diagnostic_age_branch is not None or architecture_child_initialization is not None
                or numerical_pooling_amendment is not None or output_policy_amendment is not None
                or fresh_initialization is not None or self.config.training_mode != "joint"
                or arm != "A2-D"):
            raise ValueError("A source-read child requires one fresh joint A2-D branch from its named dense parent.")
        if source_read_child_auxiliary_fit is not None and source_read_child is None:
            raise ValueError("A source-read auxiliary-fit callback is valid only during its declared child attachment.")
        if diagnostic_age_branch is not None and (
                branch_from_checkpoint is None or resume_checkpoint is not None
                or allow_microbatch_change or mathematical_route_branch is not None
                or joint_core_revision is not None):
            raise ValueError("A diagnostic age branch requires a fresh child of one named checkpoint.")
        if joint_core_revision is not None and (
                branch_from_checkpoint is None or resume_checkpoint is not None
                or allow_microbatch_change or mathematical_route_branch is not None
                or self.config.training_mode != 'joint' or arm != 'J-H'):
            raise ValueError('A joint core revision applies only to a fresh J-H joint child.')
        if mathematical_route_branch is not None and (
                branch_from_checkpoint is None or resume_checkpoint is not None
                or allow_microbatch_change or arm not in ("full_detail", "adaptive_detail")):
            raise ValueError("A mathematical route declaration applies only to a new matched warmup child.")
        source_path = (
            resolve_checkpoint(output, resume_checkpoint)
            if resume_checkpoint is not None
            else resolve_checkpoint(output, branch_from_checkpoint)
            if branch_from_checkpoint is not None
            else None
        )
        if source_read_child is not None and source_path is not None:
            try:
                source_path.resolve().relative_to(output)
            except ValueError:
                pass
            else:
                raise ValueError("A source-read child must write to a distinct run directory from its dense parent.")
        source_payload: Mapping[str, Any] | None = None
        restored_rng: Mapping[str, Any] | None = None
        if source_path is not None:
            from honf_runtime.compat import load_trusted_checkpoint

            source_payload = load_trusted_checkpoint(source_path, map_location="cpu")
            if fresh_initialization is not None and (
                    source_payload.get("experiment_identity", {}).get("fresh_initialization")
                    != fresh_initialization):
                raise ValueError("Fresh matched resume checkpoint does not preserve its original initialization receipt.")
        start_epoch = 1 if source_payload is None else int(source_payload["epoch"]) + 1
        if resume_checkpoint is not None and int(source_payload["epoch"]) >= stop_after:
            raise ValueError("Exact resume stop must advance the saved epoch.")
        if not start_epoch <= stop_after <= self.config.total_epochs:
            raise ValueError("Requested stop must advance the saved age and remain inside the common horizon.")
        if not identity:
            raise ValueError("A sealed nonempty experiment identity is required.")
        model.to(self.device)
        # The warmup stage declares the stable parameter groups for the whole
        # run, including router parameters while warmup executes all fine
        # reads. All-fine execution does not imply frozen router gradients.
        initial_phase = "joint" if self.config.training_mode == "joint" else "warmup"
        group_specs = tuple(provider.optimizer_groups(model, arm, initial_phase))
        sampling_dataset_id = (
            _sampling_dataset_id(provider_identity)
            if self.config.sampling_version == SamplingKey.CASE_EPOCH_VERSION else ""
        )
        sealed_identity = {
            **dict(identity),
            "provider_identity": provider_identity,
            "engine_config": _engine_config_payload(self.config),
            "selection_policy": asdict(self.selection),
            "optimizer_schedule_contract": [_optimizer_spec_payload(spec) for spec in group_specs],
        }
        if self.config.sampling_version == SamplingKey.CASE_EPOCH_VERSION:
            sealed_identity["sampling_dataset_id"] = sampling_dataset_id
        amendment = None
        if source_payload is not None:
            if source_read_child is not None:
                from .source_read_child import prepare_source_read_child

                amendment = prepare_source_read_child(
                    model, source_payload, sealed_identity, source_read_child,
                    source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    source_path=source_path,
                    case_keys=tuple(provider.epoch_cases(start_epoch, self.config.seed)),
                    stop_after=int(stop_after))
            elif architecture_child_initialization is not None:
                from .architecture_child import prepare_native_routing_child

                amendment = prepare_native_routing_child(
                    model, source_payload, sealed_identity, architecture_child_initialization,
                    source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    case_keys=tuple(provider.epoch_cases(start_epoch, self.config.seed)),
                    stop_after=int(stop_after))
            elif diagnostic_age_branch is not None:
                amendment = _diagnostic_age_branch_amendment(
                    source_payload.get("experiment_identity", {}), sealed_identity,
                    diagnostic_age_branch,
                    source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    case_keys=tuple(provider.epoch_cases(start_epoch, self.config.seed)),
                    stop_after=int(stop_after))
                if (source_payload.get("arm") != arm
                        or int(source_payload.get("epoch", -1)) != diagnostic_age_branch["source_epoch"]):
                    raise ValueError("Diagnostic arm or source age differs from its declared checkpoint.")
            elif joint_core_revision is not None:
                amendment = _branch_joint_locality_amendment(
                    source_payload.get('experiment_identity', {}), sealed_identity, joint_core_revision)
                if (source_payload.get('arm') != arm
                        or int(source_payload.get('epoch', -1)) != joint_core_revision['source_epoch']
                        or stop_after != joint_core_revision['review_stop']):
                    raise ValueError('Joint revision must branch from its declared common age to its review stop.')
            elif mathematical_route_branch is not None:
                amendment = _branch_mathematical_route_amendment(
                    source_payload.get("experiment_identity", {}), sealed_identity, mathematical_route_branch)
            elif numerical_pooling_amendment is not None:
                if any(row.get("kind") == "explicit_content_pooling_numerical_repair"
                       for row in source_payload.get("resume_amendments", [])):
                    raise ValueError("This lineage already used its numerical pooling source amendment.")
                amendment = _content_pooling_source_amendment(
                    source_payload.get("experiment_identity", {}), sealed_identity,
                    numerical_pooling_amendment,
                    source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    source_epoch=int(source_payload["epoch"]), stop_after=stop_after)
            elif output_policy_amendment is not None:
                from .training_output_policy import validate_output_policy_amendment

                amendment = validate_output_policy_amendment(
                    source_payload.get("experiment_identity", {}), sealed_identity, output_policy_amendment,
                    source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    source_epoch=int(source_payload["epoch"]), stop_after=stop_after,
                    run_directory=str(output),
                )
            else:
                amendment = _resume_identity_amendment(source_payload.get("experiment_identity", {}), sealed_identity,
                    allow_microbatch_change=allow_microbatch_change)
            if resume_checkpoint is not None:
                if source_payload.get("arm") != arm:
                    raise ValueError("Exact resume arm differs from the saved checkpoint.")
            elif (joint_core_revision is None and diagnostic_age_branch is None
                  and architecture_child_initialization is None and source_read_child is None):
                if source_payload.get("arm") != "warmup":
                    raise ValueError("Matched arms may branch only from the shared warmup checkpoint.")
                if int(source_payload.get("epoch", -1)) != self.config.warmup_epochs:
                    raise ValueError("Matched arms must branch at the declared common warmup boundary.")
            if architecture_child_initialization is None and source_read_child is None:
                model.load_state_dict(source_payload["model_state_dict"], strict=True)
                if fresh_initialization is not None:
                    _apply_fresh_native_validation_policy(model)
            optimizer_seed = source_payload["optimizer_state_by_name"]
            restored_rng = source_payload["rng_state"]
            restore_provider_state = getattr(provider, "load_training_state_dict", None)
            provider_state = source_payload.get("provider_training_state", {})
            if source_read_child is not None:
                from .source_read_child import source_read_child_provider_state

                provider_state = source_read_child_provider_state(provider, provider_state)
                amendment["exact_native_provider_state_verified"] = True
            elif architecture_child_initialization is not None:
                from .architecture_child import architecture_child_provider_state

                provider_state = architecture_child_provider_state(provider, provider_state, arm)
                amendment["exact_non_arm_provider_state_verified"] = True
            if callable(restore_provider_state):
                restore_provider_state(provider_state)
        if resume_checkpoint is not None:
            _consume_acknowledged_clean_stop(output, clean_stop_name, layout)
        optimizer, optimizer_specs = self._make_optimizer(
            model, provider, arm, start_epoch, optimizer_seed, group_specs=group_specs)
        if source_read_child is not None:
            from .source_read_child import verify_source_read_child_optimizer

            verify_source_read_child_optimizer(
                model, optimizer, source_payload["optimizer_state_by_name"],
                source_read_child["added_parameter_names"])
            amendment["exact_inherited_moments_verified"] = True
            amendment["new_optimizer_states_at_attachment"] = 0
        elif architecture_child_initialization is not None:
            from .architecture_child import verify_architecture_child_optimizer

            verify_architecture_child_optimizer(
                model, optimizer, source_payload["optimizer_state_by_name"],
                architecture_child_initialization["added_parameter_names"])
            amendment["exact_inherited_moments_verified"] = True
            amendment["new_optimizer_states_at_attachment"] = 0
        if (resume_checkpoint is not None or joint_core_revision is not None
                or diagnostic_age_branch is not None):
            saved_group_names = source_payload.get("optimizer_group_names")
            current_group_names = [group["group_name"] for group in optimizer.param_groups]
            saved_groups = source_payload["optimizer_state_by_name"].get("groups", [])
            current_groups = named_optimizer_state(model, optimizer).get("groups", [])
            if (saved_group_names != current_group_names or saved_groups != current_groups
                    or source_payload.get("optimizer_schedule_contract")
                    != [_optimizer_spec_payload(spec) for spec in optimizer_specs]):
                raise ValueError("Exact resume optimizer group membership/order differs from its checkpoint.")
        if restored_rng is not None:
            restore_rng_state(restored_rng)
        if source_read_child is not None:
            from .source_read_child import run_source_read_child_auxiliary_fit

            rng_devices = ()
            if self.device.type == "cuda":
                rng_devices = (torch.cuda.current_device() if self.device.index is None
                               else self.device.index,)
            amendment["auxiliary_fit_receipt"] = run_source_read_child_auxiliary_fit(
                source_read_child_auxiliary_fit, model, provider, optimizer, source_payload,
                amendment,
                added_parameter_names=source_read_child["added_parameter_names"],
                train_case_keys=tuple(provider.epoch_cases(start_epoch, self.config.seed)),
                rng_devices=rng_devices)
        history = list(source_payload.get("history", [])) if source_payload is not None else []
        if fresh_initialization is not None and source_payload is not None:
            first_epoch_row = next((row for row in history if int(row.get("epoch", -1)) == 1), None)
            first_epoch_audit = (first_epoch_row.get("observed_stream_audit")
                                 if isinstance(first_epoch_row, Mapping) else None)
            if not isinstance(first_epoch_audit, Mapping):
                raise ValueError("Fresh matched exact resume is missing its observed epoch-1 stream audit.")
            if (first_epoch_audit.get("schema") != "observed_fresh_training_stream_v1"
                    or first_epoch_audit.get("case_visits") != int(first_epoch_row["case_visits"])
                    or first_epoch_audit.get("optimizer_updates") != int(first_epoch_row["optimizer_updates"])
                    or not all(_is_sha256_hex(first_epoch_audit.get(name)) for name in (
                        "case_order_sha256", "query_sampling_sha256", "observed_mask_sha256",
                        "observed_denominators_sha256"))):
                raise ValueError("Fresh matched resume epoch-1 stream audit is malformed or incomplete.")
        resume_amendments = list(source_payload.get("resume_amendments", [])) if source_payload is not None else []
        if amendment is not None:
            amendment = {**amendment, "source_epoch": int(source_payload["epoch"]), "next_epoch": start_epoch,
                "source_checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "preserved_case_visits": sum(int(row["case_visits"]) for row in history),
                "preserved_optimizer_updates": sum(int(row["optimizer_updates"]) for row in history),
                "recorded_unix": time.time()}
            resume_amendments.append(amendment)
            amendment_name = ('architecture_child_initialization' if architecture_child_initialization is not None else
                              'source_read_child' if source_read_child is not None else
                              'numerical_pooling_amendment' if numerical_pooling_amendment is not None else
                              'output_policy_amendment' if output_policy_amendment is not None else
                              'diagnostic_age_branch' if diagnostic_age_branch is not None else
                              'joint_core_revision' if joint_core_revision is not None else
                              'mathematical_route_amendment' if mathematical_route_branch is not None
                              else 'microbatch_amendment')
            _atomic_json(layout.write_path(f"{amendment_name}_epoch_{start_epoch:04d}.json",
                category='logs' if (joint_core_revision is not None or diagnostic_age_branch is not None
                                    or architecture_child_initialization is not None
                                    or source_read_child is not None
                                    or numerical_pooling_amendment is not None
                                    or output_policy_amendment is not None) else None), amendment)
        best_field = float(source_payload.get("best_field_score", float("inf"))) if source_payload else float("inf")
        best_guarded = float(source_payload.get("best_response_guarded_score", float("inf"))) if source_payload else float("inf")
        branch_parent = (copy.deepcopy(source_payload.get("branch_parent"))
                         if source_payload is not None and resume_checkpoint is not None else None)
        if branch_from_checkpoint is not None:
            branch_parent = {
                "path": str(source_path),
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "identity": source_payload.get("experiment_identity"),
                "epoch": int(source_payload["epoch"]),
            }
            if source_read_child is not None:
                branch_parent.update(
                    source_read_child=copy.deepcopy(amendment),
                    inherited_case_visits=sum(int(row["case_visits"]) for row in source_payload.get("history", [])),
                    inherited_optimizer_updates=sum(int(row["optimizer_updates"]) for row in source_payload.get("history", [])),
                    inherited_history_scope="dense A2-D physical history through e500; child physical rows begin at e501",
                )
            elif joint_core_revision is None:
                history = []
            else:
                branch_parent.update(
                    inherited_case_visits=sum(int(row['case_visits']) for row in history),
                    inherited_optimizer_updates=sum(int(row['optimizer_updates']) for row in history),
                    joint_core_revision=copy.deepcopy(amendment),
                    inherited_history_scope='parent function through the common boundary',
                )
            if diagnostic_age_branch is not None:
                branch_parent.update(
                    diagnostic_age_branch=copy.deepcopy(amendment),
                    inherited_case_visits=sum(int(row['case_visits']) for row in source_payload.get('history', [])),
                    inherited_optimizer_updates=sum(int(row['optimizer_updates']) for row in source_payload.get('history', [])),
                )
            if architecture_child_initialization is not None:
                branch_parent.update(
                    architecture_child_initialization=copy.deepcopy(amendment),
                    inherited_case_visits=sum(int(row['case_visits']) for row in source_payload.get('history', [])),
                    inherited_optimizer_updates=sum(int(row['optimizer_updates']) for row in source_payload.get('history', [])),
                    inherited_history_scope='parent P function; child history counts only new visits and updates',
                )
            best_field = best_guarded = float("inf")
        diagnostic_contract = (
            branch_parent.get("diagnostic_age_branch")
            if isinstance(branch_parent, Mapping) else None
        )
        if diagnostic_contract is not None:
            if (stop_after not in diagnostic_contract["review_epochs"]
                    or stop_after > diagnostic_contract["review_stop"]):
                raise ValueError("A diagnostic continuation cannot extend its declared reviews or update budget.")
            evidence = Path(diagnostic_contract["evidence_path"]).read_bytes()
            if hashlib.sha256(evidence).hexdigest() != diagnostic_contract["evidence_sha256"]:
                raise ValueError("Diagnostic selection evidence changed before continuation.")
        architecture_contract = (
            branch_parent.get("architecture_child_initialization")
            if isinstance(branch_parent, Mapping) else None
        )
        if architecture_contract is not None and (
                stop_after not in architecture_contract["review_epochs"]
                or stop_after > architecture_contract["final_epoch"]):
            raise ValueError("Architecture child continuation must stay within its sealed reviews and update horizon.")
        source_read_child_contract = (
            branch_parent.get("source_read_child")
            if isinstance(branch_parent, Mapping) else None
        )
        if source_read_child_contract is not None:
            from .source_read_child import verify_source_read_child_continuation

            verify_source_read_child_continuation(source_read_child_contract, stop_after)
        active = {
            "status": "running", "arm": arm, "identity": sealed_identity, "device": str(self.device),
            "pid": os.getpid(),
            "process_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
            "start_epoch": start_epoch, "requested_stop": int(stop_after),
            "started_unix": float(os.getenv("PROCESS_STARTED_UNIX", time.time())),
            "engine_started_unix": time.time(),
            "branch_parent": branch_parent,
        }
        if resume_amendments:
            active["resume_amendments"] = resume_amendments
        if resume_checkpoint is None and layout.canonical_writes:
            identity_path = layout.path("experiment_identity.json")
            marker_path = layout.path("artifact_layout.json")
            if not marker_path.exists():
                _atomic_json(marker_path, {
                    "artifact_layout_version": 1,
                    "run_dir": str(output),
                })
            if not identity_path.exists():
                _atomic_json(identity_path, sealed_identity)
            software_path = layout.path("software.json")
            if not software_path.exists():
                _atomic_json(software_path, environment_snapshot())
            source_state_path = layout.path("source_state.json")
            if not source_state_path.exists():
                _atomic_json(source_state_path, source_state_snapshot(PROJECT_ROOT))
        _atomic_json(layout.write_path("active_process.json"), active)
        if resume_checkpoint is not None:
            restored = {"event": "checkpoint_restored", "source_epoch": start_epoch - 1,
                "next_epoch": start_epoch, "microbatch_cases": self.config.microbatch_cases,
                "effective_cases": self.config.effective_cases,
                "preserved_case_visits": sum(int(row["case_visits"]) for row in history),
                "preserved_optimizer_updates": sum(int(row["optimizer_updates"]) for row in history)}
            print(json.dumps(restored, sort_keys=True), flush=True)
            if bool(getattr(sys.stderr, "isatty", lambda: False)()):
                print(f"Restored epoch {start_epoch - 1}; next {start_epoch}/{stop_after}; "
                      f"microbatch {self.config.microbatch_cases}, effective {self.config.effective_cases}; "
                      f"retained {restored['preserved_case_visits']} TRAIN visits, "
                      f"{restored['preserved_optimizer_updates']} optimizer updates", file=sys.stderr, flush=True)
        total_started = time.perf_counter()
        stopped_at = start_epoch - 1
        latest_epoch = int(source_payload["epoch"]) if resume_checkpoint is not None else 0
        completed_visits = sum(int(row["case_visits"]) for row in history)
        completed_updates = sum(int(row["optimizer_updates"]) for row in history)
        clean_stopped = False
        previous_phase = self.config.stage_for_epoch(start_epoch - 1) if start_epoch > 1 else None
        try:
            for epoch in range(start_epoch, int(stop_after) + 1):
                # Callbacks (preparation, validation, or a prior calibration hook)
                # may leave the module in evaluation mode. The engine owns the
                # mode for every optimization epoch.
                model.train(True)
                phase = self.config.stage_for_epoch(epoch)
                stage_receipt = None
                if phase != previous_phase:
                    on_phase_start = getattr(provider, "on_phase_start", None)
                    if callable(on_phase_start):
                        stage_receipt = on_phase_start(
                            model=model, arm=arm, epoch=epoch, phase=phase,
                            temperature=self.config.temperature_for_epoch(epoch))
                    if stage_receipt is not None:
                        _atomic_json(layout.write_path(f"stage_receipt_epoch_{epoch:04d}_{phase}.json"), stage_receipt)
                # A stage callback may run calibration/evaluation in no-grad mode.
                # Reassert training mode after it and before any optimizer work.
                model.train(True)
                previous_phase = phase
                self._set_schedules(optimizer, group_specs, epoch)
                training_memory = None
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)
                    training_memory = {
                        "baseline_allocated_bytes": torch.cuda.memory_allocated(self.device),
                        "baseline_reserved_bytes": torch.cuda.memory_reserved(self.device),
                    }
                epoch_started = time.perf_counter()
                ordered_cases, sampling_dataset_id = self.epoch_case_order(provider, epoch)
                if diagnostic_contract is not None and (
                        len(set(ordered_cases)) != len(ordered_cases)
                        or set(ordered_cases) != set(diagnostic_contract["fit_case_keys"])
                        or completed_updates + math.ceil(len(ordered_cases) / self.config.effective_cases)
                        > diagnostic_contract["scientific_update_limit"]):
                    raise ValueError("Diagnostic case membership or optimizer update budget changed during fitting.")
                if architecture_contract is not None and (
                        len(ordered_cases) != len(architecture_contract["fit_case_keys"])
                        or set(ordered_cases) != set(architecture_contract["fit_case_keys"])
                        or completed_updates + math.ceil(len(ordered_cases) / self.config.effective_cases)
                        > architecture_contract["scientific_update_limit"]):
                    raise ValueError("Architecture child case membership or optimizer update cap changed during fitting.")
                order_hash = _case_order_digest(ordered_cases)
                capture_fresh_stream = fresh_initialization is not None and epoch == 1
                observed_mask_hashes: list[dict[str, Any]] = []
                observed_query_hashes: list[dict[str, Any]] = []
                observed_denominator_records: list[dict[str, Any]] = []
                update_rows: list[dict[str, Any]] = []
                update_count = math.ceil(len(ordered_cases) / self.config.effective_cases)
                progress = tqdm(
                    total=len(ordered_cases), desc=f"epoch {epoch}/{stop_after}", unit="case",
                    dynamic_ncols=True, file=sys.stderr,
                    disable=not self.config.progress_enabled, leave=False,
                )
                try:
                    for update_index, group_slice in enumerate(self.epoch_update_slices(len(ordered_cases))):
                        update_cases = ordered_cases[group_slice]
                        microbatches = self.prepare_update_microbatches(
                            provider, arm=arm, phase=phase, epoch=epoch,
                            update_index=update_index, update_cases=update_cases,
                            sampling_dataset_id=sampling_dataset_id)
                        losses, work, denominators, sampling_hash = self._run_update(model, provider, optimizer, microbatches,
                            phase=phase, arm=arm, epoch=epoch, update_index=update_index)
                        if capture_fresh_stream:
                            if sampling_hash is None:
                                raise ValueError("Fresh matched epoch 1 requires observed receiver/query-ID hashes.")
                            observed_mask_hashes.append({
                                "update_index": update_index,
                                "sha256": _observed_mask_digest(microbatches),
                            })
                            observed_query_hashes.append({
                                "update_index": update_index,
                                "sha256": sampling_hash,
                            })
                            observed_denominator_records.append({
                                "update_index": update_index,
                                "denominators": {name: float(value) for name, value in sorted(denominators.items())},
                            })
                        update_rows.append({"losses": losses, "work": work, "case_count": len(update_cases),
                                            "microbatch_count": len(microbatches),
                                            "query_sampling_sha256": sampling_hash})
                        loss_value = sum(float(value) for value in losses.values())
                        lr_values = ",".join(
                            f"{group['group_name']}:{float(group['lr']):.1e}" for group in optimizer.param_groups
                        )
                        progress.set_postfix({
                            "updates": f"{len(update_rows)}/{update_count}", "loss": f"{loss_value:.4g}",
                            "lr": lr_values, "phase": phase,
                        }, refresh=False)
                        progress.update(len(update_cases))
                        _atomic_json(layout.write_path("progress.json"), {
                            "status": "training", "epoch_in_progress": epoch, "completed_epoch": epoch - 1,
                            "phase": phase, "updates_this_epoch": len(update_rows),
                            "case_visits_this_epoch": sum(item["case_count"] for item in update_rows),
                            "completed_case_visits": completed_visits,
                            "completed_optimizer_updates": completed_updates,
                            "latest_checkpoint_epoch": latest_epoch, "updated_unix": time.time(),
                        })
                finally:
                    progress.close()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    training_memory.update({
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                        "scope": "complete epoch including data transfer, forward, backward and optimizer; excludes monitoring",
                    })
                train_seconds = time.perf_counter() - epoch_started
                row: dict[str, Any] = {
                    "epoch": epoch,
                    "phase": phase,
                    "temperature": self.config.temperature_for_epoch(epoch),
                    "arm": arm,
                    "case_visits": len(ordered_cases),
                    "optimizer_updates": len(update_rows),
                    "microbatches": sum(item["microbatch_count"] for item in update_rows),
                    "partial_update_cases": len(ordered_cases) % self.config.effective_cases,
                    "learning_rates": {group["group_name"]: float(group["lr"]) for group in optimizer.param_groups},
                    "train_seconds": train_seconds,
                    "train_losses": _mean_mapping([item["losses"] for item in update_rows]),
                    "work_counts": _sum_mapping([item["work"] for item in update_rows]),
                    "case_order_sha256": order_hash,
                }
                if training_memory is not None:
                    row["training_cuda_memory"] = training_memory
                sample_hashes = [item["query_sampling_sha256"] for item in update_rows
                                 if item["query_sampling_sha256"] is not None]
                if sample_hashes:
                    row["query_sampling_sha256"] = hashlib.sha256(
                        json.dumps(sample_hashes, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()
                if capture_fresh_stream:
                    if not sample_hashes or len(observed_query_hashes) != len(update_rows):
                        raise ValueError("Fresh matched epoch-1 sampling audit is incomplete.")
                    row["observed_stream_audit"] = {
                        "schema": "observed_fresh_training_stream_v1",
                        "sampler_namespace_sha256": _canonical_sha256(
                            fresh_initialization["sampler_namespace"]),
                        "case_order_sha256": order_hash,
                        "case_visits": len(ordered_cases),
                        "optimizer_updates": len(update_rows),
                        "microbatches": sum(item["microbatch_count"] for item in update_rows),
                        "query_sampling_sha256": row["query_sampling_sha256"],
                        "query_sampling_by_update_sha256": observed_query_hashes,
                        "observed_mask_sha256": _stream_records_digest(observed_mask_hashes),
                        "mask_sha256_by_update": observed_mask_hashes,
                        "observed_denominators_sha256": _stream_records_digest(observed_denominator_records),
                        "observed_denominators_by_update": observed_denominator_records,
                        "observed_denominator_totals": _sum_mapping([
                            record["denominators"] for record in observed_denominator_records]),
                        "effective_cases": self.config.effective_cases,
                        "microbatch_cases": self.config.microbatch_cases,
                        "sampling_dataset_id": sampling_dataset_id,
                    }
                if stage_receipt is not None:
                    row["stage_receipt"] = stage_receipt
                clean_stop_request = _read_clean_stop_request(output, clean_stop_name)
                clean_stop_requested = clean_stop_request is not None
                review = (epoch % self.config.monitor_every == 0 or epoch in self.config.monitor_epochs
                          or epoch == stop_after or clean_stop_requested)
                improved_field = improved_guarded = False
                if review:
                    validation_started = time.perf_counter()
                    if fresh_initialization is not None:
                        _apply_fresh_native_validation_policy(model)
                    metrics = self._evaluate(model, provider, arm, phase, epoch)
                    validation_terms = dict(metrics.pop("validation_loss_terms", {}))
                    validation_weights = dict(metrics.pop("validation_loss_term_weights", {}))
                    aggregation = metrics.pop("validation_loss_aggregation", None)
                    train_terms = dict(row["train_losses"])
                    matched_terms = sorted(set(train_terms) & set(validation_terms))
                    training_only_terms = sorted(set(train_terms) - set(validation_terms))
                    validation_only_terms = sorted(set(validation_terms) - set(train_terms))
                    training_reduction = (
                        "arithmetic mean of macro-update means; each optimizer update has equal weight, "
                        "including any partial final update"
                    )
                    validation_reduction = aggregation or "No held-out objective losses were recorded by this provider."
                    objective = {
                        "scope": "exposed VALIDATION panel evaluated with deployed hard inference",
                        "training_phase": phase,
                        "validation_route_phase": "hard",
                        "training_reduction": training_reduction,
                        "validation_reduction": validation_reduction,
                        "training_prediction_mode": (
                            "phase-specific optimization forward; provider loss terms may blend main and fine paths"
                        ),
                        "validation_prediction_mode": (
                            "deployed hard inference; validation terms use the provider-defined hard-route outputs"
                        ),
                        "directly_comparable": False,
                        "comparability_note": (
                            "Shared term names do not imply identical predictions or reductions: TRAIN losses are "
                            "averaged per optimizer update over changing model states; VALIDATION losses pool "
                            "valid elements at the review epoch using hard inference."
                        ),
                        "terms": validation_terms,
                        "term_weights": validation_weights,
                        "matched_terms": matched_terms,
                        "training_only_terms": training_only_terms,
                        "validation_only_terms": validation_only_terms,
                        "training_matched_total": (sum(train_terms[name] for name in matched_terms)
                                                   if matched_terms else None),
                        "validation_matched_total": (sum(validation_terms[name] for name in matched_terms)
                                                     if matched_terms else None),
                        "is_full_training_objective": not training_only_terms and not validation_only_terms,
                        "is_full_training_objective_note": (
                            "This flag reports term-name coverage only; directly_comparable separately describes "
                            "whether values use the same prediction path and reduction."
                        ),
                        "aggregation": aggregation,
                        "definition": (
                            "Shared-name subtotal includes only terms measured on both sides, using the provider's "
                            "objective weights. It is not a directly comparable objective: reductions differ and "
                            "provider prediction paths may differ. Hard validation omits full-detail replay and any "
                            "train-only PDE or auxiliary terms."
                        ),
                    }
                    if self.config.training_mode == "joint":
                        objective.update({
                            "scope": "fixed exposed DEV panel evaluated with the deployed joint model",
                            "validation_route_phase": "joint",
                            "training_prediction_mode": "all physical heads and shared blocks train jointly",
                            "validation_prediction_mode": "same deployed joint prediction path",
                            "comparability_note": "TRAIN averages macro-update means over changing joint weights; exposed DEV pools valid elements at fixed review weights. Neither is independent TEST evidence.",
                            "definition": "Provider-defined physical objectives; TRAIN averages macro-update means, DEV pools valid elements. Stored TRAIN response/operator targets remain separate.",
                        })
                    loss_metadata = getattr(provider, "loss_metadata", None)
                    if callable(loss_metadata):
                        objective["term_definitions"] = dict(loss_metadata())
                    row["validation_objective"] = objective
                    metrics["loss_objective"] = objective
                    row["validation"] = metrics
                    row["validation_seconds"] = time.perf_counter() - validation_started
                    field_score = float(metrics[self.selection.field_metric])
                    if not math.isfinite(field_score):
                        raise FloatingPointError("Nonfinite declared field selector.")
                    improved_field = field_score < best_field
                    if improved_field:
                        best_field = field_score
                    guard_key = self.selection.response_guard_metric
                    guard_pass = guard_key is None or float(metrics[guard_key]) <= self.selection.maximum_response_ratio
                    improved_guarded = guard_key is not None and guard_pass and field_score < best_guarded
                    if improved_guarded:
                        best_guarded = field_score
                history.append(row)
                print(json.dumps(row, sort_keys=True, default=str, allow_nan=False), flush=True)
                milestone = ((review if self.config.checkpoint_epochs is None
                              else epoch in self.config.checkpoint_epochs)
                             or epoch == stop_after or improved_guarded)
                curve_due = self.config.curve_every is not None and (
                    (epoch == 1 and (self.config.training_mode != "joint"
                                     or self.config.write_initial_artifacts))
                    or epoch % self.config.curve_every == 0 or epoch == stop_after or clean_stop_requested)
                should_save = (review or milestone or curve_due
                               or (self.config.latest_every is not None and epoch % self.config.latest_every == 0))
                if should_save:
                    sampler_state = {
                        "seed": self.config.seed,
                        "completed_epoch": epoch,
                        "next_epoch": epoch + 1,
                        "next_case_cursor": 0,
                        "last_case_order_sha256": order_hash,
                        "case_visits_this_epoch": len(ordered_cases),
                        "optimizer_updates_this_epoch": len(update_rows),
                    }
                    payload = self._checkpoint_payload(model, optimizer, identity=sealed_identity, arm=arm, epoch=epoch,
                        history=history, best_field=best_field, best_guarded=best_guarded, sampler_state=sampler_state,
                        group_specs=optimizer_specs,
                        provider_training_state=getattr(provider, "training_state_dict", dict)())
                    if resume_amendments:
                        payload["resume_amendments"] = resume_amendments
                    if branch_parent is not None:
                        payload["branch_parent"] = branch_parent
                    _atomic_json(layout.write_path("history.json"), history)
                    if milestone:
                        _atomic_torch_save(layout.write_path(f"epoch_{epoch:04d}_model.pt"), payload)
                    validation_sidecar_due = (
                        epoch == stop_after or clean_stop_requested
                        or (self.config.checkpoint_epochs is None
                            or epoch in self.config.checkpoint_epochs)
                    )
                    if review and validation_sidecar_due:
                        _atomic_json(layout.write_path(f"validation_epoch_{epoch:04d}.json"), row["validation"])
                    if review:
                        if improved_field:
                            _atomic_torch_save(layout.write_path("best_by_field_mse_model.pt"), payload)
                        if improved_guarded:
                            selected_checkpoint = layout.write_path(f"epoch_{epoch:04d}_model.pt")
                            _atomic_json(layout.write_path("best_by_response_guarded_selection.json"), {
                                "selector": "response_guarded_field_score",
                                "field_metric": self.selection.field_metric,
                                "field_score": field_score,
                                "guard_metric": guard_key,
                                "guard_value": float(metrics[guard_key]),
                                "maximum_guard_ratio": self.selection.maximum_response_ratio,
                                "epoch": int(epoch),
                                "checkpoint": str(selected_checkpoint.relative_to(output)),
                                "checkpoint_sha256": hashlib.sha256(selected_checkpoint.read_bytes()).hexdigest(),
                            })
                    # Advance the resume cursor only after required milestone
                    # and selector artifacts exist, including the terminal age.
                    _atomic_torch_save(layout.write_path("latest_model.pt"), payload)
                    latest_epoch = epoch
                if curve_due:
                    _atomic_json(layout.write_path("history.json"), history)
                    curve_rng = capture_rng_state()
                    try:
                        _render_loss_curves(
                            history,
                            layout.write_path("loss_curves.png").parent,
                            self.selection.field_metric,
                            **({"loss_metadata": provider.loss_metadata()}
                               if callable(getattr(provider, "loss_metadata", None)) else {}),
                        )
                    finally:
                        restore_rng_state(curve_rng)
                completed_visits += len(ordered_cases)
                completed_updates += len(update_rows)
                _atomic_json(layout.write_path("progress.json"), {
                    "status": "epoch_completed", "completed_epoch": epoch, "phase": phase,
                    "completed_case_visits": completed_visits, "completed_optimizer_updates": completed_updates,
                    "latest_checkpoint_epoch": latest_epoch, "updated_unix": time.time(),
                })
                stopped_at = epoch
                if clean_stop_requested:
                    acknowledgement = {
                        "request_id": clean_stop_request["request_id"],
                        "requested_unix": clean_stop_request.get("requested_unix"),
                        "requested_by": clean_stop_request.get("requested_by"),
                        "acknowledged_unix": time.time(),
                        "epoch": epoch,
                        "phase": phase,
                    }
                    _atomic_json(layout.write_path("clean_stop_acknowledged.json"), acknowledgement)
                    request_tag = hashlib.sha256(clean_stop_request["request_id"].encode("utf-8")).hexdigest()[:16]
                    _atomic_json(layout.write_path(f"clean_stop_consumed_{request_tag}.json"), {
                        "request": clean_stop_request,
                        "acknowledgement": acknowledgement,
                        "consumed_on_resume": False,
                    })
                    (output / clean_stop_name).unlink(missing_ok=True)
                    clean_stopped = True
                    break
            elapsed = time.perf_counter() - total_started
            active.update(status="clean_stopped" if clean_stopped else "completed",
                          completed_epoch=stopped_at, ended_unix=time.time(), engine_seconds=elapsed,
                          process_seconds=max(0.0, time.time() - float(active["started_unix"])))
            _atomic_json(layout.write_path("active_process.json"), active)
            guarded_summary = best_guarded if math.isfinite(best_guarded) else None
            _atomic_json(layout.write_path("fit_summary.json"), {
                **active,
                "case_visits": sum(int(item["case_visits"]) for item in history if int(item["epoch"]) >= start_epoch),
                "optimizer_updates": sum(int(item["optimizer_updates"]) for item in history if int(item["epoch"]) >= start_epoch),
                "cumulative_case_visits": completed_visits,
                "cumulative_optimizer_updates": completed_updates,
                "cumulative_microbatches": sum(int(item["microbatches"]) for item in history),
                "cumulative_partial_update_cases": sum(int(item["partial_update_cases"]) for item in history),
                "best_field_score": best_field,
                "best_response_guarded_score": guarded_summary,
                "branch_parent": branch_parent,
            })
            return {**active, "completed_epoch": stopped_at, "best_field_score": best_field,
                    "best_response_guarded_score": guarded_summary, "branch_parent": branch_parent}
        except BaseException as exc:
            active.update(status="failed", completed_epoch=stopped_at, ended_unix=time.time(),
                          engine_seconds=time.perf_counter() - total_started,
                          process_seconds=max(0.0, time.time() - float(active["started_unix"])),
                          exception_type=type(exc).__name__, exception_message=str(exc))
            try:
                _atomic_json(layout.write_path("active_process.json"), active)
            except OSError:
                pass  # Preserve the original failure when its receipt cannot be written.
            raise


def _mean_mapping(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([float(row[key]) for row in rows if key in row])) for key in keys}


def _sum_mapping(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            result[key] = result.get(key, 0.0) + float(value)
    return result
