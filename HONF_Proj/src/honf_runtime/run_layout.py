"""Canonical locations and legacy reads for managed training-run artifacts.

New runs write each artifact once into its category directory. Existing runs
that predate the layout marker continue writing at their established root paths
when they already contain root-level training artifacts; moving those files is
an explicit migration operation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RUN_ARTIFACT_LAYOUT_VERSION = 1

_CATEGORIES = (
    "checkpoints",
    "metrics",
    "plots/training",
    "plots/diagnostics",
    "evaluations",
    "comparisons",
    "logs",
    "configs",
    "environment",
)
_CATEGORY_ROOTS = tuple(sorted(_CATEGORIES, key=len, reverse=True))
_CHECKPOINT_ALIASES = {
    "best_total.pt": "best_model.pt",
    "best_field.pt": "best_by_field_mse_model.pt",
    "best_temperature.pt": "best_by_temperature_mse_model.pt",
    "best_autonomous.pt": "best_predicted_model.pt",
    "latest.pt": "latest_model.pt",
}
_SELECTOR_FILES = {
    "best": "best_model.pt",
    "best_total": "best_model.pt",
    "best_field": "best_by_field_mse_model.pt",
    "best_by_field_mse": "best_by_field_mse_model.pt",
    "best_temperature": "best_by_temperature_mse_model.pt",
    "best_by_temperature_mse": "best_by_temperature_mse_model.pt",
    "best_autonomous": "best_predicted_model.pt",
    "best_predicted": "best_predicted_model.pt",
    "latest": "latest_model.pt",
}
_LEGACY_ROOT_MARKERS = (
    "active_process.json",
    "best_by_field_mse_model.pt",
    "best_by_response_guarded_selection.json",
    "best_by_temperature_mse_model.pt",
    "best_model.pt",
    "best_predicted_model.pt",
    "fit_summary.json",
    "history.json",
    "last.pt",
    "latest_model.pt",
    "loss_curves.png",
    "loss_history.csv",
    "metrics.csv",
    "progress.json",
    "summary.json",
)
_EPOCH_CHECKPOINT = re.compile(r"^epoch_\d+_model\.pt$", re.IGNORECASE)
_VALIDATION_RECORD = re.compile(r"^validation_epoch_\d+\.json$", re.IGNORECASE)
_CANONICAL_PREFIXES = _CATEGORY_ROOTS


def _safe_relative(name: str | Path) -> Path:
    value = Path(name)
    if value.is_absolute() or not value.parts:
        raise ValueError(f"Run artifact name must be a nonempty relative path: {name!r}")
    if any(part in {"..", "."} for part in value.parts):
        raise ValueError(f"Run artifact path cannot traverse directories: {name!r}")
    return value


def _canonical_relative(name: str | Path, *, category: str | None = None) -> Path:
    relative = _safe_relative(name)
    if category is not None:
        if category not in _CATEGORIES:
            raise ValueError(f"Unknown run artifact category {category!r}.")
        return Path(category) / relative

    as_posix = relative.as_posix()
    if any(as_posix == prefix or as_posix.startswith(prefix + "/") for prefix in _CANONICAL_PREFIXES):
        return relative

    if relative.parts[0] in {"diagnostics", "diagnostic_plots"}:
        return Path("plots/diagnostics", *relative.parts[1:])

    basename = relative.name
    lower = basename.lower()
    if lower in _SELECTOR_FILES.values():
        return Path("checkpoints") / basename
    if lower in _CHECKPOINT_ALIASES:
        return Path("checkpoints") / _CHECKPOINT_ALIASES[lower]
    if lower.endswith((".pt", ".pth", ".ckpt")) or _EPOCH_CHECKPOINT.match(basename):
        return Path("checkpoints") / basename

    if lower in {"metrics.csv", "loss_history.csv", "history.json", "summary.json", "fit_summary.json"}:
        return Path("metrics") / basename
    if lower == "best_by_response_guarded_selection.json":
        return Path("evaluations/selection") / basename
    if lower.startswith("validation_epoch_") and lower.endswith(".json"):
        return Path("evaluations/validation") / basename
    if lower.startswith(("evaluation_", "eval_")) or lower in {"evaluation.json", "evaluation_manifest.json"}:
        return Path("evaluations") / relative
    if lower.startswith(("comparison_", "compare_")) or lower in {
        "comparison.json",
        "comparison.csv",
        "historical_comparison.json",
    }:
        return Path("comparisons") / relative
    if lower in {
        "active_process.json",
        "clean_stop_acknowledged.json",
        "config_validation.json",
        "fit_summary.json",
        "progress.json",
        "resource_sessions.json",
    } or lower.startswith(("clean_stop_consumed_", "first10_forecast", "microbatch_amendment_", "stage_receipt_")):
        return Path("logs") / basename
    if lower in {
        "formal_profile_binding.json",
        "fit_identity.json",
        "formal_recipe.json",
        "paired_recipe.json",
        "experiment_identity.json",
        "recovery_config.json",
    }:
        return Path("configs") / basename
    if lower.startswith(("core_source", "case_source", "config_", "cli_overrides", "resolved_config", "recipe")):
        return Path("configs") / basename
    if lower in {"software.json", "source_state.json", "environment.json", "hardware.json"}:
        return Path("environment") / basename
    if lower in {"flow_learning.pdf", "thermal_learning.pdf"}:
        return Path("plots/training") / basename
    if lower.startswith(("loss_curve", "loss_curves", "training_curve", "training_curves")):
        return Path("plots/training") / basename
    if lower.startswith(("diagnostic_", "routing_", "support_", "interaction_")) and lower.endswith(
        (".png", ".pdf", ".svg")
    ):
        return Path("plots/diagnostics") / relative
    if lower.endswith((".png", ".pdf", ".svg")):
        return Path("plots/training") / relative
    if lower.endswith((".log", ".jsonl")):
        return Path("logs") / relative
    if lower in {"artifact_layout.json", "run_manifest.json", ".training.lock", "clean_stop_request.json"}:
        return relative
    if lower in {"config.json", "runtime_config.json"} or lower.startswith("runtime_config_"):
        return Path("configs") / basename
    if lower in {"launch_formal.sh"}:
        return Path("configs") / basename
    if lower in {
        "launch_receipt.json",
        "recovery_startup_verified.json",
        "resume_startup_verified.json",
        "startup_verified.json",
    }:
        return Path("logs") / basename
    if lower.endswith("_metrics_schema.json") or lower in {"loss_history_schema.json", "training_metrics_schema.json"}:
        return Path("metrics") / basename
    raise ValueError(f"No canonical run-artifact category is defined for {str(name)!r}.")


class RunLayout:
    """Resolve canonical and legacy paths inside a single run directory."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.canonical_writes = self._detect_canonical_writes()

    def _detect_canonical_writes(self) -> bool:
        layout_marker = self.run_dir / "artifact_layout.json"
        if layout_marker.is_file():
            try:
                marker = json.loads(layout_marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                marker = {}
            marker_version = marker.get("artifact_layout_version", marker.get("version", 0))
            if int(marker_version or 0) >= RUN_ARTIFACT_LAYOUT_VERSION:
                return True

        manifest_path = self.run_dir / "run_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            if int(manifest.get("artifact_layout_version", 0) or 0) >= RUN_ARTIFACT_LAYOUT_VERSION:
                return True

        if any((self.run_dir / name).is_file() for name in _LEGACY_ROOT_MARKERS):
            return False
        if any(_EPOCH_CHECKPOINT.match(path.name) for path in self.run_dir.glob("epoch_*_model.pt")):
            return False
        if any(_VALIDATION_RECORD.match(path.name) for path in self.run_dir.glob("validation_epoch_*.json")):
            return False
        return not any(
            (self.run_dir / name).is_dir() and any((self.run_dir / name).rglob("*"))
            for name in ("diagnostics", "diagnostic_plots")
        )

    def path(self, name: str | Path, *, category: str | None = None) -> Path:
        """Return the canonical path for a known artifact basename or path."""

        return self.run_dir / _canonical_relative(name, category=category)

    def _legacy_write_path(self, name: str | Path, canonical: Path) -> Path:
        relative = _safe_relative(name)
        if relative.parts[0] in {"diagnostics", "diagnostic_plots"}:
            return self.run_dir / relative
        canonical_relative = canonical.relative_to(self.run_dir)
        canonical_text = canonical_relative.as_posix()
        if _VALIDATION_RECORD.match(relative.name) or relative.name.lower() == "best_by_response_guarded_selection.json":
            return self.run_dir / relative
        if canonical_text.startswith("plots/diagnostics/"):
            return self.run_dir / "diagnostic_plots" / Path(canonical_text).relative_to("plots/diagnostics")
        if canonical_text.startswith("evaluations/"):
            return self.run_dir / canonical_relative
        if canonical_text.startswith("comparisons/"):
            return self.run_dir / canonical_relative
        if canonical_text.startswith(("configs/", "environment/")):
            return self.run_dir / canonical_relative
        if any(relative.as_posix() == prefix or relative.as_posix().startswith(prefix + "/")
               for prefix in _CANONICAL_PREFIXES):
            return self.run_dir / canonical.name
        return self.run_dir / relative

    def _legacy_read_paths(self, name: str | Path, canonical: Path) -> tuple[Path, ...]:
        relative = _safe_relative(name)
        canonical_relative = canonical.relative_to(self.run_dir)
        basename = relative.name
        candidates: list[Path] = []

        if relative.parts[0] in {"diagnostics", "diagnostic_plots"}:
            candidates.extend((self.run_dir / "diagnostic_plots" / Path(*relative.parts[1:]),
                               self.run_dir / "diagnostics" / Path(*relative.parts[1:])))
        elif canonical_relative.parts[:2] == ("plots", "diagnostics"):
            suffix = Path(*canonical_relative.parts[2:])
            candidates.extend((self.run_dir / "diagnostic_plots" / suffix,
                               self.run_dir / "diagnostics" / suffix))
        else:
            candidates.append(self.run_dir / relative)

        # Older producers used both the run root and already-established
        # category directories. Check either spelling after the canonical path.
        candidates.append(self.run_dir / canonical_relative)
        candidates.append(self.run_dir / basename)
        if canonical_relative.parts[0] == "checkpoints":
            candidates.extend((self.run_dir / "checkpoints" / basename,
                               self.run_dir / _CHECKPOINT_ALIASES.get(basename.lower(), basename)))
        deduplicated: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                deduplicated.append(candidate)
        return tuple(deduplicated)

    def read_path(self, name: str | Path, *, category: str | None = None) -> Path:
        """Return the current producer's artifact path with legacy fallbacks.

        Marked runs are canonical-first. In an unmarked legacy run, some old
        finalizers left stale copies in category directories while active
        writers continued replacing root-level metrics, checkpoints, and
        receipts. Prefer the path that the current legacy writer updates for
        those artifacts; config and environment snapshots have always been
        written into their category directories and remain canonical-first.
        """

        canonical = self.path(name, category=category)
        candidates = [canonical]
        if not self.canonical_writes:
            canonical_category = canonical.relative_to(self.run_dir).parts[0]
            if canonical_category not in {"configs", "environment"}:
                producer_path = self.write_path(name, category=category)
                candidates.insert(0, producer_path)
            candidates.extend(self._legacy_read_paths(name, canonical))
        else:
            candidates.extend(self._legacy_read_paths(name, canonical))
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file() or candidate.is_dir():
                return candidate
        return canonical

    def write_path(self, name: str | Path, *, category: str | None = None) -> Path:
        """Return the producer path, preserving unmarked legacy run history."""

        canonical = self.path(name, category=category)
        if self.canonical_writes:
            return canonical
        return self._legacy_write_path(name, canonical)

    def ensure(self) -> None:
        """Create the established artifact category directories."""

        for category in _CATEGORIES:
            (self.run_dir / category).mkdir(parents=True, exist_ok=True)


def resolve_checkpoint(run_dir: str | Path, selector_or_name: str | Path) -> Path:
    """Resolve selectors and checkpoint paths, preferring canonical storage.

    Existing absolute or relative paths are returned unchanged. For a missing
    path under a run, canonical category candidates are checked before legacy
    root filenames, which also supports selectors after an explicit migration.
    """

    root = Path(run_dir).expanduser().resolve()
    supplied = Path(selector_or_name).expanduser()
    raw = str(selector_or_name).strip()
    selector = raw.lower()

    if supplied.is_absolute():
        direct = supplied.resolve()
        if direct.is_file():
            return direct
        try:
            supplied = direct.relative_to(root)
        except ValueError:
            return direct
    elif supplied.parent != Path("."):
        direct = supplied.resolve()
        if direct.is_file():
            return direct
        under_run = root / supplied
        if under_run.is_file():
            return under_run.resolve()
        direct_resolved = direct
        try:
            supplied = direct_resolved.relative_to(root)
        except ValueError:
            # A missing arbitrary relative path is preserved so the eventual
            # loader reports its original location instead of inventing a
            # checkpoint below this run.
            if supplied.parts[0] not in {"checkpoints", "Checkpoints"}:
                return direct_resolved

    filename = _SELECTOR_FILES.get(selector)
    if filename is not None:
        aliases = [alias for alias, target in _CHECKPOINT_ALIASES.items() if target == filename]
        if RunLayout(root).canonical_writes:
            candidates = [root / "checkpoints" / filename]
            candidates.extend(root / "checkpoints" / alias for alias in aliases)
            candidates.append(root / filename)
            candidates.extend(root / alias for alias in aliases)
        else:
            # Unmarked runs may contain hard-linked canonical mirrors from the
            # old finalizer. Their root checkpoint remains the exact-resume
            # cursor until an explicit migration marks the run canonical.
            candidates = [root / filename]
            candidates.extend(root / alias for alias in aliases)
            candidates.append(root / "checkpoints" / filename)
            candidates.extend(root / "checkpoints" / alias for alias in aliases)
        return next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[0])

    relative = _safe_relative(supplied)
    basename = relative.name
    canonical_relative = _canonical_relative(relative)
    is_canonical = RunLayout(root).canonical_writes
    legacy = root / relative.name if relative.parts[0] == "checkpoints" and len(relative.parts) > 1 else root / relative
    if basename.lower() in _CHECKPOINT_ALIASES:
        target = _CHECKPOINT_ALIASES[basename.lower()]
        canonical_alias = root / "checkpoints" / target
        legacy_alias = root / target
        if is_canonical:
            candidates = [canonical_alias, root / "checkpoints" / basename, root / canonical_relative, legacy_alias, root / basename]
        else:
            # A root-level familiar checkpoint name is authoritative in an
            # unmarked legacy run, even when an old finalizer left a stale
            # canonical mirror behind.
            candidates = [legacy_alias, root / basename, legacy, canonical_alias, root / "checkpoints" / basename]
    else:
        candidates = [root / canonical_relative, legacy] if is_canonical else [legacy, root / canonical_relative]
    candidates = list(dict.fromkeys(candidates))
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[0])
