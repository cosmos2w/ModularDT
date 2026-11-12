"""Case-neutral DATA2 binding, byte budgets and checkpoint storage estimates."""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def _ensure_workspace_storage_bindings(project: Path | None = None) -> dict[str, str]:
    """Repair the declared DATA2 workspace links before any run output is written."""
    data2 = Path("/data2")
    if not data2.is_dir() or not os.path.ismount(data2):
        raise RuntimeError("/data2 is not mounted; refusing to create contextual-frontier outputs.")
    data2_device = data2.stat().st_dev
    root = project or Path(__file__).resolve().parents[2]
    relocated = data2 / "wanglz/ModularDT/relocated_workspace/HONF_Proj"
    bindings = (
        (root / "Trained_Results", relocated / "Trained_Results", True),
        (root / "diagnostics", relocated / "diagnostics", True),
        (root / "Case_WindFarm/diagnostics/generated/native_role_catalogues_v1",
         relocated / "Case_WindFarm/diagnostics/generated/native_role_catalogues_v1", False),
    )
    checked: dict[str, str] = {}
    for link, target, create_target in bindings:
        target = target.resolve(strict=False)
        if not target.is_relative_to(data2.resolve()):
            raise RuntimeError(f"Declared DATA2 target escaped /data2: {target}.")
        if target.exists() and target.stat().st_dev != data2_device:
            raise RuntimeError(f"Declared target is not on the mounted /data2 filesystem: {target}.")
        if not target.exists():
            if not create_target:
                raise RuntimeError(f"Required shared DATA2 cache is missing: {target}.")
            target.mkdir(parents=True, exist_ok=True)
        resolved_target = target.resolve(strict=True)
        if resolved_target.stat().st_dev != data2_device:
            raise RuntimeError(f"Resolved DATA2 target is on the wrong filesystem: {resolved_target}.")

        valid_link = False
        if link.is_symlink():
            try:
                valid_link = link.resolve(strict=True) == resolved_target
            except FileNotFoundError:
                valid_link = False
        elif link.exists():
            raise RuntimeError(f"Expected a workspace symlink at {link}, found a non-symlink entry.")
        if not valid_link:
            if link.is_symlink():
                link.unlink()
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(resolved_target, target_is_directory=True)
        if link.resolve(strict=True) != resolved_target or link.resolve(strict=True).stat().st_dev != data2_device:
            raise RuntimeError(f"Workspace DATA2 binding verification failed for {link}.")
        checked[str(link)] = str(resolved_target)
    return checked

def _model_state_bytes(model: torch.nn.Module) -> int:
    return sum(int(tensor.numel()) * int(tensor.element_size())
               for tensor in model.state_dict().values() if torch.is_tensor(tensor))

def _storage_estimate(model: torch.nn.Module, *, completed_epoch: int, stop_after: int,
                      config: Mapping[str, Any], fresh_namespace: bool = False,
                      checkpoint_epochs: Sequence[int] | None = None,
                      model_bytes=None) -> tuple[int, int]:
    """Conservatively include model, named Adam moments, aliases, and atomic-save space."""
    state_bytes = max(1, (model_bytes or _model_state_bytes)(model))
    payload_bytes = 4 * state_bytes + 8 * 1024**2
    declared_checkpoints = (config["training"]["checkpoint_epochs"] if checkpoint_epochs is None
                           else checkpoint_epochs)
    milestones = sum(
        1 for epoch in declared_checkpoints
        if completed_epoch < int(epoch) <= stop_after
    )
    terminal = int(stop_after not in declared_checkpoints)
    if completed_epoch == 0 or fresh_namespace:
        alias_count = 2  # latest and best-field are distinct retained files
    else:
        alias_count = 0  # both aliases already exist and are atomically replaced
    new_payload_count = milestones + terminal + alias_count
    history_and_plots = 64 * 1024**2
    estimated_new = new_payload_count * payload_bytes + history_and_plots
    peak_temporary = payload_bytes + 32 * 1024**2
    return estimated_new, peak_temporary

def _tree_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = [name for name in names if not (Path(directory) / name).is_symlink()]
        for name in files:
            item = Path(directory) / name
            if not item.is_symlink():
                try:
                    total += item.stat().st_size
                except FileNotFoundError:
                    continue
    return total

def verify_storage(config: Mapping[str, Any], *, output: str | Path | None = None,
                   estimated_bytes: int = 0, peak_temporary_bytes: int = 0,
                   tree_bytes=None) -> dict[str, Any]:
    """Verify configurable /data2 output, total budget and free-space headroom."""
    data2 = Path("/data2")
    if not data2.is_dir() or not os.path.ismount(data2):
        raise RuntimeError("/data2 is not mounted; refusing to write research artifacts.")
    storage = config["storage"]
    root = Path(storage["output_root"]).expanduser().resolve(strict=False)
    data2_stat = data2.stat()
    parent = root
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if parent.stat().st_dev != data2_stat.st_dev:
        raise RuntimeError("Configured research root does not resolve to the mounted /data2 filesystem.")
    if output is not None:
        resolved_output = Path(output).expanduser().resolve(strict=False)
        try:
            resolved_output.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Run output must be inside {root}.") from error
    usage = os.statvfs(data2)
    free_bytes = int(usage.f_bavail * usage.f_frsize)
    headroom = int(storage["minimum_free_headroom_bytes"])
    if free_bytes - int(estimated_bytes) < headroom:
        raise RuntimeError("The requested run would breach configured /data2 free-space headroom.")
    tree_bytes = tree_bytes or _tree_bytes
    current_bytes = tree_bytes(root) if root.exists() else 0
    related_bytes = 0
    for raw in storage.get("related_output_roots", ()):
        related = Path(raw).expanduser().resolve(strict=True)
        if not related.is_relative_to(data2.resolve()) or related.stat().st_dev != data2_stat.st_dev:
            raise RuntimeError("Related research evidence must remain on the mounted /data2 filesystem.")
        if related == root or related.is_relative_to(root) or root.is_relative_to(related):
            raise ValueError("Related evidence roots must not overlap the scientific run root.")
        related_bytes += tree_bytes(related)
    current_bytes += related_bytes
    output_budget = int(storage["max_output_bytes"])
    if current_bytes + int(estimated_bytes) > output_budget:
        raise RuntimeError(
            f"Research output budget exceeded: {current_bytes} existing + {estimated_bytes} estimated bytes "
            f"> {output_budget}."
        )
    peak_limit = int(storage["max_peak_temporary_bytes"])
    if int(peak_temporary_bytes) > peak_limit:
        raise RuntimeError("Estimated peak temporary space exceeds the configured maximum.")
    return {
        "data2_mount": str(data2.resolve()),
        "output_root": str(root),
        "filesystem_device": int(data2_stat.st_dev),
        "free_bytes_before": free_bytes,
        "output_bytes_before": current_bytes,
        "related_evidence_bytes_before": related_bytes,
        "estimated_new_bytes": int(estimated_bytes),
        "output_budget_bytes": output_budget,
        "estimated_peak_temporary_bytes": int(peak_temporary_bytes),
        "maximum_peak_temporary_bytes": peak_limit,
        "minimum_free_headroom_bytes": headroom,
    }

def _prepare_storage(config: Mapping[str, Any], output: Path, *, estimated_bytes: int,
                     peak_temporary_bytes: int, bindings: Mapping[str, str] | None = None,
                     verifier=None) -> dict[str, Any]:
    snapshot = (verifier or verify_storage)(
        config, output=output, estimated_bytes=estimated_bytes,
        peak_temporary_bytes=peak_temporary_bytes,
    )
    if snapshot["free_bytes_before"] - estimated_bytes - peak_temporary_bytes < snapshot[
            "minimum_free_headroom_bytes"]:
        raise RuntimeError("Contextual-frontier output plus atomic-save space would breach DATA2 headroom.")
    snapshot["workspace_bindings"] = dict(bindings or {})
    return snapshot
