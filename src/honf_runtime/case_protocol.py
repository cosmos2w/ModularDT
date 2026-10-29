"""Interfaces between generic entry points and a case-specific package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

if False:  # pragma: no cover - imported only by static type checkers.
    from .config_loader import ConfigBundle


@dataclass(frozen=True)
class WorkflowRequest:
    """Normalized common CLI inputs passed to a case workflow."""

    workflow: str
    device: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    epochs: int | None = None
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    resume_checkpoint: str | None = None
    initialize_checkpoint: str | None = None
    local_checkpoint: str | None = None
    checkpoint: str | None = None
    saved_root: str | None = None
    dataset: str | None = None
    output_dir: str | None = None
    extra_args: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class LocalModuleSpec:
    """Portable description of one case-owned differentiable sub-module."""

    module_id: str
    schema_version: int
    module_parameter_names: tuple[str, ...]
    port_feature_names: tuple[str, ...]
    query_coordinate_names: tuple[str, ...]
    target_names: tuple[str, ...]
    latent_dim: int
    dataset_ids: tuple[str, ...]
    model_factory: str
    checkpoint_loader: str
    train_workflow: str
    evaluate_workflow: str
    coupling_adapter: str
    frozen_in_parent: bool = True
    embedded_in_parent_checkpoint: bool = True

    def __post_init__(self) -> None:
        if not self.module_id or int(self.schema_version) <= 0:
            raise ValueError("A local module requires a non-empty ID and positive schema version.")
        if int(self.latent_dim) <= 0 or not self.target_names or not self.port_feature_names:
            raise ValueError("A local module requires positive latent width and non-empty port/target schemas.")


@runtime_checkable
class CasePlugin(Protocol):
    """Minimum implementation required from every case package."""

    case_id: str
    display_name: str
    version: str

    def validate_config(self, bundle: "ConfigBundle") -> None:
        """Reject unknown/inconsistent case-owned settings before writes."""

    def inspect_launch(self, bundle: "ConfigBundle", request: WorkflowRequest) -> Mapping[str, Any]:
        """Validate resources and return human-readable launch facts."""

    def train(
        self,
        bundle: "ConfigBundle",
        request: WorkflowRequest,
        *,
        run_dir: Path,
    ) -> int:
        """Run a named training workflow in an already-reserved run directory."""

    def evaluate(self, bundle: "ConfigBundle", request: WorkflowRequest) -> int:
        """Run a named evaluation or post-processing workflow."""
