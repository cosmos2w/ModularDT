"""Human-readable launch review and safe confirmation behavior."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from .case_protocol import WorkflowRequest
from .config_loader import ConfigBundle
from .paths import project_relative


def print_launch_summary(
    bundle: ConfigBundle,
    request: WorkflowRequest,
    case_facts: Mapping[str, Any],
    *,
    proposed_output: str,
) -> None:
    """Print the exact configuration/resources that will be launched."""

    print("\n=== HONF launch review ===")
    print(f"workflow        : {request.workflow}")
    print(f"model family    : {bundle.effective['model_family']}")
    print(f"core profile    : {bundle.core.get('profile_name', '')}")
    print(f"core config     : {project_relative(bundle.core_source)}")
    print(f"case            : {bundle.effective['case']['id']}")
    print(f"case profile    : {bundle.case.get('profile_name', '')}")
    print(f"case config     : {project_relative(bundle.case_source)}")
    if bundle.experiment_source is not None:
        print(f"experiment      : {project_relative(bundle.experiment_source)}")
    print(f"config sha256   : {bundle.config_hash}")
    for key, value in case_facts.items():
        print(f"{str(key):16}: {value}")
    training = bundle.effective.get("training", {})
    print(f"device          : {request.device or training.get('device') or 'auto'}")
    print(f"epochs          : {request.epochs or training.get('epochs')}")
    print(f"run id          : {request.run_id or bundle.effective.get('Run_ID')}")
    print(f"output          : {proposed_output}")
    if bundle.overrides:
        print(f"CLI overrides   : {bundle.overrides}")
    print("==========================\n")


def confirm_launch(*, assume_yes: bool, dry_run: bool) -> bool:
    """Return whether execution may continue after review.

    Non-interactive callers must opt in with ``--yes``.  This avoids a batch
    job hanging on a prompt and makes unattended launches deliberate.
    """

    if dry_run:
        print("[dry-run] configuration and resources validated; no run was started.")
        return False
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        raise RuntimeError("Confirmation is required in non-interactive mode; pass --yes after reviewing --dry-run.")
    answer = input("Proceed? [y/N] ").strip().lower()
    return answer in {"y", "yes"}
