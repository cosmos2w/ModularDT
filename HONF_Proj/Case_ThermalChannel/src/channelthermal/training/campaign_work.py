"""Record shared-core forward calls and returned physical-phase work ledgers."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch


class CampaignForwardWork(AbstractContextManager):
    """Measure actual hard/soft prepare/read calls during a bounded forward.

    The recorder leaves core computations unchanged and restores bound methods
    before backward. Activation-checkpoint recomputation is outside this scope.
    Ledger counts are taken from returned diagnostics, never inferred from the
    primary query count or from a multiplier applied to hard execution.
    """

    def __init__(self, core: Any):
        self.core = core
        self.records: dict[str, dict[str, dict[str, Any]]] = {}

    def _record(self, kind: str, output: Any, prepared: Any = None) -> None:
        mode = getattr(self.core.backend, "permission_mode", "hard")
        state = getattr(prepared if prepared is not None else output, "backend_state", {})
        phase = f"P{state['hypergraph_phase']}" if isinstance(state, dict) and "hypergraph_phase" in state else "unlabelled"
        record = self.records.setdefault(mode, {}).setdefault(phase, {"prepare_calls": 0, "read_calls": 0, "ledgers": {}})
        record[kind + "_calls"] += 1
        auxiliary = getattr(output, "interaction_aux", {})
        for key, value in auxiliary.items():
            # These are the scalar integer/count diagnostics from the executed
            # backend, aggregated by the shared read across its actual chunks.
            if not key.startswith("hypergraph_") or not any(key.endswith(suffix) for suffix in (
                "unique_pairs", "eligible_pairs", "repeated_paths_removed",
                "near_mandatory_pairs", "near_full_pairs", "executed_rows", "padded_rows", "fine_calls",
                "allocated_rows", "executed_eligible_pairs", "skipped_eligible_pairs", "attention_cells",
            )):
                continue
            if torch.is_tensor(value) and value.numel() == 1:
                count = float(value.detach().cpu())
            elif isinstance(value, (int, float)):
                count = float(value)
            else:
                continue
            record["ledgers"][key] = record["ledgers"].get(key, 0.) + count

    def __enter__(self):
        self._prepare = self.core.prepare
        self._read = self.core.read
        self._prepare_owned = "prepare" in self.core.__dict__
        self._read_owned = "read" in self.core.__dict__

        def prepare(*args, **kwargs):
            output = self._prepare(*args, **kwargs)
            self._record("prepare", output)
            return output

        def read(*args, **kwargs):
            output = self._read(*args, **kwargs)
            prepared = args[0] if args else kwargs["prepared"]
            self._record("read", output, prepared)
            return output

        self.core.prepare, self.core.read = prepare, read
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for name, owned, original in (("prepare", self._prepare_owned, self._prepare), ("read", self._read_owned, self._read)):
            if owned:
                setattr(self.core, name, original)
            else:
                delattr(self.core, name)
        return False


def merge_forward_work(total: dict, measured: dict) -> None:
    """Accumulate measured phase calls/counts across complete native epochs."""

    for mode, phases in measured.items():
        for phase, values in phases.items():
            result = total.setdefault(mode, {}).setdefault(phase, {"prepare_calls": 0, "read_calls": 0, "ledgers": {}})
            for key in ("prepare_calls", "read_calls"):
                result[key] += values[key]
            for key, value in values["ledgers"].items():
                result["ledgers"][key] = result["ledgers"].get(key, 0.) + value
