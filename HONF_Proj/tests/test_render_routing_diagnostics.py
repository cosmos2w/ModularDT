"""CPU regressions for strategy-aware routing diagnostic labels."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "tools" / "diagnostics"), str(_ROOT / "src")]

import render_routing_diagnostics as renderer


def test_strategy_label_and_ledger_metadata_preserve_legacy_fallback() -> None:
    assert renderer._payload_strategy({}) == ("module_hubs", "Module-hub")
    assert renderer._payload_strategy({"routing_strategy": "mean_shift"}) == (
        "mean_shift",
        "Mean-shift",
    )
    assert renderer._strategy_label("module-hubs") == "Module-hub"


def test_infer_metadata_uses_ledger_strategy_for_old_map_contract() -> None:
    data = {
        "p0_port__hub_coords": [[0.0, 0.0]],
        "p0_port__module_source_A": [[1.0]],
        "p0_port__environment_source_A": [[1.0]],
    }

    inferred = renderer._infer_metadata(
        data,
        {"architecture": "routed_pairwise_honf", "routing_strategy": "mean_shift"},
        "0273",
    )

    assert inferred["routing_strategy"] == "mean_shift"


def test_index_title_uses_strategy_label(tmp_path: Path) -> None:
    index = tmp_path / "index.html"
    renderer._write_index(
        index,
        {
            "routing_strategy_label": "Mean-shift",
            "anchors": [],
            "endpoint_links": {},
            "status": "ok",
            "rendered_anchor_count": 0,
        },
    )

    text = index.read_text(encoding="utf-8")
    assert "HONF Mean-shift routing diagnostics" in text
    assert "<h1>Mean-shift routing diagnostics</h1>" in text
