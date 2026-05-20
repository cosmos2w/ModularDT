"""Static audit of existing demo forward models.

This script uses simple text-pattern scanning plus a curated checklist. It does
not import demo modules, execute model code, or modify demo folders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SANDBOX_ROOT.parent
RESULTS_DIR = SANDBOX_ROOT / "results"


TARGETS = {
    "multicylinder_model": REPO_ROOT / "0_Demo_MultiCylinder/src/model.py",
    "channelthermal_model": REPO_ROOT / "1_Demo_ChannelThermal/src/_models_forward/model.py",
    "channelthermal_train_config": REPO_ROOT / "1_Demo_ChannelThermal/Configs/train_global_config_template.json",
}


PATTERNS = {
    "hypergraph organizer": ["Hypergraph", "A_mh", "A_eh", "hyper_state", "organizer"],
    "environment tokens": ["env_tokens", "num_env_tokens", "environment"],
    "module tokens": ["module_tokens", "module_centers", "module_present"],
    "periodic geometry": ["periodic", "minimum-image", "phase", "tau"],
    "dynamic tokens": ["dynamic_global_token", "dynamic_hyper_base", "dynamic"],
    "mean/residual split": ["pred_mean", "pred_residual", "residual"],
    "nonperiodic geometry": ["nonperiodic", "wall", "inlet", "outlet"],
    "local surrogate": ["LocalModuleSurrogate", "local_surrogate", "port"],
    "interface/internal heads": ["interface", "internal", "port"],
    "duplicate or edge pruning": ["DISABLE_EDGE", "duplicate", "disable_edge"],
    "direct shortcut candidates": ["direct", "organizer_bottleneck_gate", "module_env_context"],
}


def main() -> None:
    audit = build_audit()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "forward_model_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (RESULTS_DIR / "forward_model_audit.md").write_text(render_markdown(audit), encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'forward_model_audit.md'}")
    print(f"Wrote {RESULTS_DIR / 'forward_model_audit.json'}")


def build_audit() -> Dict[str, object]:
    files: Dict[str, object] = {}
    all_hits: Dict[str, List[str]] = {}
    for label, path in TARGETS.items():
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        hits = scan_patterns(text)
        files[label] = {"path": str(path.relative_to(REPO_ROOT)), "exists": path.exists(), "pattern_hits": hits}
        for key, value in hits.items():
            if value:
                all_hits.setdefault(key, []).append(label)

    classifications = {
        "core_common": [
            "module token encoding",
            "environment token grid",
            "soft A_mh module-hyperedge incidence",
            "soft A_eh environment-hyperedge incidence",
            "hyper_state aggregation",
            "query-to-hyper field decoding",
        ],
        "channel_specific": [
            "nonperiodic wall/inlet/outlet geometry",
            "thermal local surrogate patch",
            "port prediction and port/global consistency",
            "interface/internal module heads",
            "thermal prior scales and heat-power heuristics",
        ],
        "multicylinder_specific": [
            "periodic minimum-image geometry",
            "phase tau encoding",
            "dynamic hyper tokens",
            "wake-aware residual branch",
            "mean/residual output split",
        ],
        "suspected_redundant_or_ablate": [
            "direct module/environment decoder shortcut",
            "near-module local context shortcut",
            "global context bypass",
            "A_me auxiliary path if hyperedge bottleneck already carries structure",
            "organizer duplicate/active-edge penalties before core behavior is established",
        ],
        "legacy_do_not_copy_initially": [
            "DISABLE_EDGE pruning",
            "case-specific port/interface losses",
            "local surrogate coupling",
            "dynamic token heads",
            "large training curricula and checkpoint policies",
        ],
    }
    return {"files": files, "pattern_summary": all_hits, "classifications": classifications}


def scan_patterns(text: str) -> Dict[str, bool]:
    low = text.lower()
    return {name: any(pattern.lower() in low for pattern in patterns) for name, patterns in PATTERNS.items()}


def render_markdown(audit: Dict[str, object]) -> str:
    lines = ["# Forward Model Audit", ""]
    lines.append("## Files")
    files = audit["files"]  # type: ignore[index]
    for label, info in files.items():  # type: ignore[union-attr]
        lines.append(f"- `{label}`: `{info['path']}` exists={info['exists']}")  # type: ignore[index]
    lines.append("")
    lines.append("## Pattern Hits")
    for label, info in files.items():  # type: ignore[union-attr]
        lines.append(f"### {label}")
        hits = info["pattern_hits"]  # type: ignore[index]
        for name, hit in hits.items():
            mark = "yes" if hit else "no"
            lines.append(f"- {name}: {mark}")
        lines.append("")
    lines.append("## Classifications")
    classifications = audit["classifications"]  # type: ignore[index]
    for name, items in classifications.items():  # type: ignore[union-attr]
        lines.append(f"### {name}")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
