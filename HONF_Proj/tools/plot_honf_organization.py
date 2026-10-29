#!/usr/bin/env python3
"""Render global HONF organization from saved prepared-scene evidence (CPU only)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", required=True, metavar="LABEL=SCENE.npz",
                        help="one or two matched scene exports with adjacent metadata JSON")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--directory-key", default="C_restricted")
    parser.add_argument("--cycle", type=int, help="actual contextual cycle to display; default last saved cycle")
    parser.add_argument("--per-role", type=int, default=2, help="shared M/E examples per role in service plots")
    parser.add_argument("--edges", type=int, choices=(1, 2, 3), default=3,
                        help="representative native rows displayed, not a learned group count")
    parser.add_argument("--e-display-threshold", type=float, default=.02,
                        help="visual-only directory weight threshold; all positive M and own E remain")
    args = parser.parse_args()
    if not 1 <= len(args.scene) <= 2:
        parser.error("Supply one or two scenes; generate separate matched reports for larger collections.")
    if args.per_role < 1 or args.e_display_threshold < 0:
        parser.error("per-role must be positive and e-display-threshold nonnegative.")
    # This command never imports the training runtime, torch or a model loader.
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / "tmp" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from honf_viz.organization_data import load_scene, require_matched
    from honf_viz.organization_overview import render_overview, render_incidence
    from honf_viz.organization_services import render_donor_services, render_incoming_neighborhoods
    from honf_viz.organization_hyperedges import render_native_hyperedges

    scenes = []
    for value in args.scene:
        name, separator, filename = value.partition("=")
        if not separator or not name or not filename:
            parser.error("Each --scene must have the form LABEL=/path/to/scene.npz.")
        scenes.append(load_scene(name, Path(filename), directory_key=args.directory_key, cycle=args.cycle))
    if len({scene.name for scene in scenes}) != len(scenes):
        parser.error("Scene labels must be distinct.")
    require_matched(scenes)
    figures = [render_overview(scenes, args.output), render_incidence(scenes, args.output)]
    figures.extend(render_donor_services(scenes, args.output, per_role=args.per_role))
    contextual = all(set(scene.gates) == {"mm", "em", "ee", "me"} for scene in scenes)
    if contextual:
        figures.append(render_incoming_neighborhoods(scenes, args.output, per_role=args.per_role))
    figures.append(render_native_hyperedges(scenes, args.output,
                                           e_threshold=args.e_display_threshold, edges=args.edges))
    evidence = {"scenes": [s.identity() for s in scenes], "query_arrays_used": False,
                "model_or_checkpoint_loaded": False, "contextual_figure_available": contextual,
                "visual_parameters": {"per_role": args.per_role, "edges": args.edges,
                                      "e_display_threshold": args.e_display_threshold},
                "figures": {item.stem: item.evidence for item in figures}}
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    text = ["# Global HONF organization", "", "This gallery shows prepared scene-wide information organization. Query invocation is a separate supplementary view. Original member identities and weights are retained; all display selections and permutations are recorded in evidence.json.", "",
            "| Scene | Case | Partition | Epoch | Checkpoint |", "| --- | --- | --- | --- | --- |"]
    for scene in scenes:
        meta = scene.metadata
        text.append(f"| {scene.name} | {meta.get('case', 'unspecified')} | {meta.get('partition', 'unspecified')} | {meta.get('epoch', 'unspecified')} | `{meta.get('checkpoint', 'not supplied')}` |")
    text.extend(["", "## Figure index", ""])
    text.extend(f"- [{item.title}](#{re.sub(r'[^a-z0-9 -]', '', item.title.lower()).replace(' ', '-')})" for item in figures)
    for item in figures:
        text.extend(["", f"## {item.title}", "",
                     f"![{item.title}](figures/{item.stem}.png)", "",
                     f"{item.caption} [PDF master](figures/{item.stem}.pdf)"])
    if not contextual:
        text.extend(["", "Actual contextual-neighborhood panels were not generated because the exports do not contain all four actual gate blocks. Missing exchange evidence is not synthesized from directory weights or transposed lookups."])
    text.extend(["", "[Numerical evidence and original-ID crosswalks](evidence.json)", ""])
    (args.output / "README.md").write_text("\n".join(text))
    print(json.dumps({"output": str(args.output), "figures": len(figures),
                      "contextual_figure_available": contextual, "query_arrays_used": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
