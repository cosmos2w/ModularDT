"""Traverse a Run-1500 checkpoint across the 90-case test population."""

from __future__ import annotations

import json
from collections.abc import Sequence

from run_run1409_occupancy_population import build_parser, run_population


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.set_defaults(
        expected_architecture="mass_competitive_group_control_honf",
        task_name="run1500_mass_competitive_population_evidence",
        board_prefix="mass_competitive_board",
        progress_label="run1500-population",
        evidence_module="mass_competitive_evidence",
    )
    output = run_population(parser.parse_args(argv))
    print(
        json.dumps(
            {
                "status": output["status"],
                "output_dir": output["output_dir"],
                "case_count": output["population"]["case_count"],
                "kcase_histogram": output["population"]["kcase_histogram"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
