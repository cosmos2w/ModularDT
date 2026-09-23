"""Generate the explicit Run-1501 sparse-incidence population evidence.

This thin entry point intentionally reuses only the maintained checkpoint and
dataset traversal.  The evidence module is Run-1501 specific: it reports the
fixed registered capacity separately from occupied source groups and query
support degree Kq, and never invents an occupancy Kplan.
"""

from __future__ import annotations

from collections.abc import Sequence

from run_run1409_occupancy_population import build_parser, run_population


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(
        evidence_module="sparse_incidence_evidence",
        expected_architecture="sparse_incidence_group_control_honf",
        board_prefix="sparse_incidence_board",
        progress_label="sparse-incidence-population",
        task_name="run1501_sparse_incidence_population_evidence",
    )
    args = parser.parse_args(argv)
    output = run_population(args)
    population = output["population"]
    print(
        "Run 1501 sparse-incidence evidence complete: "
        f"cases={population['case_count']} "
        f"Kq_mean={population['query_degree_mean']['mean']:.4f} "
        f"RM={population['module_RM_support']['mean']:.4f} "
        f"RE={population['environment_RE_support']['mean']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
