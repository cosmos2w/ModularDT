"""Thin CLI entry point for explicit Run-1503 candidate organization evidence."""

from __future__ import annotations

from collections.abc import Sequence

from run_run1503_organization_evidence import build_parser, run_population


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.description = __doc__
    output = run_population(parser.parse_args(argv))
    print(
        "Run 1503 candidate organization evidence complete: "
        f"cases={output['population']['case_count']} "
        f"Kq_mean={output['population']['query_degree_mean']['mean']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
