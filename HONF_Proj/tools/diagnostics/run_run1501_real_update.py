"""Run two real predicted-port batches and one optimizer update for Run 1501."""

from __future__ import annotations

import json
from collections.abc import Sequence

from run_run1409_occupancy_prelaunch import _jsonable, build_parser, run_prelaunch


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.set_defaults(
        expected_architecture="sparse_incidence_group_control_honf",
        task_name="run1501_sparse_incidence_real_update",
        router_stats_key="sparse_incidence_router_gradient_update",
    )
    payload = run_prelaunch(parser.parse_args(argv))
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
