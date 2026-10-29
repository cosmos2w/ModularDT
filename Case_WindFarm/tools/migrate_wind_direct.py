#!/usr/bin/env python
"""Create the hash-bound, self-contained Run2202 e5000-to-P migration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "HONF_Proj"
CASE = PROJECT / "Case_WindFarm"
RUN_ID = "Run_2202_20261008_194400_w3_h128_full_train420_q4096_e5000"
RUN_DIR = PROJECT / "Trained_Results/WindFarm/HONF_Forward_Runs" / RUN_ID
DEFAULT_CHECKPOINT = RUN_DIR / "checkpoints/epoch_5000_model.pt"
DEFAULT_NORMALIZATION = (
    CASE / "diagnostics/generated/unified_refinement_20261008/wind_w1_manual_full"
    / "prepared_data/train_only_normalization.json"
)

# This migration is archival/inference-only; never initialize CUDA in this CLI.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(CASE / "src"))

from windfarm.migration import migrate_run2202_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--identity", type=Path, default=None)
    parser.add_argument("--source-state", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh migration artifact path; select an explicitly authorized ignored output location.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("Migration construction must remain CPU-only.")
    receipt = migrate_run2202_checkpoint(
        checkpoint_path=args.checkpoint,
        normalization_path=args.normalization,
        output_path=args.output,
        identity_path=args.identity,
        source_state_path=args.source_state,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
