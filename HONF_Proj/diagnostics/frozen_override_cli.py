"""Compatibility import wrapper for maintained diagnostic CLI helpers."""
from pathlib import Path
import runpy

globals().update(
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools" / "diagnostics" / Path(__file__).name)
    )
)
