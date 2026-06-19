"""CHANNELTHERMAL-SPECIFIC import bootstrap for direct scripts.

Inputs are direct executions of `src_new/train.py` or `src_new/evaluate.py`.
The output is a temporary Python import path containing only directories inside
`src_new`. This module is not reusable across domains because it reflects the
ChannelThermal standalone source layout.
"""

from __future__ import annotations

from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

for path in (PROJECT_DIR, SRC_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
