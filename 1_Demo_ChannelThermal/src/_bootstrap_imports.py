from __future__ import annotations

"""Import-path bootstrap for direct ChannelThermal scripts.

The runnable entrypoints stay directly under ``src/`` while helper/model
modules live in private subdirectories. Import this module before project-local
imports in entrypoint scripts.
"""

from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parent
PRIVATE_IMPORT_DIRS = (
    "_helpers_forward",
    "_helpers_inverse",
    "_models_forward",
    "_models_inverse",
)


for name in reversed(PRIVATE_IMPORT_DIRS):
    path = SRC_DIR / name
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
