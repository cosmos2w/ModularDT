from __future__ import annotations

"""Compatibility entry point for the mode-aware multi-cylinder preprocessor.

Older scripts invoke this inert-named file.  The implementation now lives in
preprocess_multicyl_dataset.py and preserves inert defaults unless active
arguments are explicitly supplied.
"""

from preprocess_multicyl_dataset import main


if __name__ == "__main__":
    main()
