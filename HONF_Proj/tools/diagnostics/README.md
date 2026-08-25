# Maintained diagnostic tools

Executable offline diagnostic source lives here. Compatibility wrappers remain
under `diagnostics/*.py` for one transition window. New generated outputs
default to `diagnostics/generated/`, which is ignored; canonical run/evaluation
artifacts continue to use the established runtime layout.

`analyze_stage7_k_audit.py` consolidates completed Run 1401/1601/1602/1603 histories with maintained accuracy, topology, checkpoint-benchmark, and retained-mass outputs, then writes machine-readable comparison data and report figures under ignored `diagnostics/generated/stage7_k_audit/`.
