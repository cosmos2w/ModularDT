# Maintained diagnostic tools

Executable offline diagnostic source lives here. These tools evaluate model
accuracy, topology, routing, checkpoint compatibility, and execution cost.
Generated reports and figures are local-only and should be written beneath an
ignored output directory or the managed run/evaluation layout.

The maintained case-adaptive residual multi-case entry point is
`evaluate_case_adaptive_residual.py`. It writes per-case CSV metrics and a
count/residual/shortcut-diagnostic JSON summary beneath the managed
`metrics/` and `diagnostics/` categories.
