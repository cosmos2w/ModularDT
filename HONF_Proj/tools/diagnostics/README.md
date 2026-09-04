# Maintained diagnostic tools

Executable offline diagnostic source lives here. These tools evaluate model
accuracy, topology, routing, checkpoint compatibility, and execution cost.
Generated reports and figures are local-only and should be written beneath an
ignored output directory or the managed run/evaluation layout.

The maintained case-adaptive residual multi-case entry point is
`evaluate_case_adaptive_residual.py`. It writes per-case CSV metrics and a
count/residual/shortcut-diagnostic JSON summary beneath the managed
`metrics/` and `diagnostics/` categories.

The Phase-2 `case_adaptive_tensor_residual` profile uses the same entry point.
Tensor diagnostics are opt-in: pass `--tensor-diagnostics` (or its alias
`--return-interaction-tensor`) when the model supports the optional
`residual_interaction_tensor` output. The summary then reports module,
environment, and content unfolding ranks; content-factor cosine; environment
rank and region separation; hard-versus-soft support gaps; K/cap statistics;
and K-by-active-module-count histograms. Without the flag, ordinary inference
keeps the compact organizer output and does not materialize the full tensor.
Managed arrays, metrics, diagnostics, and figures remain under the canonical
evaluation artifact layout.
