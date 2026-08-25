# Result and provenance contract

The run store groups artifacts by case and model family. A directory name uses
`Run_<four-digit-id>_<local-timestamp>_<safe-name>`, while `run_manifest.json`
contains a globally unique UUID and UTC creation time. Existing IDs are never
silently reused.

Before execution the store snapshots the core source, case source, explicit CLI
overrides, resolved effective configuration, source-file hashes, deterministic
configuration hash, software environment, and Git commit/dirty state. Status
transitions through `created`, `running`, and `completed` or `failed`; failures
include type, message, traceback, and the last recoverable completed epoch.
Completion inventories all written checkpoint selectors.
The manifest also records the final completed epoch and workflow-owned
`best_*` summary metrics. New manifests also carry the human-readable run name
and explicit UTC start, update, and end timestamps.

Case workflows retain checkpoint-compatible root filenames. On completion, the
runtime hard-links (or copies when linking is unavailable) checkpoint and metric
artifacts into canonical `checkpoints/` and `metrics/` subtrees, so historical
commands remain valid. Managed training writes plots directly to
`plots/training/` and `plots/diagnostics/`; it does not create a second
`diagnostic_plots/` tree or mirror root PNGs.

Single-case evaluation writes one timestamped job under
`evaluations/single_case/` and records that exact directory in the run manifest:

```text
evaluations/single_case/<case>_<timestamp>/
├── summary.json
├── evaluation_manifest.json
├── fields/
├── organization/
├── routing/
├── topology/
├── plans/
├── metrics/
├── arrays/
└── diagnostics/
```

Only categories requested by the evaluation are created. Each figure has one
canonical filename; legacy aliases such as `organizer_visualization.png` and
`organization_matrices.png` are not copied. The manifest inventories every
artifact with its category. Historical `eval_global/`, `eval_local/`,
`diagnostic_plots/`, and root-level plot trees remain readable and are not
automatically moved or deleted. Comparison outputs remain explicit
user-selected artifacts because they can draw from several source runs.

Single-run evaluation validates the source manifest and loads that run's
immutable `configs/resolved_config.json` before applying dataset selection or
case evaluation defaults. The checkpoint remains authoritative for model
architecture, normalization, and embedded Stage-A state. Multi-run comparison
validates every checkpoint/schema and uses one explicitly selected dataset.

Generated result directories, checkpoints, tables, plots, arrays, and local
resource maps are ignored by Git. README files and `.gitkeep` markers preserve
the intended hierarchy in a clean clone.
