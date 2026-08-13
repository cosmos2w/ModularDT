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

Case workflows retain their checkpoint-compatible root filenames. On
completion, the runtime hard-links (or copies when linking is unavailable)
those artifacts into canonical `checkpoints/`, `metrics/`, and `plots/`
subtrees, so legacy evaluators and release tooling can coexist. Evaluation creates a
timestamped `eval_global/` or `eval_local/` child and records its absolute path
in the run manifest. Comparison outputs are explicit user-selected artifacts
because they can draw from several source runs.

Single-run evaluation validates the source manifest and loads that run's
immutable `configs/resolved_config.json` before applying dataset selection or
case evaluation defaults. The checkpoint remains authoritative for model
architecture, normalization, and embedded Stage-A state. Multi-run comparison
validates every checkpoint/schema and uses one explicitly selected dataset.

Generated result directories, checkpoints, tables, plots, arrays, and local
resource maps are ignored by Git. README files and `.gitkeep` markers preserve
the intended hierarchy in a clean clone.
