# Configuration composition

One core launch profile references one case profile. The core profile owns the
workflow, model family, logical dataset selection, HONF architecture, generic
optimizer/runtime settings, checkpoint policy, and run identity. The case
profile owns resource loading, physical parameters, local dependencies, losses,
evaluation defaults, and local-module specifications.

Composition is deterministic: the two source objects are deep-copied into an
effective compatibility payload, an optional strict experiment overlay and
explicit CLI overrides are applied, and a canonical SHA-256 hash is
calculated. Source profiles, the overlay, resolved payload, and overrides are
stored in the run. Top-level unknown sections and a case ID mismatch fail
before a directory is reserved.

An overlay is passed with `--experiment-overlay`. It has separate `core` and
`case` namespaces and may modify only keys already declared by the selected
source profiles; it cannot create fields or cross ownership boundaries. The
archived ablations under `src/config_core/forward/experiments/` demonstrate the
format. The run stores `experiment_overlay.json` plus its source hash.

Paths use explicit anchors:

- `project://...` is relative to `HONF_Proj`;
- `config://...` is relative to the configuration containing it;
- an absolute path remains absolute;
- a plain relative CLI path is resolved by the runtime's documented path rule.

The dataset selected by a core profile is a logical ID, never an HDF5 path.
The case manifest defines its identity/schema and the ignored local map defines
its machine-specific location. CLI `--dataset` is an explicit evaluation-only
override.

Training is confirmation-gated. `--dry-run` performs validation and prints the
exact proposed run without writes; `--yes` is required for unattended use.
Run IDs are normalized to four digits and cannot collide within a result
family.

An empty validation split is an error. Reusing training samples for validation
requires `allow_train_as_validation: true` in the applicable case dataset or
local-module namespace and is announced as a warning.
