# Adding a case plugin

A case plugin is the boundary between reusable HONF execution and a physical
problem. Its factory is named in the case configuration, for example
`channelthermal.plugin:create_plugin`, and is loaded dynamically after both
profiles are validated.

The object must expose a stable `case_id`, display name, version, launch
inspection, training dispatch, and evaluation dispatch as described by
`honf_runtime.case_protocol.CasePlugin`. Launch inspection must resolve and
validate external resources before confirmation. Training receives the
composed configuration, a common `WorkflowRequest`, and an already reserved run
directory. Evaluation receives the same case-neutral request plus untouched
case-specific CLI arguments.

Keep these elements in the case package:

- a manifest-backed dataset registry and exact data schemas;
- dataset readers, normalization, and reproducible sampling;
- adapters producing the generic `honf_forward_core.config.BatchData` fields;
- dataset-derived resolution of `auto` architecture dimensions;
- physical model wrappers and field/channel interpretation;
- losses, metrics, plots, and post-processing;
- optional named local modules, their training/evaluation, and coupling logic;
- checkpoint metadata/migration hooks specific to the case.

The reusable core must never import the case. The generic entry points must
never branch on a case ID. Prove this boundary with a tiny synthetic plugin
that can be installed, dry-run, adapt a batch, execute inference, and calculate
a loss without modifying core/runtime files.

For each real case, commit a case JSON schema, data manifest, HDF5/array schema
documentation, an example location map, and tests for resource validation,
normalization, all supported forward modes, checkpoint compatibility, and
evaluation artifacts. Machine paths, data, checkpoints, and run outputs remain
untracked.

Each optional sub-module should expose `honf_runtime.case_protocol.LocalModuleSpec`.
That immutable record names its input/port/query/target schemas, latent width,
datasets, model/checkpoint factories, workflows, coupling adapter, freeze
policy, and whether its state is embedded into parent checkpoints.
