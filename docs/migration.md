# Migrating a physical case to modular HONF

Keep physical data, checkpoints, and historical runs outside the maintained
source tree. `Case_ThermalChannel` is the reference for separating case-owned
physics from the reusable core and runtime.

Command mapping:

| Workflow | Modular command |
|---|---|
| Train an optional local module | `train.py --config src/config_core/forward/local_module_thermal_disk.json` |
| Train the coupled forward model | `train.py --config <core-profile>` |
| Evaluate a local module | `evaluate.py --workflow local_module --config <local-profile>` |
| Evaluate the forward model | `evaluate.py --workflow forward --config <core-profile>` |
| Compare checkpoints | `evaluate.py --workflow compare --config <core-profile>` |

Compose a core launch profile with a case-owned profile such as
`Case_ThermalChannel/configs/case_default.json`. Resolve external datasets by
logical ID through a machine-local ignored location map. Managed output roots
are also local-only; model parameters, decoder options, optional local-module
architecture, normalization, physical losses, and checkpoint selection belong
in versioned configuration and checkpoint metadata.

## Forward architecture modes

Forward checkpoint configuration is now behavior-descriptive. A saved model
configuration that predates these fields is normalized to:

```text
organizer_mode = fixed_projection
mechanism_state_mode = residual_concat
field_assembly_mode = context_fusion
module_assignment_normalizer = softmax
environment_assignment_normalizer = softmax
query_assignment_normalizer = softmax
routing_execution = dense
```

This inference happens before strict dataclass construction and resume-config
comparison. It does not create parameters, rename state-dict keys, or redirect
the checkpoint to another computation path. The existing
`enhanced_honf_pairwise.json` profile remains unchanged. The separate
`adaptive_sparse_additive.json` profile declares its architecture modes
explicitly, does not enable an edge-count loss, and uses `gathered` execution
after passing the full-limit equivalence and bounded CUDA benchmark gates.

## Inverse topology modes

Inverse checkpoint model configurations that predate topology modes resolve to
`plan_token_mode=indexed`, `plan_conditioning_mode=ordered_flat`, and
`matching_mode=canonical`. They instantiate the same edge embedding and
ordered flattened-plan projection under the same state-dict paths.

The separate `train_inverse_topology_set_template.json` selects
`exchangeable_set`, `set_cross_attention`, and Sinkhorn training assignment.
It contains no learned edge-index embedding or ordered flattened-plan module,
and the fixed-width joint corrector is disabled. Before use, replace the
all-zero SHA-256 placeholder with the exact adaptive forward checkpoint digest.
Training then requires a topology-set dataset bound to
`honf_topology_signature` schema version 3 and the same digest; compact-plan
schema-v1 datasets are rejected for this mode instead of being silently mixed.
