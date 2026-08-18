# Migration from Demo 1

The original `1_Demo_ChannelThermal/src_HONF_CL` remains an immutable reference
until the modular implementation and complete reruns are accepted. No dataset
or historical run is copied into the release source tree.

Command mapping:

| Previous command | Modular command |
|---|---|
| `src_HONF_CL/train_local.py` | `train.py --config src/config_core/forward/local_module_thermal_disk.json` |
| `src_HONF_CL/train.py` | `train.py --config <core-profile>` |
| `src_HONF_CL/evaluate_local.py` | `evaluate.py --workflow local_module --config <local-profile>` |
| `src_HONF_CL/evaluate.py` | `evaluate.py --workflow forward --config <core-profile>` |
| `src_HONF_CL/compare_models.py` | `evaluate.py --workflow compare --config <core-profile>` |

Old combined JSON files are replaced by a core launch profile plus
`Case_ThermalChannel/configs/case_default.json`. Old `Data_Saved` paths are
replaced by logical IDs and an ignored location map. Old `Saved_Model_*`
directories are replaced by the structured `Trained_Results/ThermalChannel`
tree. The model parameter names, decoder options, Stage-A architecture,
normalization behavior, physical losses, checkpoint selectors, and all
post-processing capabilities are retained.

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
