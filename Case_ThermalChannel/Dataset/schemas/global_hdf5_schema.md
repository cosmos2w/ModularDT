# `thermal_channel_global_hdf5_v1`

Required root keys are `case_ids`, `splits`, `cases`, `channel_order`,
`interface_condition_feature_names`, `interface_target_names`, and
`normalization`.

Required per-case tensors consumed by the migrated workflow are:

| Name | Shape/role |
|---|---|
| `sampled_points` | `[N,2+F]`, coordinates followed by target channels |
| `module_centers` | `[M,2]` |
| `heat_powers` | `[M]` |
| `module_present` | `[M]` binary mask |
| `interface_condition` | `[M,P,C]` |
| `interface_target` | `[M,P,2]` |
| `module_internal_temperature` | `[M,H,W]` |
| `module_internal_mask` | `[H,W]` |
| `x_grid`, `y_grid`, `steady_field` | full-grid evaluation arrays |

Material attributes and the serialized case configuration provide Reynolds
number, inlet velocity, domain dimensions, and six ordered material values.
See `../PHYSICS_AND_DATA.md` for semantic definitions.
