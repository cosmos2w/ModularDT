# `thermal_disk_local_hdf5_v1`

Required root arrays are `case_ids`, `splits`, `module_params`, `port_tokens`,
`internal_query_points`, `internal_temperature_targets`, `interface_targets`,
and `normalization`. Feature-name arrays declare all non-spatial columns.

The active contract uses parameter width 7, port width 5, interface target
width 2, and 64 interface points. Optional `local_grid`, `local_mask`, solver,
modal, raw-target, and roughness arrays support detailed evaluation.
