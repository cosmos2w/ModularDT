# Native Wind relation-anchored interface

`WindRelationAnchoredAdapter` retains individual native turbine records and predicts the three velocity components together. It uses a configurable three-dimensional environment lattice and trilinear query coverage. `state_payload()` and `from_state_payload()` preserve the adapter configuration and train-fitted transforms; pass the model state dictionary on reload. Scene batching pads and masks turbine records and preserves their original IDs.

`windfarm.training.relation_anchored_task.build_wind_relation_anchored_task(recipe, device)` accepts the historical `wind_relation_anchored_r15_2_v1` and `wind_relation_anchored_r15_2_v2` schemas with their original settings. The generic `wind_relation_anchored_v1` schema supports a declared 3-D environment shape, epoch horizon, query counts, query tile, and training and validation microbatches on the existing development or full-training membership. It retains seed 42, the native five-role objective, FP32, an effective batch of 24, and train-only normalization and calibration. The generic schema does not authorize a training run.

The following is a portable recipe template; replace the path placeholders with the existing local inputs and compatible train-fitted calibration files. Add execution and checkpoint settings required by the selected runner. `full_train1000` selects the existing full membership; for that protocol set the catalogue RAM bound to 137438953472 bytes and provide `existing_transform_cache`. The protocol name identifies the data binding; `total_epochs` is the explicitly declared horizon for the generic schema.

```json
{
  "schema": "wind_relation_anchored_v1",
  "dataset": "wind",
  "architecture": "relation_anchored",
  "protocol": "development",
  "seed": 42,
  "primary_queries": 4096,
  "validation_queries": 4096,
  "effective_batch_size": 24,
  "microbatch_size": 8,
  "validation_microbatch_size": 4,
  "gradient_accumulation": true,
  "total_epochs": 100,
  "precision": "fp32",
  "optimizer_bias_norm_exclusion": true,
  "query_tile_size": 1024,
  "catalogue_cache_ram_bytes": 34359738368,
  "environment_shape": [4, 4, 4],
  "data_root": "<native-data-directory>",
  "derived_root": "<native-derived-directory>",
  "pilot_root": "<existing-pilot-directory>",
  "role_scale_cache_dir": "<train-calibration-directory>",
  "existing_role_scale_cache": "<existing-train-role-scale-json>"
}
```

The physical microbatch is at most 24 and `gradient_accumulation` must be true exactly when it is smaller than 24. Query counts must cover all five protected roles. Existing calibration files must match their train data bindings or pass the documented exact replay checks. The native geometry catalogue must already be prepared: this provider reads verified entries without building or writing, and reports missing or incompatible entries. Legacy cache callers retain their existing fallback policy. Geometry, cache recipe, array precision, and native target sampling remain part of the existing data contract.
