# R0 Thermal classic native-input reconciliation

**Status:** a checkpoint-native input-transformation defect was identified, and a versioned replay now applies the corrective transform; actual-HDF5 input parity and a focused regression test pass, while corrected model-prediction parity and the full DEV22 replay remain pending. The historical Thermal1404/1804 physical-output values in the 2026-10-12 multiview report are quarantined until that replay completes.

## Finding and bounded disposition

The prior multiview evaluator supplied physical `heat_powers` and `interface_condition` directly to the public classic predictor, although both stated checkpoints save `normalize_inputs=true` and checkpoint-owned global normalization statistics. The maintained public dataset reader normalizes those two arrays for these checkpoints; it derives `local_module_params` and `teacher_port_tokens` from the raw physical arrays before normalizing the global inputs. Consequently, matching checkpoint, target, mask, or model-state hashes alone did not establish equivalent predictor inputs.

The R0 clock-bound checkpoint-native model-prediction comparison was **not completed by the 12:59:00 UTC, 2026-10-13 bound**. At 12:26:53.364910777 UTC, a CPU-only actual-HDF5 check established exact equality between the new helper and the public dataset reader for normalized `heat_powers`/`interface_condition`, and for raw-derived local parameters/teacher port tokens, on DEV0291 and DEV0687 for both checkpoints. That receipt records zero model constructions, predictions, GPU calls, optimizer steps, and solver calls; it is input-transform parity, not prediction parity.

No corrected classic RMSE values are established here. A quartet previously relayed as normalized native RMSE values (approximately 0.588/0.304/0.294/0.280) has no retained R0 model-prediction receipt or result binding and is withdrawn as unsupported; the only R0 receipt explicitly records `prediction_calls=0`. Do not use that quartet as model output or as a replacement score.

The maximum raw-versus-checkpoint-normalized input differences in those cases were 2.913795948 for heat power and 8848.708984375 for interface condition. The helper leaves physical targets, geometry, source-slot order, presence, and support arrays unchanged. A three-test focused suite also checks the actual HDF5 reader contract, the disabled-normalization path, and rejection of missing required checkpoint statistics.

## Bound identities and preserved evidence

| Checkpoint | Raw checkpoint SHA-256 | Model-state SHA-256 | Local-surrogate SHA-256 | Saved normalizer semantic SHA-256 | Input/target normalization |
|---|---|---|---|---|---|
| Thermal1404, literal e5000 | `71a588ab4c2d9bb624f4975f50a0874d575f6ba6687f68e15869806ce512cbef` | `43987257250166b03674b600ceb375611019553842e31a5e4f37002385f67fe4` | `1981fac0abbbe27f0521dd17d90d5b7deccb31e3e980a620dcbb9bf8b297988a` | `5cbe6473e6449e403626520e63ad1ca1cf283cd802d257d904c42e877c4f9129` | true / true |
| Thermal1804, literal e5000 | `c6d369ddb3b731ac5879ce4deccbb0e81986e09cbda6139e151ffad33628aeaa` | `f7ba2912c799ad62a64b5b73be7908cb6f3b102d4212cbdc6c1ae8ee29882e40` | `1981fac0abbbe27f0521dd17d90d5b7deccb31e3e980a620dcbb9bf8b297988a` | `5cbe6473e6449e403626520e63ad1ca1cf283cd802d257d904c42e877c4f9129` | true / true |

The immutable input-parity receipt is `/data/wanglz/ModularDT/diagnostics/r1_sparse_reads_20261013_115851/receipts/r0_thermal_classic_input_parity_cpu_v1.json` (raw SHA-256 `0be7d85bdaa87456098a4f3ae75f8879ec1f6f47d281c4ddaeaff6c1160547ec`). It binds the exact frozen v12 evaluator, v12 manifest/result, fixed panel, HDF5 source size/mtime, case-array hashes, native masks/support counts, old representative-array hashes, both checkpoint identities above, and per-channel SSE/count records. The old arrays were not modified.

For reference only, the now-quarantined v12 raw-input temperature errors were:

| Checkpoint | Case | Common-mask count | Saved raw-input temperature SSE | Saved raw-input temperature RMSE |
|---|---:|---:|---:|---:|
| Thermal1404 e5000 | DEV0291 | 7,837 | 183,279.14702091555 | 4.835947797092058 |
| Thermal1404 e5000 | DEV0687 | 7,463 | 104,021.40181073947 | 3.733400920712485 |
| Thermal1804 e5000 | DEV0291 | 7,837 | 135,839.98297319852 | 4.163311305630041 |
| Thermal1804 e5000 | DEV0687 | 7,463 | 185,761.71594544902 | 4.989090549755598 |

These are the saved outputs of the mismatched raw-input path; they are not corrected baseline scores. The receipt retains corresponding `u`, `v`, `p`, and `omega` SSE, bias, count, and RMSE so the discarded path remains auditable.

## Corrected replay queued

The durable helper `channelthermal.evaluation.loading.checkpoint_native_inputs_from_physical_sample` applies only the saved heat-power and interface-condition transforms, after raw-derived local features exist. It feeds the unchanged v12 classic loader, `predict_case`, physical denormalization, common fluid/stencil mask, and auxiliary metric reducers. The versioned replay is restricted to the two exact classic checkpoints and all 22 fixed DEV cases; it uses the same query grid, predicted local ports, native contexts, own normalizers, and supports, with 44 full receiver-grid predictions total. It does not rerun other mature systems or alter saved arrays.

The inert preflight passed with no checkpoint payload loads, model construction, target reads, predictions, GPU calls, optimizer steps, or solver calls. Root's queued execution command is:

```bash
PYTHONPATH=HONF_Proj/src:HONF_Proj/Case_ThermalChannel/src:HONF_Proj/tools NVIDIA_TF32_OVERRIDE=0 /home/wanglz/miniconda3/envs/ModularDT/bin/python -B /data/wanglz/ModularDT/diagnostics/r1_sparse_reads_20261013_115851/evaluation/replay_thermal_classic_normalized_inputs_v1.py --execute --manifest /data/wanglz/ModularDT/diagnostics/r1_sparse_reads_20261013_115851/evaluation/thermal_classic_normalized_inputs_manifest_v1.json
```

The versioned runner has raw SHA-256 `3ac632040a4d6eebd7d38b7710c9ee4a0ea581eb0ca21245582af4d3236c6567`; its manifest has raw SHA-256 `b9dda96b2fff05744893c5e62b31d2e0d4bf567abcf166979d5cbc120ce2db67` and semantic SHA-256 `3a6d50418177edf3388190542fba563ed81ef31e77a96827ab865f6a08857c04`. The focused regression command is:

```bash
PYTHONPATH=HONF_Proj/src:HONF_Proj/Case_ThermalChannel/src:HONF_Proj/tools /home/wanglz/miniconda3/envs/ModularDT/bin/python -B -m pytest -q -p no:cacheprovider HONF_Proj/tests/test_checkpoint_native_inputs.py
```

Until the queued replay emits and validates the full DEV22 result, the report's Thermal1404/1804 field, interface, and material-output comparisons remain invalidated. This addendum does not alter any checkpoint, target, mask, historical array, or mature Wind comparison.
