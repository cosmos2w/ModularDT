# R0 Thermal classic native-input reconciliation

**Status:** a checkpoint-native input-transformation defect was identified, and the new helper passes exact actual-HDF5 input parity on the two probe cases; the post-bound corrected GPU attempt failed the established interface-output tolerance on Thermal1804/DEV0687, so the full paired DEV22 classic replay is incomplete and all classic physical-output comparisons remain quarantined.

## Finding and bounded disposition

The prior multiview evaluator supplied physical `heat_powers` and `interface_condition` directly to the public classic predictor, although both stated checkpoints save `normalize_inputs=true` and checkpoint-owned global normalization statistics. The maintained public dataset reader normalizes those two arrays for these checkpoints; it derives `local_module_params` and `teacher_port_tokens` from the raw physical arrays before normalizing the global inputs. Consequently, matching checkpoint, target, mask, or model-state hashes alone did not establish equivalent predictor inputs.

The R0 clock-bound checkpoint-native model-prediction comparison was **not completed by the 12:58:51 UTC, 2026-10-13 bound**. At 12:26:53.364910777 UTC, a CPU-only actual-HDF5 check established exact equality between the new helper and the public dataset reader for normalized `heat_powers`/`interface_condition`, and for raw-derived local parameters/teacher port tokens, on DEV0291 and DEV0687 for both checkpoints. That receipt records zero model constructions, predictions, GPU calls, optimizer steps, and solver calls; it is input-transform parity, not prediction parity.

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

## Post-bound corrected replay attempt and quarantine

The R0 clock-bound model-prediction comparison was not completed by 12:58:51 UTC on 2026-10-13. Later CPU and root-wrapped GPU attempts used the corrected checkpoint-native input transform with the original public predictor and physical outputs. These checks exceeded the planned 60-minute R0 stage window; they are retained as a stage-window deviation and do not change its quarantine disposition or qualify a timely reconciliation.

The CPU-only helper check remains narrowly scoped: on DEV0291 and DEV0687 for Thermal1404 and Thermal1804, it exactly matched the maintained HDF5 reader's normalized heat powers and interface conditions and its raw-derived local module parameters and teacher port tokens. It used two checkpoint metadata payload reads but constructed no models and made no predictions, GPU calls, optimizer steps, or solver calls. This qualifies the input transform only.

After the R0 deadline, a separate actual CPU prediction replay used the public HDF5 reader and corrected helper on DEV0291/DEV0687 for both exact checkpoints. All field-grid, interface and material outputs were byte-identical over 12 helper/public/repeated-public predictions, with zero SSE, RMSE and relative L2 differences. This qualifies those CPU probe predictions and does not qualify the complete GPU classic board. Its retained runner is `evaluation/thermal_classic_cpu_public_helper_parity_v3.py` (SHA-256 `cde445215b6307bde8aa8d59f67d44d27c9be30b04ed8262f8ecc1a9d6188327`); the actual result is `evaluation/thermal_classic_cpu_public_helper_parity_v3.json` (SHA-256 `8b11f590d6f3e148766ae670bf78c75d5a7a25ac4b1a17e9c1f5462bed3f5cc9`). It made no CUDA, optimizer or solver calls.

The later GPU attempt compared the public predictor and helper path on DEV0291 and DEV0687 for both checkpoints and also recorded a repeated public-path call. The three public/helper probe pairs for Thermal1404/DEV0291, Thermal1404/DEV0687, and Thermal1804/DEV0291 passed the established FP32 output tolerance. Thermal1804/DEV0687 failed the interface tolerance (`atol=2e-6`, `rtol=2e-5`): the public/helper maximum absolute interface difference was `7.2479248046875e-5` with a maximum elementwise tolerance ratio of `1.9186795`; the field-grid maximum difference was `7.62939453125e-6` with ratio `0.6567075`, and material-temperature maximum difference was `1.1444091796875e-5` with ratio `0.1813643`, both within tolerance. The repeated public-path interface comparison reached `8.487701416015625e-5` with ratio `2.5582394`, also outside tolerance, while the normalized input hashes for the helper and public paths were identical. This records same-path output variation on the failing probe; it does not identify its mechanism or establish that the helper alone caused the discrepancy.

The v3 attempt stopped at Thermal1804/DEV0687 after about 9.64 seconds. It retained a Thermal1404 literal-e5000 partial JSON for the full 22-case DEV panel and 42 per-case array files, plus four public/helper diagnostic rows. It did not produce a complete paired 44-case board or a combined comparison result. Those partial values are retained as failure evidence and are not reported as completed classic metrics or corrected baseline scores. No further classic replay or remedy is queued under this bounded R0.

The exact GPU-run evidence is `evaluation/replay_thermal_classic_normalized_inputs_v3.py` (SHA-256 `794590bdc25bc73b481e4e3ecc607eb5961b29aadcdc0450cc5e0c585c9d55e2`), `evaluation/thermal_classic_normalized_inputs_manifest_v3.json` (raw SHA-256 `972568c43078f996267819cde9d99d56254173d8e40f9ac5b8cdc93d031ae902`; semantic SHA-256 `7f4e942732d6c537f5346eb196353d477a2a2f2542453238616466c78124aebb`), `logs/thermal_classic_corrected_dev22_v3.log` (SHA-256 `564610fcdc01078a92f8a9ac2b5b471882c7ec431efa48c0fb71cfac203ddfb9`), and `evaluation/thermal_classic_normalized_dev22_v3/gpu_public_helper_parity_diagnostics.jsonl` (SHA-256 `a2973578017ddb9723cfd939087a482e52de5e5a676ea9b2efc505def9f67617`). The retained single-checkpoint partial board is `evaluation/thermal_classic_normalized_dev22_v3/thermal1404_literal_e5000_partial.json` (SHA-256 `3359da50ae54c3710ea7d26b4a003be598bf6e74c9a1ce6837598f78fa8e4953`). The actual-HDF5 CPU input receipt remains `/data/wanglz/ModularDT/diagnostics/r1_sparse_reads_20261013_115851/receipts/r0_thermal_classic_input_parity_cpu_v1.json` (SHA-256 `0be7d85bdaa87456098a4f3ae75f8879ec1f6f47d281c4ddaeaff6c1160547ec`).

The old raw-input Thermal1404/1804 arrays and their recorded metrics remain preserved but invalid for checkpoint-native comparison. The corrected-path partial outputs and all parity deltas remain quarantined because the strict interface check failed and the paired DEV22 replay did not complete; no classic replacement metrics are asserted here. The helper, targets, masks, historical arrays, checkpoints, and mature Wind comparisons were not changed.
