# Stage-7 K-scaling audit evaluation: Runs 1401, 1601, 1602, and 1603

## Technical summary

Run 1602 completed its managed continuation from epoch 500 to epoch 5000 and produced a scientifically interesting but non-promotable K=4 result. Its best-by-field checkpoint is epoch 4890. On the complete 90-case held-out test split, that checkpoint improves whole-fluid pooled normalized MSE by 2.21%, median-case MSE by 4.75%, and case-p95 MSE by 4.13% relative to accepted Run 1401 epoch 4585. It improves whole-fluid `u`, `v`, `p`, and temperature MSE by 19.59%, 15.01%, 6.69%, and 2.90%, respectively, while worsening `omega` by 6.26%.

The aggregate accuracy gain does not satisfy the complete promotion contract. Run 1602 best worsens near-interface pooled MSE by 7.03% relative to Run 1401 best, including 10.14% worse `u` and 11.58% worse `omega`, and its median query effective rank falls from 3.100 at its original epoch-495 best to 2.588 at epoch 2500 and 2.518 at epoch 4890. This is below the established `>3.0` topology gate. Query edge-column cosine also rises to 0.544, narrowly inside the `<0.55` ceiling, while pairwise-map effective rank falls to 1.975. The K=4 model therefore gains global accuracy while compressing query/pairwise organization and losing near-interface fidelity.

Run 1401 epoch 4585 remains the scientific K=6 baseline. Run 1602 should be retained as a completed K=4 research reference, not promoted and not trained further under the present audit. Run 1601 remains the exact fused K=6 execution control, Run 1603 remains stopped/rejected at epoch 500, and K=12 remains unjustified. Fused gathered beta-0.98 remains the promoted efficient evaluation/deployment path because it reduces prepared-decoder runtime by 38.45–40.32% and peak allocated memory by 58.78–58.82% for the mature checkpoints while preserving full-support output exactly and changing pooled MSE only at numerical-noise scale.

Controlled dense inference does not show a material K=4 speed advantage. Run 1602 best is 3.34% faster than Run 1401 best in prepared decoding but only 0.63% faster in full-forward median, and it is effectively tied with the fused K=6 control in prepared decoding. The continuous Run 1602 epoch-2500-to-5000 checkpoint interval corresponds to approximately 4.53 seconds per epoch, which does not establish a training-speed gain over Run 1401's descriptive 4.36 seconds per epoch. No causal training-throughput conclusion should be drawn without a same-GPU forward/backward benchmark.

## Contents

1. [Scope and decision rules](#1-scope-and-decision-rules)
2. [Provenance and resume integrity](#2-provenance-and-resume-integrity)
3. [Checkpoint inventory](#3-checkpoint-inventory)
4. [Validation convergence](#4-validation-convergence)
5. [Complete-split accuracy](#5-complete-split-accuracy)
6. [Topology and routing quality](#6-topology-and-routing-quality)
7. [Computational efficiency](#7-computational-efficiency)
8. [Compatibility and validation](#8-compatibility-and-validation)
9. [Interpretation](#9-interpretation)
10. [Limitations and robustness](#10-limitations-and-robustness)
11. [Reproducibility](#11-reproducibility)
12. [Final K-audit decision](#12-final-k-audit-decision)

## 1. Scope and decision rules

This report extends the original epoch-500 K-scaling audit with Run 1602 milestones at epochs 2500 and 5000 and its final best-by-field checkpoint at epoch 4890. The continuation itself was launched earlier at the user's explicit request; the evaluation described here launched no additional training and did not modify checkpoints, model code, data, optimizer policy, routing policy, or scientific configuration.

- Accuracy and topology use all 90 held-out test cases, checkpoint-owned normalization, and channel order `u, v, p, omega, temperature`.

- Matched maturity comparisons use epochs 500, 2500, and 5000 for Runs 1401 and 1602. Best-checkpoint comparison uses accepted Run 1401 epoch 4585 and Run 1602 epoch 4890, both selected by validation field MSE.

- The established topology bands remain unchanged: environment edge-column cosine `<0.20`, environment effective rank `>3.5`, query edge-column cosine `<0.55`, and query effective rank `>3.0`. The K=4 result is not granted a relaxed or K-normalized gate after seeing the result.

- Promotion requires a useful Pareto result across validation stability, complete-split and near-interface accuracy, channel balance, topology, strict checkpoint/resume behavior, and computation. A lower aggregate test MSE alone is insufficient.

## 2. Provenance and resume integrity

### 2.1 Run inventory

| Run | Managed directory | Final status | Completed epoch | Scientific role |
|---|---|---:|---:|---|
| 1401 | `Run_1401_20260823_151126_stage7_modern_structured_context` | completed | 5000 | accepted K=6 scientific baseline |
| 1601 | `Run_1601_20260824_112400_stage7_fused_query_module` | completed | 500 | exact fused K=6 execution control |
| 1602 | `Run_1602_20260824_130922_stage7_k4_fused_audit` | completed after resume | 5000 | completed K=4 research audit; not promoted |
| 1603 | `Run_1603_20260824_135513_stage7_k8_fused_audit` | completed/stopped | 500 | rejected K=8 audit |

Run 1602 retained its original managed run directory and resolved configuration. The continuation restored the absolute managed `latest_model.pt`, resumed at epoch 501, and finished with manifest status `completed`, exit code 0, and `last_completed_epoch=5000`. The loader restored the optimizer rather than reinitializing it; the checkpoint inventory remains one optimizer parameter group with 101 populated per-parameter state records, and the training restore log reported 128 optimizer tensors containing 2,472,482 scalars.

Run 1602's source commit remains `a95e26dc2c0bdbed4fb620182ce014ff91787e8d`, its config SHA-256 remains `07ad3eb36b23d52f2c03c37ab30d007aa0db70c68d600838b1ce7f61afb7f1d0`, and its data, seed, optimizer, loss, Stage-A checkpoint, hidden width, fixed organizer, softmax assignments, residual/raw hyperedge state, context fusion, legacy pair kernel, and fused dense training execution remain unchanged from the original K=4 overlay.

### 2.2 Resume boundary evidence

| Check | Result |
|---|---|
| Managed directory reused | pass |
| Resume checkpoint | absolute path to Run 1602 `latest_model.pt` at epoch 500 |
| First continued epoch | 501 |
| Optimizer restored | pass; one group, populated state preserved |
| Epoch-2500 milestone written | pass |
| Epoch-5000 milestone written | pass |
| Final manifest | `completed`, exit code 0, epoch 5000 |
| Strict state inventory | 237 keys at epochs 500, 2500, 5000, and best |

## 3. Checkpoint inventory

| Label | Epoch | Checkpoint SHA-256 | State keys | Optimizer groups/states |
|---|---:|---|---:|---:|
| Run 1401 epoch 2500 | 2500 | `ddd467d8d53db3c4cd311ccce71ed8d13ed79d5c9d04e1c56b5386625abb6ae9` | 237 | checkpoint-owned |
| Run 1401 epoch 5000 | 5000 | `cee978f0461db928b72c3b6b66cb0ef56647674f82d6ef2660a8380754a829d6` | 237 | checkpoint-owned |
| Run 1401 accepted best | 4585 | `5be150bd6b4fc79599af62c767fba84490ba50edcc8cc8ce85026ae27a1846b3` | 237 | 1 group |
| Run 1602 epoch 500 | 500 | `4feecee889a5b9c01dac02e2e32a40a51f98128e51570479fc9bf974ec4f76aa` | 237 | 1 / 101 |
| Run 1602 epoch 2500 | 2500 | `7264bfa2645b06cb182b84e62c7dc2434e16c7bb7fa370755d494c5e88001f7c` | 237 | 1 / 101 |
| Run 1602 epoch 5000 | 5000 | `10a4c6803c2a21b85bef390d12f0246c0022623ae17910bfdacbbff541a12c26` | 237 | 1 / 101 |
| Run 1602 best by field | 4890 | `50e3852f9d1f923d7952f9b255fea40b7f484dbd740662966df8e39da46d54cc` | 237 | 1 / 101 |

Run 1602's best validation field MSE is `1.525806e-3`, 1.05% above Run 1401's `1.509966e-3`. Its best validation temperature MSE is `1.998218e-3`, 5.02% above Run 1401's `1.902618e-3`. The best total validation loss occurs earlier at epoch 2015 (`1.718251e-2`), while the best temperature checkpoint occurs at epoch 4975; the separation of those optima reinforces the need for field, temperature, topology, and regional checks rather than a single-score promotion.

## 4. Validation convergence

All table values are 50-epoch trailing medians ending at the named epoch.

| Epoch | Run 1401 field | Run 1602 field | Run 1602 change | Run 1401 temperature | Run 1602 temperature | Run 1602 change |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | `2.298699e-2` | `2.669332e-2` | +16.12% | `1.508924e-2` | `1.692972e-2` | +12.20% |
| 2500 | `3.410506e-3` | `3.523399e-3` | +3.31% | `3.740256e-3` | `3.821148e-3` | +2.16% |
| 5000 | `2.198345e-3` | `1.940238e-3` | **-11.74%** | `2.289453e-3` | `2.387931e-3` | +4.30% |

![Extended training convergence](generated/stage7_k_audit/20260824_210713_extended/synthesis/training_convergence_extended.png)

Run 1602 catches and then surpasses Run 1401 on the trailing field metric late in training, but it does not surpass Run 1401 on the trailing temperature metric. The field improvement is therefore real but not uniformly shared across objectives. The continuation did not show an instability or loss spike at the resume boundary.

## 5. Complete-split accuracy

### 5.1 Matched milestones

| Epoch | Run 1401 whole-fluid MSE | Run 1602 whole-fluid MSE | Run 1602 change | Run 1401 near-interface MSE | Run 1602 near-interface MSE | Run 1602 change |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | `1.2705e-2` | `1.1985e-2` | -5.67% | `5.7182e-2` | `5.3330e-2` | -6.74% |
| 2500 | `1.898541e-3` | `1.647064e-3` | **-13.25%** | `7.343517e-3` | `6.695052e-3` | **-8.83%** |
| 5000 | `1.295047e-3` | `1.227927e-3` | -5.18% | `4.716835e-3` | `5.141875e-3` | **+9.01%** |

Run 1602 is strongest relative to Run 1401 at epoch 2500, where both whole-fluid and near-interface aggregates improve. From epoch 2500 to 5000 it continues improving whole-fluid accuracy but loses the near-interface advantage, indicating a late-training redistribution of error rather than uniform refinement.

### 5.2 Best-by-field checkpoints

| Metric | Run 1401 e4585 | Run 1602 e4890 | Run 1602 change |
|---|---:|---:|---:|
| Pooled normalized fluid MSE | `9.537943e-4` | **`9.327516e-4`** | **-2.21%** |
| Median case normalized fluid MSE | `7.058953e-4` | **`6.723648e-4`** | **-4.75%** |
| Case p95 normalized fluid MSE | `2.739247e-3` | **`2.626072e-3`** | **-4.13%** |
| Near-interface pooled normalized MSE | **`4.398427e-3`** | `4.707725e-3` | **+7.03%** |

### 5.3 Best-checkpoint channel tradeoff

| Region/channel | Run 1401 e4585 | Run 1602 e4890 | Run 1602 change |
|---|---:|---:|---:|
| Fluid `u` | `3.290504e-4` | **`2.645739e-4`** | -19.59% |
| Fluid `v` | `2.697891e-4` | **`2.293005e-4`** | -15.01% |
| Fluid `p` | `7.953542e-4` | **`7.421598e-4`** | -6.69% |
| Fluid `omega` | **`1.646094e-3`** | `1.749096e-3` | +6.26% |
| Fluid temperature | `1.728684e-3` | **`1.678628e-3`** | -2.90% |
| Near-interface `u` | **`3.437391e-4`** | `3.785845e-4` | +10.14% |
| Near-interface `v` | `1.157346e-3` | **`9.747690e-4`** | -15.78% |
| Near-interface `p` | `1.376781e-3` | **`1.244036e-3`** | -9.64% |
| Near-interface `omega` | **`1.627311e-2`** | `1.815718e-2` | +11.58% |
| Near-interface temperature | `2.841156e-3` | **`2.784056e-3`** | -2.01% |

![Extended accuracy comparison](generated/stage7_k_audit/20260824_210713_extended/synthesis/milestone_accuracy_extended.png)

The independent data-quality audit found exactly 17,280 accuracy rows: 8 checkpoint labels × 90 cases × 4 regions × 6 channel aggregates, with no duplicate checkpoint/case/region/channel keys. Recomputing pooled MSE directly from raw normalized SSE and value counts reproduced the published Run 1401 best, Run 1401 epoch-5000, Run 1602 best, and Run 1602 epoch-5000 values exactly.

## 6. Topology and routing quality

Values are medians over all 90 cases using the final organizer pass and selected representation.

| Checkpoint | Environment cosine | Environment rank | Query cosine | Query rank | Region separation | Pairwise-map rank | Gate result |
|---|---:|---:|---:|---:|---:|---:|---|
| Run 1401 e500 | 0.155 | 4.129 | 0.301 | 4.525 | 0.267 | 3.411 | pass |
| Run 1602 e500 | **0.069** | 3.836 | 0.389 | 3.151 | **0.297** | 2.589 | pass; query rank near floor |
| Run 1401 e2500 | 0.131 | 4.539 | 0.385 | 3.917 | 0.251 | 3.041 | pass |
| Run 1602 e2500 | **0.095** | 3.896 | 0.525 | **2.588** | **0.266** | 2.066 | **fail query-rank gate** |
| Run 1401 e5000 | 0.120 | 4.794 | 0.418 | 3.716 | **0.261** | 2.837 | pass |
| Run 1602 e5000 | **0.104** | 3.869 | 0.542 | **2.520** | 0.254 | 1.974 | **fail query-rank gate** |
| Run 1401 best e4585 | 0.122 | 4.733 | 0.419 | 3.717 | **0.261** | 2.862 | pass |
| Run 1602 best e4890 | **0.104** | 3.870 | 0.544 | **2.518** | 0.256 | 1.975 | **fail query-rank gate** |

![Extended topology trajectory](generated/stage7_k_audit/20260824_210713_extended/synthesis/topology_trajectory_extended.png)

K=4 preserves distinct environment assignments and remains comfortably above the environment-rank floor, but its query organization becomes increasingly correlated after epoch 500. Query effective rank drops 20.1% from 3.151 at epoch 500 to 2.520 at epoch 5000, query cosine rises from 0.389 to 0.542, and pairwise-map rank drops from 2.589 to 1.974. The absolute query-rank gate was set before the extension and is not weakened here.

This is not a numerical or all-edge collapse: environment rank remains 3.87 of a maximum four, query cosine still narrowly passes, and region separation remains nonzero. It is a task-relevant dimensional compression that coincides with better global regression and worse near-interface `u`/`omega`. The result supports diagnosing the representation tradeoff separately, not promoting the checkpoint as the default scientific model.

The topology artifact contains 4,320 unique rows: 8 checkpoints × 90 cases × 3 organizer passes × 2 representations. Independent median recomputation reproduced the four central query-rank statistics exactly.

## 7. Computational efficiency

### 7.1 Controlled dense GPU-1 benchmark

All rows were measured in one process on an NVIDIA RTX 6000 Ada Generation, case 0273, 8,192 queries, 10 warmups, and 40 synchronized iterations.

| Checkpoint | Prepared median / p95 | Full-forward median / p95 | Total parameters | State keys |
|---|---:|---:|---:|---:|
| Run 1401 best | `6.528 / 6.723 ms` | `25.228 / 27.371 ms` | 3,508,649 | 237 |
| Run 1601 best | `6.315 / 6.728 ms` | `25.522 / 27.184 ms` | 3,508,649 | 237 |
| Run 1602 best | `6.310 / 6.416 ms` | `25.069 / 26.615 ms` | 3,507,621 | 237 |
| Run 1602 epoch 5000 | **`6.295 / 6.546 ms`** | `25.092 / 26.667 ms` | 3,507,621 | 237 |
| Run 1603 best | `6.502 / 6.646 ms` | **`25.142 / 26.169 ms`** | 3,509,677 | 237 |

K=4 removes only 1,028 parameters, or 0.029% of the model. Relative to accepted Run 1401, Run 1602 best reduces prepared median by 3.34% and full-forward median by 0.63%. Relative to fused K=6 Run 1601, it changes prepared median by -0.08% and full-forward median by -1.78%. These are small benchmark-scale differences; the rerun also changes the apparent K=8 ordering relative to the original benchmark, confirming that sub-2% full-forward differences should not be interpreted as stable architecture speedups.

### 7.2 Gathered beta-0.98 execution

| Checkpoint | Fused dense median | Gathered median | Runtime reduction | Dense/gathered peak | Memory reduction | Full-support max difference |
|---|---:|---:|---:|---:|---:|---:|
| Run 1401 best | `6.301 ms` | `3.761 ms` | 40.32% | `384.52 / 158.49 MiB` | 58.78% | `0.0` |
| Run 1602 best | `6.315 ms` | `3.887 ms` | 38.45% | `384.26 / 158.23 MiB` | 58.82% | `0.0` |
| Run 1602 epoch 5000 | `6.312 ms` | `3.775 ms` | 40.19% | `384.26 / 158.23 MiB` | 58.82% | `0.0` |

Mean retained beta mass is at least `0.999999991` and mean active module-route reduction remains 0% for these five-module cases. Pooled relative MSE changes range from `-3.44e-9` to `-2.87e-9`, and exact full-support gathered parity is zero max absolute difference. The efficiency gain therefore comes from the gathered/vectorized execution layout at the current problem size, not from scientific sparsification or removal of active module routes.

![Extended efficiency comparison](generated/stage7_k_audit/20260824_210713_extended/synthesis/efficiency_extended.png)

### 7.3 Training throughput evidence

Run 1602's epoch-2500 and epoch-5000 checkpoint modification times are 17:53:28 and 21:02:11 EDT, a continuous 2500-epoch interval of approximately 4.53 seconds per epoch. Run 1401's full manifest averages 4.36 seconds per epoch, while the original Run 1602 epoch-500 manifest segment averaged 4.55 seconds per epoch. These records do not substantiate a training-speed improvement. GPU identity, co-tenancy, cache state, checkpoint overhead, and logging were not controlled, so they also do not prove a slowdown.

## 8. Compatibility and validation

| Gate | Result |
|---|---|
| Run 1602 strict checkpoint loading at 2500, 5000, and best | pass |
| Run 1602 237-key inventory | pass |
| Run 1602 optimizer-resume semantics | pass |
| Run 1000 golden numerical replay | exact; 237 keys; 1 optimizer group |
| Run 1401 golden numerical replay | exact; 237 keys; 1 optimizer group |
| Dense versus full-support gathered output | exact, max absolute difference `0.0` |
| Full repository tests | 283 passed, 1 expected skip in 10.67 seconds |
| Expected skip | local inverse joint-training artifacts unavailable |

The compatibility evidence does not show a regression in historical Run 1000, accepted Run 1401, ThermalChannel coupling, configuration loading, inverse-facing contracts, strict state loading, or evaluation-only gathered overrides.

## 9. Interpretation

### 9.1 What K=4 improves

K=4 is a credible compact scientific alternative in aggregate regression terms. It reaches a slightly better whole-fluid test result than the accepted K=6 checkpoint, improves the distribution median and p95 rather than only a few large cases, and improves four of five whole-fluid channels. Its epoch-2500 checkpoint is particularly balanced, beating matched Run 1401 in both whole-fluid and near-interface pooled MSE.

### 9.2 Why K=4 is not promoted

The late K=4 model trades away query/pairwise dimensional diversity and near-interface vorticity fidelity. The query-rank failure is persistent at epochs 2500, 5000, and best, not a one-checkpoint fluctuation. The near-interface regression also appears at epoch 5000 and best. Promoting K=4 would therefore weaken an established structural gate and accept a localized physics-facing regression in exchange for a small aggregate MSE improvement.

### 9.3 What actually improves computation

Reducing K from six to four is not a useful primary compute lever in the current implementation because the expensive legacy pair MLP is shared and the two-edge reduction removes only 1,028 parameters. Fused gathered execution supplies the material real-case improvement: roughly 38–40% lower prepared runtime and 59% lower peak allocation without changing checkpoint state or predictions. The next efficiency work should continue on sparse context-fusion execution while keeping K=6 science frozen.

### 9.4 Recommended follow-up

- Keep Run 1401 epoch 4585 as the scientific forward baseline and keep the fused K=6 overlay as the exact training/evaluation execution profile.

- Mark Run 1602 as a completed, non-promoted K=4 research reference; retain epoch 2500 and epoch 4890 because they expose the accuracy/topology tradeoff from different points.

- Do not continue Run 1602 beyond 5000, do not resume Run 1603, and do not launch K=12 under this audit.

- If K-dependent topology thresholds are studied later, define and validate them prospectively across multiple seeds and K values; do not retroactively relax the current `>3.0` query-rank gate.

- Proceed to sparse context-fusion execution using the accepted K=6 science, with explicit route-removal, parity, regional-accuracy, and topology safeguards.

## 10. Limitations and robustness

- Runs 1401 and 1602 are single-seed training runs. The 2.21% aggregate best-checkpoint difference is smaller than a robust architecture claim would require without seed replication.

- Best checkpoints are selected on validation field MSE, not test data, but comparison among multiple checkpoints and metrics still creates researcher degrees of freedom. The matched 2500/5000 tables are therefore as important as the best-checkpoint table.

- Topology gates are absolute and were originally calibrated around K=6. They remain binding for this audit; whether a K-normalized rank measure is scientifically preferable is a separate prospective question.

- The complete split has 90 cases and two environment-resolution regimes, but it remains one ThermalChannel dataset. No out-of-distribution geometry or operating-condition split was added.

- Inference timings are controlled single-case measurements, while training throughput is reconstructed from manifests/checkpoint times and is descriptive only.

- Current five-module cases retain effectively all beta mass and remove no active module routes at beta 0.98. The gathered speedup is real execution efficiency, but it is not yet evidence of workload sparsity.

## 11. Reproducibility

Initial epoch-500 evidence is under `diagnostics/generated/stage7_k_audit/20260824_143720/`. Extended evidence is under ignored `diagnostics/generated/stage7_k_audit/20260824_210713_extended/`, with maintained synthesis source at `tools/diagnostics/analyze_stage7_k_audit.py`.

The extended evaluation used the maintained diagnostics rather than duplicated scripts or generated result folders in source control:

```bash
conda run --no-capture-output -n ModularDT python tools/diagnostics/evaluate_stage5_accuracy.py --checkpoint run1401_e2500=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/epoch_2500_model.pt --checkpoint run1401_e5000=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/epoch_5000_model.pt --checkpoint run1401_best=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/best_by_field_mse_model.pt --checkpoint run1601_best=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1601_20260824_112400_stage7_fused_query_module/best_by_field_mse_model.pt --checkpoint run1602_e2500=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1602_20260824_130922_stage7_k4_fused_audit/epoch_2500_model.pt --checkpoint run1602_e5000=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1602_20260824_130922_stage7_k4_fused_audit/epoch_5000_model.pt --checkpoint run1602_best=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1602_20260824_130922_stage7_k4_fused_audit/best_by_field_mse_model.pt --checkpoint run1603_best=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1603_20260824_135513_stage7_k8_fused_audit/best_by_field_mse_model.pt --device cuda:1 --split test --query-batch-size 8192 --max-cases 90 --output-dir diagnostics/generated/stage7_k_audit/20260824_210713_extended/accuracy
```

```bash
conda run --no-capture-output -n ModularDT python tools/diagnostics/evaluate_topology_quality.py --checkpoint run1401_e2500=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/epoch_2500_model.pt --checkpoint run1401_e5000=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/epoch_5000_model.pt --checkpoint run1401_best=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1401_20260823_151126_stage7_modern_structured_context/best_by_field_mse_model.pt --checkpoint run1601_best=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1601_20260824_112400_stage7_fused_query_module/best_by_field_mse_model.pt --checkpoint run1602_e2500=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1602_20260824_130922_stage7_k4_fused_audit/epoch_2500_model.pt --checkpoint run1602_e5000=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1602_20260824_130922_stage7_k4_fused_audit/epoch_5000_model.pt --checkpoint run1602_best=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1602_20260824_130922_stage7_k4_fused_audit/best_by_field_mse_model.pt --checkpoint run1603_best=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1603_20260824_135513_stage7_k8_fused_audit/best_by_field_mse_model.pt --device cuda:1 --split test --query-batch-size 8192 --max-cases 90 --organizer-passes --render-case-id 0273 --render-case-id 0653 --output-dir diagnostics/generated/stage7_k_audit/20260824_210713_extended/topology
```

```bash
conda run -n ModularDT python tools/diagnostics/analyze_stage7_k_audit.py
conda run -n ModularDT python tools/diagnostics/replay_forward_golden.py --device cpu
conda run -n ModularDT pytest -q
```

## 12. Final K-audit decision

| Item | Final decision |
|---|---|
| Scientific baseline | retain Run 1401 K=6 best-by-field epoch 4585 |
| Efficient exact execution | retain fused query-module dense overlay |
| Efficient evaluation/deployment | retain gathered beta-0.98 override |
| Run 1601 K=6 fused | completed execution control; no continuation needed |
| Run 1602 K=4 fused | completed 5000-epoch research audit; retain evidence, do not promote or continue |
| Run 1603 K=8 fused | rejected/stopped at epoch 500 |
| K=12 | do not launch |
| Next scientific/engineering phase | sparse context-fusion execution on frozen K=6 science |

Run 1602 demonstrates that K=4 can match or slightly exceed K=6 aggregate accuracy in one seed, but it does so with a failed query-rank gate and worse near-interface error. The binding decision is therefore to preserve Run 1401 as the scientific baseline, preserve Run 1602 as research evidence, and move forward with sparse execution work without changing the accepted K=6 model behavior.
