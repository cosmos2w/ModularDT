# Runs 1406 and 1407 best-before-5000 comparative evaluation

**Compared models:** Run 1406 (`group_control_pairwise_honf`), Run 1407 (`phase_shared_group_control_honf`), historical Run 1404 (`routing_only_pairwise`), and Dense Run 1804 (`dense_pairwise_field`).

**Evaluation date:** 2026-09-20.

**Primary checkpoint policy:** each run's validation-field-MSE-selected checkpoint before or at epoch 5000. Exact epoch 5000 is a sensitivity analysis, not the primary selection rule.

**Scope:** 90-case predicted-port accuracy, convergence, synchronized inference and optimizer-step cost, training-memory dynamics, and learned routing and organization.

## Executive conclusion

The mature result is a genuine trade-off rather than one universal winner.

- **Accuracy:** validation-selected pooled fluid relative L2 is **0.02896 for Dense 1804**, **0.03065 for Run 1407**, **0.03250 for Run 1406**, and **0.03460 for Run 1404**. Run 1407 improves on Run 1406 by **5.69%** and wins 59/90 paired cases, but Dense remains 5.51% better and wins 63/90 against 1407. Run 1407's initially different convergence was real: it caught and modestly passed Run 1406 only late in training.
- **Checkpoint fluctuation matters:** at exact epoch 5000, Run 1406 is better than Run 1407 on 67/90 cases and in pooled error, reversing their validation-selected ordering. A single endpoint would therefore give the wrong answer to the requested best-before-5000 question.
- **Inference:** Run 1407's shared controller reduces preparation by **8.4%** relative to 1406, making its full forward **1.8% faster**. Its P2 decode is nevertheless **17.0% slower** than 1406. Run 1407 remains 12.4% slower than Dense for one-shot full forward; Run 1404 is fastest.
- **Training:** Runs 1406 and 1407 are effectively tied on the canonical M12 update (1146.6 and 1145.9 ms) and both are about **45.6% faster** than Dense. Their M12 after-forward allocation is also about 13.7--13.9% below Dense.
- **Routing:** Run 1406 remains execution-dense: its mean query degree is 5.999/6 and environment unique support is 99.98%. Run 1407 forms selective query routes (mean degree 3.340/6) and 73.22% environment support, but the QE executor remains rectangular. Module support is still 100% of valid sources; the hybrid executor mainly removes padding for small-module cases. Across the whole population, this removes only about **2.2% of all module-plus-environment fine rows**.
- **Organization:** both group-control models are functionally organized but not six-way physically specialized. Run 1407 occupies exactly two of six module groups in every case; Run 1406 does so in 88/90 cases, with one- and three-group exceptions. Frozen uniform-query interventions materially change predictions, so routing is used; however, group identities are permutation ambiguous and the evidence does not establish physical causality.
- **Memory changes:** the large Run-1406 drop at epoch 501 is the accepted QE activation-checkpoint boundary introduced at resume, not routing. Run 1407's early 17--23 GiB transitions are consistent with its route-dependent hybrid module executor crossing the selected/rectangular threshold. Once mature, 64 repeated fixed M12 updates show no growth in cleanup allocation. Smaller shared one-epoch dips are consistent with workload/allocator high-water effects, and the measured trace provides no evidence of a controller-state leak.

The practical verdict is: **Dense 1804 remains the primary fluid-accuracy and one-shot-inference reference; Run 1407 is the best routed accuracy result and has cleaner selective query routing; Run 1406 retains the better prepared P2 decoder; and Run 1404 remains the latency floor.** A future scientific run is not justified merely to obtain more sparsity. If one is commissioned, it should target the single unresolved bottleneck: **environmental QE support/execution, where 1407 learns logical sparsity but still executes the full rectangle.**

![Mature accuracy, convergence, and recorded memory](../../diagnostics/generated/run1406_run1407_best5000_comparison_20260920/figures/convergence_accuracy_memory.png)

## 1. Experimental controls and checkpoint policy

### 1.1 Why the primary checkpoints are not exact epoch 5000

The checkpoint was selected independently for each run by its training-run validation field MSE. The 90-case evaluation population was not used to choose an epoch. This answers "best accuracy achieved before 5k" without selecting on the final evaluation cases.

| Model | Selected epoch | Validation field MSE | Exact endpoint retained? |
|---|---:|---:|---|
| Run 1404 | 4890 | 0.00187127 | epoch 5000 sensitivity |
| Run 1406 | 4797 | 0.00194972 | epoch 5000 sensitivity |
| Run 1407 | 4782 | 0.00193070 | epoch 5000 sensitivity |
| Dense 1804 | 4738 | **0.00152102** | epoch 5000 sensitivity |

Every selected checkpoint strict-loaded through the trusted loader and retained optimizer and RNG state. No training process or checkpoint was modified during this evaluation.

### 1.2 Accuracy protocol

All checkpoints were evaluated on the same established 90-case test/development holdout with:

- 8,192 query points per case;
- predicted ports, not ground-truth port substitution;
- identical physical normalization and case population;
- pooled and equal-case fluid metrics;
- near/far, component, interface, port, and temperature metrics; and
- debug maps outside timed measurements.

### 1.3 Cost protocol

The controlled cost benchmark used physical GPU 1 and cases 0273 and 0653:

- Q=8,192, receiver chunk 2,048;
- two warmups and five synchronized repetitions for inference;
- full forward, preparation plus one query, and prepared P2 decode measured separately;
- a separate CUDA-event pass for semantic phase attribution;
- real predicted-port B48/Q1024 M1 and M12 updates; and
- one warmup plus 64 measured disposable updates per workload.

The semantic CUDA-event regions are nested and should not be added. Training used a fresh disposable optimizer with the maintained loss path; neither model nor optimizer state was written back.

### 1.4 Data-quality reconciliation

The reduction handled three historical artifacts explicitly:

- Run 1404 repeats epochs 851--854 and Run 1407 repeats epoch 137. The later row was retained; all scientific fields in each duplicate were identical, while wall-time or peak-memory fields differed.
- Dense 1804's post-resume rows contain 324 values under the original 286-column header. Its final three values are the predicted validation total, field, and temperature metrics; these were recovered explicitly.
- Dense memory after epoch 500 is not interpreted because of that schema drift. Checkpoint metadata remains authoritative for the selected epoch and metric.

## 2. Predictive accuracy

### 2.1 Validation-selected whole-population accuracy

| Model | Pooled fluid rel. L2 | Equal-case mean | Median | P95 | Maximum | Worst case |
|---|---:|---:|---:|---:|---:|---|
| Run 1404 | 0.03460 | 0.03244 | 0.02811 | 0.06028 | 0.06690 | 0297 |
| Run 1406 | 0.03250 | 0.02875 | 0.02361 | **0.05439** | 0.08054 | 0297 |
| Run 1407 | 0.03065 | 0.02717 | 0.02180 | 0.05731 | 0.06501 | 0295 |
| Dense 1804 | **0.02896** | **0.02562** | **0.02075** | **0.05281** | **0.06398** | 0298 |

Paired-case evidence is consistent with the pooled result:

| Candidate | Baseline | Candidate wins | Baseline wins | Mean relative-L2 delta |
|---|---|---:|---:|---:|
| Run 1407 | Run 1406 | **59** | 31 | -0.00158 |
| Run 1407 | Dense 1804 | 27 | **63** | +0.00155 |
| Run 1406 | Dense 1804 | 18 | **72** | +0.00314 |
| Run 1406 | Run 1404 | **69** | 21 | -0.00369 |

Run 1407 therefore recovers another 5.69% over Run 1406, but not the remaining dense-baseline gap. Its maximum is lower than Run 1406's, although Run 1406 has the better P95; neither routed model dominates the other in every tail metric.

### 2.2 Component behavior

| Pooled relative L2 | Run 1404 | Run 1406 | Run 1407 | Dense 1804 | Best |
|---|---:|---:|---:|---:|---|
| Fluid field | 0.03460 | 0.03250 | 0.03065 | **0.02896** | 1804 |
| Near interface | **0.03337** | 0.04122 | 0.03749 | 0.03504 | 1404 |
| Far fluid | 0.03839 | 0.02788 | 0.02738 | **0.02537** | 1804 |
| U | 0.02323 | **0.01148** | 0.01355 | 0.01264 | 1406 |
| V | 0.01970 | 0.01417 | 0.01462 | **0.01287** | 1804 |
| Pressure | 0.03625 | 0.02352 | **0.02319** | 0.02486 | 1407 |
| Vorticity | **0.04064** | 0.04885 | 0.04452 | 0.04203 | 1404 |
| Field temperature | 0.04873 | 0.04419 | 0.04198 | **0.03867** | 1804 |
| Internal temperature | 0.03377 | 0.02699 | 0.02708 | **0.02680** | 1804 |
| Surface temperature | 0.04482 | 0.04033 | 0.03946 | **0.03898** | 1804 |
| Normal heat flux | 0.13784 | 0.11712 | 0.11259 | **0.10615** | 1804 |
| Final environment T | 0.07025 | 0.06940 | 0.06729 | **0.06699** | 1804 |
| Final effective h | 0.05001 | 0.04689 | **0.04682** | 0.04813 | 1407 |

The routed models have real component wins: Run 1406 on U and Run 1407 on pressure and effective h. Run 1404 retains strong near-interface and vorticity behavior. Dense wins most temperature and aggregate fluid metrics.

Run 1407 also has the best reported all-domain relative L2 (0.34606, versus 0.37386/0.38135/0.43058 for 1404/1406/Dense). That secondary aggregate includes the internal-module region and has less secure attribution than the primary fluid mask; it is reported, but it does not replace the fluid-domain ranking.

![Component accuracy relative to Dense](../../diagnostics/generated/run1406_run1407_best5000_comparison_20260920/figures/component_accuracy_vs_dense.png)

### 2.3 Exact epoch-5000 sensitivity

| Exact epoch 5000 | Pooled fluid rel. L2 | Equal-case mean | Wins versus Dense |
|---|---:|---:|---:|
| Run 1404 | 0.03549 | 0.03335 | 6/90 |
| Run 1406 | 0.03223 | 0.02861 | 27/90 |
| Run 1407 | 0.03385 | 0.03066 | 13/90 |
| Dense 1804 | **0.02966** | **0.02643** | -- |

Run 1406 beats Run 1407 on 67/90 endpoint cases. This is not a contradiction: their validation and test curves fluctuate at the mature scale, and their best field checkpoints occur at different epochs. The selected-best result is the appropriate primary comparison; the endpoint result bounds policy sensitivity.

### 2.4 Convergence and the early Run-1407 lag

Run 1407 was genuinely different, not simply broken. Its trailing-100 validation-field median relative to Run 1406 was 1.218x at epoch 50, 3.663x at 100, 0.968x at 250, 0.899x at 500, 1.068x at 1000, 1.199x at 2500, and 0.987x at 5000. The relationship is non-monotone, but the final trailing-100 medians are close: 0.002470 for 1407 and 0.002503 for 1406.

This supports the user's decision to continue the same run: Run 1407 needed a longer horizon to become competitive. It does not show that phase sharing is uniformly superior; the best-before-5000 gain is modest and endpoint ordering still fluctuates.

## 3. Computation efficiency

### 3.1 Synchronized inference

Values are means of the two per-case medians.

| Phase | Run 1404 | Run 1406 | Run 1407 | Dense 1804 |
|---|---:|---:|---:|---:|
| Full forward | **25.38 ms** | 38.90 ms | 38.21 ms | 33.98 ms |
| Preparation + one query | 22.45 ms | 28.66 ms | 26.26 ms | **21.85 ms** |
| Prepared P2 decode | **6.39 ms** | **12.03 ms** | 14.08 ms | 13.02 ms |

Run 1407's phase-shared controller does exactly what its design predicts: preparation improves 8.4% over 1406. That gain is large enough to make full forward 1.8% faster than 1406, but not enough to match Dense. The prepared P2 path regresses by 17.0% relative to 1406 and 8.1% relative to Dense.

Run 1406 preserves the best group-control P2 result: it is 7.6% faster than Dense once prepared. Its full forward is still 14.5% slower because preparation cost dominates the saved P2 time. Run 1404 remains the unambiguous latency floor, but it is the least accurate primary checkpoint.

![Inference and training cost](../../diagnostics/generated/run1404_1406_1407_1804_best5000_performance_20260920/performance_comparison.png)

### 3.2 Where the forward time goes

The separate CUDA-event diagnostic gives the following mean nested regions:

| Semantic region | Run 1406 | Run 1407 | Dense 1804 |
|---|---:|---:|---:|
| Encode | 0.81 ms | 0.86 ms | 0.82 ms |
| P0 prepare | 3.82 ms | 4.92 ms | 2.19 ms |
| P1 prepare | 3.74 ms | **1.75 ms** | 2.15 ms |
| P2 prepare | 3.66 ms | **1.73 ms** | 2.35 ms |
| P0 read | 2.46 ms | 2.97 ms | 1.98 ms |
| P1 read | 2.37 ms | 2.95 ms | 1.98 ms |
| P2 read | **11.95 ms** | 13.88 ms | 12.75 ms |
| Local surrogate | 5.52 ms | 5.51 ms | 5.73 ms |
| Local fusion | 0.44 ms | 0.41 ms | 0.44 ms |
| P2 QM subregion | **2.58 ms** | 4.10 ms | n/a |
| P2 QE subregion | **8.01 ms** | 8.30 ms | n/a |

These regions are nested rather than additive. They still localize the result:

1. Run 1407 eliminates repeated controller construction at P1/P2, but P0 must construct the shared controller and six-bit support metadata.
2. Run 1407's P2 penalty is concentrated in QM and selected-support dispatch; QE remains almost unchanged and rectangular.
3. Encoding, the local surrogate, and local fusion do not explain the routed full-forward gap.
4. Dense has cheaper preparation and P0/P1 reads; Run 1406's prepared P2 win is therefore insufficient to win one-shot forward latency.

![Forward phase attribution](../../diagnostics/generated/run1404_1406_1407_1804_best5000_performance_20260920/phase_breakdown.png)

### 3.3 Canonical optimizer updates

| Model | M1 median | M12 median | M12 after-forward allocated | M12 reserved high-water |
|---|---:|---:|---:|---:|
| Run 1404 | **325.74 ms** | **647.05 ms** | 23.79 GiB | 24.41 GiB |
| Run 1406 | 535.42 ms | 1146.62 ms | 21.32 GiB | 25.33 GiB |
| Run 1407 | 537.95 ms | 1145.91 ms | **21.27 GiB** | **24.77 GiB** |
| Dense 1804 | 966.16 ms | 2106.45 ms | 24.71 GiB | 28.33 GiB |

Run 1406 and 1407 are indistinguishable at practical M12 precision. They are 45.6% faster than Dense and retain about 13.7--13.9% less activation allocation after the forward. Phase sharing does not materially change training memory: the live P0 controller graph is retained by design, while both group-control runs benefit from the same accepted complete-QE checkpoint boundary.

Run 1404 is much faster, but its M12 allocation is higher than the two mature group-control executors. This is a useful warning that parameter count and training activation memory do not rank architectures identically.

Model size follows the expected design boundary: Run 1404 has 3,508,649 total parameters (2,473,510 trainable), Runs 1406 and 1407 each have 3,987,140 (2,952,001 trainable), and Dense 1804 has 5,430,548 (4,395,409 trainable). Run 1407 therefore introduces no parameters relative to Run 1406; its cost difference is executor/data-flow behavior.

## 4. Sparse routing and organization

### 4.1 Population-level topology

| Statistic across 90 cases | Run 1406 | Run 1407 |
|---|---:|---:|
| Mean positive query degree / 6 | 5.999 | **3.340** |
| Mean effective query groups | 4.022 | **2.383** |
| Mean normalized query entropy | 0.771 | **0.479** |
| Mean module source degree | 1.355 | 1.355 |
| Mean environment source degree | **1.224** | 2.212 |
| Occupied module groups / 6 | 2.000 | 2.000 |
| Occupied environment groups / 6 | 4.122 | 6.000 |
| Module unique support / valid pairs | 100.00% | 100.00% |
| Environment unique support / valid pairs | 99.98% | **73.22%** |
| Mean query mass on module-empty groups | 0.329 | **0.121** |
| Mean query mass on environment-empty groups | 0.540 | **0.000** |

Run 1407 succeeds at the intended query-side change. It cuts positive query degree by 44%, lowers entropy, removes environment-empty-group mass, and makes the environmental logical support materially selective. Run 1406 has sparse source-to-group assignments but nearly all-six query access, so source sparsity collapses back to an almost full pair set.

![Routing support across 90 cases](../../diagnostics/generated/run1406_run1407_best5000_routing_organization_20260920/figures/routing_support_summary.png)

### 4.2 Logical support is not executed work

Run 1407 deliberately keeps different support and execution counters:

- module support covers every valid active module, but the hybrid executor gathers small M=3/M=5 cases and uses the rectangular M=12 path for the mature M=7/M=10 cases;
- population-weighted module execution is 62.96% of the padded module rectangle but 129.52% of the valid active-module rectangle, because the rectangular large-module cases also evaluate padding;
- environment logical support is 73.22%, yet QE executes 100% of the Q-by-192 rectangle; and
- including both source families, executed fine rows are about 97.8% of the full module-plus-environment rectangle.

Thus 1407 learns real logical sparsity but obtains only a small net row reduction. The expensive environmental path dominates total work. Support counts must not be relabeled as executed rows.

Run 1406's map/evidence path can use a partial support reader, but its normal timed/training path is rectangular. Its mature support is dense enough that this distinction is negligible for QE and does not establish sparse runtime.

### 4.3 Did meaningful organization form?

There is positive evidence of functional organization:

- source and query memberships contain exact zeros and are normalized;
- route statistics vary systematically with module count and geometry;
- Run 1407 reuses exactly the same P0 controller at P1/P2, while refreshing fine source values and K/V as designed;
- forcing uniform query-group access changes predicted fields by relative L2 0.155--0.453 on the four Run-1406 anchors and 0.237--0.715 on Run-1407, so learned query routing is not ignored; and
- consistently permuting the learned group code labels changes output only at floating-point scale, as expected for an exchangeable group labeling.

There are also important negative limits:

- Run 1407 occupies exactly two module groups in all 90 cases, while Run 1406 does so in 88/90 cases and has one one-group and one three-group exception;
- Run 1407 fills all environment groups but retains broad source membership;
- group labels are permutation ambiguous across cases;
- correlations with module count partly reflect the structured case design; and
- no learned route is evidence of recovered physical causality.

The defensible conclusion is therefore **functional, case-responsive organization with partial collapse, not a discovered six-part physical decomposition**. Run 1407 improves query selectivity and phase consistency, but the environment remains the central sparsity bottleneck.

## 5. Why recorded memory changes only occasionally

### 5.1 What the training CSV records

`peak_cuda_memory_mb` is a per-epoch maximum of allocated CUDA memory across training and validation after peak statistics are reset. It is not reserved memory and not an instantaneous end-of-step value. A single high-memory batch or executor branch therefore sets the value for the whole epoch.

### 5.2 Run 1406: a code boundary, not routing

| Segment | Median epoch peak | Range |
|---|---:|---:|
| Epochs 1--500 | 29.08 GiB | 28.77--29.14 GiB |
| Epochs 501--5000 | 23.27 GiB | 22.95--23.36 GiB |

The exact epoch-500-to-501 boundary drop is 5.78 GiB; the rounded segment medians in the table differ by 5.81 GiB. The drop occurs when the resumed run adopted the accepted complete-QE activation-checkpoint boundary. Earlier controlled parity work showed that the boundary preserved outputs and first derivatives exactly while reducing retained saved tensors. This is an implementation/resume boundary, not a learned routing transition.

### 5.3 Run 1407: early hybrid-path transitions, then a plateau

| Segment | Median epoch peak | Range |
|---|---:|---:|
| Epochs 1--50 | 17.33 GiB | 17.18--21.23 GiB |
| Epochs 51--100 | 23.22 GiB | 21.69--23.27 GiB |
| Epochs 101--5000 | 23.22 GiB | 22.90--23.30 GiB |

Run 1407's early controller changes which query/module pairs satisfy the six-bit support test. The hybrid executor selects gathered module execution only when support is at most half of the padded rectangle. During epochs 1--60, recorded peak memory correlates strongly with module-group incidence (Pearson 0.83). This is consistent with increasingly broad module support crossing the discrete executor threshold and causing a few batches to set a higher epoch peak. It is observational attribution supported by the executor policy and the trace, not a claim that every early jump has one route tensor as its sole cause.

At the mature checkpoint, module support covers all valid sources. M=3/M=5 cases gather and M=7/M=10 cases run rectangular; a training epoch containing a large-module batch therefore records the same high-water plateau.

### 5.4 What the 64-step trace rules out

With checkpoint and workload fixed, all four models return to one invariant post-cleanup allocated-memory value across 64 consecutive updates. Reserved memory moves through a small number of CUDA allocator plateaus and then remains cached. No OOM or allocation retry occurs.

This fixed-workload trace provides no evidence of persistent controller-state accumulation or an ordinary memory leak; it cannot exclude an input-order-dependent or slower leak outside the measured horizon. Small dips at the same epochs in 1406, 1407, and Dense are instead consistent with deterministic batch/case composition and allocator high-water effects. `nvidia-smi` reserved/context memory should not be compared directly with the training CSV's live allocated peak.

![Controlled memory dynamics](../../diagnostics/generated/run1404_1406_1407_1804_best5000_performance_20260920/m12_trace/memory_dynamics.png)

## 6. Development decision

### 6.1 What Run 1407 established

Run 1407 is not a failed implementation:

1. it strict-loads, trains stably, and reaches a validation-selected field result 5.69% better than Run 1406;
2. its shared controller reduces repeated P1/P2 preparation as intended;
3. prototype-plus-case-control keys produce materially selective query routes;
4. routing materially affects predictions; and
5. its optimizer-step cost and activation allocation retain the Run-1406 advantage over Dense.

### 6.2 What it did not establish

It did not make the whole executor sparse or faster than Dense inference:

- valid module support remains complete;
- environmental support remains too broad for the calibrated gathered QE;
- only about 2.2% of combined fine rows are removed;
- support-index and gathered-module overhead make prepared P2 slower; and
- Dense still wins the primary accuracy metric.

### 6.3 Is another run justified?

No additional run is justified merely to repeat this controller, change a threshold, or add a corrective routing loss. The existing evidence is already clear enough: query sparsity is not the limiting factor.

If a future scientific run is authorized, it should isolate one architectural bottleneck only: **environmental QE source support and its executor**. A valid candidate would need to reduce actual environment rows while retaining exact fine K/V refresh and competitive fidelity. It should not expand per-group triples, conflate support with execution, or change the Cg+CM+CE scientific contract. Whether phase-static P0 control itself affects optimization could be studied separately, but it is a different ablation and should not be mixed into the QE-execution question.

## 7. Reproducibility and artifacts

### 7.1 Primary artifacts

- Accuracy and raw 90-case tables: `diagnostics/generated/run1404_1406_1407_1804_best5000_accuracy_20260920/`
- Consolidated checkpoint, convergence, and accuracy reduction: `diagnostics/generated/run1406_run1407_best5000_comparison_20260920/`
- Cost, semantic phase, and 64-step memory profile: `diagnostics/generated/run1404_1406_1407_1804_best5000_performance_20260920/` (the corrected workload-tagged trace is under `m12_trace/`)
- 90-case routing, phase, correlations, and frozen interventions: `diagnostics/generated/run1406_run1407_best5000_routing_organization_20260920/`

The run directories and checkpoints were read in place. No managed training run was launched by this evaluation.

### 7.2 Commands

Primary best-checkpoint accuracy:

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src \
conda run --no-capture-output -n ModularDT python evaluate.py --workflow compare \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1404_20260916_092508_routing_only_pairwise/best_by_field_mse_model.pt \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1406_20260919_132837_low_dimensional_group_control_executor_optimized_rerun/best_by_field_mse_model.pt \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1407_20260919_174751_phase_shared_prototype_group_control/best_by_field_mse_model.pt \
  --checkpoint-path Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/best_by_field_mse_model.pt \
  --label Run1404_best_field --label Run1406_best_field \
  --label Run1407_best_field --label Run1804_best_field \
  --split test --case-ratio 1.0 --query-batch-size 32768 \
  --local-port-condition-mode predicted --return-routing-maps --save-debug-npz \
  --debug-case-id 0273 --debug-case-id 0653 \
  --debug-case-id 0680 --debug-case-id 0298 \
  --anchor-case-id 0273 --anchor-case-id 0653 \
  --anchor-case-id 0680 --anchor-case-id 0298 \
  --skip-figures --device cuda:0 \
  --output-dir diagnostics/generated/run1404_1406_1407_1804_best5000_accuracy_20260920 \
  --dataset /data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5 \
  --saved-root Trained_Results/ThermalChannel/HONF_Forward_Runs
```

The exact-5000 sensitivity used the same command with the four checkpoint paths replaced by `epoch_5000_model.pt` and output under `endpoint5000/`. The full argument vectors are retained in both comparison manifests.

Reductions:

```bash
PYTHONPATH=src:Case_ThermalChannel/src \
conda run --no-capture-output -n ModularDT python \
  tools/diagnostics/reduce_run1404_1406_1407_1804_accuracy.py

PYTHONPATH=src:Case_ThermalChannel/src \
conda run --no-capture-output -n ModularDT python \
  tools/diagnostics/analyze_run1406_run1407_best5000_comparison.py
```

Performance and memory on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src \
conda run --no-capture-output -n ModularDT python \
  tools/diagnostics/profile_best5000_performance.py \
  --device cuda:0 --trace-warmups 1 --trace-steps 64 \
  --output diagnostics/generated/run1404_1406_1407_1804_best5000_performance_20260920
```

After the trace rows were given explicit M1/M12 workload labels, the 64-step memory pass was repeated without re-running inference:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:Case_ThermalChannel/src \
conda run --no-capture-output -n ModularDT python \
  tools/diagnostics/profile_best5000_performance.py \
  --device cuda:0 --trace-only --trace-warmups 1 --trace-steps 64 \
  --output diagnostics/generated/run1404_1406_1407_1804_best5000_performance_20260920/m12_trace
```

Routing and organization on physical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:Case_ThermalChannel/src \
conda run --no-capture-output -n ModularDT python \
  tools/diagnostics/diagnose_run1406_run1407_routing_organization.py \
  --checkpoint 1404=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1404_20260916_092508_routing_only_pairwise/best_by_field_mse_model.pt \
  --checkpoint 1406=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1406_20260919_132837_low_dimensional_group_control_executor_optimized_rerun/best_by_field_mse_model.pt \
  --checkpoint 1407=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1407_20260919_174751_phase_shared_prototype_group_control/best_by_field_mse_model.pt \
  --checkpoint 1804=Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1804_20260905_081349_dense_pairwise_field_adaptation/best_by_field_mse_model.pt \
  --device cuda:0 --query-count 8192 \
  --output-dir diagnostics/generated/run1406_run1407_best5000_routing_organization_20260920
```

Focused validation:

```bash
PYTHONPATH=src:Case_ThermalChannel/src \
conda run --no-capture-output -n ModularDT pytest -q \
  tests/test_reduce_run1404_run1407_accuracy.py \
  tests/test_analyze_run1406_run1407_best5000_comparison.py \
  tests/test_run1406_run1407_routing_organization.py
```

## 8. Limitations

- The 90 established cases are a held-out development population, not new CFD verification.
- GPU timings cover one physical GPU and two representative anchor workloads; ratios are empirical for this hardware and software stack.
- CUDA-event regions are nested and instrumentation adds overhead; use them for attribution, not additive wall-time reconstruction.
- The fixed 64-step trace repeats a canonical workload. It diagnoses allocator stability but cannot reproduce every case ordering seen during 5000 epochs.
- Frozen routing interventions test model dependence, not retrained accuracy or physical causal validity.
- Group labels are exchangeable, so cross-case claims concern support and distributions rather than group identity.
