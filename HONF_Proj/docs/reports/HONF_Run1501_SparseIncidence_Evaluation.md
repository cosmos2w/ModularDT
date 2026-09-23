# HONF Run 1501 sparse-incidence evaluation

## Decision summary

Run 1501 implements the requested sparse-incidence adaptive HONF architecture
without a case-level prototype plan, deletion variable, top-k rule, auxiliary
gate, or sparsity loss. Training completed normally at the mandated epoch-500
stop. The exact endpoint reaches pooled fluid relative-L2 0.11149; the explicit
saved-best-total policy is epoch 444 and is materially better at 0.09891. The
preferred Run-1501 checkpoint is therefore the labeled epoch-444 saved best,
not a relabeled endpoint. No continuation beyond epoch 500 was launched.

The central interim result is already clear. Learned source/query support is
meaningfully smaller than the dense logical graph, but the maintained
rectangular executor still runs the full fine-row rectangles. A diagnostic
support-selected executor reduced row counts on the two anchor cases yet was
9--11% slower and used 30--35% more incremental peak CUDA memory at epoch 50.
At epoch 500 it remains 6--25% slower and uses 25--32% more memory. Therefore
Run 1501 retains the rectangular reader.

## Executed identity and evidence policy

| Item | Executed value |
|---|---|
| Run | `Run_1501_20260922_211056_sparse_incidence_adaptive_honf` |
| Run UUID | `3d329165-dc98-4764-b37d-d28feb850017` |
| Architecture | `sparse_incidence_group_control_honf` |
| Capacity / control width | `K=12`, `D=16` |
| Dataset | `packed_dataset.h5`, 690 cases (600 train / 90 test) |
| Dataset SHA-256 | `4224093c22a67af4adfecc8b21d53548e4263ec2254c230dc83c89526b36da05` |
| Device policy | `cuda:0`, normal CUDA visibility |
| Local-port policy | predicted |
| Population formation sample | all 90 test cases, fixed `Q=1024` queries/case |
| Accuracy comparison | all 90 test cases, full fixed-grid prediction |
| Anchor executor cases | 0273 and 0653, `Q=8192` |

The 90-case test holdout is used here as a development comparison set, not as
an untouched final benchmark. Results are one training seed. Group indices are
permutation-ambiguous, and learned routing organization is not evidence of a
physical causal graph.

## Architecture contract

For each physical phase, source assignment is recomputed locally. Content
logits produce an entmax-1.5 proposal, followed by exactly one geometry
refinement at one quarter of the domain diagonal and a final entmax-1.5
assignment. Only the continuous final assignments enter the forward path.
There is no `Kplan`.

For query location \(x\), the prototype-anchored key and query route are

\[
k_h = \operatorname{RMS}\!\left(c_h + W_h h_h\right),\qquad
q_x = \operatorname{RMS}\!\left(W_q[\phi(x),g]\right),
\]

\[
\ell_{xh}=\frac{q_x^\top k_h}{\sqrt{D}}+
b_{\mathrm{geom}}(x,h),\qquad
r_x=\operatorname{masked\ sparsemax}(\ell_x).
\]

The mask contains phase-occupied groups only. The direct 16-bit support
intersection computes logical source/query pairs without a \(2^K\) lookup
table. The production forward path remains the historical rectangular fine
executor, preserving the Dense and Run-1406 reader contract.

## Prelaunch validation

The frozen-query diagnostic used 90 cases at `Q=1024` and found mean
\(K_q=2.3151\), 1.7378 effective query groups, module support fraction
\(R_M=0.4431\), and environment support fraction \(R_E=0.5723\). The reported
0.5603 relative RMS prediction change is a route perturbation sensitivity,
not a prediction-accuracy metric.

The real prelaunch probe executed batches of 24 and 48 queries plus one AdamW
update. It observed finite gradient norm 0.014996 and parameter-update norm
0.039740; all 12 rows were nonzero and unequal. This probe created no managed
run or checkpoint and used CPU, so it is gradient/shape evidence rather than a
GPU cost claim.

## Continuation gates

| Gate | Validation evidence | Formation / accuracy evidence | Decision |
|---|---|---|---|
| Epoch 10 | managed training and exact checkpoint completed | no collapse or nonfinite state | continue unchanged |
| Epoch 50 | validation total 0.69976; validation field 0.36950 | mean \(K_q=2.9254\); full 90-case matched comparison completed | continue to stronger physical gate |
| Epoch 150 | validation total 0.17119; validation field 0.09376 | all 90 cases improved from exact epoch 50; pooled fluid relative-L2 0.46960 -> 0.22783 | continue unchanged to epoch 500 |
| Epoch 500 | validation total 0.05883; validation field 0.02197 | exact pooled fluid relative-L2 0.11149; best-total e444 gives 0.09891 | stop; prefer labeled best-total e444; no continuation |

At epoch 150, the 90-case case-mean fluid relative-L2 decreased from 0.46374
to 0.22186, near-interface relative-L2 from 0.41378 to 0.23337, far-fluid
relative-L2 from 0.46825 to 0.22507, internal-module-cell relative-L2 from
0.38758 to 0.15086, and fluid temperature relative-L2 from 0.51233 to 0.19172.

![Recorded Run 1501 training trajectory](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1501_20260922_211056_sparse_incidence_adaptive_honf/evaluations/summary_figures/training_curves.png)

## Formation evidence

### Epoch 50

Across 92,160 sampled queries, every query had nonempty support. Mean
\(K_q=2.9254\), mean effective query groups was 2.1008, and the pooled
histogram was

`1:1204, 2:34219, 3:34927, 4:15633, 5:4737, 6:1149, 7:257, 8:32, 9:2`.

All 12 source groups were occupied in every case. This is occupied registered
capacity, not a case plan. Mean source-support fractions were
\(R_M=0.6997\) and \(R_E=0.7148\); mean module/environment source degrees were
4.9022 and 5.9531. Assignment ranks were 5.833 (module), 12 (environment), and
10.833 (query). Geometry was nontrivial: observed-to-mass-preserving-shuffle
radius ratios were 0.6074 for module sources and 0.5019 for environment
sources. These are learned compactness diagnostics, not causal labels.

The execution distinction is decisive. At `Q=1024`, the maintained reader ran
12,288 module rows and 196,608 environment rows per case. Mean module padding
was 789 rows and environment padding was zero. Logical unique-pair reduction
therefore did not become executed row reduction.

### Epoch 150

Every query again had support. Mean \(K_q\) increased modestly to 3.1182 and
effective query groups to 2.1897. The pooled histogram was

`1:948, 2:22333, 3:41445, 4:21138, 5:5074, 6:1060, 7:153, 8:8, 9:1`.

All 12 source groups remained occupied. Mean \(R_M=0.7200\),
\(R_E=0.6733\); module/environment source degrees were 5.1128 and 5.4400.
Observed-to-shuffled compactness ratios were 0.6055 (module) and 0.4830
(environment). The rectangular reader row counts were unchanged.

### Epoch 500

Every query again had support. The cross-checkpoint formation summary is:

| Exact epoch | Mean Kq | Effective query groups | RM | RE | Mean module unique pairs | Mean environment unique pairs | Rectangular module / environment rows |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 2.9254 | 2.1008 | 0.6997 | 0.7148 | 4,198.1 | 140,534.4 | 12,288 / 196,608 |
| 150 | 3.1182 | 2.1897 | 0.7200 | 0.6733 | 4,292.2 | 132,385.7 | 12,288 / 196,608 |
| 500 | 3.1673 | 2.2822 | 0.7497 | 0.6277 | 4,479.4 | 123,419.6 | 12,288 / 196,608 |

At epoch 500 the pooled Kq histogram is

`1:342, 2:20170, 3:41797, 4:24255, 5:4817, 6:732, 7:45, 8:2`.

Per-case mean Kq spans 2.9082 (case 0686) to 3.6563 (case 0644), rather than
collapsing to one case-independent degree. Case 0273 has mean Kq 3.5146,
RM/RE 0.9235/0.7079; case 0653 has mean Kq 3.2637, RM/RE
0.7654/0.6237. Across all cases, the interior coordinate region generally has
higher Kq than the normalized inlet band. These region names are descriptive
coordinate bins, not causal physical regimes.

![Logical support and maintained rectangular work across exact checkpoints](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1501_20260922_211056_sparse_incidence_adaptive_honf/evaluations/summary_figures/support_vs_rectangular_execution.png)

![Per-case Kq spread across all 90 cases](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1501_20260922_211056_sparse_incidence_adaptive_honf/evaluations/sparse_incidence_population_epoch0500_q1024/figures/kq_case_spread.png)

![Representative learned hypergraph organizations for cases 0273 and 0653](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1501_20260922_211056_sparse_incidence_adaptive_honf/evaluations/sparse_incidence_population_epoch0500_q1024/figures/representative_hypergraph_organization__0273__0653.png)

The representative board shows source geometry, query-support Kq, sorted
query-to-group weights, and source incidence side by side. Learned group IDs
are permutation-ambiguous; the visualization describes organization but does
not identify causal interactions.

![Casewise Kq by descriptive coordinate region](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1501_20260922_211056_sparse_incidence_adaptive_honf/evaluations/sparse_incidence_population_epoch0500_q1024/figures/kq_case_coordinate_regions.png)

## Matched fidelity and cost

At the exact epoch-50 checkpoints, pooled full-grid fluid relative-L2 was:

| Model | Pooled fluid relative-L2 | Run 1501 case wins / losses | Mean synchronized case time (s) | Mean incremental peak CUDA MiB |
|---|---:|---:|---:|---:|
| Run 1804 dense | 0.37203 | 5 / 85 | 0.2002 | 81.45 |
| Run 1406 | 0.42440 | 4 / 86 | 0.3482 | 118.32 |
| **Run 1501** | **0.46960** | -- | 0.5016 | 101.50 |
| Run 1404 | 0.61834 | 89 / 1 | 0.0547 | 557.38 |
| Run 1500 mass | 0.63058 | 90 / 0 | 0.4914 | 124.72 |

Run 1500 is retained as development context, not as a formal mature baseline.
The synchronized cost scope is one `predict_case` call per matched case and
excludes dataset I/O, checkpoint loading, metrics, and plotting.

At epoch 500, the exact endpoint and the explicit saved-best-total policy give:

| Model / policy | Checkpoint epoch | Pooled fluid relative-L2 | Case-mean fluid relative-L2 | Near / far mean relative-L2 | Internal mean relative-L2 | Temperature mean relative-L2 | Mean case time (s) | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Run 1804 dense | 500 | **0.09874** | 0.09611 | 0.09126 / 0.09915 | 0.12389 | 0.11275 | 0.1976 | 81.45 |
| **Run 1501 saved-best total** | **444** | **0.09891** | **0.09608** | 0.09685 / 0.09959 | **0.07835** | **0.10737** | 0.5323 | 93.96 |
| Run 1406 | 500 | 0.10949 | 0.10600 | 0.11911 / 0.10004 | 0.14539 | 0.11643 | 0.3517 | 124.56 |
| Run 1501 exact endpoint | 500 | 0.11149 | 0.10760 | 0.11127 / 0.10912 | 0.11977 | 0.14250 | 0.5359 | 94.49 |
| Run 1404 | 500 | 0.15073 | 0.14543 | 0.13033 / 0.16486 | 0.18551 | 0.18308 | 0.0549 | 557.38 |

The saved-best Run1501 checkpoint beats Dense in 48/90 cases, Run1406 in
83/90, Run1404 in 90/90, and the exact Run1501 endpoint in 80/90. Its pooled
fluid error is only 0.17% above Dense, while it has lower internal and
temperature case-mean errors. However, it remains 2.69x slower than Dense and
1.51x slower than Run1406 in this evaluator, so it is a fidelity result rather
than an acceleration result.

![Exact-checkpoint accuracy and measured evaluation cost](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1501_20260922_211056_sparse_incidence_adaptive_honf/evaluations/summary_figures/accuracy_evolution_and_comparators.png)

## Diagnostic support-selected executor

At epoch 50, the diagnostic implementation executed each supported pair once
and was compared with the rectangular reference using three warmups and ten
timed repetitions per anchor case.

| Case | Rectangular rows (module / env) | Selected rows (module / env) | Selected / rectangular median latency | Selected / rectangular peak memory | Max prediction difference |
|---|---:|---:|---:|---:|---:|
| 0273 | 98,304 / 1,572,864 | 17,304 / 1,186,109 | 1.1135 | 1.3498 | 2.40e-5 |
| 0653 | 98,304 / 1,572,864 | 31,607 / 1,136,193 | 1.0937 | 1.2980 | 5.72e-6 |

The predictions agree within the tested `5e-5` numerical tolerance but are not
bitwise identical. Despite fewer logical rows, the materialized selected path
was slower and used more memory. This rejects an executor switch at epoch 50.
The same benchmark will be repeated at epoch 500 because support remains below
dense.

At epoch 500, case 0273 uses 22,609 / 1,108,279 selected module/environment
rows versus 98,304 / 1,572,864 rectangular rows, but the selected path is
1.2523x slower and uses 1.3186x peak memory. Case 0653 uses 31,300 / 983,631
selected rows, but is 1.0630x slower and uses 1.2522x peak memory. Maximum
prediction difference across both cases is 1.29e-5.

![Selected versus rectangular executor at epochs 50 and 500](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1501_20260922_211056_sparse_incidence_adaptive_honf/evaluations/summary_figures/selected_vs_rectangular_benchmark.png)

## Answers to the plan questions

1. **Are source groups coherent?** Yes in the limited learned-geometry sense:
   assignment ranks are nontrivial and radii are smaller than
   mass-preserving shuffles. This does not identify physical mechanisms.
2. **Is query routing genuinely local and sparse?** Yes. Every query has
   support, mean \(K_q\) is far below 12, and support varies across queries.
3. **What is the \(K_q\) distribution?** It is broad but concentrated on two
   to four groups. At epoch 500, per-case means span 2.908--3.656 and only 47
   of 92,160 queries use seven or eight groups.
4. **Is unique support below dense support?** Yes for both source phases.
5. **Is source overlap still the bottleneck?** It is substantial,
   particularly for the environment phase; logical paths exceed unique pairs
   through multiplicity.
6. **Does fidelity improve with training?** Strongly from 50 to 150 on all 90
   cases. Saved-best e444 reaches pooled error 0.09891, but exact e500 regresses
   to 0.11149, so checkpoint policy matters.
7. **Is exact selected execution worthwhile?** Not with the current
   materialization: lower row count did not yield lower latency or memory.
8. **What is the tradeoff?** Run 1501 learns descriptive sparse incidence and
   its best-total checkpoint nearly matches Dense fidelity, but it is slower
   than Dense and Run1406 because the production executor stays rectangular.
9. **What would justify source-sparse execution next?** A fused or segmented
   executor must demonstrate end-to-end latency and peak-memory wins at equal
   predictions; row-count reduction alone is insufficient.

## Reproducible artifacts

- Managed run: `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_1501_20260922_211056_sparse_incidence_adaptive_honf`
- Epoch-50 formation: `evaluations/sparse_incidence_population_epoch0050_q1024`
- Epoch-150 formation: `evaluations/sparse_incidence_population_epoch0150_q1024`
- Epoch-500 formation: `evaluations/sparse_incidence_population_epoch0500_q1024`
- Epoch-50 matched comparison: `evaluations/matched_epoch0050_90case`
- Epoch-50/150 candidate comparison: `evaluations/candidate_epoch0050_epoch0150_90case`
- Epoch-500 matched endpoint/best-policy comparison: `evaluations/matched_epoch0500_90case`
- Epoch-50 selected executor: `evaluations/selected_executor_epoch0050_q8192.json`
- Epoch-500 selected executor: `evaluations/selected_executor_epoch0500_q8192.json`
- Final summary figures: `evaluations/summary_figures`
- Summary renderer: `tools/diagnostics/render_run1501_report_summary.py`

## Limitations carried forward

- The population evidence describes final refined assignments and P2
  source/query incidence; proposal-versus-refined assignment matrices are
  verified algebraically in focused tests but not exported as a population
  artifact.
- The `Q=1024` population pass uses one decoder chunk. Fine-row ledger keys
  are not generalized to aggregate across multiple query chunks.
- Formation measurements at `Q=1024`, full-grid fidelity at `Q=8192`, and
  executor timing are different evidence scopes and must not be conflated.
- Sparse support is learned behavior. It is neither physical causality nor
  proof of hardware sparsity.
- Group labels are permutation-ambiguous across cases and checkpoints.
- Only one seed was trained.
