# Tree faithfulness and response transfer on fixed25_v1

This round made the Tree receiver index and its structural objective depend on physical measures, separated collective-control donors from all-source planning, and trained a strong direct-pair control with the same fine backbone and source-local inputs. Fresh Tree-L stopped at 100 epochs; Tree-F and Pair-F completed the reviewed matched 200-epoch stage. A fresh Run1804-like dense architecture completed all 1,000 selected-data epochs. The candidate plan permits at most 500 epochs, but that ceiling is not a requirement to spend the allowance.

| Goal | What changed | Gains | Misses and limits | Measured scope |
|---|---|---|---|---|
| Predictor | Fresh faithful Tree, direct Pair, legacy Tree and requested dense baseline | At200, Tree-F reduces fluid/material/surface temperature RMSE 30.4%/35.0%/32.1% versus Pair-F200; every measured role is finite in all 22 cases | Pair is better in velocity, pressure functional and effective h; Dense200 remains stronger across broad physical roles, and Dense1000 has a longer budget | Fixed 150 train / all 22 exposed validation, exact retained ages, native benchmark units |
| Organizer | Canonical measure index; mass-weighted cost; both control donor types and planning/phase provenance | Rebuilt physical/representation invariance and correctly pulled-back derivatives pass; trained 200 controls improve thermal accuracy relative to identity | Geometry-matched grouping benefit is weak; no learned eligible-source reduction or demonstrated speedup; global planning remains disclosed | All 22 light rebuilt checks; four detailed cases, task VJPs, interventions and physical executor hooks |
| Inverse | Frozen native predictors; observed-only feasible trust proposals with live continuous physics | All 32 ten-attempt trails complete; observed-only feasible reuse and numerical-rank diagnostics work, with frozen predictors | Graph updates do not improve held fit or call efficiency over joint;23/32 trails leave selected-training heat range; candidate physical validity is unmeasured | Four cases × two input-only starts × four declared modes; completed ten-attempt trails |

**A — added organizer value:** Partial thermal added value appears at 200, including an independently trained Pair comparison and strong trained-control reliance. Geometry-matched action replacement changes representative fluidT RMSE only 0.119%; no unique-grouping, sparse-work or speedup claim follows.

**B — faithful interpretation:** The final trained Tree passes all 22 permutation/unequal-refinement physical and effective-representation checks and 64/64 detailed derivative checks. Both donor types and phase/planning ancestry are exposed. Original failed derivatives and legacy failures remain visible.

**C — response transfer:** Known-null flow consistency misses: both matched candidates have larger u/v/p/omega heat increments at 200 than 100, despite separately improved section-pressure increments. Nonzero held physical heat-response labels are absent inside the selected data. No new solve, Wind training, full-data expansion or formal run was launched.

## Protocol and exact lineages

The [revised execution plan](../../../UpgradePlan/ThemalChannel/20261004_041750Z_HONF_Tree_Faithfulness_and_Response_Transfer_Development_Plan.md) and [development guide](../../guides/Thermal_Tree_Faithfulness_Development.md) retain the [fixed quarter protocol](../../guides/Thermal_Model_Development_Protocol.md). The manifest is `/data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json`, semantic fingerprint `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`. Exactly 150 original train cases and 22 original validation cases are reused. Global normalization fits only those selected training cases; the frozen Stage-A and its own local normalization stay unchanged.

Selection seed is 20261004; initialization and query seed are 0. Q1024, FP32, effective batch 48 and the absolute 1,000-epoch schedule remain fixed. Each completed development epoch visits all 150 selected cases in19 microbatches, reaches four ordinary optimizer boundaries and provides 153,600 primary fluid queries. Query/bucket streams are case/epoch/seed based, independent of architecture-specific RNG consumption. All 22 supply statistics; four input-selected representatives supply detailed Q8192 arrays: 0277(M3), 0291(M5), 0294(M7), 0687(M10). Validation counts are 6/6/6/4; selected training has only four cases each at these M values and also preserves its other module-count strata.

| Arm / run | Active trainable scalars | Accepted final age | Primary case visits | Ordinary optimizer updates | Primary fluid queries |
|---|---:|---:|---:|---:|---:|
| Tree-L /3103 |3,252,792 |100 |15,000 |400 |15,360,000 |
| Tree-F /3204 |3,336,120 |200 |30,000 |800 |30,720,000 |
| Pair-F /3202 |3,207,565 |200 |30,000 |800 |30,720,000 |
| Dense-D25 /3101 |4,395,409 |1,000 |150,000 |4,000 |153,600,000 |

All run directories are below `/data/wanglz/ModularDT/thermal_development/HONF_Forward_Runs/ThermalChannel/HONF_Forward_Runs/`. Literal identities:

- `Run_3103_20261004_003323_thermal_development25_h-tree_v1`.
- `Run_3204_20261004_005022_thermal_faithfulness25_tree-f_v2_tie_repair`.
- `Run_3202_20261004_004520_thermal_faithfulness25_pair-f_v1`.
- `Run_3101_20261004_003323_thermal_development25_b-native_v1`.

Pair width 80 was fixed from parameter counts before outcomes; it has 3.85% fewer active trainable scalars than Tree-F. Common same-shaped physical tensors use the maintained fresh B-fine initializer; new controls are independently seeded. Neither candidate inherits mature physical weights. Dense's core configuration exactly matches the archived Run1804 core—hidden 256/message 128/four heads/environment 24×8/chunk 128 with activation checkpointing—while using fresh weights and the selected dataset. Its larger capacity and maintained native feature path make it a requested strong benchmark, not an isolated test of collective grouping. The archived initial Run1804 configuration says 500 epochs; that number is not its mature checkpoint age. Historical Run1804 is unchanged and is not relabelled as a fair selected-data control.

Checkpoint/latest/best-field/plot cadence is 100 epochs. Selection uses minimum native trainer `val_field_mse` among saved monitoring checkpoints only. Exact-age comparisons and selected-weight comparisons remain distinct. The retained best-field aliases are Tree-L100, Tree-F200, Pair-F200 and Dense1000; scores are 0.624928,0.170335,0.155381 and 0.0192751 respectively. No unsaved favorable epoch enters selection. Detailed claims below use the declared same-age weights, rather than mixing a best checkpoint's fields with another checkpoint's graph or timing.

## Predictor: reconstruction gains, tails and costs

The first 100 epochs use the original physical objective without atlas or null-response callbacks. All 22 native-unit equal-case mean RMSEs were:

| Native-unit all 22 equal-case RMSE | Tree-L100 | Tree-F100 | Pair-F100 | Dense100 |
|---|---:|---:|---:|---:|
| Fluid temperature | 4.52385 | 5.47149 | 4.36374 | 5.34786 |
| u | 0.232852 | 0.244209 | 0.243545 | 0.127904 |
| v | 0.0281279 | 0.0279413 | 0.0262921 | 0.00707376 |
| p | 0.105049 | 0.100178 | 0.108354 | 0.0550785 |
| omega | 0.586242 | 0.573579 | 0.552392 | 0.359167 |
| Material temperature | 3.50903 | 3.74484 | 3.79236 | 4.22994 |
| Surface temperature | 4.11125 | 4.32049 | 4.5131 | 4.56745 |
| q-normal proxy | 7.61902 | 7.70462 | 7.80012 | 7.77735 |
| Final outside temperature | 4.29745 | 4.66377 | 4.51045 | 5.09881 |
| Final heat-transfer coefficient | 1.78405 | 1.62985 | 1.32948 | 1.7199 |
| Inlet/outlet pressure difference | 0.135377 | 0.120159 | 0.114684 | 0.0545179 |
| Module material peak | 3.41351 | 3.19651 | 4.06559 | 3.75698 |

Tree-F100 has 25.4% worse fluid-temperature RMSE than independently trained Pair-F100, while material/surface temperature improve by 1.25%/4.27% and module-peak error improves 21.4%. Tree-F/Tree-L changes several components together and cannot attribute a difference to one mechanism. Finite/nonzero task-control VJPs establish that controls are connected; they do not establish predictive utility.

The actual e100 review was recorded at 06:48:53.868UTC, before the first e101 launch. It authorized a matched extension only to 200 to test the unresolved response-consistency objective. Tree-F's structural measure policy and both candidates' common native physical denominator policy changed explicitly from 1 to 2 at 101; optimizer/RNG, membership, normalization and absolute schedule were restored. Both candidates use the same selected 0348 atlas callback and two selected-training heat-null cases per epoch atQ 256. Dense uses the maintained physical amendment/0348 callback without the new null term. Its post 100 results therefore have a different disclosed scalar objective.

The actual [local e200 review](../../../../../../../../../data/wanglz/ModularDT/thermal_development/tree_faithfulness_20261004/e200_review.json) stops both candidates at completed 200. Thermal predictor gains and learned-control reliance are real, but an unchanged 300–500 extension would mainly pursue field fitting without resolving the failed null-response question or supplying missing positive physical labels. The 500 ceiling remains in the revised plan for a separately justified matched stage. No late coefficient sweep, new objective, seed or model was launched. Final executor and inverse measurements bind the same saved 200 weights.

| Native role, equal-case RMSE | Tree-F200 | Pair-F200 | Dense200 | Dense1000 (longer) |
|---|---|---|---|---|
| Fluid u | 0.129963 | 0.103451 | 0.0684936 | 0.0284574 |
| Fluid v | 0.0155973 | 0.0106756 | 0.00665559 | 0.0036049 |
| Fluid p | 0.0465292 | 0.0289651 | 0.0273815 | 0.0139807 |
| Fluid omega | 0.395732 | 0.310923 | 0.280689 | 0.120396 |
| Fluid temperature | 2.13424 | 3.06548 | 1.76051 | 0.949971 |
| Material temperature | 2.2824 | 3.51092 | 1.35528 | 0.865606 |
| Surface temperature | 2.53879 | 3.73971 | 1.59222 | 0.974319 |
| Normal-flux proxy | 6.12631 | 6.2886 | 6.30705 | 3.58755 |
| Final outside temperature | 2.11724 | 3.15704 | 1.82759 | 1.52459 |
| Final effective h | 1.36326 | 1.26512 | 1.34864 | 0.725213 |
| Pressure difference | 0.0229555 | 0.0173515 | 0.0131414 | 0.0084145 |
| Module peak | 2.52742 | 3.85557 | 1.25699 | 0.841038 |

Tree-F200 fluid/material/surface temperature errors improve 61.0%/39.1%/41.2% from its 100 checkpoint and 30.4%/35.0%/32.1% versus Pair-F200. It also reduces final outside-temperature and flux-proxy errors versus Pair. Pair 200 remains better in u, v, p, omega, pressure difference and effective h; its fluid-temperature error improves 29.8% from 100. At the same 200 age, Dense fluid-temperature error is 1.76051, material 1.35528 and surface 1.59222, so the faithful Tree still misses that broader benchmark. Dense1000 improves fluid/material/surface to 0.949971/0.865606/0.974319; its 400–900 role regressions show that more epochs do not improve every role monotonically. The 1000 checkpoint wins the declared normalized selector among the ten saved milestones.

| Role | Arm / age | Mean | Case p90 | Case max | Pooled RMSE |
|---|---|---|---|---|---|
| Fluid u | Tree-F200 | 0.129963 | 0.145027 | 0.158025 | 0.130401 |
| Fluid u | Pair-F200 | 0.103451 | 0.125473 | 0.126691 | 0.104313 |
| Fluid u | Dense200 | 0.0684936 | 0.0784635 | 0.0916496 | 0.0689143 |
| Fluid u | Dense1000 (longer) | 0.0284574 | 0.0345671 | 0.0399813 | 0.0289255 |
| Fluid temperature | Tree-F200 | 2.13424 | 2.81342 | 3.21179 | 2.18174 |
| Fluid temperature | Pair-F200 | 3.06548 | 4.05141 | 5.74297 | 3.18663 |
| Fluid temperature | Dense200 | 1.76051 | 2.32507 | 2.46871 | 1.80186 |
| Fluid temperature | Dense1000 (longer) | 0.949971 | 1.34089 | 1.43596 | 0.990704 |
| Material temperature | Tree-F200 | 2.2824 | 3.15729 | 3.93766 | 2.5299 |
| Material temperature | Pair-F200 | 3.51092 | 4.47756 | 5.48785 | 3.65425 |
| Material temperature | Dense200 | 1.35528 | 2.2342 | 2.84547 | 1.68475 |
| Material temperature | Dense1000 (longer) | 0.865606 | 1.28717 | 1.70536 | 0.949488 |
| Pressure difference | Tree-F200 | 0.0229555 | 0.0505045 | 0.0618671 | 0.0282187 |
| Pressure difference | Pair-F200 | 0.0173515 | 0.0348169 | 0.0458595 | 0.0220618 |
| Pressure difference | Dense200 | 0.0131414 | 0.0356868 | 0.0445602 | 0.017984 |
| Pressure difference | Dense1000 (longer) | 0.0084145 | 0.0137535 | 0.0183222 | 0.00971881 |

Tails are case RMSE tails, not pointwise uncertainty intervals. Pooled RMSE uses native sample counts and is reported separately from equal-case means. All retained all 22 ordinary summaries have zero nonfinite cases in every measured role. The detailed reference is the stored analytic/shared-grid benchmark; its q-normal channel is a proxy, and its port coefficient is an effective quantity. No independent SI calibration, continuum-flux conservation or CFD-design validity is asserted.

| M / cases | Arm / age | Fluid T | Material T | Fluid u | Pressure difference |
|---|---|---|---|---|---|
| 3 / 6 | Tree-F200 | 1.80348 | 1.83519 | 0.118297 | 0.0265641 |
| 3 / 6 | Pair-F200 | 2.72026 | 3.05459 | 0.103776 | 0.0139039 |
| 3 / 6 | Dense200 | 1.58593 | 0.768773 | 0.0614271 | 0.0138235 |
| 3 / 6 | Dense1000 (longer) | 0.86548 | 0.759006 | 0.0263441 | 0.00417502 |
| 5 / 6 | Tree-F200 | 2.27334 | 2.26389 | 0.126243 | 0.0316437 |
| 5 / 6 | Pair-F200 | 3.2341 | 3.8156 | 0.107865 | 0.0216111 |
| 5 / 6 | Dense200 | 1.6703 | 1.15994 | 0.0648228 | 0.0128823 |
| 5 / 6 | Dense1000 (longer) | 0.903519 | 0.885513 | 0.0276119 | 0.00786413 |
| 7 / 6 | Tree-F200 | 2.1558 | 2.34282 | 0.135483 | 0.0148785 |
| 7 / 6 | Pair-F200 | 3.24146 | 3.71631 | 0.105373 | 0.0234819 |
| 7 / 6 | Dense200 | 1.7392 | 1.58083 | 0.0710692 | 0.0182504 |
| 7 / 6 | Dense1000 (longer) | 1.02881 | 0.961996 | 0.028281 | 0.0113852 |
| 10 / 4 | Tree-F200 | 2.3894 | 2.89035 | 0.144761 | 0.0166255 |
| 10 / 4 | Pair-F200 | 3.06643 | 3.43029 | 0.093462 | 0.00693781 |
| 10 / 4 | Dense200 | 2.18964 | 2.18972 | 0.080736 | 0.00484361 |
| 10 / 4 | Dense1000 (longer) | 1.02812 | 0.851061 | 0.0331605 | 0.0111432 |

Per-M evidence remains limited to the selected 6/6/6/4 validation cases. Complete near/far, initial/final port, role, pooled and case statistics are preserved in each checkpoint's `evaluation/*/summary.json` and the compact `evaluation/milestone_readout.json`; these local evidence files are indexed below.

![Learning curves, true case visits, measured process time, paired wrapper latency and merged GPU allocation](../../../diagnostics/generated/tree_faithfulness_20261004/figures/01_learning_and_measured_exposure.png)

**Figure 1.** Candidate exact-age 200 fluidT RMSE is 2.13424/3.06548 for Tree/Pair; Dense1000 is 0.949971 at five times the candidate epoch count. Successful Tree training/validation takes 2.9694 h versus Pair 0.4978 h; merged main GPU allocation is 5.1410 h. Paired 200 Q8192 wrapper medians are 835.56/903.21 ms for dense/subset Tree versus 185.68 ms for Pair onM 3; no subset latency gain appears. Curves show actual selected-data exposure, not source-dataset epochs. Native all 22 metrics are separate from the normalized selector. Process time includes contention; merged allocation counts co-resident jobs once and includes censored/failed attempts. Active GPU compute time was not measured. The dense 1000 reference has a longer budget.

![Four fixed representative physical fields and residuals on native spatial grids](../../../diagnostics/generated/tree_faithfulness_20261004/figures/02_physical_fields_and_residuals_at_e200.png)

**Figure 2.** Exact 200 checkpoints are shown on the four input-selected cases; all 22 means are Tree/Pair fluidT 2.13424/3.06548 and materialT 2.28240/3.51092. The structured signed residuals remain visible, and Pair retains better velocity fidelity. Dense1000 has smaller temperature residuals but uses five times the candidate epoch budget. Reference, Tree-F and Pair-F temperature, signed temperature/velocity residuals and native disk-material residuals use shared masks and limits. The dense 1000 temperature residual is a longer-budget reference. All four cases—including difficult cases—are shown. Material spatial coordinates were checked against the packed disk mask and exact stored row order; module indices are not treated as physical positions.

## Organizer: representation consistency and actual utility

Tree-F's canonical coordinate/role index sums physical anchor masses, preserves every original fine atom and keeps coincident blocks together. Boundaries use mass-weighted child centroids; node content and structural occupancy integrate receiver/source measures. Query-role counting measure is declared separately. Physical module counting measure, environmental quadrature, active-frontier complexity and integer executed rows are distinct quantities.

Controls use membership-weighted module and environmental summaries, receiver geometry/role, prescribed context, phase and donor-presence flags. All-source embeddings remain planning inputs. The `source_local_v3` adapter uses each module's own heat with the fixed selected-training feature scale 1.7748545408248901; total/mean/maximum heat are absent from prescribed/background context. This does not remove ordinary Stage-A thermal physics. Both control donor types are exported, with P1/P2 transport/local-refinement ancestry disclosed. Learned computational support is not physical causality.

Two bounded representation remedies were required. The initial Tree-F3201 encountered an FP32 mass-cut near tie and stopped after two complete epochs plus a partial third. The deterministic cut repair uses canonical total-mass-scaled dtype tolerances; fresh 3204 starts at zero exposure. At trained 100, whole-wrapper outputs were stable but equal-tie fine-row min/max bounds still distributed coordinate subgradients by row count. Node extrema now use canonical blocks with mass-weighted coordinate pullback. This changes derivatives while preserving all four saved baseline forward fields/ports exactly at the same weights; legacy behavior stays separately replayable. With child masses `w_i = alpha_i * w_parent` at tied coordinates, a canonical centroid has child-coordinate derivative `alpha_i`. Summed child coordinate/feature gradients and coefficient-weighted mass gradients therefore pull back to the parent variation; raw row-count tie gradients do not preserve that convention.

The late supplemental Tree-L100 light panel reused four saved cases and measured the other 18: permutation passes 22/22; unequal refinement fails 22/22, with maximum temperature change 0.145009. Its 72 new wrapper calls were collected after the first decision. The legacy diagnostic explicitly injects child masses before encoding: parent/total mass and parent physical lengths are preserved, and all 22 ordinary-versus-annotated baselines are bit-identical. These failures are not caused by changed uniform quadrature. They test the complete index-conditioned encoder/planner/frontier/membership/control rebuild and do not isolate geometry alone. The ordinary historical wrapper's omission of explicit environmental masses remains a separate replay limitation. The original trained Tree-F failure is retained: all 22 permutation/30–70 refinement predictions and effective representations passed, but 12 split-coordinate tensor checks failed across four cases and three refinements; only 52/64 detailed tensor checks passed. The same-weight amendment made 64/64 pass, maximum coordinate discrepancy 1.04308e-7 and maximum tolerance ratio 0.03763. Its 40 necessary wrapper calls are charged. Pointwise FP32 equivalence uses `|delta| <= 2e-5 + 2e-5 * |reference|`; the original maximum physical difference 6.29425e-5 is not an absolute 2e-5 claim. Gradient tolerances are separately declared in saved receipts.

At200, all 22 physical/effective-representation permutation and 30/70 whole-rebuild checks pass; all four equal/multiple split panels and 64/64 first-gradient tensors pass. Maximum physical discrepancy across these variants is 7.72476e-05 under pointwise mixed absolute/relative tolerances; maximum first-gradient tolerance ratio is 0.11591. The complete watcher attempted 321 native wrappers:94 invariance, four locality and 223 bounded physical-path calls. Its raw keyword-query counter missed positional physical-path queries; source-verified primary query rows are 321×14=4,494, while P0/P1/P2 auxiliary roles remain additional. Preserved raw counts and the explicit correction prevent 1372 from being presented as the total.

![Actual atoms, rebuilt boundaries, physical refinement errors and correctly pulled-back derivative checks](../../../diagnostics/generated/tree_faithfulness_20261004/figures/03_equivalent_representation_and_derivatives.png)

**Figure 3.** The final faithful physical/representation panel passes 22/22; the late legacy unequal-refinement panel fails 22/22, maximum native temperature change 0.145009. Final detailed derivatives pass 64/64 after the mass-consistent pullback repair, while the original 10052/64 result is retained. The native spatial atom plots and rebuilt boundaries come from the saved actual diagnostic arrays. Exact duplicate refinement preserves coordinates/features/parent length and explicitly partitions physical mass. It is different from displacing a physical source. Genuine changed-input tree switches are reported separately; they are not used to loosen source-equivalence tolerance. Tree-L's trained legacy split discrepancy remains a compound-control result rather than being erased by the faithful path.

The final saved actual donor inventory contains 96 excluded module-donor/group entries onQE across the four representatives, while E control donors remain full. The original selectedP 0/QM derivative probes find no exclusion and remain vacuous for a zero-path test. A bounded late supplement selects actualQE exclusions on0291P0,0294P0 and 0687P1. With realized planning, membership, control geometry and source identities frozen, all five actually excludedM donors have exactly zero content-to-selected-control Jacobians, all 17 admittedM and 576 admittedE donors have nonzero Jacobians. Excluded-source planner-path gradient norms are 0.00437214/0.00250922/0.000736231. The supplement makes threeQ 14 wrappers/42 primary queries and 48 control-vector plus three planner VJPs. This verifies conditional latent-content locality; ordinary rebuilt planning and multi-hop P1 ancestry still carry global information. All 952,700/952,700 native eligible value pairs across all typed routes and auxiliary material/port paths in these three Q14-primary wrappers remain admitted, so conditional control exclusion is not physical value pruning.

The ordinary 200 utility panel reuses four saved normal arrays and makes 20 intervention wrappers. Learned-control reliance changes sign from 100: fluidT normal/identity is 2.02649/3.56504 (+75.9% error), material 2.17589/3.19369 (+46.8%), surface 2.41249/3.69245 (+53.1%). Geometry-reference action replacement preserves receiver weight multisets and source cardinalities and changes fluidT only 0.119%; rewiring raises it7.26%. These interventions distinguish trained reliance from evidence for unique collective geometry. No binary pair count or dense-executor fine row changes in any of the five controls.

| Fixed-weight intervention | Normal T RMSE | Intervention T RMSE | Change | Fine-row delta (sum) |
|---|---|---|---|---|
| control_identity_fixed_access | 2.02649 | 3.56504 | +75.922% | 0 |
| full_access_fixed_controls | 2.02649 | 2.10097 | +3.675% | 0 |
| geometry_reference_actions | 2.02649 | 2.02891 | +0.119% | 0 |
| effective_rewire | 2.02649 | 2.17355 | +7.257% | 0 |
| root_union | 2.02649 | 2.09633 | +3.446% | 0 |

Final native Q14 field-loss probes on0277/0687 give Pair 20/20 direct-control tensors finite/nonzero (L2.5951/2.4905). Tree hard gain-score controls are 12/12 finite/nonzero (L2.5880/1.0646), organizer parameters 53/55 (L2.1016/.5678); the two discrete planner tensors are unused by this hard path. The official hard-value/soft-organizer route gives 55/55 finite/nonzero (L2.2038/1.1903) at the same hard field losses. Persistent weights/buffers and flags are unchanged, parameter.grad remains unset and no optimizer updates occur. These are measured trained task VJPs, not per-epoch gradient logs.

![Actual value and dual control donor support, all-source planning dependencies, phase ancestry and same-weight utility controls](../../../diagnostics/generated/tree_faithfulness_20261004/figures/04_value_control_paths_and_utility.png)

**Figure 4.** The final saved P2 graph identifies native value support, both control donor types and separately shown global planning ancestry. Four-case mean fluidT rises 2.02649→3.56504 when controls become identity; geometry-matched actions give 2.02891. All five interventions retain native binary pairs and dense fine rows. The stronger independently trained Pair comparison remains the all 22 table above, rather than being replaced by an ablation. The graph and support counts come from measured phase archives. Allocated capacity, active frontier, source-bearing groups, nonredundant actions, receiver participation and native unique pairs are reported separately. The utility interventions use the ordinary dense executor; their zero fine-row changes do not imply the subset executor cannot pack invalid columns. A nonzero task Jacobian or large ablation response cannot replace the separately trained Pair-F comparison.

### Physical executor work and latency

| Case / Q8192 | Executor | Wrapper median ms | Prepared-P2 median ms | Physical fine rows | Physical MLP calls | Incremental CUDA peak MiB |
|---|---|---:|---:|---:|---:|---:|
| 0277 | Pair-F | 185.68 | 135.97 | 1,998,768 | 161 | 81.58 |
| 0277 | Tree-F dense-mask | 835.56 | 614.26 | 1,998,768 | 161 | 45.27 |
| 0277 | Tree-F subset | 903.21 | 640.76 | 1,905,708 | 164 | 45.71 |
| 0687 | Pair-F | 179.50 | 133.14 | 1,998,768 | 161 | 81.58 |
| 0687 | Tree-F dense-mask | 872.35 | 618.26 | 1,998,768 | 161 | 58.98 |
| 0687 | Tree-F subset | 915.39 | 642.01 | 1,978,088 | 164 | 58.99 |

At200, the full-grid subset removes 93,060/20,680 actual rows onM 3/M10:4.6559%/1.0346%. Exact rectangle partition reconstruction attributes every saving to inactive source columns, with zero closure of physically eligible source-value pairs. Inactive receiver rows 5,265/1,212 and MM self rows 9/30 still execute. Subset packing increases physical MLP calls 161→164 and is 8.10%/4.93% slower than dense-mask Tree in median complete-wrapper latency. Tree dense-mask is 4.50/4.86 times Pair latency in these two cases. SmallQ 14 complete-wrapper medians are Pair 49.92/49.49 ms, Tree dense 249.84/275.27 ms, subset 262.23/288.58 ms. All four dense/subset output-and-first-gradient parity panels pass. These are physical forward rows/calls, not GPU hardware kernel counts or unique eligible pairs. The full wrapper includes P0/P1/P2 and native material/port work beyond the nominal fluid Q. Prepared-P2 decode, encoder/preparation, nested organizer, physical MLP calls and memory have separate recorded scopes. Instrumented inclusive timings are not additive; timing repeats are separate from diagnostic hook forwards. All weights were resident during paired comparison, so incremental allocated memory is meaningful while exclusive active-model memory is unavailable. Dense/subset field and first-gradient parity use declared native tensor-space/channel-scaled tolerances; normalized internal tensors are not called uniformly physical units.

## Response transfer: known-null misses and missing positive labels

The benchmark generator constructs analytic flow before heat. Four stored historical controls verify exact zero u/v/p/omega differences with nonzero temperature differences 1.19294–2.38589. The excluded 0001 family is read-only physics diagnosis, not a selected training/validation population. This contract is specific to the benchmark; it is not a general claim for buoyant thermal flow.

The 172 selected cases form 171 exact input geometry/prescribed-context families. The sole duplicate 0133/0493 has identical heat; there are zero held different-heat reference families. Only 0348 intersects the training atlas. **C-positive-response transfer remains unavailable.** Thermal surrogate sensitivity is not response accuracy.

The all 22 null panel reuses 172 exactly identified feasible transfers: two input-selected donor pairs, fractions 0.1/0.2 and both signs, except ineligible variants counted explicitly. Geometry, context, identities and total heat remain fixed. Each arm/age uses 194 native wrappers, Q256 fluid plus native material/ports and 32 rows per pressure section. This pressure functional differs from the ordinary section-mean error and inverse two-point diagnostic.

| Arm / saved age | u null RMS | v null RMS | p null RMS | omega null RMS | T sensitivity (model-only) | Section-pressure |increment| | |---|---|---|---|---|---|---| | Tree-L100 (late) | 0.000365058 | 6.07009e-05 | 0.000344359 | 0.00113372 | 0.00995782 | 0.000121511 | | Tree-F100 | 0.000458066 | 0.000136249 | 0.000370412 | 0.00263115 | 0.0182384 | 0.000269291 | | Pair-F100 | 0.000271811 | 5.32454e-05 | 0.000191219 | 0.000961373 | 0.0138148 | 0.000109791 | | Tree-F200 | 0.00244782 | 0.00014518 | 0.000770427 | 0.00295796 | 0.0556479 | 0.000165412 | | Pair-F200 | 0.00164483 | 8.06074e-05 | 0.000333945 | 0.00199777 | 0.038129 | 9.1225e-05 | | Dense1000 (late, longer) | 0.00121311 | 0.000103348 | 0.000585874 | 0.0040203 | 0.0909105 | 0.000166227 |

Tree-F100→200 known-null u/v/p/omega mean increments worsen×5.344/1.066/2.080/1.124; Pair worsens×6.051/1.514/1.746/2.078. At200, Tree exceeds Pair by×1.488/1.801/2.307/1.481. The section-pressure absolute increment separately improves 38.6% in Tree and 16.9% in Pair; it does not erase the pointwise flow misses. Tree 200 fluid/interface/material thermal sensitivities are 0.05565/0.09108/0.12886 and Pair 0.03813/0.07822/0.12240, all model-only.

The late Tree-L100 and Dense1000 controls each use exactly the same 172 input-defined variants and 194 wrappers, collected after the first decision. Their feature mode is padding_invariant_v2, fixed_heat_scale=None and neither has null training; Dense has the longer 1000 budget. Its excellent ordinary field fit does not enforce the known heat-null contract. These controls are not relabelled as matched source_local_v3 response-objective arms.

Both successful 101–200 candidates retain fixed null coefficient 0.1. Tree records 800 charged null wrappers including hard/soft-organizer shadows,102,400 physical before/after fluid queries and 309,920 requested role queries,256.444 s; Pair 400 wrappers with the same requested query counts,52.030 s. Requested query counters exclude shadow replay passes, whose physical work is separately recorded. Tree calibration native/null gradient norms 8.44570/2.92701e-4 give weighted ratio 3.46568e-6 (0.0003466%), and its weighted scalar mean 2.64065e-5 has mean ratio 4.84446e-5 to native total loss.  Pair calibration native/null gradient norms are 2.76271/1.328997e-4, so weighted null/native ratio 4.81049e-6 (0.000481%). Its mean weighted null scalar is 5.72738e-6 and mean scalar ratio to native total loss 1.10992e-5. This is a weak force at one measured calibration point; there are no later gradient norms/cosines, and scalar ratios do not establish later force. The null term runs at the last accumulation boundary, not every ordinary optimizer update. No matched no-null continuation was trained, so worsening is not proof that the auxiliary term caused it. No coefficient tuning or sweep followed.

![Matched 100-to-final all 22 heat-null increments, case distributions, model-only thermal sensitivity and an actual fixed-total heat direction](../../../diagnostics/generated/tree_faithfulness_20261004/figures/05_bounded_heat_responses_at_e200.png)

**Figure 5.** All 22 known-null increments are larger at 200 than 100 in both candidates in all four flow channels. The final u increment is 0.00244782 for Tree and 0.00164483 for Pair. Late legacy 100 and dense 1000 controls are visibly labelled with different ages/features/objectives. Nonzero thermal responses avoid an all-heat-insensitive collapse but do not measure positive physical response fidelity. Zero reference increments are scaled by fixed selected-training channel scales, never by a zero denominator. Hatched bars retain 100 and solid bars show the final matched age. Thermal points show a nonzero ordinary learned response without asserting perturbed-temperature accuracy. The saved 0277 direction preserves total heat 2.59413 with|delta|0.0291021 in native heat units.

A prepared, unexecuted follow-on request contains the four fixed representatives, each a new aligned baseline and two opposite feasible fixed-total transfers: at most 12 benchmark solves. It specifies exact slots, amplitudes and packed-FP32 receiver alignment. It requires separate authorization; none was executed here. Longer selected-data fitting cannot by itself supply these missing physical labels.

## Inverse: completed surrogate reuse and its limits

All 32 fresh trails complete 320 attempted updates at the saved 200 checkpoints; zero new solves and no failed/reused launch. Tree 24 trails use 715 trajectory forwards/240 VJPs, Pair 8 use 208/80. Shared uniform-rank probes use eight forwards/48 Jacobian VJPs, charged once:931 forwards+368 VJPs=1,299 calls in this inverse scope. Model weights and persistent buffers are bitwise unchanged, no parameter.grad or training update; prepared caches are outside that certificate.

The 100 saved-only preflight found no proper module graph block. At200 the learnedQE control memberships provide 50/80 subfull graph proposals, with 30/80 full-joint fallbacks; randomized controls preserve those exact per-attempt cardinalities and also have 30 fallbacks. These are unions of exposed phase-current module value/control donors. Full environmental summaries, global planning and P1 ancestry remain disclosed, so a selected module block is not a physically isolated subsystem and does not reduce the complete forward executor. The proposal selector never uses hidden heat.

Tree joint/graph/random mean held RMSE is 2.40320/2.42160/2.45457 after fitting, from the common 2.49510 start mean. Graph performs no better than joint in this bounded comparison, with equal 319 trajectory calls; random reaches lower observed error but worse held error. Pair improves every observed trail, yet held errors improve only 4/8 and worsen 4/8. Tree joint and graph improve held errors 3/8, worsen 2/8 and leave 3 unchanged; size-random improves 2/8, worsens 3/8 and leaves 3 unchanged. The cases/starts below retain misses instead of hiding them in a mean.

All saved allocations are nonnegative and fixed-total within maximum Float 64-summed deviation 2.14577e-6 from the public task total (the separate start-relative drift is 1.25170e-6); maximum saved per-attempt fraction change is below 0.05000003, differing from 0.05 only by roundoff. Nevertheless Tree 15/24 and Pair 8/8 trails (23/32) have at least one accepted/repeated saved endpoint outside the selected-training native heat range[0.5036706,1.9991537]. Some input-only interior starts already lie outside that range. Unaccepted trial heat arrays are not persisted, so that count covers saved endpoints, not every proposed design. Radius/total feasibility is not a support-membership or design-validity certificate. Hidden original heat is read only after optimization for the discrepancy below.

There are 571 trial evaluations, including 362 rejected Tree fixed-topology active-set continuations with negative/nonfinite membership,114 valid observed-nonimproving proposals (Tree 42/Pair 72), and 95 improving accepted rebuilds. No frozen-improving candidate fails the ordinary rebuild. All 225 rejected attempts retain the anchor because neither permitted trial is valid and improving. The 95 accepted steps pass ordinary rebuild checks; ten accepted rebuilds change 164 discrete entries (joint 3, graph 0, size-random 7). The strict native continuation policy is retained. In particular, Tree 0294 uniform and 0687 interior joint/graph starts do not improve within the bounded attempts. This is a completed miss, not an unfinished optimization. The observed GPU 2 inverse process interval 08:49:01–08:53:15UTC is 254.374 s (0.0706595 allocated process hours including load/I/O); it is separate from main training allocation and is not active GPU compute time.

The interface uses six input-defined observed temperatures and six disjoint held temperatures, snapped to distinct native grid rows; two pressure points are diagnostics only. Starts are uniform and seed 0 interior simplex, both input-only. Each attempt has a maximum 0.05 change in any module's allocation fraction, at most two trial radii and fixed-total nonnegative heat. This is a local bound per attempt, not a cumulative heat-change bound. Proposals freeze declared discrete P0/P1/P2 topology while recomputing continuous physics. An ordinarily rebuilt trial must also improve observed MSE before acceptance; held residuals never select steps. Rejected refreshes, failed topology trials, forwards and VJPs are charged. Model parameters/buffers and training flags are restored and no forward optimizer is updated.

| Mode /8 trails | Observed RMSE mean | Held RMSE mean | Hidden heat RMSE (post-hoc) | Held better/worse/tie | Accepted/rejected | Full fallback /80 | Trajectory Fwd+VJP | Outside heat range |
|---|---|---|---|---|---|---|---|---|
| Tree joint | 2.5834 → 2.2068 | 2.4951 → 2.4032 | 0.3441 → 0.3962 | 3/2/3 | 14/66 | 0 | 319 | 5 |
| Tree graph | 2.5834 → 2.2277 | 2.4951 → 2.4216 | 0.3441 → 0.4009 | 3/2/3 | 12/68 | 30 | 319 | 5 |
| Tree size-random | 2.5834 → 2.0941 | 2.4951 → 2.4546 | 0.3441 → 0.4233 | 2/3/3 | 21/59 | 30 | 317 | 5 |
| Pair joint | 2.9926 → 1.8950 | 2.6829 → 2.5331 | 0.3441 → 0.7459 | 4/4/0 | 48/32 | 0 | 288 | 8 |

Temperature residuals use native benchmark temperature units; hidden heat discrepancy uses native heat units and only active modules. Means are over the same four cases/two starts; the three Tree modes are correlated, not 24 independent cases. Trajectory calls exclude the separately charged shared rank probes.

| Case | Start | Tree joint obs/held | Tree graph obs/held | Tree size-random obs/held | Pair joint obs/held |
|---|---|---|---|---|---|
| 0277 | 0 (uniform) | 1.5631 / 1.8239 | 1.5631 / 1.8239 | 1.5631 / 1.8239 | 1.5257 / 1.6769 |
| 0277 | 1 (interior) | 1.5797 / 1.8121 | 1.5797 / 1.8121 | 1.5797 / 1.8121 | 1.5253 / 1.6752 |
| 0291 | 0 (uniform) | 1.2983 / 1.9347 | 1.3152 / 1.9926 | 1.2484 / 2.2198 | 1.2982 / 2.5134 |
| 0291 | 1 (interior) | 1.3133 / 1.9747 | 1.3133 / 1.9747 | 1.3133 / 1.9747 | 1.2662 / 2.3315 |
| 0294 | 0 (uniform) | 2.0403 / 2.4523 | 2.0403 / 2.4523 | 2.0403 / 2.4523 | 0.9748 / 2.7820 |
| 0294 | 1 (interior) | 1.3210 / 2.2358 | 1.4780 / 2.2540 | 0.8079 / 2.4085 | 1.1342 / 2.5541 |
| 0687 | 0 (uniform) | 3.2806 / 2.9570 | 3.2736 / 3.0282 | 2.9422 / 2.9103 | 3.2192 / 3.2108 |
| 0687 | 1 (interior) | 5.2582 / 4.0350 | 5.2582 / 4.0350 | 5.2582 / 4.0350 | 4.2162 / 3.5209 |

The uniform-point observation Jacobian gives:

| Case | Fixed-total free dimensions | Tree rank | Pair rank | Tree condition | Pair condition |
|---|---|---|---|---|---|
| 0277 | 2 | 2 | 2 | 2.056 | 2.852 |
| 0291 | 4 | 4 | 4 | 12.67 | 15.08 |
| 0294 | 6 | 6 | 6 | 141.6 | 122.5 |
| 0687 | 9 | 6 | 6 | not defined | not defined |

Rank thresholds are saved per arm/case. These are local diagnostics at the uniform probe, not global uniqueness certificates for all trajectories.

![All completed inverse trails versus charged native calls, actual heat allocations, held misses, rank limits and full-joint fallback](../../../diagnostics/generated/tree_faithfulness_20261004/figures/06_bounded_inverse_reuse.png)

**Figure 6.** All 32 completed trails are shown against charged trajectory calls. Tree graph has 50 proper-block attempts/30 full fallbacks and no held-fit advantage over joint; Pair held errors worsen in four of eight trails. Native-coordinate heat allocations are surrogate outputs;23/32 trails leave selected-training heat support. Shared rank work is charged once separately, andM 10 measured rank 6 remains below nine free directions. These are surrogate heat allocations fitted to stored observations at the original design. Candidate physical reference fields are unavailable. Hidden original heat is used only for post-hoc error reporting, not proposal selection. Fixed-total free dimensions are 2/4/6/9; atM 10 six temperatures can have rank at most 6 below nine free directions, regardless of optimization success. Reported sample/design differences are not independently validated designs or generative diversity.

## Contention, revisions and charged work

Initial candidate forecasts were based on first five actual epochs, not an assumed fourfold quarter-data speedup. Tree-L completed onGPU 1; first-screen Tree-F sharedGPU 2 with Dense. Pair-F used availableGPU 1. The e100 measured process allocations were 0.827/1.647/0.227 hours for Tree-L/Tree-F/Pair-F. The reviewed first paired continuation placed Tree-F beside an untouched external 13.75GiB GPU 1 process and Pair beside Dense onGPU 2. The initial censored 3201 additionally consumed 300 complete visits/eight updates/307,200 primary queries, plus 104–112 partial-third-epoch visits and two completed update boundaries; in-flight exposure is bracketed rather than invented.

The first resumed attempt exposed seven absent null-loss CSV columns. Pair's epoch 101 finished 150 visits/four updates/153,600 primary queries plus four null wrappers/1,024 fluid/3,904 total requested role queries, but no checkpoint was saved. Tree stopped during its 11th microbatch after 80–88 visits/one completed update/81,920–90,112 fluid queries, before its null callback. Both resumed their intact 100 states after a lossless schema extension: old cell order/values are preserved, new historical cells blank, original CSV backed up and atomically replaced after native resume validation. Raw telemetry is keyed by attempt/PID, since repeated 101 records must not be summed as accepted age.

The second Tree-F GPU 1 attempt completed unsaved 101–103 (450 visits/12 updates/460,800 primary queries) with train times 65.07/135.31/391.09 seconds and 24 null wrappers including soft shadows. Partial 104 had 128–136 visits/two completed update boundaries. This observed contention was not causally isolated. Those unsaved weights were not recoverable: CSV/log/native telemetry were archived, working CSV restored to 100, and Tree-F restarted from its 100 checkpoint onGPU 2 only after both Dense1000 and Pair 200 exited. The successful GPU 2 first five epochs averaged 50.19 train-plus-validation seconds, revising the then remaining forecast to 76 minutes. No external job or GPU 0 was touched.

The e100 paired benchmark paused only our exact verified Dense process for 109.4905 seconds, with independent watchdog/finally resume; native epoch wall times retain that pause. Its continuous allocation already includes the pause and is not counted a second time. All four main trainers have exited 0. Accepted exposure is 225,000 visits/6,000 ordinary updates/230.4M primary fluid queries. Main-process allocation sums 7.318136 h; merged assigned GPU-slot allocation is 5.141042 h (GPU 1 1.120557 h, GPU 2 4.020486 h), including failed/censored work and the dense pause. Successful final Tree 101–200 train-plus-validation time is 1.328116 h; Pair is 0.276267 h. The saved `final_work_evidence_index.{json,md}` reconciles 15 ordinary all 22 panels/330 wrappers; six known-null panels/1,164 wrappers;40 new utility wrappers with cached normals reused; finalB 324 attempted wrappers/4,536 source-verified primary queries; and final paired execution 136 wrappers/120 prepared decodes. Accepted validation contributes 33,000 case visits/33.792M primary queries. Separate successful train-atlas callbacks record 2,400 wrapper calls/9,778,340 requested role queries; heat-null callbacks 1,200 wrappers including shadows/619,840 requested physical-role queries. These counters have different scopes and are not combined into a fictitious work total. Dense training fine rows, validation internal execution, activation-checkpoint backward recomputation, hardware kernels, calibration VJP totals and active compute hours remain unmeasured. Censored attempts add 900 complete visits/24 updates/921,600 primary queries plus the three explicitly bracketed partial fragments. Active GPU compute hours remain unmeasured. Completed, discarded, startup, calibration, soft-shadow, atlas, null, ordinary evaluation, representation, utility, execution and inverse work are separate in saved receipts; no launch is relabelled completion.

## Evidence index, validation and next decision

All numerical evidence is below `/data/wanglz/ModularDT/thermal_development/tree_faithfulness_20261004/`. Main entries:

| Evidence | Local path under campaign root |
|---|---|
| Actual decisions | `e100_review.json`, archived original review, `e200_review.json` |
| All 22 roles/per-M/tails/selection | `evaluation/milestone_readout.json`; each `evaluation/*/summary.json` |
| Checkpoint/runtime binding | `dense_1804_architecture_binding.json`, `profiles_e200/preparation_receipt.json` |
| Rebuilt representation/derivatives/locality | `e100/tree-f/representation/`, `e100/tree-f/derivative_amendment/`, `e200/tree-f/representation/` |
| Trained utility | `evaluation/organizer_utility_e100/`, `evaluation/organizer_utility_e200/` |
| Native executor measurements | `e100/paired_execution_gpu2/`, `e200/paired_execution_gpu2/` |
| Null responses and force | `evaluation/*_null/`, `pair_f_e200_null_force_review.json`, `final_e200_null_force_and_work_review.json`, final response receipts |
| Inverse complete trails/rank/work | `evaluation/tree_f_e200_inverse/`, `evaluation/pair_f_e200_inverse/`, `final_e200_inverse_receipt.json` |
| Allocation and discarded work | `allocation_intervals.json`, `failed_e101_logging_attempt1/`, `gpu1_contention_attempt2/` |
| Physical-response boundary | `response_physics_audit/receipt.json`, `follow_on_reference_request_fixed4.json` |

The original UpgradePlan source remains locally ignored under the existing repository rule; the tracked development guide carries the revised 500-epoch budget and review cadence. Generated arrays/checkpoints and one-time renderers remain local and ignored. Exactly six selected figure masters are retained as PDF with small PNG companions required for inline Markdown and source-only JSON manifests; superseded presentation exports were removed after visual validation. All images below resolve in this checkout. They are deliberately absent from a fresh remote checkout under the repository upload rule.

| Selected main figure | PDF master |
|---|---|
|1 Learning and work |[PDF](../../../diagnostics/generated/tree_faithfulness_20261004/figures/01_learning_and_measured_exposure.pdf) |
|2 Physical fields |[PDF](../../../diagnostics/generated/tree_faithfulness_20261004/figures/02_physical_fields_and_residuals_at_e200.pdf) |
|3 Equivalent representation |[PDF](../../../diagnostics/generated/tree_faithfulness_20261004/figures/03_equivalent_representation_and_derivatives.pdf) |
|4 Paths and utility |[PDF](../../../diagnostics/generated/tree_faithfulness_20261004/figures/04_value_control_paths_and_utility.pdf) |
|5 Heat responses |[PDF](../../../diagnostics/generated/tree_faithfulness_20261004/figures/05_bounded_heat_responses_at_e200.pdf) |
|6 Inverse reuse |[PDF](../../../diagnostics/generated/tree_faithfulness_20261004/figures/06_bounded_inverse_reuse.pdf) |

The main integration suite passed 325 tests with two optional native-resource skips. After the derivative/policy/resume changes, affected suites passed 152 tests with one resource skip; the logging/schema guard round passed 53. These suites overlap and their counts are not additive. Native startup exercises actual low/high-M Adam updates, same physical initialization and save/reload with restored optimizer; generic 3-D compatibility and the FP64 soft-permission quotient are retained. Final trained task/control probes are reported as measured inference/gradient evidence, not as logged training norms: disabled per-epoch norm columns contain NaN placeholders. All six selected figures were visually inspected against saved evidence and their source manifests; the CommonMark renderer resolves six embedded images and the six PDF index links, with no remaining placeholders or missing local targets. Predictor QA independently matches 224 displayed numeric table cells and percentage claims; Organizer/Response/Inverse scope checks are recorded with the local evidence. No additional model pass was used for presentation.

Predictor next work should preserve the measured thermal/velocity tradeoff and the strong selected-data dense reference. Organizer next work should require an action effect and actual eligible-row reduction before a speed or sparse-design claim. Response next work needs training-only force calibration with a new declared matched protocol and authorized positive reference labels; the prepared 12-solve request remains unexecuted. Inverse reuse should retain rank, held-error and physical-reference limits rather than treat optimizer success as design validity. The durable source/tests/configuration/guide/report were committed and pushed on the non-default branch after auditing the whole outgoing commit history, running the existing artifact gate and verifying remote/local tips. Numerical outputs, figures and one-time renderers were not uploaded.
