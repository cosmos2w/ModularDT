# HONF Run 1506: continuous functional coalescence

Run 1506 is the single fresh `1503-v4` candidate following the [continuous functional coalescence plan](../../UpgradePlan/HONF_Run1503_v4_Continuous_Functional_Coalescence_Plan.md). The run uses the same 600 training and 90 development cases, seed, K=12, D=16, fine physical reader, predicted-port curriculum, optimizer, losses, and normalizers as Run 1502. The dataset's `test` split supplies sampled validation and the full-grid evaluations here; these are **development**, not independent test results. One seed does not estimate run-to-run variation. Checkpoints, numerical tables, figures, and one-time diagnostic runners remain in local ignored directories.

## What changed

The opt-in `continuous_functional_coalescence_honf` architecture retains the Run 1502 provisional source memberships and fine module/environment interactions. In each P0/P1/P2 prepared phase, a fixed eleven-node binary tree acts on the twelve original finite query-access logits. For node C, `M_C = s_C I + (1-s_C) P_C`, where `P_C` averages access functions in the subtree. The bottom-up product `T` transforms parent logits continuously. Outside the transition interval `T=I`; at `s_C=0`, the descendant rows are mathematically identical for every query, so an integer quotient packs them without introducing a new prediction. The tree limits available mergers to its subtrees; it is not equivalent to the prior unrestricted convex-clustering objective.

The contraction score compares a hypothetical tied route with the parent using separate environment, physical-port, and phase-current outside-temperature probes and four source-action Gram blocks. The local source-action RMS taper closes below 0.02, retains the parent above 0.06, and has endpoint-flat transitions between them. The combined post-transform discrepancy is recorded separately. The score is a finite-probe read-input diagnostic, not a uniform field-error bound. At epoch 1–50 the backend calls the parent exactly; the scales ramp smoothly over epochs 51–150 and remain fixed over 151–500.

Each packed class stores its proposal multiplicity. Query sparsemax preserves virtual proposal density and sums it into compact mass; source incidence sums over leaves, while the source-resolved control moment sums `A_sk t_k` before fine source modulation. Thus compact and virtual executions represent the same **constructed** access functions and latent source-control action. This algebraic equivalence does not establish CFD conservation or physical-field accuracy. The rectangular executor remains the reference. The v4 path avoids v3's iterative ADMM planner and its GPU row-identity synchronizations, without changing historical v2/v3 checkpoints or security behavior.

## Structural prior and provenance

The tree was formed once from the exact Run 1505 e500 checkpoint's **raw pre-coalescence** access functions on the 600 training IDs only. Three prepared phases per ID yielded 1,800 records; every pair had 1,790–1,800 eligible observations. The pair score used the same four-block source-action metric as the live model. Deterministic complete linkage produced the eleven nested subsets stored in the Run 1506 profile and checkpoint. Pair-score minimum/10th percentile/median/90th percentile/maximum were 0.203206/0.492515/0.728923/1.219890/1.319433. They all exceed the live 0.06 keep scale; formation after fresh training was therefore uncertain. No development-case routing image selected the tree, and no Run 1505 neural or optimizer weights initialized Run 1506. The local extraction evidence is under `diagnostics/generated/run1503_v4_tree_train_only/`; its one-off builder and raw case data are excluded from GitHub.

A retrospective check of Run 1505's exact e500 **development** formation CSV found 429 accepted non-singleton classes across 270 case-phase records: proposal pairs `[0,10]` occurred 209 times, `[1,8]` 187 times, and `[6,7]` 33 times. All three are nodes of the previously fixed train-built tree. This checks coverage of those old recurring pair *types*; it neither selected the tree from development cases nor says fresh Run 1506 weights make those functions redundant. The old v3 solver's accepted mergers and the v4 read-input score use different criteria.

## Read-only diagnosis of retained Run 1505

The five-case same-weight P0/P1/P2 parent-access interventions were performed on Run 1505's distinct e500 and e473 checkpoints. Forcing parent access only at P2 changed the full field but left raw and final ports, P1 outside temperature, and interface/internal outputs bitwise unchanged in these cases; the separately requested P2 port-global consistency read can change. P0/P1 interventions changed downstream fields and ports, but the five-case effects were small and mixed. These are dependencies within a fixed trained model, not an explanation of the difference between independently trained Run 1502 and Run 1505. The local record is `diagnostics/generated/run1503_v4_port_audit/panel/`.

At the known feasible case-0641 e500 geometry bracket, x=2.5641326904–2.5642242432, forced opposite partitions at the **same physical coordinate** changed the normalized fluid field by about 3.05% relative L2 while P0/P1 were fixed. Source aggregates and final field access changed; this confirms a v3 hard-switch mechanism, not a v4 boundary claim. The Run 1505 natural Q8192 full application median was 574.39 ms over five synchronized repetitions; a fixed-plan replay median was 333.56 ms with output parity. The latter is only a same-input diagnostic lower bound, not a deployable speedup or a measured v4 result. The local decomposition is `diagnostics/generated/run1503_v4_boundary_cost/boundary_cost_report.json`.

## Implementation and preflight

The maintained training/configuration dispatch accepts the new architecture and strictly serializes the tree. Phase probes are built from physical port coordinates and phase-current outside-temperature locations, with environmental coordinates and measures from the encoded case. The source-moment quotient remains differentiable; physical source coordinates and source-local environmental value modulation remain unchanged. A mixed B48 M1/M2 and an M10-only B2 ordinary predicted-port optimizer step on physical GPU 2 had finite loss, gradients, optimizer updates, and nonzero gradients through the physical port head and backend. The mixed step took 1.0464 s and peaked at 6.296 GB allocated; the M10 step took 0.5039 s and peaked at 1.051 GB. These are bounded preflights, not epoch or application benchmarks.

Focused tests cover tree permutation and packing, multiplicity-correct routing, compact/virtual output and first-gradient parity, source moments, phase-current probe dispatch, exact e1–50 parent behavior, and controlled closure/outer-identity boundaries. The combined focused suite passed 65 tests in 7.01 s with Ruff and config dry-run checks. Controlled boundary tests establish the **implemented taper algebra** under injected scores; real trained physical-design continuity requires the later sweep.

## Managed run and epoch-50 review

The one fresh run is `Run_1506_20260924_192303_1503_v4_continuous_functional_coalescence` on physical `cuda:2`, UUID `e80beaae-878d-4680-9847-21483688dd79`, with source commit `946ccbe81571d6c51c8293d1c1a9424a593da6fa`. It began from the Run 1502 seed and settings, not a model checkpoint warm start. The managed e50 stage completed normally and saved the exact `epoch_0050_model.pt` plus optimizer state. All 50 train/validation loss and field/temperature series are finite and match the corresponding Run 1502 series **exactly**; all 280 shared model tensors and 453 optimizer tensors match bitwise. Missing-sample `nan` entries in the optional gradient logger have the same pattern as Run 1502; recorded e50 gradient/update values are finite.

At e50, train total loss was 0.799973, sampled validation total loss 0.698150, field MSE 0.384350, and temperature MSE 0.211413. The full 90-case Q8192 predicted-port evaluator compared the exact Run 1506 and Run 1502 e50 checkpoints on ordered, identically normalized cases. All 6,390 paired values across 71 recorded physical metrics were exactly equal. The Q1024 P0/P1/P2 formation survey found all twelve proposals occupied, R=12, and zero transition or closed nodes in all 270 case-phase records, as scheduled. At full Q8192 P2, executed padded width was twelve. Lower effective K reflects sparse query participation, not functional class removal. The Q1024 compact Kq/Keff means were 2.735/1.997 (P0), 2.587/1.948 (P1), and 2.947/2.157 (P2); the Q8192 P2 means were 2.956/2.166.

The e1–50 median logged train epoch time was 8.56 s versus 8.39 s for historical Run 1502, consistent with the exact parent dispatch and ordinary run-to-run timing noise. During the active e51–102 ramp, the corresponding medians were 11.02 s versus 8.52 s. These are logged training epoch costs, not a synchronized paired inference benchmark; the new score/probe work does carry a material training overhead.

One e50 **phase-only** diagnostic issue was found: local interface predictions had been denormalized but their target had not. The first generated P0/P1 surface-temperature and heat-flux error columns were explicitly invalidated; outside-temperature/port-token metrics and all maintained Q8192 physical metrics were unaffected. A corrected all-90 Q1024 phase-only rerun under `diagnostics/generated/run1503_v4_checkpoint_review/e50_phase_corrected_20260925T0015/` restored those columns in physical units. Its P0/P1 mean surface-temperature RMSE was 5.1509/3.1358 and mean heat-flux RMSE was 10.3869/7.4190. This correction does not affect the exact parent-tracking result.

## Epoch-150 review

The managed e150 stage completed normally. Its exact checkpoint records epoch and selection 150, finite values in all 280 floating model tensors and 453 floating optimizer tensors, and optimizer state. Sampled validation field/temperature MSE was 0.090705/0.036930, versus Run 1502's exact e150 0.067827/0.033932. The single stage's best-total, best-field, best-temperature, and best-predicted files all select **e148** and contain identical model tensors, distinct from the e150 endpoint. This selection difference matters more than a rounded e150 scalar.

The exact e150 checkpoint's Q1024 all-90 survey found **zero closed nodes and zero transition nodes** in all 270 P0/P1/P2 records. P1 and P2 had R=12 in every case; P0 had R=12 in 89 cases and R=11 in one case because a proposal was unoccupied, not because functions merged. The smallest of 2,970 node scores was 0.259 RMS, above the 0.06 keep threshold; the combined post-transform score was zero because all T matrices stayed at identity. P2 Q8192 compact/virtual query support both averaged 3.039, and compact/virtual Keff both averaged 2.168. This is sparse query participation within twelve provisional functions, **not dynamic K reduction**. The rectangular full-grid P2 width remained twelve.

The taper is flat at `gamma=0` above its keep threshold, so it supplies no direct contraction gradient for these observed e150 nodes. Ordinary reconstruction training can still change their source-action scores through shared neural parameters. Whether any score enters the transition region later is an empirical question for the bounded diagnostic horizon; no threshold or tree change is made within this run.

The maintained full Q8192 evaluator paired exact Run 1506 e150 with exact Run 1502 e150 on the same ordered 90 cases and normalization. The table uses **equal-case mean** errors, not pooled relative L2. Positive change means the candidate error increased.

| Physical metric | Run 1502 e150 | Run 1506 e150 | Candidate worse cases |
| --- | ---: | ---: | ---: |
| Fluid-field normalized relative L2 | 0.20020 | 0.22820 | 79/90 |
| Physical fluid-temperature RMSE | 1.35140 | 1.33086 | 36/90 |
| Physical interface heat-flux RMSE | 14.17700 | 14.46530 | 79/90 |
| Physical final port-temperature RMSE | 1.33966 | 1.46388 | 69/90 |

The fluid-field equal-case deficit rises from +0.00857 at M=3 to +0.05468 at M=10; field, port, and flux mean deficits occur in all four M=3/5/7/10 strata, except the listed fluid-temperature component. These observed differences belong to independently trained checkpoints. Since the e150 operator formed no contraction, they are **not evidence of a merger changing the field**. A controlled same-weight active-path test with scores forced above the keep threshold gave R=12, zero transitions, field max-absolute difference 2.98e-8, and relative L2 difference 6.27e-8 against the parent; this supports numerical identity of the no-contraction reader on that CPU sample. It does not prove the two trained optimization trajectories are identical after epoch 50.

The distinct selected **e148** checkpoint also has no transition or closed nodes across 270 phases; its smallest node RMS score is 0.239. Its all-90 full-grid equal-case fluid-field relative L2, heat-flux RMSE, and final port-temperature RMSE are 0.20174, 14.18445, and 1.35626. Relative to the candidate's own exact e150 weights, e148 improves those metrics in 81, 86, and 62 cases. For context, they are near Run 1502 **e150** values, but that is a cross-epoch comparison and not a matched-stage win. No e148 Run 1502 full-grid checkpoint was used.

The exact e150 physical/formation tables and spatial/population figures remain local under `diagnostics/generated/run1503_v4_checkpoint_review/e150_all90_20260924T2359/`; the distinct e148 evidence is under `diagnostics/generated/run1503_v4_checkpoint_review/e148_selected_all90_20260925T0001/`. The corrected phase-local interface fidelity columns use denormalized targets.

### Exact execution and bounded physical geometry

The separate Q8192 rectangular benchmark used physical GPU 2 (RTX 6000 Ada; PyTorch 2.6.0+cu124), maps off, two warmups and three synchronized repetitions, then reversed model/case/scope order. It covered the five-case M=3/3/5/7/10 panel at both native inner/outer chunk 128 and matched chunk 2048. Every one of 60 model/case/chunk/pass rows completed. Values below are medians of row wall-time medians in milliseconds; ratios are medians of paired candidate/Run 1502 rows.

| Scope and chunk | Run 1502 e150 | Run 1505 e150 | Run 1506 e150 | Run 1506 / Run 1502 |
| --- | ---: | ---: | ---: | ---: |
| Complete application, native 128 | 305.85 | 578.19 | 402.62 | 1.33× |
| Complete application, matched 2048 | 61.04 | 294.61 | 119.76 | 1.97× |
| Phase preparation, matched 2048 | 52.33 | 281.19 | 117.35 | 2.29× |
| Prepared P2 decode, matched 2048 | 16.24 | 19.02 | 18.81 | 1.15× |

Thus v4 removes much of v3's planning penalty (v3/parent complete-application ratios were 1.88× native and 4.80× at 2048), yet it does not reach parent latency. A separate maps-off Q=1 phase diagnostic on the same panel recorded median v4 backend preparation around 22 ms per phase, including about 3.3 ms for the three probe routes, 4.6 ms for node scoring, and 4.1 ms for the combined diagnostic. These intervals are components of preparation, not additional application timings. On case 0273, parent/v3/v4 all physically executed 98,304 module and 1,572,864 environmental fine rows in the rectangular reader, despite different logical support and v3's R=11. Low compact query support does not imply fewer executed geometry rows. The complete local benchmark is `diagnostics/generated/run1506_exact_execution_benchmark/run1506_comparison_e150_five_cases_corrected.json`.

A read-only physical-GPU sweep of case 0641 plus M=5/7/10 cases 0653/0673/0686 used 70 feasible module-slot-0 positions in bounded ±0.02 intervals, fixed 16-query sets, predicted ports, and full P0/P1/P2 forwards. All outputs were finite. It observed **no native taper or packing signature change**; all phases retained R=12 with zero transition nodes. Minimum node RMS over the four sweeps was 0.349/0.443/0.650/0.575 by case. At the old Run 1505 case-0641 bracket of width 9.155e-5, v4's physical field changed by relative L2 2.47e-5 with the same phase signatures on both sides. This compares two positions away from any v4 closure and therefore does **not** establish physical continuity across a trained v4 merger. No CFD-reference derivative is available. The bounded record is `diagnostics/generated/run1503_v4_boundary_cost/run1506_e150_physical_boundary_four_strata.json`.

The e150 endpoint has no merger and broad field/port/flux deficits, so it does not support maturation or an automatic 5,000-epoch continuation. The distinct e148 selection is closer to the parent in full-grid metrics, v4 is much cheaper than v3, and the requested 500-epoch horizon is diagnostic. The same run therefore continues without changing its tree, scales, losses, or seed, to determine whether late training ever enters the functional transition region and how selection behaves at the bounded stop.

## Epoch-500 review and decision

The single Run 1506 candidate stopped normally at **exact epoch 500**. Its manifest records `completed`, exit code zero, the original UUID, and last completed epoch 500; the training window exited. All 500 sequential rows of tracked train/validation total loss, field MSE, and temperature MSE are finite. The exact checkpoint contains finite values in all 280 floating model and 453 floating optimizer tensors and retains its optimizer state. Sampled validation field/temperature MSE at e500 was 0.033112/0.014508; these are not full-grid physical errors. Median logged e151–500 training epoch time was 11.16 s versus 8.59 s for corresponding historical Run 1502 epochs, an unmatched run-log comparison.

The saved total-loss and predicted selections have identical model weights at **e459**; field selection is distinct **e474** and temperature selection is distinct **e461**. None is the exact e500 model. The Q1024 all-90 P0/P1/P2 surveys for **each of these four distinct checkpoints** found all twelve proposals occupied, R=12, zero transition nodes, and zero closed nodes in every one of 270 case-phase records per checkpoint. Minimum node RMS scores were 0.188 (e500), 0.199 (e459), 0.230 (e474), and 0.208 (e461), all above the 0.06 keep scale. The exact e500 combined post-transform discrepancy is zero because every T remains identity. This fixed criterion did not form a single functional merger by the diagnostic stop.

At exact e500, Q1024 P0/P1/P2 mean compact support was 2.698/2.715/3.104 and mean compact Keff was 2.066/2.081/2.243; virtual values were identical. On the Q8192 P2 full grid, the 90-case mean per-case compact support was 3.1105 (range 2.7366–3.6293) and mean Keff was 2.2474 (range 2.0041–2.5854), again identical to virtual support/Keff because every multiplicity is one. Registered K, occupied K, compact R, and executed padded width were all **12**. These statistics describe sparse query participation within twelve access functions; they are **not** dynamic-K function removal or a reduction in rectangular physical work. Mean Q8192 unique module/environment query-source pairs were 36,799/589,218, while the fine reader still executed 98,304/1,572,864 rows per case; mean module padding was 50,517 rows and environment padding zero. Fewer unique logical pairs did not translate into fewer executed fine geometry rows.

The maintained Q8192 evaluator paired exact Run 1506 e500 with exact Run 1502 e500 on the same ordered 90 development cases and normalization. Values below are **equal-case means**, with the final column counting cases in which v4 error is higher. Positive paired change means worse error. This table does not mix selected checkpoints with the endpoint.

| Physical metric | Run 1502 e500 | Run 1506 e500 | V4 worse cases |
| --- | ---: | ---: | ---: |
| Fluid-field normalized relative L2 | 0.10568 | 0.12839 | 69/90 |
| Near-interface normalized relative L2 | 0.09875 | 0.12244 | 81/90 |
| Far-fluid normalized relative L2 | 0.11626 | 0.12961 | 60/90 |
| Physical u relative L2 | 0.04039 | 0.03333 | 15/90 |
| Physical v relative L2 | 0.08455 | 0.11336 | 84/90 |
| Physical pressure relative L2 | 0.09910 | 0.16891 | 80/90 |
| Physical vorticity relative L2 | 0.12874 | 0.14652 | 70/90 |
| Physical fluid-temperature RMSE | 0.83272 | 0.76577 | 36/90 |
| Physical interface heat-flux RMSE | 15.68070 | 15.49852 | 33/90 |
| Physical final port-temperature RMSE | 0.90591 | 0.85983 | 35/90 |

The global-field mean deficit is +0.02271 and is driven especially by vorticity, v, pressure, and near-interface errors; physical temperature, surface temperature, interface flux, and final-port means improve. The field deficit increases by module count and the M=10 final port mean reverses the overall port improvement:

| Active modules M (cases) | V4 field L2 (Δ vs parent) | V4 flux RMSE (Δ) | V4 final port T RMSE (Δ) |
| --- | ---: | ---: | ---: |
| 3 (25) | 0.09105 (−0.00006) | 14.7675 (−0.0274) | 0.7478 (−0.0916) |
| 5 (25) | 0.11860 (+0.01352) | 15.6647 (−0.1716) | 0.8344 (−0.0998) |
| 7 (25) | 0.14490 (+0.03265) | 16.0253 (−0.3221) | 0.8644 (−0.0589) |
| 10 (15) | 0.17940 (+0.05943) | 15.5619 (−0.2246) | 1.0813 (+0.1408) |

Each distinct selected checkpoint had its **own** Q1024 formation and Q8192 physical evaluation over all 90 cases. The table shows equal-case physical means. Run 1502 e500 is displayed only as fixed context: comparisons from e459/e461/e474 to parent e500 are **cross-epoch**, not matched-stage wins.

| Checkpoint | Field fluid L2 | Flux RMSE | Final port T RMSE | Minimum node RMS |
| --- | ---: | ---: | ---: | ---: |
| Run 1502 exact e500 | 0.10568 | 15.68070 | 0.90591 | — |
| Run 1506 exact e500 | 0.12839 | 15.49852 | 0.85983 | 0.188 |
| Run 1506 total/predicted e459 | 0.09870 | 15.44680 | 0.83151 | 0.199 |
| Run 1506 field e474 | 0.09891 | 15.04375 | 1.10432 | 0.230 |
| Run 1506 temperature e461 | 0.10260 | 15.38028 | 0.79221 | 0.208 |

Relative to the candidate's own exact e500 weights, e459 reduces field error in 82/90 cases, flux error in 58/90, and final-port error in 55/90. E474 reduces field and flux error in 83/90 and 82/90 cases but **raises** final-port error in 79/90; its port mean is worse in every M=3/5/7/10 stratum. E461 has the lowest final-port mean but a higher field mean than e459/e474. Selection sensitivity remains material. None of these independently saved weights produced dynamic R, so no selected-output combination is presented as one model.

Local ignored `diagnostics/generated/run1503_v4_checkpoint_review/e500_all90_20260925T0137/` contains exact and selected per-case/phase tables, paired deltas, a manifest, and `readout_e500.md`. Its 90-case Q8192 population figures show compact Keff, virtual Keff, R, and Kq distributions for each checkpoint. Full-grid spatial boards for cases 0273/0641/0653/0673/0678/0686 show active support, effective K, dominant access class, maximum class mass, and routing entropy for v4 and exact Run 1502. The layouts cover M=3/5/7/10 and include the earlier case-0653 routing comparison. Color regions mark dominant **learned routing mass**, not merged functional classes or physical causal regions. No figures or raw arrays were committed.

Key local images in that directory are `figures/q8192_population_metrics_e500_n90.png` and the matched `figures/routing_support_e500_0653.png` / `figures/routing_support_run1502_e500_0653.png` pair; analogous filenames cover the other listed cases and selected checkpoints.

### Exact execution and bounded physical geometry at e500

The synchronized, maps-off GPU 2 benchmark used RTX 6000 Ada, PyTorch 2.6.0+cu124, Q8192, native inner/outer chunk 128 and matched 2048. The five-case M=3/3/5/7/10 panel used two warmups, three repetitions, and reversed model/case/scope order; all 60 rows completed. A separate all-90 application-only pass used one warmup and two repetitions; all 540 rows completed. Numbers are medians of row wall-time medians in milliseconds, and the final column is the median paired v4/parent ratio by case (and pass for the five-case panel).

| Scope / panel / chunk | Run 1502 e500 | Run 1505 e500 | Run 1506 e500 | V4 / parent |
| --- | ---: | ---: | ---: | ---: |
| Complete application / five / native 128 | 316.73 | 612.52 | 412.94 | 1.32× |
| Complete application / five / matched 2048 | 59.73 | 313.29 | 118.03 | 1.99× |
| Phase preparation / five / matched 2048 | 51.24 | 301.84 | 118.13 | 2.33× |
| Prepared P2 decode / five / matched 2048 | 16.30 | 19.38 | 18.76 | 1.16× |
| Complete application / all 90 / native 128 | 304.07 | 613.79 | 404.44 | 1.329× |
| Complete application / all 90 / matched 2048 | 60.23 | 306.55 | 120.48 | 1.990× |

The all-90 paired v4/parent application ratio ranges from 1.319–1.354 across M strata at native 128 and 1.980–2.020 at matched 2048. V4 removes much of the v3 planner penalty (all-90 v3/parent ratios 1.990× native and 5.059× matched), but its probe/Gram and transform preparation remains slower than the parent and rectangular physical rows do not shrink. Benchmark JSON remains local under `diagnostics/generated/run1506_exact_execution_benchmark/`.

The exact e500 read-only physical-GPU geometry sweep covered 70 feasible module-slot-0 positions in bounded ±0.02 intervals for cases 0641/0653/0673/0686 (M=3/5/7/10), fixed 16-query sets, predicted ports, and full P0/P1/P2 forwards. All outputs were finite. There was **no native taper or packing signature change**; every phase kept R=12 and zero transitions. Minimum node RMS scores over these four scans were 0.326/0.459/0.774/0.644. At the old v3 case-0641 bracket of width 9.155e-5, the v4 predicted field difference was 1.63e-5 relative L2 with identical phase signatures on both sides, versus the old v3 forced hard-partition difference of about 3.05e-2 at a fixed coordinate. The v4 points were **not** near a v4 closure, so this does not establish continuity at a learned merger or CFD sensitivity fidelity. The local record is `diagnostics/generated/run1503_v4_boundary_cost/run1506_e500_physical_boundary_four_strata.json`.

### Research decision

Do **not** continue this candidate to 5,000 epochs. The 500-epoch diagnostic shows no case-dependent functional reduction at the endpoint or any selected checkpoint, a broad exact-endpoint fluid-field deficit that worsens with M, and application latency 1.33–1.99× the parent despite a large improvement over v3 planning. Selected e459/e461/e474 weights show useful physical tradeoffs, but these do not establish the intended organizer mechanism, and the scanned geometries never approached a native closure. More epochs of the same flat-above-threshold taper are not supported by the measured criterion scale; shared-parameter training could still move scores, so this is a research judgment rather than proof that closure is impossible. A separate future candidate should revisit the source-action score/threshold calibration and how contraction receives a learning signal, using training-only evidence and a fresh bounded physical-boundary review. The current run, historical runs, security behavior, and raw diagnostic outputs remain preserved locally. No 5,000-epoch continuation was launched.
