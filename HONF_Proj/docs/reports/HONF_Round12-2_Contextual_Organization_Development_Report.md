# Round12-2 measured development: smaller frontiers, global-root collapse, and no full-data recommendation

Round12-2 replaced the Thermal-specific affine temperature path with a joint physical-field predictor, supplied prescribed original-module heat before Transformer interaction, and learned a reversible contextual retrieval frontier. Both matched scratch runs completed their first100 development epochs normally. The learned J-H model constructs a smaller reusable group directory and preserves the four global priority temperature/peak error bands relative to the same-age J-D leaf control. It nevertheless collapses to one large group per bank in all four detailed scenes, reads every original E record at every positive E query, and fails bounded functional-sharing and geometry-response checks. Original value storage/execution does not contract with this directory. This is a measured negative organizer screen, with useful implementation and predictor evidence; it is not a successful selective hypergraph organization result.

**Decision: stop this bounded development screen at the completed epoch100 checkpoints; recommend zero full TRAIN600 scratch100 candidates.** The2000-epoch master horizon is a ceiling, and the conditional full-data allocation remains unused. No further training, control portfolio, solver call, dataset generation, TEST access or inverse-design campaign was launched after this review. The100-epoch predictors are immature: this decision rejects promotion of the present organizer objective, not the possibility of a useful joint HONF or of later predictor recovery.

| Area | Gain actually measured | Miss / limit | Decision |
| --- | --- | --- | --- |
| Predictor | Same-checkpoint full native exposed DEV22 evaluation; all four priority global mean/p90 preservation bands pass against J-D | Local and stratum regressions; simplified-Thermal source laws and geometry/mixed responses remain inaccurate | Preserve both100-epoch references; do not infer mature physical fidelity |
| Organizer | Contextual candidate/stop implementation, executed direct C[K,N], reversible K<N and receiver/query reuse | Global-root collapse; learned M group route inactive; E source reads become dense; root/split equivalence fails | No validated selective functional grouping; do not promote this objective unchanged |
| Inverse / decisions | Live source and geometry derivative probes expose actual model dependence and seams | No physical geometry-gradient fidelity or reliable candidate-selection evidence | No verified design or inverse recommendation |
| Cost | Isolated fixed-panel preparation/decoding/backward and executed-row measurements; fast/reference reader parity | Lower measured latency accompanies more E source edges and much higher peak memory | Report the bounded tradeoff; do not infer efficiency from K |

## 1. Identity, scope and completed execution

The governing [Round12-2 plan](../../UpgradePlan/HONF_Round12-2_Contextual_Adaptive_Hypergraph_Organization_Work_Plan.md), [fixed25 protocol](../guides/Thermal_Model_Development_Protocol.md), [J-D recipe](https://github.com/cosmos2w/ModularDT/blob/7981c173e8b1d29ee2b3addb7b972d0f15003542/HONF_Proj/configs/research/thermal_contextual_frontier_jd_dev_v1.json) and [J-H recipe](https://github.com/cosmos2w/ModularDT/blob/7981c173e8b1d29ee2b3addb7b972d0f15003542/HONF_Proj/configs/research/thermal_contextual_frontier_jh_dev_v1.json) define this separately initialized campaign. Its normalization is fitted only on fixed TRAIN150; its22 exposed development cases are not full VAL89 or independent TEST evidence. Original U-H, zero-merge Round12, amended compact local100, scratch v1 and soft_only v2 remain separate lineages. None supplies weights or normalization to these runs, and no favorable property is borrowed from one of their checkpoints.

Both scientific arms started from matched scratch seed0 tensors and primary case/query schedules, with Q1024, effective batch48/nominal micro24, constant AdamW3e-4/WD1e-5 and absolute master horizon2000. Epochs1–20 force leaves; epochs21 onward sample legal reversible frontiers for J-H, with deterministic zero-logit stops at evaluation. J-D retains leaves. Structural multiplier0.1→1 and temperature2→0.5 use the original absolute1–200 schedule, not a segment-local restart; at100 the multiplier is0.546448. Policy and predictor remain trainable, with no permanent hard cut or freeze.

| Arm | Exact measured checkpoint | Completed exposure | Terminal versus selected-best |
| --- | --- | --- | --- |
| J-D | [thermal_contextual_frontier_jd_dev_v1/checkpoints/epoch_0100_model.pt](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/_bk/11_20261024_Contextual_Organization_Development/thermal_contextual_frontier_jd_dev_v1/checkpoints/epoch_0100_model.pt) |100 epochs;15000 primary case visits;400 Adam updates;15,360,000 primary fluid-query rows | Only declared monitoring point100; latest and best-by-field tensors equal epoch100 |
| J-H | [thermal_contextual_frontier_jh_dev_v1/checkpoints/epoch_0100_model.pt](../../Trained_Results/ThermalChannel/HONF_Forward_Runs/_bk/11_20261024_Contextual_Organization_Development/thermal_contextual_frontier_jh_dev_v1/checkpoints/epoch_0100_model.pt) |100 epochs;15000 primary case visits;400 Adam updates;15,360,000 primary fluid-query rows | Same100/latest/best identity; no later selected checkpoint is substituted |

Both runs launched at19:11:18 UTC on24 October2026 from source commit082cdcc on physicalGPU0 and exited0 after their declared100-epoch segment. J-H engine training time was8507.517s (2.3632 process-hours), J-D14249.442s (3.9582 process-hours). They overlapped on the same GPU, so the6.3214-hour sum is neither device-busy GPU-hours nor a controlled training-speed comparison. Their common GPU0 training window was about3.96 hours. Training sessions/PIDs and commands are historical launch identities, not currently running jobs. No GPU1/GPU2 compute was introduced by this screen.

The [saved-history audit](../../diagnostics/round12_2_execution_20261024/contextual_frontier_saved_history_audit_20261024.json) verifies all233 model tensors equal among each arm's100/latest/best aliases. It also reconstructs identical per-case query draws and case orders throughout100 epochs despite packing changes. J-H used existing OOM rollback at epochs21 and100, repacking24→12→6; its old history reports1186 microbatches but successful retries actually executed1192, versus700 for J-D. This bookkeeping defect does not change primary case visits or Adam updates. The durable runner now reports successful retry packing and fresh-stream masks; historical files are preserved unchanged, with the correction in the audit. Failed attempts constructed six microbatches and96 repeated case batches, which are not all completed physical-forward exposures.

Extra exposure is explicit: each arm performs100 response microbatches (200 perturbed endpoint-scene visits) before the same primary Adam step,100 ordinary functional comparisons (50M/50E;300 native forward calls),288 no-step native calibration-case visits at1/21/100, six calibration-response scene visits, and12 calibration functional comparisons (36 native forwards). These are additional to15000 primary visits. Provider batch-construction counters15100/15196 must not be relabelled as primary training case visits.

## 2. What the model and objective actually execute

One joint trunk predicts $(u,v,p,T)$ from a shared context. The existing case adapter derives $\omega$, surface/material temperature, q proxy and original-module maxima through native interpolation, curl and receiver geometry. Original records, coordinates, measures, individual values and all original M source identities remain present. Prescribed heat enters each M record before two maintained geometry-aware Transformer cycles (width128, four heads, MM→EM→EE→ME). Raw E features use coordinates and boundary distances without occupancy/nearest-module heuristics. Both banks can carry cross-field contextual information; neither is assigned exclusively to temperature or flow.

This removes the universal affine-temperature/heat-independent-flow restriction discussed in plan Section3.1. Original source IDs identify input axes; they do not require allocating an E-by-source response-factor bank. E-only final access can depend on module controls through M→context→E. It still has finite contextual capacity and is not a proof of physical decoupling, local physical causality or correct multiphysics coupling. The simplified Thermal references contain no evidence for absent coupling mechanisms. Both arms retain dense encoder N² work; applied U-H contextual incidence inside encoder exchanges was not implemented in this revision.

Context and spatial four-neighbor nominations build a typed spanning binary candidate forest. A legal reversible frontier defines live K. Its direct membership C[K,N] reads original same-type contextual records within ancestry plus the declared spatial halo. Queries score admissible live groups through smooth geometric windows and independent null routes, then execute selected original-record keys/values. Ordinary retrieval constructs neither the retired native N×N retrieval directory nor old query-to-native-seed logits before pooling. Candidate distance work and the dense encoder remain separate quadratic operations.

The optimizer combines native fixed-frontier loss, conditional native score-function policy gradients, separately normalized expected-K capacity, stopped-outer-probability functional discrepancy and detached risk supervision. No-step TRAIN calibration uses the same complete bank parameter inventory, fills missing VJPs with zero, targets2% of conditional native policy norm per term/bank, and caps the combined structural contribution downward at10% of full native gradient after scheduling. Calibration is not an unbiased guarantee of optimizing the uncapped expected loss. The old U-H work controller's epoch3000 targets are disabled because they do not describe this reader; expected K is therefore a structural proxy with no trained penalty on actual final source edges. This explicit departure is central to the negative result.

Existing-reference heat and single-geometry supervision occurs every fourth primary update without an extra optimizer step. Only0348 intersects selected TRAIN150 and is used for fitting/scales;0001/0318/0333 remain reference-only. Families0304/0320/0335/0350 are response-validation-only. All eight atlases originate in source TRAIN, including the four held out from response fitting: they are not DEV22/VAL89 or TEST cases. Common fluid masks, original module IDs and local receiver identities are joined before differencing. Mixed geometry never enters backpropagation.

## 3. Native prediction: global preservation passes, local failures remain

The same explicit100 checkpoints were evaluated on all22 exposed DEV cases using8192 fluid grid points,64 surface angles per module and all3096 saved material points per module. Fluid/physical masks and native observation rules are retained. The endpoint metric is per-case RMSE across differences of separately recomputed original-module temperature maxima. The values below are equal-case means/p90/worst, not an aggregate field score. Units are the packed dataset's native velocity, pressure, vorticity, temperature and q-proxy units; no SI conversion or physical-validity tolerance is inferred.

| Quantity | J-D mean / p90 / worst (case) | J-H mean / p90 / worst (case) | Mean / p90 ratio |
| --- | --- | --- | --- |
| u | 0.0852497 / 0.104162 / 0.106934 (0277) | 0.0862311 / 0.107072 / 0.115317 (0644) | 1.0115 / 1.0279 |
| v | 0.00708812 / 0.00883539 / 0.00913257 (0684) | 0.00700764 / 0.00861032 / 0.00898146 (0684) | 0.9886 / 0.9745 |
| p | 0.0384433 / 0.0428695 / 0.0477393 (0677) | 0.0386359 / 0.0431219 / 0.052114 (0677) | 1.0050 / 1.0059 |
| omega | 0.181619 / 0.207742 / 0.215364 (0686) | 0.177438 / 0.206611 / 0.222314 (0686) | 0.9770 / 0.9946 |
| temperature_fluid | 2.98487 / 3.75704 / 4.57165 (0663) | 2.85991 / 4.26793 / 4.6129 (0655) | 0.9581 / 1.1360 |
| temperature_surface | 2.27394 / 3.01731 / 4.53765 (0663) | 2.36539 / 3.28698 / 4.27151 (0663) | 1.0402 / 1.0894 |
| temperature_material | 2.29194 / 2.87289 / 4.23698 (0663) | 2.26616 / 3.05557 / 3.98418 (0663) | 0.9888 / 1.0636 |
| q_proxy | 4.89451 / 6.50235 / 7.53413 (0677) | 4.46613 / 6.60915 / 6.83763 (0677) | 0.9125 / 1.0164 |
| module_peak_endpoint_rmse | 2.36326 / 3.13827 / 3.70055 (0663) | 2.2828 / 3.17107 / 3.72694 (0645) | 0.9660 / 1.0105 |

Fluid/surface/material temperature and endpoint each pass mean≤1.10J-D and p90≤1.15J-D. For example fluid-temperature mean improves2.98487→2.85991 while its p90 worsens3.75704→4.26793 (+13.60%); surface mean worsens2.27394→2.36539. These are descriptive preservation bands against another immature jointly trained model, not evidence of sufficient absolute fidelity or significance. Routine training-side sampled validation summaries use a different receiver panel and are not substituted for this native board.

![Figure1](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/comparison_native_priority_ratios100.png)

Figure1. Full native exposed DEV22, both explicit epoch100 checkpoints. All nine component mean/p90 ratios are shown, with the four priority quantities and their prespecified1.10/1.15 descriptive bands. Passing the global bands does not remove the stratum, regional or physical-response failures below. [PDF master](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/comparison_native_priority_ratios100.pdf).

![Figure2](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/native_fields_residuals100.png)

Figure2. Input-selected exposed DEV0684/M10 at epoch100, native u and fluid-temperature fields and signed prediction-minus-reference residuals. Physical masks, original module outlines and common native-unit color scales are retained. This case shows spatial error structure that global means do not resolve; it is an illustrative case rather than a population selection by error. [PDF master](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/native_fields_residuals100.pdf).

### Module-count strata

The exposed panel contains M3/M5/M7/M10 with6/6/6/4 cases. Other TRAIN module counts have no DEV22 cases and receive no invented stratum estimate. Each cell below gives J-D→J-H mean and p90; all nine quantities remain visible. Per-case and worst-stratum identities are retained in the linked native result JSONs.

| Quantity | M3 mean; p90 | M5 mean; p90 | M7 mean; p90 | M10 mean; p90 |
| --- | --- | --- | --- | --- |
| u | 0.08624→0.08486; 0.1069→0.1061 | 0.09027→0.09213; 0.1034→0.1074 | 0.08259→0.08537; 0.09429→0.09644 | 0.08022→0.08072; 0.08356→0.08475 |
| v | 0.005393→0.005413; 0.005731→0.005615 | 0.007102→0.007123; 0.007287→0.007566 | 0.007675→0.007461; 0.008311→0.007855 | 0.008729→0.008546; 0.009098→0.008958 |
| p | 0.03641→0.0362; 0.04017→0.03988 | 0.03731→0.03746; 0.04178→0.04326 | 0.03953→0.04029; 0.04381→0.04738 | 0.04154→0.04156; 0.04266→0.04286 |
| omega | 0.1514→0.1478; 0.1713→0.1623 | 0.1859→0.1752; 0.197→0.1835 | 0.1933→0.1926; 0.2111→0.2084 | 0.2029→0.2025; 0.2128→0.2152 |
| temperature_fluid | 2.759→2.436; 3.687→3.16 | 3.256→2.866; 3.716→4.087 | 3.212→3.401; 4.302→4.373 | 2.577→2.675; 2.731→2.951 |
| temperature_surface | 1.701→1.792; 2.115→2.444 | 2.59→2.425; 3.061→3.197 | 2.591→2.764; 3.638→3.61 | 2.184→2.538; 2.764→3.268 |
| temperature_material | 1.95→1.871; 2.514→2.72 | 2.498→2.309; 2.686→2.748 | 2.49→2.539; 3.462→3.46 | 2.2→2.385; 2.754→3.151 |
| q_proxy | 4.145→3.759; 5.448→4.975 | 5.164→4.081; 6.329→5.19 | 5.394→5.145; 7.316→6.76 | 4.865→5.085; 5.729→6.316 |
| module_peak_endpoint_rmse | 2.22→2.134; 2.823→2.878 | 2.68→2.348; 3.022→2.605 | 2.374→2.409; 3.386→3.413 | 2.087→2.219; 2.579→3.106 |

The global pass conceals priority stratum misses: M3 surface-temperature p90 increases15.58%; M10 surface mean/p90 increase16.19%/18.27%, and M10 peak p90 increases20.40%. These exposed strata contain only6/4 cases and are descriptive regressions, not significance estimates. They remain failures even though the global priority bands pass.

## 4. Organizer: actual directory contraction without selective local grouping

The input-only TRAIN organizer review panel covers24 scenes across available module counts; bounded detailed native execution/interventions use0006/M1,0097/M4,0337/M7 and0231/M12. All four deterministic100 frontiers have live and allocated K_M=1 and K_E=1, versus original M=1/4/7/12 and E=192. M1 has no reducible M pair. E expected K is1.00474/1.00210/1.00157/1.00191 and root stop probability0.995314/0.997914/0.998434/0.998104. This is real direct organizational contraction, not a soft expected-K surrogate. It does not prove selective redundancy: the detailed scenes all select the largest available groups.

Recorded sampled primary-preparation K/N falls from leaves during1–20 to M0.7156/E0.3337 at21, M0.1821/E0.0061 at50, and M0.1763/E0.00535 at100. The M average reflects variable original module count and is consistent with roughly one group per scene; it is not17.6% query access. The late trajectory is stable near roots despite ongoing predictor improvement. This supplies a concrete reason to review the objective rather than mechanically extend to2000.

Actual trained-context dependence was also checked on TRAIN0231/M12 using the same J-H100 weights on CPU, without decoding, optimization or physical labels. Setting encoded contextual states to zero while preserving original coordinates/lengths/measures/IDs changes the candidate edge set by18 M and862 E edges (symmetric difference), and changes all12/192 contextual neighbor lists. A separate fixed-original-tree intervention zeros only the contextual part of each node descriptor: root stop logits change9.697→−3.594 for M and7.870→−3.481 for E, with maximum internal-logit changes13.601/11.351. Thus contextual information really influences both nomination and policy on the trained checkpoint. Zero context is an out-of-distribution diagnostic, not an alternative physical input or proof of useful learned grouping; it does not identify a unique cause of root collapse. The [context-dependence record](../../diagnostics/round12_2_execution_20261024/review100/context_dependence100.json) preserves this distinction.

![Figure3](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/comparison_training_k100.png)

Figure3. Saved training history through100: an unweighted sum of eight reported native component losses, followed by sampled primary-prepare live K/N for each bank. The loss sum excludes organizer/auxiliary terms and the separate module-peak observable, so it is not the full weighted optimizer objective. The20-epoch leaf warmup ends before both banks collapse; lower K alone is not functional grouping. [PDF master](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/comparison_training_k100.pdf).

All original sources and values survive. The actual0231 C_M[1,12] has seven positive entries and C_E[1,192] has all192; the remaining physical M records remain available through the protected local route. A single prepared forest/C is reused across native receiver roles and query chunks. Learned M group access is empty in all four detailed scenes; M final reads come from the declared local envelope. Every query with positive E access reads all192 E originals. In the256-query bounded traces this yields34944/35712/37056/38016 E source edges. E-only final rows exist, but they do not read only nearby E records, and their encoded E values still inherit module information through the dense context.

![Figure4](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/organization_actual100.png)

Figure4. Actual J-H100 TRAIN0231/M12 ancestry, direct C amplitudes, stop probabilities and256-query final E access. The dendrogram is truncated only for display; retained arrays contain the full forest. The root covers192 original E columns, every positive E query reads all192, and learned M-group access is empty while local M access remains. This supports executed model organization/dependence, not physical causality or a compact value executor. [PDF master](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/organization_actual100.pdf).

### Bounded interventions and gradient diagnosis

For each detailed scene, two input-selected internal nodes per eligible bank compare a parent stop with its descendant frontier, retain the same encoded originals and the other bank, reconstruct C/windows/access normalization, run the complete native receiver union and restore ordinary state. E candidate coverage is2/191 internal nodes per scene, not an exhaustive claim. The root/split fluid-temperature RMS change divided by ordinary native RMSE is0.7716/0.3234/0.5313/0.4409; surface0.3075/0.3187/0.4322/0.1822 and material0.2498/0.3488/0.3742/0.1387. All exceed the prespecified0.1 functional-invariance scale. Root splits reduce full-native E edges by about19–44%, but also change the learned function. Some deeper nodes pass; they do not validate the selected roots.

M parent/split changes are at floating-point noise scale (roughly1e-6×native RMSE) while the learned M-group route is empty. This is evidence of an unused control path, not useful functional sharing. The100 ordinary training probes include50M comparisons, so the inactive terminal route must not be described as no M calibration or no M probing at all.

At the last J-H update, native full-gradient norm is0.883709, raw structural norm0.0225205 and scheduled norm0.0123063, below cap0.0883709 (cap scale1). Thus the total10% cap is not binding at that update. Capacity/native policy cosines are−0.13185 for M and−0.02805 for E; score-function/native cosines−0.14257/−0.04507. E functional-share cosine is+0.11275 and risk−0.00720. These are observed signed competitions, not a unique causal attribution. The final pre-ramp M share coefficient13350.18 arises from a tiny calibrated gradient, versus E0.12278; this is automatic norm matching, not manual amplification, but can amplify nearly null/noisy directions. Complete norm/zero/coverage records remain in the audit. Separate bank calibration and richer policy inputs do not guarantee identifiability or selective organization.

The strongest supported mechanism is an objective mismatch: minimizing expected K rewards the roots, while this implementation charges no actual query/source-edge penalty. A root can cheaply nominate an entire bank and delegate fine discrimination to original-record query/key scores. Dense E reads then preserve much native utility despite broad organization; the dormant M-group path can be organized without affecting the prediction. Native loss, capacity, stochastic estimator variance, limited functional coverage and learned gate saturation may all contribute. No single-factor ablation was performed, so none is asserted as the sole cause. The proposed future amendment in Section9 is not a completed result.

## 5. Regional E-only quality

Four DEV examples0277/M3,0287/M5,0294/M7 and0684/M10 are selected by input strata before errors. Upstream means x<minimum module-center x−2r; far means distance to every module>3r; near means distance≤2r; downstream means x>maximum module-center x+2r. The common E-only mask requires actual positive E and zero M source reads for both models at every interpolation contributor and all five curl-stencil nodes. Reference-valid masks determine the denominator; nonfinite predictions invalidate a metric rather than silently disappearing. All measured predictions are finite.

| DEV case | Upstream physical / common E-only rows | Upstream fluid-T RMSE J-D→J-H | Far common E-only rows; T RMSE J-D→J-H | Downstream common E-only rows; T RMSE J-D→J-H |
| --- | --- | --- | --- | --- |
|0277|384 /155|2.3043→1.7600|5376;4.2976→2.4891|2650;5.3484→3.2040|
|0287|3072 /1044|1.1536→1.0969|3659;2.1866→1.6194|922;3.2659→2.5633|
|0294|1472 /143|1.2674→1.5857|2501;3.8568→3.2575|346;6.0206→5.0445|
|0684|576 /205|2.5639→3.6161|2394;2.6222→2.4165|813;2.1059→2.4088|

The protected M envelope leaves no common E-only near-module rows. Upstream temperature regresses25% and41% in0294/0684 despite passing global temperature bands. E-only final retrieval is executed, but nearby-E-only structured indexing is not achieved: every positive E route still opens all192 originals. Region masks/counts and all five fluid components remain in the retained regional JSON/NPZ.

## 6. Physical responses: heat improves relatively, geometry and mixed responses remain weak

Both explicit100 checkpoints replay the eight unchanged saved physical-reference families:88 native calls per arm comprise baseline, four single-geometry states, four mixed corners and heat± for every family. This yields32 single first differences, eight mixed central derivatives, eight raw four-corner contrasts and16 heat differences. Receivers use256 common physical fluid rows,16 surface angles and all3096 module-local material points. Module maxima are recomputed over common valid material receivers before differencing. Source-column correspondence and receiver joins are retained. No new solver, physical data or TEST was used.

Primary preservation was declared before candidate inspection: for each priority temperature/peak/q component and fixed cohort/response set, compare median absolute RMSE≤1.10J-D and p90 absolute RMSE≤1.15J-D. The other four flow/vorticity components retain the same ratio diagnostics without becoming a new promotion criterion. Pool the raw saved differences within the cohort/set before taking each arm's quantiles, then divide arm statistics. Do not average directions/families first, take a median of ratios, or replace RMSE by secondary MAE. Heat pools its two signs, single geometry its four directions, and mixed central/raw contrasts remain separate. Relative error is reported only above the fixed TRAIN-derived near-null floor; absolute errors remain primary. These bands describe preservation against J-D, not sufficient physical fidelity.

The following priority table gives J-H/J-D median/p90 absolute-RMSE ratios; † marks failure of either band. All nine components, their absolute errors and per-reference identities remain in the primary JSON and Figure5.

| Cohort / set (reference count) | Fluid T | Surface T | Material T | q proxy | Peak T |
| --- | --- | --- | --- | --- | --- |
| Fit0348 / heat (2) | 0.967/0.965 | 0.928/0.927 | 0.915/0.918 | 0.994/0.995 | 0.935/0.933 |
| Fit0348 / single_geometry (4) | 1.131/1.081† | 1.082/0.998 | 1.158/1.007† | 0.761/1.036 | 1.450/1.187† |
| Fit0348 / mixed_central (1) | 1.387/1.387† | 1.004/1.004 | 1.039/1.039 | 0.851/0.851 | 1.121/1.121† |
| Fit0348 / mixed_raw (1) | 1.387/1.387† | 1.004/1.004 | 1.039/1.039 | 0.851/0.851 | 1.121/1.121† |
| TRAIN reference-only / heat (6) | 1.058/1.085 | 1.296/1.277† | 1.287/1.244† | 0.904/1.183† | 1.289/1.173† |
| TRAIN reference-only / single_geometry (12) | 0.976/0.712 | 1.151/0.802† | 1.061/0.837 | 0.927/0.691 | 1.190/0.793† |
| TRAIN reference-only / mixed_central (3) | 1.111/1.926† | 0.749/1.369† | 0.670/1.402† | 0.654/0.905 | 0.577/1.088 |
| TRAIN reference-only / mixed_raw (3) | 0.948/2.054† | 1.142/1.452† | 1.243/1.545† | 0.654/0.905 | 1.234/1.253† |
| Response-validation-only / heat (8) | 0.896/0.869 | 1.072/0.834 | 1.066/0.850 | 0.874/0.810 | 1.074/0.891 |
| Response-validation-only / single_geometry (16) | 1.121/0.724† | 0.944/0.756 | 0.944/0.775 | 0.835/0.978 | 0.948/0.714 |
| Response-validation-only / mixed_central (4) | 1.474/1.273† | 1.338/1.025† | 1.318/1.055† | 0.663/0.552 | 1.202/0.953† |
| Response-validation-only / mixed_raw (4) | 1.474/1.273† | 1.338/1.025† | 1.318/1.055† | 0.663/0.552 | 1.202/0.953† |

Only fit0348 heat passes all nine components. The other11 cohort×response-set groups each fail at least one component. The four response-validation-only families pass all five priority heat bands but fail heat-flow/vorticity components; their single-geometry fluid-temperature median ratio is1.121, while mixed-central priority fluid/surface/material/peak ratios are1.474/1.338/1.318/1.202. Thus heat preservation does not establish geometry or mixed interaction preservation.

![Figure5](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/comparison_response_fidelity100.png)

Figure5. The eight existing source-TRAIN physical-reference families, separated into fit0348, three TRAIN-reference-only and four response-validation-only families. Every physical component is shown for pooled single geometry, heat, mixed central derivative and raw four-corner contrast. Points compare each arm’s raw-record absolute-RMSE median/p90, with frozen1.10/1.15 bands. All predictions are finite; mixed/near-null relative errors are handled separately and are not rescued by this descriptive comparison. [PDF master](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/comparison_response_fidelity100.pdf).

### Absolute versus relative response accuracy

For the four response-validation-only families, the following values describe J-H against physical references, not against J-D. Relative error is RMSE/reference-response RMS only when the fixed TRAIN near-null criterion admits it; excluded relative entries still retain absolute errors. The small mixed-derivative denominators make relative ratios unstable, so the absolute columns must be read first. Units for finite heat/geometry differences are their native physical component units; mixed central derivatives additionally divide by the saved geometry-step product, whereas raw four-corner contrasts do not.

| Response set | Component | Absolute RMSE median | Relative error median / p90 | Relative entries better than zero-response predictor |
| --- | --- | --- | --- | --- |
|Heat|Fluid T|0.1991|0.812 /0.829|8/8|
|Heat|Surface T|0.2370|0.524 /0.834|8/8|
|Heat|Material T|0.2453|0.475 /0.649|8/8|
|Heat|q proxy|0.3693|0.400 /0.449|8/8|
|Heat|Peak T|0.2939|0.448 /0.491|8/8|
|Single geometry|Fluid T|0.3426|2.747 /9.743|2/16|
|Single geometry|Surface T|0.2925|1.396 /3.728|5/16|
|Single geometry|Material T|0.2933|1.676 /4.520|5/16|
|Single geometry|q proxy|1.0586|1.060 /1.271|5/16|
|Single geometry|Peak T|0.3074|2.847 /5.556|4/15|
|Mixed central|Fluid T|2.0390|75.17 /75.17|0/1|
|Mixed central|Surface T|1.7622|70.87 /98.42|0/3|
|Mixed central|Material T|1.9612|93.82 /107.93|0/3|
|Mixed central|q proxy|3.1260|49.79 /205.08|0/3|
|Mixed central|Peak T|1.9714|90.23 /112.53|0/2|

These measurements support bounded useful heat differences for this panel but poor geometry and mixed sensitivity. The sole fitted response family and100 response microbatches provide limited supervision. No full-population response accuracy, original U-H affine coefficient recovery, or physically faithful geometry gradient follows.

### Original controls, source laws and cache application

Bounded probes on0348 and0304 use64 common fluid rows,32 local material rows and16 angles. Original source AD retains every physically present input column. Disjoint even/odd original-column heat partitions a+b=h are fully re-prepared for h,a,b,0. In these simplified homogeneous-boundary Thermal cases, zero heat implies zero temperature rise and heat does not alter flow; these are case-specific analytical checks, not universal HONF laws or new physical solver references.

| Arm / family | Zero-heat fluid-T RMS error | Heat-additivity fluid-T RMS residual | Full-heat minus zero-heat u RMS | Full-heat minus zero-heat p RMS |
| --- | --- | --- | --- | --- |
|J-D0348|6.45522|0.87711|0.025065|0.016102|
|J-H0348|5.95251|1.78875|0.033669|0.017621|
|J-D0304|2.87673|0.15605|0.008771|0.005728|
|J-H0304|2.76761|0.16611|0.012651|0.008209|

Both joint architectures fail these case laws at100. Zero heat and the disjoint heat partitions need not be covered by routine native training; these are bounded analytical control/generalization checks, not interpolation-accuracy estimates on the native DEV22 distribution. Removing a universal hard-coded heat decomposition permits broader field coupling but transfers these simplified-case properties to data coverage and case-owned training constraints. Their failure in J-D as well as J-H separates this concern from organizational compression; it is not evidence that hard-coded universal heat-affinity should be reinstated.

Fixed-complete-input cached preparation versus fresh inference has zero measured output RMS discrepancy. Cached application skips the encoder but re-enters both neural M/E value readers once. Changed heat or geometry requires full re-preparation. There is no retained exact affine response-factor application in this joint class; old U-H/current-weight coefficient-cache results cannot be transferred to it. Cache parity is distinct from the physical-reference errors above.

### Live AD/FD and topology seams

On each of0348/0304, eight mean physical readouts retain all original-source AD columns and one bounded geometry-axis AD/central-FD probe. These are mean-readout diagnostics, not pointwise or endpoint-max derivative validation. Default FP32 steps0.001r and0.0005r preserve the observed candidate/frontier/donor/query support, but show precision sensitivity: J-H maximum geometry AD/FD absolute discrepancy is0.000743/0.006684 on0348 and0.009551/0.015880 on0304. On0304 q proxy, AD0.077339 compares with FD0.086890/0.061459. They must not be reported as a clean tiny-step numerical pass.

A separate numerical qualification promotes the same saved weights to FP64 solely for this probe; it does not replace FP32 field/response results. With the same fine steps, maximum geometry discrepancy falls to4.95e-7/1.24e-7 for J-H0348 and9.83e-6/2.46e-6 for J-H0304 (J-D similarly improves). This supports a roundoff/truncation contribution. Coarser FP32 steps0.01r/0.005r cross candidate/query/source-support seams in0304 and produce discrepancies0.441/0.502 for J-H and0.740/0.728 for J-D. Native interpolation-index seams were not separately instrumented, so support changes are not asserted as the unique cause. Piecewise model AD/FD consistency supplies no physical geometry-gradient fidelity; the saved geometry-reference comparisons remain the relevant negative evidence.

## 7. Executed cost and memory

Isolated physicalGPU0/RTX6000Ada measurements use the same TRAIN0231/M12 input and full native8192-fluid/64-angle/all-material panel, one warmup and three repeats. They measure different trained functions with matched hardware/input; the comparison is not a causal K-only speed experiment. The separate fast/reference reader test uses the same trained function and32 fixed queries, with maximum differences9.54e-7 (J-D) and1.43e-6 (J-H), below rtol/atol1e-5.

| Boundary / work | J-D100 | J-H100 | Scope |
| --- | --- | --- | --- |
|Prepare median|195.28ms|95.09ms|Includes dense context/candidates/direct bank|
|Core decode median|710.91ms|87.02ms|Selected original records|
|Native extraction median|1.38ms|1.35ms|Native receiver transforms|
|Expected-K policy VJP median|6.72ms|6.58ms|Proxy derivative; J-D hypothetical inactive policy|
|Native-output backward median|183.01ms|113.48ms|Quadratic native-output scalar, no optimizer|
|Complete measured wall median|1108.75ms|305.67ms|One measured prepare/decode/extract/VJP/backward pass|
|Peak allocated memory, max of repeats|487484928B|5887032320B|J-H approximately12.08× J-D|
|E bounds tested|1572864|8192|Metadata work|
|E neural group scores|13728|8192|Actual group scoring|
|E original-source query edges|52716|1275264|J-H24.19× more|
|E nonlinear value-union rows|192|192|100% of original E rows in both|
|M nonlinear value-union rows|12|12|100% of original M rows in both|
|M original-source query edges|7741|7741|Protected local route; learned M group access0|

The backward scalar sums mean squares of physical native outputs and includes the compatibility fluid-temperature view; it is not the complete weighted training objective or optimizer macro. No old L_work derivative or actual executor-work derivative runs here. Faster measured decoding can coexist with more original-source edges because shape, grouping and vectorization differ; higher source-edge storage explains a plausible memory pressure but was not isolated as the sole cause. Dense encoder N² remains. Query tiling512 bounds intermediate shapes but does not make192 E-value rows into one physical value.

![Figure6](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/cost_work_tradeoff100.png)

Figure6. Isolated full-native TRAIN0231/M12 cost/work comparison on GPU0. Stage medians, maximum peak allocation and executed E-count ratios expose the tradeoff: J-H has lower latency in this bounded test,24.19× more E source edges and12.08× peak memory. Different trained functions are compared; same-function fast/reference parity is a separate numerical qualification. No whole-training speed or sparse-value-executor claim follows. [PDF master](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/cost_work_tradeoff100.pdf).

## 8. Completed validation, resources and preservation

Initial qualification is kept separate from learning: an original per-query Python reader failed to complete a largest-M forward after about four minutes; its failure log remains. The replacement tiled reader completed actual native forward/backward and strict maintained save/reload. Disposable qualification epochs1–2 visited300 cases/eight Adam updates and did not initialize the scientific runs. Its original three CUDA RNG streams were restored; scientific scratch runs deliberately expose only GPU0 UUID and contain one stream, which future resumes must preserve.

The finite ceiling remains62 GPU-hours: qualification2, paired development24 (12/arm), conditional full-data screens32 (16/selected arm) and scientific validation4. Known scientific training process wall is6.3214 hours with overlapping GPU execution, not measured hardware busy time. Qualification includes222.817s and107.271s maintained epochs,20.8055s high-module forward/backward sizing and earlier failed/forward checks. Native, response, organization, regional, derivative and isolated-cost workers fit within the23:34:43–23:50:32 UTC validation envelope (0.26357 wall-hours, a conservative GPU0 window rather than summed process or busy time); the later context-only diagnostic used CPU. No GPU-hour ceiling or automatic extension was invoked. Device energy/busy utilization was not continuously profiled, and CPU/report preparation time must not be presented as GPU training time. Full-data training/evaluation was not performed because no organizer candidate earned promotion.

Storage was frozen at8GiB retained output,1GiB temporary and200GiB minimum DATA2 free headroom; initial DATA2 free3219533967360B. Existing TRAIN/reference stores and DATA2 symlinks were reused without dataset copies. Both scientific runs retain all100 history rows, normal validation history, loss figures at10, their declared100 monitoring checkpoint and latest/best resumable states; qualification states/failure evidence and all numerical scientific outputs remain. Generated data/figures and one-time renderers stay ignored on DATA2, outside Git. Final byte accounting and the cleanup receipt are recorded below after exact allowlisted cleanup.

Final cleanup removed18 superseded response PNGs (4098431 payload bytes), the completed final pytest tree (572 regular files,446 directories,35 symlinks;16105916 payload bytes), and two earlier completed pytest trees (44 regular files and six internal symlinks;1042370 payload bytes), totaling21246717B in these closeout passes. Earlier documented round cleanup removed27156303B; the cumulative accounted removal is48403020B. No scientific checkpoint, numerical response array, dataset or one-time source helper was removed. Retained payloads are63026286B for J-D,63122984B for J-H,80007266B for qualification, and72963800B for review100 at the cleanup snapshot; the latter includes six selected PDF/PNG pairs totaling2152698B and raw response summaries/arrays totaling26699980B. Final DATA2 free space at that snapshot is3218759266304B (about2.93TiB), comfortably above the200GiB floor. Other live jobs also write to DATA2, so capacity change is not an exclusive attribution of this campaign's storage.

The [exact cleanup receipt](../../diagnostics/round12_2_execution_20261024/contextual_frontier_round_cleanup_20261024.json) records ownership/inodes/links, removed-file hashes, absence of active handles,37 protected-file postchecks in the final addendum, prior receipts and before/after capacity. Report hashes in that receipt are dated cleanup-preflight snapshots; the subsequent final byte-accounting prose is an authorized report revision, not a claim of report immutability. Original response JSON plot catalogs are preserved as historical records; their18 preliminary raster entries are explicitly retired to the selected response figure in that receipt rather than silently rewritten. The final report embeds only retained figures. All scoped temporary trees are absent after cleanup; numerical/provenance evidence, histories, qualification failures and declared checkpoints remain.

Affected implementation tests cover legal frontier/exact cover, contextual/control dependence, bank inventories/zero VJPs/calibration, native module/receiver identities and masks, query chunk/reference parity, fixed-state intervention/restoration, original source derivatives, strict checkpoint/RNG recovery, OOM rollback and successful-retry history accounting. The final affected runner/organization/training/progress suite passes96 tests in49.49s. Earlier model/forest/response boundary tests and native executions remain qualification evidence; software tests do not substitute for the scientific failures above. Ruff finds34 existing findings across the five affected Python files, exactly matching HEAD's per-rule/message baseline with no new findings; this is not a clean whole-file lint claim. Two small durable corrections accompany the review: conservative E-only curl-contributor masks/nonfinite metric handling, and successful-retry packing/mask bookkeeping. Neither changes the saved100-epoch weights or rewrites their history.

Unrelated working-tree changes and the user's later U-H report commits are preserved. The original untracked Round12 work plan is not added. No historical run, migration backup, physical-reference array or protected GPU job is deleted or modified. Delivery audits the entire outgoing commit range and uses the existing repository pre-push gate; local/remote tip verification is part of closeout.

## 9. What to carry forward, and what remains unproved

The completed screen establishes a reusable generic joint-field/contextual-frontier implementation and a clear failure mode: a small organizer can retain a large implicit original-record read. It does not establish selective redundancy, useful local E-only indexing, recovered geometry/mixed physics, mature universal field accuracy, physical causality, inverse candidate quality, compressed value execution or an end-to-end speedup. The correct number of promoted full-data candidates is zero. Continuing the identical objective just to improve scalar field loss would not answer why the learned structure selects roots and leaves M group routing unused.

A focused future amendment should first make the tradeoff between dictionary size and query/source work explicit, preserving this exact matched control and the bounded physical board. For a fixed candidate tree, let $p_v$ be its stop probability and $s_{viq}$ an input-geometric indicator that stopping at v could expose original source i to q. The probability of no potential exposure can be computed recursively:

$$
F_{\mathrm{leaf},iq}=1-s_{\mathrm{leaf},iq},\qquad F_{v,iq}=p_v(1-s_{viq})+(1-p_v)F_{l,iq}F_{r,iq},\qquad \overline W=\frac{1}{QN}\sum_{q,i}(1-F_{\mathrm{root},iq}).
$$

This deduplicates potential ancestry/halo coverage under the declared independent legal-frontier policy, with candidate topology and geometric supports held fixed for the derivative. If learned query routes are assumed open, it is a structural upper-bound proxy, not actual executed source work. Fixed protected M reads must be accounted separately; penalizing an inactive M group route can otherwise be meaningless. Replacing expected K by this term alone may simply favor leaves, so a future controlled lineage must declare both dictionary and source-work tradeoffs, calibrate only on TRAIN, retain gradient caps, and verify actual edges and field/response utility. This equation is a proposed diagnostic/design direction, not implemented or fitted evidence from this campaign.

The second independent issue is case-owned physical supervision: broader joint coupling should coexist with evidence that the simplified Thermal zero-heat/heat-independence laws are learned where applicable. Additional input/response coverage or explicit case-owned constraints may help, but the present screen tests neither remedy. Universal hard-coded affinity, VAL-tuned thresholds, forced K, silent changes to original source identities and broad architecture searches are not justified by these results. Contextual candidate nomination itself has discrete seams and bounded capacity; smooth read windows alone cannot eliminate them.

No further run is left waiting to launch. Both100-epoch states remain resumable under their original namespace, optimizer, absolute schedules and dataset binding if a later user decision authorizes a specific continuation. This closeout answers the current organizer screen with a measured negative and preserves the useful joint predictor/reader infrastructure for a focused next revision.

## 10. Short figure and evidence index

| Figure | Scope | Retained master |
| --- | --- | --- |
|1|All-nine DEV22 preservation|[PDF](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/comparison_native_priority_ratios100.pdf)|
|2|Native fields and signed residuals|[PDF](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/native_fields_residuals100.pdf)|
|3|Cut-free learning and K trajectory|[PDF](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/comparison_training_k100.pdf)|
|4|Actual ancestry/C/query reads|[PDF](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/organization_actual100.pdf)|
|5|Physical responses and failures|[PDF](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/comparison_response_fidelity100.pdf)|
|6|Fidelity/work interpretation and costs|[PDF](../../diagnostics/round12_2_execution_20261024/review100/selected_figures/cost_work_tradeoff100.pdf)|

The six Python figures above have visually inspected PDF masters and necessary PNG companions for Markdown. Numerical sources are [J-D native](../../diagnostics/round12_2_execution_20261024/review100/jd_native100/result.json), [J-H native](../../diagnostics/round12_2_execution_20261024/review100/jh_native100/result.json), [primary response preservation](../../diagnostics/round12_2_execution_20261024/review100/primary_response_preservation100.json), [J-D physical responses](../../diagnostics/round12_2_execution_20261024/review100/jd_responses100/response_summary.json), [J-H physical responses](../../diagnostics/round12_2_execution_20261024/review100/jh_responses100/response_summary.json), [actual organizer receipt](../../diagnostics/round12_2_execution_20261024/review100/jh_organization100/receipt.json), [regional quality](../../diagnostics/round12_2_execution_20261024/review100/regional_quality100.json), [J-D costs](../../diagnostics/round12_2_execution_20261024/review100/jd_cost100.json), [J-H costs](../../diagnostics/round12_2_execution_20261024/review100/jh_cost100.json), [J-D derivatives/source law](../../diagnostics/round12_2_execution_20261024/review100/jd_derivative_source100.json), [J-H derivatives/source law](../../diagnostics/round12_2_execution_20261024/review100/jh_derivative_source100.json), [J-D precision qualification](../../diagnostics/round12_2_execution_20261024/review100/jd_derivative_precision100.json), [J-H precision qualification](../../diagnostics/round12_2_execution_20261024/review100/jh_derivative_precision100.json) and the [compact comparison/trajectory export](../../diagnostics/round12_2_execution_20261024/review100/contextual_frontier_100_comparison.json). Historical Round12 interpretation remains in the [consolidated report](HONF_Round12_Consolidated_Scientific_Report.md). These DATA2-backed local artifacts are retained evidence, not uploaded files.
