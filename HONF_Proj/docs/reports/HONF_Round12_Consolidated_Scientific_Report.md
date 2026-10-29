# Round 12 consolidated report: compact retrieval recovers fields and heat responses, while geometry decisions remain unreliable

Round 12 first introduced a separate retrieval-group bank over the existing Thermal U-H model. The original learning objective produced no merges; a targeted stop-gradient amendment then halved the E bank, initially damaged field quality, and recovered useful prediction and heat-response quality after further training. The same explicit amended local 100 checkpoint now supplies the central evidence: a genuinely smaller reusable bank, passing global native-field preservation bands, and useful original-source heat responses. It does not establish selective redundancy, reliable mixed geometry responses, dependable design ranking or an end-to-end speedup.

On 24 October 2026, this consolidated report became the only top-level report. The four phase reports are retained in the [dated Round12 archive](_bk/20261023_Round12_Compact_Hypergraph_Thermal/); the [historical reading map](_bk/20261024_Report_Navigation/README.md), [document chronology](_bk/20261024_Report_Navigation/DOCUMENT_CHRONOLOGY.md) and [23 October archive review](_bk/20261024_Report_Navigation/DOCUMENT_ARCHIVE_REVIEW_20261023.md) remain available with their historical evidence and status boundaries preserved. Existing figures, numerical evidence and checkpoints retain their locations.

This report consolidates the work plan and all four Round 12 reports into one scientific interpretation. Historical terminal, selected-best and interim results remain distinct; later completed measurements supersede earlier “not yet evaluated” statements only for the checkpoints actually measured. The new scratch1000 run is a separate experiment and contributes no results here; its logs, GPU usage, epochs and checkpoints were not inspected for this synthesis.

The separately authorized scratch revision now preserves v1 through a maintained clean stop at epoch467, including latest resumable state and all467 history rows. The new [soft-only v2 recipe](../guides/_bk/20261029_Guide_Consolidation/03_R12_R13_Predecessor_Recipes/Thermal_Compact_SoftOnly_Scratch1000_Recipe.md) keeps K=N for all1000 epochs, trains policy and predictor jointly, uses bankwise TRAIN-only calibration and bounded existing-reference geometry supervision, and reserves terminal/selected-best validation inside the run. Its [qualification evidence](../../diagnostics/thermal_compact_softonly_full1000_scratch_v2_preparation/qualification_final/qualification.json) is software/budget evidence only:96 input-selected TRAIN cases cover every available module-count category,252 M candidates and9216 E candidates; three no-step calibrations and four disposable macros pass strict recovery, with28,432.85MiB peak allocation and a25.72-hour conservative training projection. No v2 prediction, functional-grouping, geometry-response, inverse-design, compression or speed conclusion follows from that qualification. The earlier scratch1000 statements above describe the synthesis date; the v1 stop and v2 recipe are a distinct continuation.

A separate 24 October organizer visualization supplement in Section6 compares explicit U-H and Run1404 epoch2500 checkpoints, with Run1404 epoch5000 as a separate mature reference. It adds input-selected scene/query evidence for the native hypergraph underlying the compact-retrieval question. It does not replace the U-H epoch1000 adaptation parent, evaluate current training progress, or supply a new compact100/v2 field, response or decision result.

The subsequently authorized [Round12-2 contextual joint-field development](HONF_Round12-2_Contextual_Organization_Development_Report.md) completed a separately initialized, matched TRAIN150/exposed DEV22 first100 screen. It changes the input/field law as well as organization and therefore uses its own J-D control. J-H passes all four global priority field-preservation bands but selects one root per bank in all four detailed scenes, leaves learned M-group access inactive and opens all192 E originals at every positive E query; functional-sharing, geometry/mixed-response and source-law failures remain. Its measured decision is zero full-data candidates and no unchanged continuation. These results do not revise the explicit compact100 results below or supply new current-weight evidence for U-H/soft_only v2.

## Conclusions and decisions

| Area | Measured conclusion | Decision |
| --- | --- | --- |
| Predictor | Compact100 passes all four global temperature/peak mean≤1.10R0 and p90≤1.15R0 bands; surface/material means improve 6.28%/8.90%. M3 peak p90 worsens 15.25%. | Use explicit compact100 as the primary compact-model result; preserve stratum/case failures and exposed-validation scope. |
| Organizer | E192→96; VAL89 C cells fall 48.51%; original physical records and source columns remain. Every eligible E sibling is accepted and every M sibling rejected. | Accept executed organizational compression; withhold selective-redundancy and layout-adaptive E-topology claims. |
| Physical responses | All eight heat controls have thermal relative-L2 below 1; geometry medians are useful but temperature tails and mixed responses remain weak. All primary geometry/mixed absolute-error preservation bands pass versus R0. | Allow the bounded fixed-layout heat-response/reuse conclusion; withhold general geometry-interaction fidelity. |
| Inverse / decisions | Compact100 objective Spearman−0.257143, feasible regret 0.524815; best90 predicts no feasible candidate. | Do not use these results to claim reliable design selection or a new verified design. |
| Executor / cost | Dense still executes 100% of original value rows. Segment saves 8.95% sampled-macro peak allocated memory but costs 22.72% more complete macro time. | Retain dense as the measured training choice; report segment as a memory/time tradeoff, with no compact-versus-R0 speed claim. |
| Campaign status | Original comparison, diagnosis, amendment pilot and matched recovery/validation are completed within their stated scopes. | Close the bounded adaptation study with a partial scientific success and explicit negative findings; do not train until desired rankings appear. |

## 1. What the plan intended to test

The [work plan](../../UpgradePlan/HONF_Round12_Compact_Hypergraph_Thermal_Work_Plan.md) selected U-H Level R only: retain the applied native encoder and learn a directly constructed, reusable retrieval bank smaller than the original M/E inventory while preserving useful physical predictions and responses. U-S e1000 was a frozen reference, not a second amendment. Level U factorization, Wind/HET-MF, new datasets/solvers, locked TEST access, forced merge quotas and broad inverse search were outside this experiment. The central question requires all three properties on one checkpoint: actual K<N, useful native fields, and useful current-weight original-record responses.

For N=M+E physical records and K retrieval controls, the executed architecture uses the following distinct axes:

$$
C(X)\in[0,1]^{K\times N},\qquad \pi(q,X)=\operatorname{entmax}_{1.5}(s(q,K)),\qquad a(q,X)=\pi(q,X)C(X).
$$

$$
g_i^M=\ell_i+(1-\ell_i)a_i^M,\qquad g_j^E=a_j^E,\qquad T(q;X,h)=\sum_{i=1}^{M}\mathcal K_i(q;X)h_i.
$$

C and group descriptors control retrieval; they do not replace individual original-record nonlinear values, donor coordinates/measures or the M physical-source columns. The thermal coefficient matrix denoted by $\mathcal K$ is distinct from the retrieval-group count K. Two MM→EM→EE→ME native contextual cycles and their N² work remain. Level R computes a new C from final prepared states, rather than literally reusing the encoder incidence. It directly scores Q×K and avoids the retired N×N retrieval directory/old query-seed logits; legitimate E and post-E M information-reader Q×N scores remain. These boundaries are implemented in [compact.py](https://github.com/cosmos2w/ModularDT/blob/7981c173e8b1d29ee2b3addb7b972d0f15003542/HONF_Proj/src/honf_forward_core/continuous_joint/compact.py), [unified.py](../../src/honf_forward_core/continuous_joint/unified.py) and [model.py](../../src/honf_forward_core/continuous_joint/model.py).

Same-type, lexicographically adjacent singleton siblings are the only eligible ancestors; odd records carry and exact-coordinate tie bundles remain unmerged. With P accepted pairs, K=N−P and K_X≥ceil(N_X/2). For M12/E192, the attainable lower bound is 102, not an arbitrary small quota; the observed 108 retains all 12 M groups and 96 E groups. Positive original-record support is the union of donors reached by positive group mass, so fewer groups need not reduce final value support. The plan deliberately kept dense execution for learning and treated cost as a separate measurement.

Native preservation uses equal-case mean≤1.10R0 and p90≤1.15R0 separately for fluid, surface and material temperature and recomputed module peaks. Current-response preservation was declared before inspecting new candidate results: median absolute response-error RMSE≤1.10R0 and p90≤1.15R0 for each temperature/q component, separately on 32 geometry and 8 mixed differences. These are descriptive research bands, not physical-validity or significance guarantees. Useful response magnitude is assessed separately through per-response relative L2 and the fraction below 1; near-null physical responses use absolute errors. The [prespecified continuation protocol](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/assessment/prespecified_protocol.json) records this distinction.

## 2. Which experiments and checkpoints the evidence belongs to

| State | Initialization / horizon | Actual organization | What it can establish |
| --- | --- | --- | --- |
| U-H/U-S e1000 | Literal existing parent checkpoints; frozen references | Native parent retrieval | Predictor/response/decision references with different architecture/training history from compact adaptation. |
| Common 20 | U-H e1000 weights, inherited normalization/controllers; fresh AdamW moments | K=N Level-R adaptation | One shared 20-epoch prefix, executed once. |
| R0 local 100 | Common 20→singleton control through 100 | K=N | Matched compression control and its selected-best 100. |
| Original R1 local 100 | Original pD tie loss; strict cut after 40; recovery to 100 | Zero accepted pairs, K=N | Negative terminal learning result, not destructive compression. |
| Original R1 best80 | Existing best-field selection from the original run | Still K=N | Comparable selected singleton predictor; no best80 response/ranking campaign. |
| Amended local 41 | Same common20; recalibrated stopped-outer-p tie;20 soft epochs+one recovery epoch | E192→96; M unchanged | Actual compact feasibility with unrecovered native fields. Missing response/ranking/cache results were subsequently completed at this same 41. |
| Amended local 50 | Exact continuation of 41;9 more epochs | Same frozen cut | Recovered interim fields/responses; comparison with R0-100 has unequal horizon. |
| Amended local 100 | Exact continuation through matched 100 | Same compact E bank | Primary same-checkpoint fields, response, structure, source/cache, geometry, decisions and cost. |
| Amended best90 | Separately evaluated existing best-field alias | Compact | Selected-best exposed analysis; does not replace terminal 100 or provide a feasible recommendation. |
| Scratch1000 | Separately authorized fresh initialization,1000-epoch recipe | No result imported | Future comparison only; this synthesis makes no completion or accuracy claim. |

The inherited U-H source is `thermal_u_h_full5000_v1/checkpoints/epoch_1000_model.pt`, with SHA256 `2efd008f4135c7e5c7d909e0174b31e6ad5b05a058c0e68914ffc727d29134d7`. Common and child runs preserve TRAIN600, canonical exposed VAL89 (excluding 0273), normalization key `7ab3441ba2616c68224f9d829a92e2db4c59658b45d3f1acf4c535333896939b`, seed 0, Q1024, effective 48/micro 24, native auxiliary/response laws and case/query schedule. The valid VAL strata are M3/M5/M7/M10 with 24/25/25/15 cases; low/high-M1/M2/M12 probes are TRAIN evidence. No independent TEST inference follows from these exposed panels.

The adaptation uses the serialized formal_joint AdamW contract: constant LR3e-4, WD1e-5, betas(0.9,0.999), epsilon 1e-8 and clip 1.0; the old provider-summary WD1e-4 is not the executed contract. Each complete epoch visits 600 cases through 12 macros of 48 and one of 24, giving 13 updates and 25 microbatches. Source scientific age 1000 is distinct from controller coordinate 4000. Forking common20 copies weights, Adam/controller/RNG/sampler/history state; only the declared arm differs. All 100 paired R0/R1 schedules match, and the amended continuation preserves those keys.

| State | Sampler epoch | Controller epoch | Macro clock | Scientific work since source, original comparison |
| --- | --- | --- | --- | --- |
| Source U-H e1000 | 1000 | 4000 | 13000 | Inherited source |
| Shared common20 | 1020 | 4020 | 13260 | 260 updates; 12,000 case-visits |
| Each local 40 stop | 1040 | 4040 | 13520 | 520 cumulative updates per endpoint |
| Each local 100 endpoint | 1100 | 4100 | 14300 | 1,300 cumulative updates; 60,000 case-visits per endpoint |
| Unique executed campaign | — | — | — | 260 + 1,040 + 1,040 = 2,340 updates; 108,000 visits |

The primary compact checkpoint is [literal amended epoch_0100_model.pt](../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/runs/R1_tie_stop/checkpoints/epoch_0100_model.pt) (24,035,697 bytes; SHA256 `b37813e616bc09758e023af45f3ddfe05a9e4b36198ac04b2b7b928f07c41267`). R0-100 has SHA256 `4fcabf766b611f3d3fac2305f0a9ef8250511fdaeb6af10b60c5466ae1d3db2d`; amended best90 has SHA256 `cdf54c7080e6a81440bfd3d9fcdcfb80120f79dfcf222569d7e6d1162c3669d0`. No field, response or cost measurement is borrowed across these states. [Terminal assessment](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/assessment/R1_tie_stop_e100_complete_assessment.json) and [best90 assessment](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/assessment/R1_tie_stop_best_e090_complete_assessment.json) retain their source bindings.

| Executed stage | New optimizer updates | New case visits | New primary queries | Summed train-loop seconds |
| --- | --- | --- | --- | --- |
| Original common20+R0/R1→100 | 2340 | 108,000 | 110,592,000 | 7826.942 |
| Amendment 21–41 | 273 | 12,600 | 12,902,400 | 924.269 |
| Amended42–100 | 767 | 35,400 | 36,249,600 | 2602.052 |
| Unique new adaptation work, all stages | 3380 | 156,000 | 159,744,000 | 11353.263 (~3.154 worker hours) |

The unique-work sum excludes frozen-parent training, evaluation, no-update calibration and disposable software qualifications, and counts common20 once; the amended R0 bootstrap at 20 adds zero training. A compact100 endpoint itself has 100 local epochs:1300 exposure updates,60000 visits and 61440000 primary queries, plus 102400 response queries. Its U-H ancestry contributes 600000 visits/13000 exposure updates/614400000 primary queries, yielding 660000 visits/14300 exposure updates/675840000 primary and 1126400 response queries. Adam was reset at adaptation initialization, so 14300 exposure updates do not mean inherited 14300-step moments. The continuation process sessions consumed 2650.046 s including loading/reviews, versus 2602.052 s train-loop time.

## 3. Why the original policy failed, and what the amendment changed

The first experiment did execute candidate generation, policy optimization, strict thresholding and recovery correctly on the tested paths, but learned rejection. A geometry-only TRAIN600+VAL89 survey finds 67963 eligible candidates: TRAIN M1595/E57600 and VAL M224/E8544. Common 20 logits are exactly−2 (p=0.1192); original pre-cut 40 logits all lie between−6.3832 and−4.9157 (median−5.4528), with mean p about 0.00423 M/0.00437 E. Two actual native TRAIN preparations match survey logits/topology exactly. The strict b>0 rule therefore accepts zero; no shortage of candidates or threshold reset explains the result. All eight policy parameters receive 260 Adam steps during 21–40 and remain frozen thereafter, while other parameters reach 1300.

For one sibling pair, p=sigmoid(b) and D(p) is the equal-donor-type discrepancy of the live blended rows. Omitting the outer equal-case reduction, the original structural derivative is:

$$
\frac{\partial L_{struct}}{\partial b}=\alpha p(1-p)\left[\frac{\lambda_{tie}}{P}\left(D+p\frac{\partial D}{\partial p}\right)-\frac{\lambda_{cap}}{N}\right].
$$

The explicit positive D term lets descent reduce the tie loss by lowering p rather than learning similar functions. Original coefficients are lambda_tie = 0.178774665 and lambda_cap = 9.9890278e-5. Ignoring row-change derivatives, mismatch above roughly 2.8e-4 favors rejection; the two representatives have mismatch medians 0.0042–0.0728. Calibration scales different participating parameter sets, so labeling both terms “2% of native gradient” does not balance their policy-head effects. Their scaled norms are 6.0129e-4 versus 9.7825e-6. The next 48 TRAIN calibration cases are 45 M1, one M2 and two M3: only 3 M pairs versus 4608 E pairs, with no M4–M12 coverage.

| Alpha | TRAIN case/type | Pairs | Explicit rejection / capacity gradient L2 | Current dL/db >0 | Proposed dL/db <0 |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0006/E | 96 | 19.6× | 96 | 96 |
| 0.05 | 0254/M | 6 | 279.2× | 6 | 6 |
| 0.05 | 0254/E | 96 | 326.5× | 96 | 96 |
| 1.0 | 0006/E | 96 | 17.2× | 96 | 96 |
| 1.0 | 0254/M | 6 | 258.2× | 6 | 6 |
| 1.0 | 0254/E | 96 | 296.6× | 96 | 96 |

All 198 tested pair-logit derivatives encourage rejection under the original structural objective at both alpha values. The explicit rejection/capacity gradient ratio is 17–327×. TRAIN expected reduction falls 5.88%→0.239% of N between 21 and 40, while the capacity surrogate rises 0.9412→0.9976. These derivatives omit native/work competition and are not a complete optimizer replay. Later diagnostics measure that competition rather than asserting this is the only cause of every policy outcome.

The separate terminal accuracy failure is nonmonotone optimization, not evidence of a destructive merge: no groups merged. Original R1 validation scores recover at 70–80 and deteriorate by 100; its best80 fields remain comparable to R0-best 100. Epoch sums of 13 pre-clip gradient norms reach 18.57 at 98 and 12.56 at 100, versus R0’s 3.53/2.06. These are sums, not largest individual-update norms. Same-weight singleton/repeated-prepare checks give C variation about 1.2–1.3e-6 and TRAIN0254 fluid-T difference RMSE about 1.1e-5 in both modes, consistent with retained FP32 reduction variability, not a hidden cut switch. No replay identifies the unique trigger of the long trajectory.

| Local epoch | R0 exposed VAL field score | R1 exposed VAL field score | R1/R0 |
| --- | --- | --- | --- |
| 30 | 0.00218874 | 0.00224755 | 1.027 |
| 40 | 0.00214337 | 0.00231978 | 1.082 |
| 60 | 0.00229347 | 0.00471532 | 2.056 |
| 70 | 0.00250762 | 0.00216758 | 0.864 |
| 80 | 0.00217806 | 0.00206739 | 0.949 |
| 90 | 0.00223762 | 0.00243188 | 1.087 |
| 100 | 0.00214090 | 0.00506705 | 2.367 |

![Original Round12 loss escape, capacity collapse, selected best and terminal instability](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/round12_diagnosis_v2.png)

Figure 1. Preserved original-run evidence: standardized exposed monitoring scores, selected-best/native-field comparison, TRAIN capacity surrogate and no-step structural policy derivatives. Original best80 predicts well but has K=N. The stopped-weight bars are a no-update proposal diagnostic here; the subsequent amended training supplies the actual compaction evidence. [PDF master](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/round12_diagnosis_v2.pdf).

The implemented amendment uses stopgrad(p)D(p). It removes the explicit p′D shortcut while retaining pD′ through the live latent/geometry blend; capacity still pushes toward acceptance. At fixed weights, old/new scalar tie losses and all tested native field, routing, access and source-factor arrays are bitwise identical on common20 and original 100 TRAIN0006/0254. Recalibrating under the same TRAIN-only procedure gives lambda_tie = 0.193474576 and lambda_cap = 9.98895787e-5. The architecture, source law, strict b>0 rule, initialization and 20/40 phases remain fixed. Focused 18 tests plus 59 integration checks pass; substituting the original formula makes the intended regression test fail with logit derivative 0.1321. Software evidence establishes the backward change, not successful physical accuracy by itself.

## 4. A real cut happened at 40; one recovery epoch was insufficient

The amended pilot adds 273 updates over 21–41 and achieves E contraction. TRAIN mean logits/probabilities are+2.47617/0.91966 for E and−3.36168/0.03353 for M. The geometry-derived TRAIN totals below agree with native low/high-M cut boards; complete VAL89 native prepares independently show 96 E groups and K99–106. Geometry-derived totals and native executed counts are not interchangeable evidence.

| TRAIN600 bank | Original records N | Eligible pairs P | Accepted pairs after cut | Groups K after cut |
| --- | --- | --- | --- | --- |
| M | 3,417 | 1,595 | 0 | 3,417 |
| E | 115,200 | 57,600 | 57,600 | 57,600 |
| Total | 118,617 | 59,195 | 57,600 | 61,017 |

TRAIN K/N is 0.51440. Fixed TRAIN0006/M1 changes 193→97 groups and 0254/M12 changes 204→108. The expected reduction surrogate reaches 43.69% of N before the cut, distinct from allocated K. Saved pre/post 40 checkpoints differ only in cut/freeze buffers, with exact policy weights, optimizer, RNG, history and clocks. At 41 all eight logits/policy tensors remain equal to 40 while 81 other compact-retrieval tensors change; policy Adam step remains 260 and other stepped parameters reach 533. Sampler/controller/joint-macro clocks are 1041/4041/13533.

Unlike the original zero-merge boundary, the amended cut materially changes the represented operator: fluid-T prediction-difference RMSE is 0.0299633 on TRAIN0006 and 0.255963 (max 1.12518) on TRAIN0254. On 0254 the native fluid-T error changes 1.43897→1.29229 at the cut and falls to 0.557453 after 41. The unchanged target/physical mask has 7318 valid exterior receivers in that pilot panel; this is separate from the later derivative probe’s fixed receiver intersection. The original zero-merge six-scene maxima were only 0.0026446/0.0029854/0.0026100/0.014565 for fluid/surface/material T/q, with corresponding source-coefficient differences 0.000391/0.000808/0.000626/0.00353. Neither boundary was an exact entmax mass-preserving merge.

![Actual amended cut effect and one-epoch TRAIN0254 recovery](../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/tie_stop_native_fields.png)

Figure 2. Fixed TRAIN0254/M12 target and physical receivers; predictions share a temperature scale, residuals share a signed scale and cut change uses its own labeled scale. Error improves after 41 but this single TRAIN geometry cannot establish population recovery. The complete local 41 VAL89 board fails all four priority temperature/peak bands. [PDF master](../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/tie_stop_native_fields.pdf).

Amended monitoring scores are 0.00228624 at 30,0.00672148 before 40 and 0.00332416 at 41. Full native fields recover by 50; its comparison with R0-100 is an interim unequal-horizon review. The subsequent same-checkpoint 50 response replay also passes the declared primary absolute-error bands, but mixed thermal response medians remain 1.6431/1.3527/1.2963/3.4723. Recovery then reaches the matched 100 endpoint without altering the frozen policy, LR, controls or physical objectives. Terminal/best90 monitoring scores are 0.0021045002/0.0020408222; sampled standardized scores do not replace native field statistics.

A bounded contingency allowed matched 150/200 reviews and at most one paired 50-epoch LR-only ablation, with ceilings 4667 new scientific updates,8 training+5 evaluation+1 cost/gradient worker hours and 12 GiB new storage. Those extensions/ablations were not used: the 100 endpoint resolves native recovery, while continued training until mixed responses/ranking pass would change the scientific stopping logic. Local 72–74 gradient excursions settled; epoch sums at 98/100 are 2.201/2.404 instead of the original R1’s 18.571 at 98. The [endpoint decision](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/assessment/recovery_endpoint_decision.json) records this measured stop.

## 5. Native predictor quality at matched 100

All nine quantities use equal-case RMSE in their own packed native units, with no invented SI conversion or cross-unit aggregate. Full native evaluation uses the 128×64 fluid grid with physical masks,64 surface/outside receivers per present module, all stored material points and recomputed valid module peaks. q remains the inherited proxy rather than independently validated physical flux. The primary three-state comparison below preserves the original negative endpoint alongside the amended result; references and interim/selected states appear in Appendix A and Figure 3.

| Native quantity | R0-100 mean / p90 | Original R1-100 mean / p90 | Compact100 mean / p90 | Compact100 worst (ID) | Compact100/R0 mean / p90 |
| --- | --- | --- | --- | --- | --- |
| u | 0.0105283 / 0.0157443 | 0.0137592 / 0.0166657 | 0.010211 / 0.0157795 | 0.0258836 (0293) | 0.9699 / 1.0022 |
| v | 0.00153183 / 0.00205454 | 0.0018379 / 0.00236064 | 0.0015667 / 0.00204027 | 0.00267901 (0284) | 1.0228 / 0.9931 |
| p | 0.00575354 / 0.00895573 | 0.00700722 / 0.00952976 | 0.00578046 / 0.00878279 | 0.0156833 (0285) | 1.0047 / 0.9807 |
| omega | 0.0275909 / 0.0402663 | 0.0310708 / 0.0424439 | 0.0270391 / 0.0391817 | 0.0492255 (0297) | 0.9800 / 0.9731 |
| Fluid T | 0.365555 / 0.529687 | 0.564668 / 0.816736 | 0.370443 / 0.489441 | 0.859091 (0297) | 1.0134 / 0.9240 |
| Surface T | 0.316832 / 0.621592 | 0.626164 / 1.04987 | 0.296921 / 0.587253 | 1.53124 (0297) | 0.9372 / 0.9448 |
| Material T | 0.343554 / 0.657756 | 0.704775 / 1.09838 | 0.31297 / 0.615878 | 1.84383 (0297) | 0.9110 / 0.9363 |
| q proxy | 1.21355 / 2.17976 | 1.47613 / 2.34926 | 1.21658 / 2.2692 | 4.90682 (0297) | 1.0025 / 1.0410 |
| Module peak | 0.31949 / 0.727995 | 0.680675 / 1.12233 | 0.320293 / 0.679361 | 2.09899 (0297) | 1.0025 / 0.9332 |

Original R1 terminal 100 fails all four priority mean/p90 bands and every M-stratum mean worsens on all nine fields. Its priority mean ratios are 1.545/1.976/2.051/2.131. The amended terminal 100 passes all four global bands: fluid/surface/material/peak mean changes are+1.34%/−6.28%/−8.90%/+0.25%, and p90 changes−7.60%/−5.52%/−6.37%/−6.68%. q p90 rises 4.10%. The amended selected-best90 also passes the four global bands but is an exposed selection result, not the terminal experiment.

![All nine native-field mean and p90 ratios for nine distinct checkpoint states](../../diagnostics/thermal_compact_round12_20261023/consolidated_20261023_v1/figures/round12_all_checkpoint_fields.png)

Figure 3. Python reaggregation of retained completed evidence, full exposed VAL89. Every cell is a same-field ratio to R0-100; no cross-field physical score is formed. Outlined cells miss the priority temperature/peak reporting bands; parent and selected/interim columns retain their separate interpretation. Original best80 remains singleton, while amended41/50/90/100 are compact. Only amended100 is the primary matched compact endpoint; all absolute statistics remain in the tables. [PDF master](../../diagnostics/thermal_compact_round12_20261023/consolidated_20261023_v1/figures/round12_all_checkpoint_fields.pdf).

| Compact100 / R0 ratios | M3 mean / p90 | M5 mean / p90 | M7 mean / p90 | M10 mean / p90 |
| --- | --- | --- | --- | --- |
| u | 0.9985 / 1.0126 | 0.9920 / 1.0096 | 0.9535 / 0.9402 | 0.9256 / 0.9293 |
| v | 1.0396 / 1.0442 | 1.0068 / 0.9329 | 1.0313 / 1.0512 | 1.0125 / 1.0165 |
| p | 1.0422 / 1.0204 | 1.0065 / 1.0216 | 0.9925 / 0.9798 | 0.9806 / 0.9437 |
| omega | 0.9663 / 0.9460 | 0.9831 / 0.9699 | 0.9882 / 0.9947 | 0.9758 / 0.9810 |
| Fluid T | 1.0015 / 0.9793 | 0.9990 / 1.0271 | 1.0151 / 1.0490 | 1.0492 / 1.0658 |
| Surface T | 0.8826 / 0.8578 | 0.9499 / 1.0354 | 0.9517 / 1.0369 | 0.9276 / 0.9315 |
| Material T | 0.8283 / 0.7777 | 0.9347 / 0.9642 | 0.9310 / 1.0915 | 0.9068 / 0.9332 |
| q proxy | 0.9824 / 0.9840 | 0.9920 / 0.9500 | 1.0129 / 1.0549 | 1.0238 / 1.0247 |
| Module peak | 1.0649 / 1.1525 | 1.0642 / 0.9850 | 0.9597 / 1.0467 | 0.9542 / 0.9294 |

The stratum board prevents a global pass from becoming a uniform-accuracy claim: M3 module-peak p90 rises 15.25%, just outside the 15% descriptive band, while its mean rises 6.49%. M10 fluid-T mean/p90 rises 4.92%/6.58%. Native worst IDs and all per-case/stratum values remain linked in the [terminal board](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/evaluation/R1_tie_stop_e100/VAL89_fields/result.json). R0 improves parent U-H means for flow, fluid T and q but not surface/material/peak; U-S remains the stronger parent surface/material reference. Parent-versus-child changes include extra training and a new scorer, so they do not isolate merging.

![Matched R0 and compact100 physical temperature maps and signed residuals on VAL0692](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/02_native_temperature_residuals.png)

Figure 4. Exposed VAL0692/M10 uses identical coordinates, target and physical-fluid mask; white interiors are excluded. Shared prediction/reference and signed residual scales retain native temperature units. Fluid-T RMSE worsens 0.30718→0.34223 despite passing global bands; compact-minus-R0 RMSE is 0.17683. Original R1 on the same case had 0.62306 fluid-T RMSE and 0.012194 u RMSE versus R0’s 0.010178; its separate physical map remains in the historical visual archive. [PDF master](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/02_native_temperature_residuals.pdf).

## 6. What the compact structure represents, and what it does not

| Executed full VAL89 quantity | Singleton R0 / original R1 | Amended compact100 |
| --- | --- | --- |
| Original N / live and allocated K | N=195–202; K=N | N=195–202; K=99–106, without single-case padding |
| M/E live groups | All original M;192 E | All M3/5/7/10;96 E |
| Eligible / accepted VAL siblings | M224/E8544; zero accepted in original R1 | M0/224; E8544/8544 accepted |
| Direct C allocation cells | 3,484,910 | 1,794,350 (−48.51%) |
| Group slots before query expansion | 17,610 | 9,066 |
| Q8192 direct group-score rows | 144,261,120 | 74,268,672 |
| Native contextual pair rows | 6,969,820 | 6,969,820 |
| Retired retrieval gather / old pre-E M seed rows | 0 / 0 | 0 / 0 |
| Retained original M/E information-score rows | 4,276,224 /139,984,896 | 4,276,224 /139,984,896 |
| Actual dense M/E nonlinear value rows | 4,276,224 /139,984,896 | 4,276,224 /139,984,896 (100% of valid candidates) |

TRAIN24 spans original N193–204 and compact K97–108, retaining every original source. Actual TRAIN0254 prepares C[108,204], with 12 M and 192 E original donor columns, and reuses that one bank across fluid, surface, outside and material roles. All physically present original source IDs remain aligned. The live [prepare/directory probe](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/structure_cost/e100_live_compact_directory_probe.json) and [typed C and access probe](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/structure_cost/e100_piC_typed_block_support_probe.json) show direct construction and no retired directory allocation. Legacy compatibility scalar tensors do not constitute an active old directory.

| Positive C memberships | TRAIN0006 M1 | TRAIN0254 M12 |
| --- | --- | --- |
| M groups → M records | 1 | 143 |
| M groups → E records | 192 | 2304 |
| E groups → M records | 96 | 580 |
| E groups → E records | 1789 | 1270 |

The M→E block reaches every E column in both fixed TRAIN scenes. At M1 it alone supports all 1572864 Q×E pairs even though E→E memberships are sparse. At M12 it supports 529536 pairs; final logical support is 71094/98304 M and 604019/1572864 E. Every original donor column is used somewhere. Across VAL89, compact equal-case positive M/E fractions are 75.95%/35.95%, with 3188384/50327207 selected rows; original R0/R1 equal-case fractions were 75.6502%/36.0486% and 75.9406%/36.3517%. Dense executes 100% in all cases. Positive support, allocation, FLOPs, peak memory and complete runtime are different measurements.

![Actual compact memberships, sibling ancestry, receiver-role reuse and original-record access](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/03_compact_structure_and_access.png)

Figure 5. Explicit compact100, TRAIN0254: C[108,204], unchanged original donor columns, merged E ancestry, query-group mass and final gates. Cyan separates M/E axes. Neural stencil samples may lie inside solids and are not physical-error receivers; role-fluid mass includes the native grid, while field metrics use the physical mask. Support bars describe positive logical access; executed dense rows remain 100%. Fixed E ancestry is not physical causality or layout-adaptive clustering. [PDF master](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/03_compact_structure_and_access.pdf).

### The learned rule contracts E but has not demonstrated selective redundancy

The amendment removes a measured reject-all escape, but acceptance of every E sibling is not selective learned functional redundancy. The bank-specific policy uses legitimate input-only symmetric local features, without target errors, heat amplitudes or case/source embeddings. At fixed domain/material/operating inputs, its fixed E grid has no explicit module-layout dependence. Hard E ancestry can therefore remain identical across layouts even while live contextual descriptors, C and query access vary. Preserving all M physical source columns does not mathematically force no M organizational merging; M rejection is an observed policy result.

| Saved state / soft phase | Bank | Eligible / accepted pairs | Native positive / negative logit gradients | Structural positive / negative | Total positive / negative | Mean native L2 | Mean structural L2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 / alpha=0.05 | E | 2112 / 0 | 1037 / 1062 | 0 / 2112 | 812 / 1300 | 5.838e-07 | 2.58e-09 |
| 20 / alpha=0.05 | M | 60 / 0 | 36 / 24 | 0 / 60 | 35 / 25 | 2.497e-07 | 4.357e-10 |
| 40 / alpha=1 | E | 2112 / 2112 | 1015 / 1081 | 0 / 2112 | 367 / 1745 | 1.853e-06 | 6.863e-07 |
| 40 / alpha=1 | M | 60 / 0 | 21 / 39 | 1 / 59 | 20 / 40 | 6.971e-07 | 6.83e-09 |

The signed-gradient panel contains 22 input-stratified TRAIN cases (two lowest IDs per available module-count category plus 0006/0254). Positive dL/db lowers acceptance under descent. At 20/alpha 0.05, structural gradients favor acceptance for all 2112 E/60 M candidates; at 40 they favor acceptance for 2112 E and 59/60 M. Native gradients compete in both directions. Remaining 13/16 E native derivatives at 20/40 are zero. All three exact work-controller hinges are zero on this panel, so its native-plus-work channel equals task-only. At 40 the case-weighted mean structural/native logit-gradient norm ratio is 0.370 E versus 0.0098 M, with the original calibration’s 3 M versus 4608 E pairs and missing high-M coverage. This supports unequal signal/coverage as limitations; it does not prove calibration alone explains historical M rejection or substitute a logit snapshot for Adam/shared-parameter trajectory analysis. Policy tensors remain frozen after 40, with eight Adam steps fixed 260 and 283 active parameters reaching 1300 at 100.

### Group intervention measures executed model dependence

On compact TRAIN0254, the largest role-balanced mean query-mass group is singleton M group 10 (original M ID5). Disabling it changes live K108→107 while allocated K remains 108, then recomputes calibration/entmax normalization. A separate copy disables the largest-used merged E group 32 (original E IDs 5/29, concatenated record indices 17/41), so a genuinely merged group is also tested. Selection uses input/query mass, not reference errors. All original model/preparation hashes, tensors and versions are restored exactly; disabled mass is zero and remaining normalization discrepancy≤2.92e-6 in FP32.

| Disabled group | ΔRMSE fluid T | ΔRMSE surface T | ΔRMSE material T | ΔRMSE q proxy | Final logical M / E rows |
| --- | --- | --- | --- | --- | --- |
| Global M singleton | 0.260612 | 0.111538 | 0.103576 | 0.777044 | 71,184 / 583,027 |
| Merged E pair | 0.0471115 | 0.00811681 | 0.00471439 | 0.0672303 | 73,032 / 604,055 |

Baseline compact logical rows are 71094 M/604019 E; actual dense execution remains 98304 M/1572864 E in both interventions. The original singleton R1 intervention removed M group 10, K204→203 at allocated 204, and changed fluid T by 0.250343 RMSE (max 2.10136), with normalization discrepancy 2.80e-6. The compact intervention is a new same-checkpoint dependence test, not inherited singleton evidence. Changes in the table exceed retained repeat variation, but neither intervention identifies physical causes or proves the accepted pairs are redundant.

### Mature U-H and Run 1404: learned organization on representative layouts

This separately scoped CPU visualization replay makes the native organization concrete before interpreting its compression. It uses `thermal_u_h_full5000_v1/checkpoints/epoch_2500_model.pt` (SHA256 `854b8fddf2b67705f8f4a75235d99db35b6a87ab07a251552a8f379271ed9bff`), Run 1404 `epoch_2500_model.pt` (`8c7fb5522c048f675af726c1dbe57690036e49b0e096d80a726351b359a6d32f`) and its separate mature `epoch_5000_model.pt` (`71a588ab4c2d9bb624f4975f50a0874d575f6ba6687f68e15869806ce512cbef`). These are explicit milestone files, not latest/best aliases. The first two figure rows match nominal age 2500; the third shows 1404 at 5000 and must not be treated as a matched-age U-H comparison. Different architectures, objectives and saved input transforms also prevent an isolated organizer-only attribution. The frozen U-H e1000 source of the Round12 adaptation remains unchanged.

The cases 0274/M3,0658/M5,0293/M7 and0692/M10 reuse the previously input-selected exposed VAL89 representatives. Before inference, 512 shared query IDs are fixed as 16 evenly spaced native grid rows by 32 columns, including endpoints. No target or reconstruction error selects a case, query or displayed group. All 512 positions remain numerical organization probes; right-hand spatial-map means use the 500/492/485/467 positions outside the original circular module geometry, respectively. This geometry mask is not a new full native observation-stencil qualification or a field-accuracy evaluation. The [four-page PDF](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/figures/1404_UH_hypergraph_comparison.pdf) retains one complete page per case, while only the low/high-M examples are embedded below.

The models share a two-stage query-to-group-to-record idea but give its axes different meanings:

$$
\beta_{qi}^{1404}=\sum_{k=1}^{6}\alpha_{qk}\overline A^{MH}_{ik},\qquad c_{\mathrm{pair}}(q)=g_p\sum_i\beta_{qi}^{1404}\psi_\theta(q,i),\qquad c_H(q)=0,
$$

$$
B^{UH}(X)\in[0,1]^{N\times N},\qquad \pi(q)=\operatorname{entmax}_{1.5}(s(q)),\qquad a(q)=\pi(q)B^{UH}(X),\qquad N=M+192.
$$

Run 1404 uses six latent routing groups and retains original query-module values; the replay confirms its direct hyperedge-value context is exactly zero. Its displayed matrix is the raw record-to-group membership A transposed, with each record normalized over six groups. The module membership used in beta additionally normalizes over donors within each group. U-H instead displays its actual calibrated native-seed-to-original-record incidence B, without inventing a row normalization. Its diagonal is imposed retrieval self identity, not learned self-importance. A and B are dimensionless organizational coefficients with different normalizations; equal colors or numerical weights do not identify equal physical strength, matched latent roles or comparable response coefficients.

The two spatial examples for 1404 are its greatest panel-mean query-weight groups. U-H shows the greatest-mean-query-weight M seed and E seed separately, preserving both broad and localized examples beside the complete matrix. Red stars mark the stored 1404 region anchor or U-H native seed position. Module circles retain original IDs and their own membership coefficients; backgrounds show the 192 original E memberships on their 24×8 grid. Every positive-member count uses exact coefficient>0, without a drawing threshold. Query effective groups are exp(-sum p log p); their color scales remain 1–6 for 1404 and 1–34 for U-H. More effective groups is not automatically better organization, and concentration is not proof that the complete scene-wide bank can be compressed to that count.

| Exposed case | M | Outside-module probes /512 | U-H live native groups | 1404 e2500 mean effective groups | U-H e2500 mean effective groups | 1404 e5000 mean effective groups | Positive U-H B entries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0274 |3 |500 |195 |4.488 |14.332 |4.773 |7.38% |
| 0658 |5 |492 |197 |4.311 |13.055 |4.218 |9.09% |
| 0293 |7 |485 |199 |3.643 |13.874 |4.053 |11.97% |
| 0692 |10 |467 |202 |3.921 |12.426 |4.168 |12.11% |

All effective-group means in this table use outside-module probes; [raw evidence](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/evidence.json) separately retains statistics over all 512 neural probes. Positive B fractions include the defined retrieval diagonal and exclude absent/padded records. Every 1404 raw M/E membership and query-to-group softmax entry is positive on these panels. U-H's sparse memberships and selective seed weights therefore establish a different executed organization, while its available group count remains 195–202 rather than a learned compact K. These native-model checkpoints supply no new compression outcome.

![Learned Run 1404 and U-H organization on exposed case0274 with three modules](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/figures/case_0274_comparison.png)

Figure5a. Exposed 0274/M3; rows 1404 e2500, U-H e2500 and separate 1404 e5000. Columns show the complete membership matrix, two representative spatial groups and query effective groups. U-H has 195 live groups and 7.38% positive B entries. Its displayed M:0 group contains all 3 M/192 E records, while E:53 contains 1 M/9 E records. Outside-module query effective-group means are 4.488/14.332/4.773 across the three rows. The coexistence of a broad M-seed group and a localized E-seed group is observed model organization, without a physical-causality or minimum-capacity claim. [PDF master, page1](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/figures/1404_UH_hypergraph_comparison.pdf).

![Learned Run 1404 and U-H organization on exposed case0692 with ten modules](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/figures/case_0692_comparison.png)

Figure5b. Exposed 0692/M10, the same row/column definitions and fixed query construction. U-H retains 202 live groups,12.11% positive B entries and 12.426 mean effective groups outside modules;1404 e2500 uses 3.921 effective groups from its fixed six. The displayed U-H M:3 group contains 10 M/192 E members, whereas E:53 contains 2 M/9 E members. Module identities and coordinates are preserved; group IDs/colors are not aligned between models or checkpoint ages. The complete PDF also retains 0658/M5 and 0293/M7 rather than selecting only these two examples.

The applied-encoder trace confirms that U-H's relation is used during value exchange rather than drawn only after dense encoding. On 0692 at 2500, cycle1 EE has 34,441 positive eligible relations out of 36,864; cycle2 has 1,978/36,864. These are two ordered stages in one fixed-weight forward pass, not a training-time sparsification trajectory. For all four 512-query panels, the actual seed-times-incidence composition, followed by the existing M local envelope, exactly reproduces final positive gate support. Stage-specific source-state versions remain distinct; the two matrices are not simultaneous physical adjacencies.

![Actual U-H applied contextual EE incidences and final composed E access on case0692](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/figures/UH_encoder_and_composed_access.png)

Figure5c. U-H e2500, exposed0692/M10. The first two panels show the actual applied EE incidence in cycles1/2; the last maps the fraction of 192 E records with positive final access. Sparse second-cycle EE relations coexist with areas of full final E access. A positive query coefficient on a broad M-seed E row can open every E record through the nonnegative support union, so sparse encoder blocks or selective seed use do not guarantee sparse composed access. The panel reports logical support and does not measure kernel-row reduction or speed. [PDF master](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/figures/UH_encoder_and_composed_access.pdf).

The design implication is to retain U-H as an actually applied, query-dependent weighted overlapping organization, while treating native group count, final-access expansion and functional sharing as separate questions. This supports Level R's separation of group controls from original physical records; it does not require pooling original values or heat amplitudes. The supplement does not establish necessary/optimal groups, selective mergeability, source-response accuracy, inverse decisions or complete runtime. Original record identities and checkpoint files remain intact. [Extraction and interpretation notes](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/README.md), [derived table](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/visualization_summary.json) and [retained numerical/provenance manifest](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/retained_manifest.json) preserve the exact inputs, arrays and claim boundaries.

### A spatial reading: entry selection and module access are different steps

The following explanatory figures reuse the same saved U-H epoch2500 arrays, with no new model inference, training or physical solves. Here an entry means one native group-control slot associated with an original M or E record. Its coordinate is an address for that control, not the spatial extent of all records linked by its membership row. The count diagrams below arrange dots by entry identity; their apparent dot-to-dot distance has no geometric meaning. Figure 5d instead places selected entries and original modules at their actual physical coordinates so that locality can be inspected directly.

![Actual physical locations of selected E entries and final M reads for three U-H queries](../../diagnostics/uh_entry_sparsity_explainer_20261024/geometry_followup/physical_entry_and_module_access.png)

Figure 5d. U-H epoch2500, exposed VAL 0692/M10, using the same three fixed queries as Figure 5f. A/B/C are the nearest saved outside-module panel probes to input-defined positions (2,1)/(6,3)/(10,5), respectively; their actual coordinates are (1.921875,1.171875)/(6.140625,3.140625)/(9.984375,5.109375). Left: all 192 E-entry locations, with exact-positive query weights encoded by blue marker size/color and zero weights shown in gray; labelled circles are original modules. Right: final positive M-read support, with orange fill and line width encoding the arithmetic mean of the four actual module reader-head coefficients. The queries open 16/25/19 E entries and zero M entries, yet read 2/8/7 original M records. Leading mean module reader weights are M7/M6=94.1%/5.9% at A,M0/M9=61.1%/23.6% at B, and M1/M3=62.2%/19.8% at C. Reader attention is dimensionless and is not a temperature contribution or a verified physical influence. [PDF master](../../diagnostics/uh_entry_sparsity_explainer_20261024/geometry_followup/physical_entry_and_module_access.pdf); [per-query gates, weights and coordinates](../../diagnostics/uh_entry_sparsity_explainer_20261024/geometry_followup/evidence.json).

The principal E-entry weights show spatial concentration: left/lower entries for A, central/upper entries for B and upper/right entries for C. At A, the largest-weight entry E53 lies at (2.75,1.875), about 1.09 coordinate units from qA. Its row includes M6 and M7. Near-entry selection can nevertheless lead to nonlocal original-record access: A also has positive final access to E158 at (7.25,4.875), about 6.49 units away, with final gate 0.65772 and mean E-reader weight 0.02286. Thus a nearby entry and a distant linked record can both participate. The current learned content/geometry scorer does not impose a nearest-neighbor ordering or a hard geometric radius on this composed directory; nonlocal relations require functional/physical validation rather than being accepted or rejected from distance alone.

For a module $i$, the executed path has three distinct quantities: entry weight, composed access and final reader weight. The first two determine its gate through

$$
a_i(q)=\sum_s\pi_s(q)B_{si},\qquad g_i^M(q)=\ell_i(q)+[1-\ell_i(q)]a_i(q).
$$

The smooth local envelope $\ell$ preserves an additional near-module access route. The reader subsequently combines the gate with learned content/geometry scores and physical information measures, after the query has been conditioned by its E read. At A, nearest module M7 is 0.79 units away, with composed access 0.58925, local envelope 0.09799 and final gate 0.62950. M6 is 1.85 units away and has the larger gate 0.92857, yet its mean reader weight is only 5.9%, compared with 94.1% for M7. Gate amplitude therefore does not rank final read strength. Zero query weight on all M entries also does not imply no M information: selected E rows can open M records, nearby modules have their local envelope, and contextual E states already contain information exchanged with M. The legacy thermal head additionally keeps original source-resolved coefficients; neither these reader weights nor the drawn links replace source-response measurements.

### An intuitive count view: many available entries, selective query use

Figure 5e shows three previously input-selected exposed validation layouts with 3/7/10 modules. In each, one query is chosen solely as the nearest saved outside-module probe to (2,1), without looking at routing weights, targets or errors. Each dot represents one actual live entry, after removing absent module padding. Gray is exact zero query weight; blue and orange are positive weights. Orange identifies the smallest descending-weight prefix carrying at least 90% of the query's total entry weight. This prefix is a descriptive concentration statistic, not a tested pruning rule. Dot positions encode entry identity rather than physical position, and entry identities are not aligned across different layouts.

![Simple entry inventory, positive support and weight concentration across three U-H layouts](../../diagnostics/uh_entry_sparsity_explainer_20261024/figures/01_three_layouts.png)

Figure 5e. U-H epoch2500, exposed VAL 0274/M3, 0293/M7 and 0692/M10. Available/open/effective entries for the displayed individual queries are 195/17/9.64, 199/15/9.78 and 202/16/9.94. Their smallest 90%-mass prefixes contain 8/9/9 entries, carrying 91.3%/91.2%/92.8% of the corresponding weights. The left column supplies physical geometry and query location; the middle supplies exact support; the right ranks every open entry weight. Effective count measures weight balance through exp(entropy), so 9.94 is neither a collection of fractional entries nor the number of allocated groups. All values here describe individual queries, not scene averages. [PDF master](../../diagnostics/uh_entry_sparsity_explainer_20261024/figures/01_three_layouts.pdf).

![The same 202-entry bank opens different subsets for three queries in one U-H layout](../../diagnostics/uh_entry_sparsity_explainer_20261024/figures/02_one_layout_three_queries.png)

Figure 5f. U-H epoch2500, exposed VAL 0692/M10, unchanged 202-entry bank. Queries A/B/C open 16/25/19 entries, have 9.94/15.29/9.21 effective entries and require 9/13/9 leading entries to reach at least 90% of weight. Identical dot positions across these three rows identify identical entries, allowing their different subsets to be compared. Their union contains 60 distinct entries; across all 467 outside-module probes, 193 of 202 entries have positive weight at least once. The header's 202 available/23.40 mean open/12.43 mean effective statistics summarize those 467 probes and do not describe any one displayed query. These are exposed validation organization probes, not independent TEST results or new physical-error measurements. [PDF master](../../diagnostics/uh_entry_sparsity_explainer_20261024/figures/02_one_layout_three_queries.pdf); [query IDs, entry identities and complete weights](../../diagnostics/uh_entry_sparsity_explainer_20261024/evidence.json).

### Reusing applied attention relations from preparation to query reading

The organization in Figures 5d–5f is connected to the preceding encoder computation. Original M and E tokens first exchange contextual information through two ordered MM→EM→EE→ME cycles, with destination named first. A learned gate $b$ modulates each stage's actual attention-value reduction. The native U-H directory then reuses the final cycle's applied gates in its corresponding typed blocks, rather than fitting an unrelated retrieval graph after encoding. Updated original-record values and the reusable relation directory remain distinct prepared objects; a query uses both. Figures 5g–5h show this continuity with the same color and record identity across the two uses. Secondary qualifications are supplied in these captions to keep the figures legible.

![Native U-H attention exchanges reuse their applied gates as the query retrieval directory](../../diagnostics/uh_attention_to_retrieval_explainer_20261024/figures/01_attention_to_retrieval_overview_v2.png)

Figure 5g. Native U-H forward-process overview, with schematic token icons and implementation-derived data flow. Orange denotes modules, blue environment entries and green reused relation gates. The upper half prepares the scene: encode own-record geometry/material and prescribed operating inputs, execute two ordered contextual cycles and assemble the final-cycle gates into B, whose rows are native entry addresses and columns are original M/E records. The lower half handles each query: obtain exact-zero-capable entry weights, compose $a=\pi B$, include the M local envelope, read prepared original E values, condition the query and read prepared original M values plus nearby-module detail before field/source-resolved thermal readout. Solid arrows show computation/value flow; dashed arrows show relation reuse. The token icons are illustrative, not the complete 202-record inventory. B reuses applied relation gates, not the full headwise normalized attention weights; same-type retrieval self entries are separately set to 1. Its four blocks come from ordered source-state stages, not a simultaneous physical adjacency. This native model still retains one entry per present record; learned group-count compression is not depicted. [PDF master](../../diagnostics/uh_attention_to_retrieval_explainer_20261024/figures/01_attention_to_retrieval_overview_v2.pdf).

![The exact same measured M6 and M7 gates update E53 and later control qA access to the original modules](../../diagnostics/uh_attention_to_retrieval_explainer_20261024/figures/02_actual_E53_reuse_trace_v2.png)

Figure 5h. Numerical example from the unchanged U-H epoch2500 checkpoint, exposed VAL 0692/M10 and the same qA=(1.921875,1.171875) as Figures 5d/5f. During the last EM stage, M6/M7 messages into E53 are modulated by gates 0.9512439966/0.9341055751. Those exact floating-point values reappear in B[E53,M6/M7]. Query weight $\pi_{E53}=0.2168742716$ contributes 0.2063003489/0.2025834662 to M6/M7 access; the other 15 selected E rows and local envelope produce final M gates 0.9285687804/0.6294999719. All M-entry weights are zero, but original M6/M7 are subsequently read. Their mean four-head reader weights are 5.9%/94.1%, distinct from these gates and from temperature-response contributions. The upper arrows show M→E value exchange; the lower arrows show query→E-row→M-access control. The later M values are read from their own original records, not relayed through E53 or replaced by a pooled E value. Every final-cycle block matches B exactly off its same-type diagonal; the self-identity rule changes 10 valid MM and 192 EE diagonal entries in this scene. Values are drawn from existing saved arrays, with no new inference or physical evaluation. [PDF master](../../diagnostics/uh_attention_to_retrieval_explainer_20261024/figures/02_actual_E53_reuse_trace_v2.pdf); [full-precision evidence and equality checks](../../diagnostics/uh_attention_to_retrieval_explainer_20261024/evidence.json).

The model therefore connects two uses of learned organization: the relations affect how context is built, and those same relation gates later help decide which original values a query reads. This is a concrete continuity between encoder and retrieval. It does not make the relation gates identical to normalized attention coefficients, force an entry's members to be nearby, or establish physical causality. The remaining compression task is to learn when several entry controls can share their function and reduce actual K while preserving useful fields/responses and measuring complete execution. Compact Level R constructs its own live C, so the native B-reuse diagram should not be transferred literally to that later representation. [Figure notes and implementation correspondence](../../diagnostics/uh_attention_to_retrieval_explainer_20261024/README.md) retain these distinctions.

### What these views imply for adaptive compression

Run 1404 and native U-H are intermediate organization designs. Run 1404 demonstrates useful learned routing through a prescribed six-group capacity and exhibits soft spatial specialization; overlapping boundaries are compatible with that role. U-H instead keeps one entry per original record and learns selective, overlapping relations that are actually used. Its broad and local relations show representational flexibility, but the native entry axis still grows with the original inventory. Exact-zero query weights alone do not remove dense directory construction, candidate scoring or value work. For the intended scalable adaptive organizer, learning which entry controls can share functionality and reducing actual live K are therefore central research requirements.

The small active fraction and concentrated weights support investigating more economical access and shared organization. They do not identify a scene-wide minimum K: even these three queries use different subsets, and 193 entries appear somewhere on the fixed 467-probe panel. Panel usage also does not prove those 193 entries are necessary; multiple entries could still implement equivalent functions. Establishing that equivalence requires isolated sharing/merging interventions, recomputed query normalization and physical field/response checks. The separate amended compact100 result in this report already establishes bounded E192→96 compression with recovered global priority fields, while retaining its selectivity, geometry-response and cost limitations. The native e2500 pictures supply neither that result nor a new compression outcome.

The [soft-only v2 recipe](../guides/_bk/20261029_Guide_Consolidation/03_R12_R13_Predecessor_Recipes/Thermal_Compact_SoftOnly_Scratch1000_Recipe.md) tests functional sharing while retaining K=N; these figures import no v2 checkpoint or progress result. The separately documented [Round12-2 contextual adaptive organization](../../UpgradePlan/HONF_Round12-2_Contextual_Adaptive_Hypergraph_Organization_Work_Plan.md) addresses contextual candidates, a learned reversible group frontier and selected original-value reads. Those are design directions, not results of the visualization. Group-count reduction, per-query support, actual value execution and complete runtime remain separate measurements; keeping original physical records does not require keeping a separate retrieval entry for every record.

## 7. Live geometry dependence and derivative limits

| Checkpoint | TRAIN case | AD mean-fluid-T / dx | Represented central FD | Relative AD/FD discrepancy | C shift RMSE | pi shift RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| Original singleton 100 | 0006 | -0.0365042 | -0.0365016 | 0.007113% | 0.00126671 | 0.0000769234 |
| Original singleton 100 | 0254 | -0.165492 | -0.165403 | 0.053827% | 0.00105395 | 0.000165602 |
| Compact100 | 0006 | -0.0389871 | -0.0389976 | 0.0268% | 0.001420 | 0.0000906 |
| Compact100 | 0254 | -0.0979207 | -0.0978510 | 0.0713% | 0.0008342 | 0.0002443 |

The actual compact model uses h 0.0048 on 8117/7315 unchanged exterior-fluid receivers for 0006/0254. Four steps shrink to 0.0006; cancellation prevents uniformly improving FP32 agreement. Its C/pi shift columns above use the separately declared legal translation 0.024, not the derivative step; live control changes are 0.03124/0.01319 and final-access changes 0.001262/0.001454. Original-probe C/pi values retain their own saved legal shift definition and must not be used as matched physical-response comparisons. Every probe rebuilds live original states, group controls and memberships. Candidate ordering, b>0 signs and ancestry remain fixed on both derivative sides; bounded legal scans find no crossing. No hard-cut jump curve was measured, no global C1/no-seam guarantee follows, and model AD/FD consistency is not physical geometry-gradient fidelity. The latter is assessed through saved solver responses below.

![Actual cut and recovery, compact-model geometry AD versus finite differences, and M/E gradient competition](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/04_cut_recovery_and_live_geometry.png)

Figure 6. The top-left sampled standardized monitoring score and summed pre-clip macro norms are not native physical RMSE or individual-update maxima. TRAIN0254 retains 108 groups after the actual 204→108 cut while field quality recovers. The derivative sweep belongs to compact100, and the 0.0098/0.370 M/E competition ratios come from pre-cut 22-TRAIN diagnostics. No original singleton derivative result is transferred to the compressed model. [PDF master](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/04_cut_recovery_and_live_geometry.pdf).

## 8. Physical response accuracy: preservation and usefulness give different answers

All checkpoints replay the unchanged TRAIN physical-reference families 0001/0318/0333/0348:32 geometry first differences (16 single-axis+16 diagonal),8 mixed geometry differences,8 heat-transfer controls and the independent-heater null. Family geometries, heat vectors, physical references, IDs and component-specific masks match R0; surface T and outside q masks are distinct. There are no new solver calls or dataset generation. The independent audit reconstructs 48 differences×8 components from retained arrays, agreeing with RMSE to 1.11e-16 and relative L2 exactly. [The corrected estimand audit](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/assessment/corrected_estimand.json) uses the ratio of RMSE distribution quantiles, Qp(error_compact)/Qp(error_R0); the quantile of paired per-response ratios is a different descriptive statistic and is not substituted for the prespecified bands.

| Absolute response-error RMSE | Geometry R0 median / p90 | Geometry 100 median / p90 | Mixed R0 median / p90 | Mixed 100 median / p90 |
| --- | --- | --- | --- | --- |
| u | 0.00389413 / 0.00542926 | 0.00384276 / 0.00514073 | 0.000639016 / 0.00110296 | 0.000798627 / 0.00115398 |
| v | 0.000579394 / 0.000704802 | 0.000573087 / 0.00071709 | 0.000119192 / 0.000226782 | 0.000134357 / 0.000253419 |
| p | 0.00214731 / 0.00282562 | 0.00214083 / 0.00290434 | 0.000399262 / 0.000623659 | 0.000432113 / 0.000622912 |
| omega | 0.012308 / 0.0193148 | 0.0118764 / 0.0190711 | 0.00271959 / 0.00377683 | 0.00321114 / 0.00415054 |
| Fluid T | 0.147224 / 0.260575 | 0.142422 / 0.268947 | 0.0143672 / 0.0245322 | 0.0134658 / 0.023698 |
| Surface T | 0.266798 / 0.532168 | 0.258848 / 0.536128 | 0.0268104 / 0.0473208 | 0.0232562 / 0.0474898 |
| Material T | 0.251169 / 0.456275 | 0.241331 / 0.4633 | 0.027274 / 0.0428217 | 0.0231933 / 0.037035 |
| q proxy | 0.964411 / 1.66542 | 0.964213 / 1.67827 | 0.144089 / 0.287491 | 0.129252 / 0.268086 |

All four primary temperature/q components meet both preservation bands on both geometry sets at compact100, and separately at 41/50/best90. The absolute units remain component-specific. Nonprimary mixed-flow median ratios for u/v/omega are 1.2498/1.1272/1.1807, exceeding the descriptive 1.10 level, while p is 1.0823; all four mixed-flow p90 ratios remain below 1.15 (1.0463/1.1175/0.9988/1.0989). Mixed v/p physical reference RMS is near 1e-10–1e-9, so relative errors of order 1e5 are not useful accuracy scores; their absolute errors above remain visible.

| Thermal component | Geometry rel-L2 median / p90 | <1 | Mixed rel-L2 median / p90 | <1 | Heat rel-L2 median / p90 | <1 |
| --- | --- | --- | --- | --- | --- | --- |
| Fluid T | 0.441402 / 1.07912 | 87.50% | 1.68405 / 6.37473 | 12.50% | 0.0621704 / 0.0800136 | 100.00% |
| Surface T | 0.811671 / 1.06977 | 78.12% | 1.39043 / 5.05196 | 37.50% | 0.0491003 / 0.0782623 | 100.00% |
| Material T | 0.804904 / 1.2309 | 68.75% | 1.3097 / 6.82245 | 50.00% | 0.0366046 / 0.0571496 | 100.00% |
| q proxy | 0.799489 / 0.982259 | 93.75% | 3.25003 / 11.859 | 0.00% | 0.117861 / 0.138155 | 100.00% |

All eight heat controls have relative-L2 below 1 for every primary thermal component; heat medians are 0.06217/0.04910/0.03660/0.11786 and all p90 values are below 0.14. Geometry thermal medians are below 1, but fluid/surface/material p90 exceeds 1. All mixed thermal medians and p90 exceed 1, and q has zero mixed controls below 1. Thus useful fixed-layout heat response and typical first-difference response do not establish uniformly useful geometry interactions. Relative L2=1 is the error from predicting zero response, not a physical-validity threshold.

| Eight heat controls: absolute RMSE | R0 median / p90 | Compact100 median / p90 |
| --- | --- | --- |
| Fluid T | 0.0111178 / 0.0234435 | 0.0137619 / 0.0271721 |
| Surface T | 0.0140422 / 0.0304839 | 0.017465 / 0.0407251 |
| Material T | 0.0159785 / 0.0272219 | 0.0166173 / 0.0353459 |
| q proxy | 0.103222 / 0.104748 | 0.101385 / 0.103366 |

The heat-error medians change+23.78%/+24.37%/+4.00%/−1.78% versus R0 for fluid/surface/material T/q. Their low relative errors still support bounded heat utility, but no post-hoc preservation threshold is invented to pass the fluid/surface increases. Conversely, preserving an already inaccurate mixed-response control does not make the response physically accurate. Starting 41 and recovery 50 response results were completed after the pilot and are included in Appendix B; none is transferred from another checkpoint.

![Geometry and mixed response error distributions, prespecified absolute preservation and true retained-cache parity](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/05_physical_responses_and_cache.png)

Figure 7. Explicit compact100 versus R0-100 on existing TRAIN physical references. Relative-L2 distributions and same-component absolute-RMSE preservation ratios answer different questions. Mixed responses stay weak despite preservation. Cached thermal endpoints agree near 1e-13 or below, while flow misses the declared 1e-10 parity band; physical endpoint errors remain far larger. Component units are not combined into a physical scalar. [PDF master](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/05_physical_responses_and_cache.pdf).

### Original source correspondence, imposed heat laws and actual cache application

TRAIN0006/0254 coefficient exports retain exactly 1/12 physically present M source columns, in original slot/ID order 0 and 0–11. Direct coefficient contraction matches fresh native thermal prediction within the declared tolerance; worst q relative discrepancy is 1.33e-6. Zero heat yields exact-zero thermal/q states. Heat superposition and source-partition additivity meet relative 1e-5 plus absolute 1e-5 tolerances; q partition max-abs 1.13e-4 is disclosed alongside relative L2=1.52e-6. Applying zero/h/2h/partitions to the same prepared state leaves routing and model state exact. Independent preparations have tiny FP32 C/pi/flow variability with stable IDs/masks/K; repeat diagnostics separate numerical reentry spread from heat dependence.

The independent-heater physical null has maximum residuals 1.91e-6 fluid T,9.54e-7 surface T,1.91e-6 q and 1.91e-6 material T, with exact-zero physical flow; promoted model mixed-heat residuals are rounding scale (original four models≤1.1e-13). Heat-transfer flow differences are absolute RMSE about 1e-8–5e-7 despite zero physical flow response. These verify the imposed affine law and heat-independent routing/flow to numerical precision, not nonlinear physical interaction fidelity.

| Retained-factor component | Cached/fresh FP64 endpoint maxabs | Worst fresh/physical endpoint RMSE | Declared 1e-10 parity |
| --- | --- | --- | --- |
| fluid_fields/u | 4.84072e-08 | 0.0101868 | fail |
| fluid_fields/v | 1.64586e-08 | 0.00183219 | fail |
| fluid_fields/p | 3.63136e-08 | 0.0058611 | fail |
| fluid_fields/omega | 3.22715e-07 | 0.0301689 | fail |
| fluid_fields/temperature | 7.10543e-15 | 0.48492 | pass |
| interface/temperature | 7.10543e-15 | 0.361849 | pass |
| interface/q_proxy | 1.06581e-13 | 1.27272 | pass |
| solid_temperature/temperature | 1.42109e-14 | 0.339126 | pass |

Actual retained-factor application has zero neural-reader reentry on all four families. Thermal T/q absolute endpoints pass the prespecified FP64 maximum-absolute 1e-10 parity threshold; all four flow components fail at 1e-8–1e-7 and the threshold is retained. Successful contraction is a property of applying learned coefficients, not of their physical accuracy: fresh/physical endpoint RMSE remains 0.36–1.27 in the terminal thermal panel. Original-model cache numbers in Appendix B use per-endpoint RMSE rather than this maximum-absolute statistic. Old precision-promoted increment summaries that did not populate affine_query cannot support cache-reuse claims; only actual-cache records do. Default-FP32 increment cancellation remains separate: original four-model worst maxabs discrepancies are 1.08e-4 fluid T,7.59e-5 surface T,8.81e-5 material T,9.34e-4 q and 3.87e-6 omega. These are not bitwise parity or all-component high-precision passes.

## 9. Saved candidate decisions remain unreliable

The replay preserves the six M3 IDs `size_matched_random_u00_c00` through`c05`, total heat 3.479647219181061, tau 2 smooth module-peak objective and pressure limit 0.13649363704785286. Feasible regret is selected saved-reference objective minus the best reference-feasible objective; it is undefined for a reference-infeasible selection or no feasible recommendation. Correct feasibility flags do not establish correct objective ranking. No geometry optimization, new candidate, new physical heat-allocation result or verified design is created by this replay.

| Checkpoint | Objective MAE | Spearman | TP/FP/TN/FN | Selected | Saved-reference feasible | Feasible regret |
| --- | --- | --- | --- | --- | --- | --- |
| U-H e1000 | 0.207589 | -0.257143 | 1/0/2/3 | c00 | True | 0.524815 |
| U-S e1000 | 0.728951 | -0.0285714 | 4/0/2/0 | c05 | True | 0.54532 |
| R0-100 | 0.317005 | -0.257143 | 1/0/2/3 | c00 | True | 0.524815 |
| Original R1-100 | 0.62299 | -0.371429 | 4/0/2/0 | c03 | True | 0.588267 |
| Amended41 | 0.359706 | -0.2 | 1/0/2/3 | c00 | True | 0.524815 |
| Amended50 | 0.196877 | -0.257143 | 1/0/2/3 | c00 | True | 0.524815 |
| Amended90 | 0.261922 | -0.257143 | 0/0/2/4 | No feasible recommendation | — | undefined |
| Amended100 | 0.225939 | -0.257143 | 1/0/2/3 | c00 | True | 0.524815 |

Compact100 lowers objective MAE0.317005→0.225939 versus R0, but Spearman stays−0.257143. It accepts only c00 and rejects three physically reference-feasible alternatives. Its selected c00 physical objective is 15.23511059; feasible c02 is best at 14.71029556, leaving regret 0.52481503. U-S and original R1 classify all six feasibility flags correctly but still have negative objective correlations and larger regrets. Best90 predicts no feasible candidate because even its minimum pressure 0.1366783 exceeds the unchanged limit; the raw scorer’s minimum-violation fallback is not interpreted as a feasible selection. [The decision-contract audit](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/assessment/candidate_decision_contract_audit.json) preserves that correction without rewriting raw historical records.

| Candidate | Physical objective | Compact100 objective | Physical pressure | Compact100 pressure | Physical feasible | Predicted feasible |
| --- | --- | --- | --- | --- | --- | --- |
| c00 | 15.2351 | 15.0405 | 0.12440362 | 0.13534665 | True | True |
| c01 | 14.8952 | 14.96 | 0.13650931 | 0.14821140 | False | False |
| c02 | 14.7103 | 15.2679 | 0.12892032 | 0.13993142 | True | False |
| c03 | 15.2986 | 15.0098 | 0.13044213 | 0.14008172 | True | False |
| c04 | 15.162 | 15.0748 | 0.14047158 | 0.15175491 | False | False |
| c05 | 15.2556 | 15.0929 | 0.12899755 | 0.14063212 | True | False |

## 10. Organizational savings and actual computational cost

The valid efficiency comparison changes only dense_weighted versus segment execution on compact100 with the same weights, normalization, inputs, heat and gates, on the same idle RTX6000 Ada GPU0 at measurement time. Three alternating-order paired repeats retain their ranges. This is a completed historical benchmark; this report does not query current GPU state. Full-native calls use 8192 fluid receivers,64 angles/module and 3096 material points/module. The 48-case TRAIN macro uses Q1024, micro 24, native auxiliaries, task backward, work-controller VJPs and clipping without an optimizer step. Loading 7.216 s and native-batch construction are excluded from steady calls; the entire corrected diagnostic worker took 49.117 s.

| Full-native TRAIN case | Backend | Median native ms | Range ms | Median cleanup + native ms | Prep ms* | Decode/extract ms* |
| --- | --- | --- | --- | --- | --- | --- |
| 0006 | dense_weighted | 262.900 | 241.38–329.73 | 424.149 | 61.046 | 179.318 |
| 0006 | segment | 302.696 | 261.05–346.97 | 439.881 | 80.420 | 248.375 |
| 0254 | dense_weighted | 284.017 | 237.68–317.54 | 428.176 | 65.936 | 183.190 |
| 0254 | segment | 287.784 | 251.84–319.54 | 435.014 | 68.497 | 200.041 |

| Full-native TRAIN case | Backend | Max allocated GiB | Max increment over baseline GiB | Max reserved GiB | Executed M / E rows |
| --- | --- | --- | --- | --- | --- |
| 0006 | dense_weighted | 1.472 | 1.447 | 28.688 | 8,192 / 1,572,864 |
| 0006 | segment | 1.589 | 1.565 | 28.688 | 8,192 / 1,572,864 |
| 0254 | dense_weighted | 1.577 | 1.553 | 28.727 | 98,304 / 1,572,864 |
| 0254 | segment | 1.366 | 1.342 | 28.727 | 71,094 / 604,019 |

Stage times marked* are separately synchronized diagnostics with boundary overhead and are not summed to reconstruct complete-call medians. The final benchmark clears graph-attached native work summaries through the maintained microbatch lifecycle; garbage collection alone had reclaimed zero while references were live. Corrected native baseline allocation is 26440192 bytes in every repeat (spread 0). Early uncorrected cost attempts remain failure/qualification evidence; only [the lifecycle-corrected JSON](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/structure_cost/cost_e100_fullnative_lifecycle_checked.json) defines these numbers. Cleanup/GC overhead about 0.14–0.18 s is explicitly visible in cleanup-inclusive medians.

Segment native-only medians are 15.14% slower at M1 and 1.33% slower at M12. M1 has full E support and no row saving; allocated peak rises 1.472→1.589 GiB. At M12 segment executes 72.3% M and 38.4% E candidate rows, with allocated peak 1.577→1.366 GiB, but it still does not improve latency. Warm-allocator reserved peaks near 28.69 GiB do not represent the incremental native call footprint; baseline, total peak and increment are separated above.

| 48-case sampled macro | Median complete s | Peak allocated GiB | Peak reserved GiB | Prep s* | Decode/extract s* | Task backward s* | Work VJP s* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense_weighted | 2.604037 | 22.971 | 28.688 | 0.3471 | 0.6488 | 0.8093 | 0.6242 |
| segment | 3.195592 | 20.915 | 28.688 | 0.3843 | 0.8339 | 1.1035 | 0.6358 |

The sampled macro baseline is 25684480 bytes. Segment lowers allocated peak 22.971→20.915 GiB (8.95%) but increases complete macro 2.604037→3.195592 s (22.72%), with reserved 28.688 GiB unchanged. All nine native outputs and 432 macro component-case checks pass their fixed numerical tolerances; masks, source axes and discrete topology are exact. Maximum native/macro relative-L2 differences are 5.82e-6/7.20e-6, and aggregate post-clipping gradient relative L2 is 1.55e-5 (M3.29e-6/E1.78e-5/shared 7.90e-5); continuous C is numerically close, not bitwise identical. These measurements establish a limited memory/time tradeoff on one trained function. They do not compare compact/R0 complete runtime, remove retained encoder N² work or support a speed claim from 48.51% fewer C cells.

![Original candidate objectives and lifecycle-corrected cost versus compact organizational work](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/06_decisions_cost_and_quality_work.png)

Figure 8. Existing six-candidate objectives and saved feasibility, same-function native and sampled-macro times, allocated memory and full-VAL field quality versus allocated C cells. Dense execution remains 100% despite organizational contraction. Segment’s macro memory reduction accompanies 22.72% slower execution. Negative objective correlation and regret 0.524815 remain visible; a saved-pool feasible selection is not a new verified design. [PDF master](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/06_decisions_cost_and_quality_work.pdf).

## 11. Scientific decisions after all completed measurements

| Question | Answer | Practical decision |
| --- | --- | --- |
| Was the Level-R capability implemented and exercised? | Yes: direct C[K,N], direct query-to-K, original donors/source columns, prepared-bank reuse, ordinary save/load/fork/resume and native backward. | Retain the capability and its mathematical axis separation. |
| Did the original objective learn compression? | No: K=N; negative structural gradient incentives explain reject-all on tested probes. | Preserve the original negative run and singleton best80 reference; do not call it compact. |
| Did the amendment produce a useful smaller model? | Yes in the bounded sense: same compact100 has K<N, recovered global fields and useful heat responses. | Use explicit 100 as the primary adaptation endpoint; keep the same-checkpoint limits attached. |
| Did it identify selective functional redundancy? | Not established: all eligible E pairs accepted, M none, input-only fixed-E topology. | Treat acceptance/gradient coverage as policy limitations, not learned physical clustering. |
| Are geometry interactions and decisions ready? | No: mixed thermal errors often exceed zero-response error; all saved-pool objective correlations are negative. | Withhold broad geometry-inverse or design-reliability claims. |
| Does a field-best model solve the decision problem? | No: best90 has no feasible recommendation under the original pressure limit. | Keep model selection objective-specific; never promote field selection into decision success. |
| Did organizational compression deliver measured speed? | No demonstrated compact/R0 speedup; segment is slower despite some allocated-memory/row savings. | Keep cost claims boundary-specific; no executor rewrite is justified solely to obtain a positive efficiency result. |
| Is more recovery needed to complete this bounded study? | No: matched 100 native recovery and all requested bounded validations are completed, with negative outcomes retained. | Do not silently use unused extensions or threshold tuning to seek a preferred answer. |
| What does the new scratch1000 run change here? | Nothing about the completed adaptation evidence. Its source, normalization fitting and exposure differ. | Evaluate it separately when requested after completion; do not borrow adaptation accuracy or inspect its progress here. |

The remaining scientific questions are selective M/E redundancy under representative calibration, robustness across seeds/conditions and independent data, useful mixed geometry responses, reliable feasibility/ranking, and cost improvements that survive complete execution. These are measured limitations or untested generalizations, not missing startup checks. This report authorizes no new experiment. The separately frozen [scratch1000 recipe](../guides/_bk/20261029_Guide_Consolidation/03_R12_R13_Predecessor_Recipes/Thermal_Compact_TieStop_Scratch1000_Recipe.md) is linked for lineage context only, with no result imported.

## 12. Selected figure index

- Figure 1: [Original objective diagnosis (PDF)](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/round12_diagnosis_v2.pdf).
- Figure 2: [Amended physical cut and recovery41 (PDF)](../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/tie_stop_native_fields.pdf).
- Figure 3: [All nine fields across nine checkpoint states (PDF)](../../diagnostics/thermal_compact_round12_20261023/consolidated_20261023_v1/figures/round12_all_checkpoint_fields.pdf).
- Figure 4: [Terminal physical temperature and signed residuals (PDF)](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/02_native_temperature_residuals.pdf).
- Figure 5: [Actual compact C, ancestry, reuse and access (PDF)](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/03_compact_structure_and_access.pdf).
- Figures5a–5b: [Native1404/U-H organization, four representative cases and explicit2500/5000 identities (four-page PDF)](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/figures/1404_UH_hypergraph_comparison.pdf).
- Figure5c: [U-He2500 applied encoder incidence and final composed access (PDF)](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/figures/UH_encoder_and_composed_access.pdf).
- Figure 5d: [Actual E-entry locations and final M reads for queries A/B/C (PDF)](../../diagnostics/uh_entry_sparsity_explainer_20261024/geometry_followup/physical_entry_and_module_access.pdf).
- Figure 5e: [Simple entry counts and query-weight concentration across three layouts (PDF)](../../diagnostics/uh_entry_sparsity_explainer_20261024/figures/01_three_layouts.pdf).
- Figure 5f: [One 202-entry bank, three different query subsets and scene means (PDF)](../../diagnostics/uh_entry_sparsity_explainer_20261024/figures/02_one_layout_three_queries.pdf).
- Figure 5g: [Applied attention gates reused as the native retrieval directory (PDF)](../../diagnostics/uh_attention_to_retrieval_explainer_20261024/figures/01_attention_to_retrieval_overview_v2.pdf).
- Figure 5h: [Exact M6/M7–E53 gate reuse from contextual exchange to qA access (PDF)](../../diagnostics/uh_attention_to_retrieval_explainer_20261024/figures/02_actual_E53_reuse_trace_v2.pdf).
- Figure 6: [Cut/recovery, live geometry and policy gradients (PDF)](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/04_cut_recovery_and_live_geometry.pdf).
- Figure 7: [Physical responses and cache application (PDF)](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/05_physical_responses_and_cache.pdf).
- Figure 8: [Candidate decisions and actual cost (PDF)](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/06_decisions_cost_and_quality_work.pdf).

The original eight selected pages remain embedded above with checkpoint/data/units/mask qualifications immediately attached. Seven reuse existing Python-rendered scientific exports without copying them; one Python page consolidates retained numerical field results. The separate24 October organizer supplement adds the four-page1404/U-H comparison PDF, two selected case PNGs and one encoder/access PDF/PNG; all four PDF pages and selected rasters were visually inspected. Figures 5d–5f add three visually inspected explanatory PDF/PNG pairs reusing the same saved U-H arrays: physical entry/M-read geometry, cross-layout count examples and within-layout query variation. Figures 5g–5h add two streamlined relation-reuse PDF/PNG pairs; secondary small-print qualifications are retained in their captions, and their superseded first visual exports are removed after validating the replacements. These five explanatory figures add no inference or new field/response measurements. Every original campaign visualization remains intact and reachable through Appendix D. PDF masters are retained, and only the PNG companions needed by the Markdown/source reports are kept. Nothing is uploaded as generated data or figures.

## Appendix A. All reference, interim and selected native field results

Every cell below is mean/p90 native RMSE on full exposed VAL89, with each quantity keeping its own unit. Parent and selected-best comparisons are descriptive;50 is an interim unequal-age review,90 is an exposed selection and 41 has only one recovery epoch. Original best80 still has zero merges and no current-weight response/ranking replay.

### A1. Frozen parent references and original singleton best80

| Native quantity | U-H | U-S | Original best80 |
| --- | --- | --- | --- |
| u | 0.0119199 / 0.0171878 | 0.010685 / 0.0161179 | 0.00994911 / 0.0147914 |
| v | 0.00164392 / 0.002188 | 0.00162947 / 0.00206717 | 0.00153675 / 0.00215688 |
| p | 0.00640242 / 0.00982884 | 0.00598773 / 0.00926582 | 0.00587325 / 0.00884477 |
| omega | 0.02954 / 0.0423427 | 0.0303119 / 0.0461279 | 0.0275151 / 0.0394023 |
| Fluid T | 0.39411 / 0.543919 | 0.421801 / 0.605792 | 0.369392 / 0.505337 |
| Surface T | 0.308197 / 0.596402 | 0.294918 / 0.572164 | 0.287659 / 0.639599 |
| Material T | 0.324324 / 0.637418 | 0.301664 / 0.544238 | 0.306267 / 0.654583 |
| q proxy | 1.24966 / 2.24418 | 1.1694 / 2.16314 | 1.2042 / 2.17456 |
| Module peak | 0.313373 / 0.76219 | 0.312638 / 0.654966 | 0.314344 / 0.70148 |

### A2. Amended compact interim/selected checkpoints

| Native quantity | Amended41 | Amended50 | Amended90 |
| --- | --- | --- | --- |
| u | 0.0121895 / 0.0164394 | 0.0105623 / 0.0158841 | 0.00985781 / 0.0155662 |
| v | 0.00173995 / 0.00220554 | 0.00161793 / 0.0020512 | 0.00153333 / 0.0019495 |
| p | 0.0066506 / 0.00933156 | 0.00604704 / 0.00855387 | 0.00578499 / 0.00830112 |
| omega | 0.0307627 / 0.0423611 | 0.0286738 / 0.040232 | 0.0272927 / 0.0395337 |
| Fluid T | 0.481472 / 0.688257 | 0.397125 / 0.522066 | 0.37348 / 0.506616 |
| Surface T | 0.452893 / 0.749043 | 0.308852 / 0.617238 | 0.287466 / 0.596436 |
| Material T | 0.487559 / 0.782557 | 0.32251 / 0.625228 | 0.302196 / 0.600846 |
| q proxy | 1.44239 / 2.29804 | 1.2692 / 2.23244 | 1.19778 / 2.26429 |
| Module peak | 0.457499 / 0.852796 | 0.340228 / 0.626296 | 0.301486 / 0.632555 |

### A3. Historical worst cases and original stratum failure

| Field | U-H worst (case) | U-S worst (case) | R0 worst (case) | R1 worst (case) |
| --- | --- | --- | --- | --- |
| u | 0.0240632 (0293) | 0.0285953 (0293) | 0.026752 (0293) | 0.0300068 (0293) |
| v | 0.0028546 (0295) | 0.00360367 (0293) | 0.00269257 (0302) | 0.00301217 (0293) |
| p | 0.0178847 (0285) | 0.0133298 (0293) | 0.0152856 (0285) | 0.0179223 (0285) |
| omega | 0.0506718 (0297) | 0.0593591 (0297) | 0.0498433 (0284) | 0.0546099 (0284) |
| Fluid T | 0.921186 (0297) | 0.883843 (0297) | 0.813639 (0297) | 1.11433 (0298) |
| Surface T | 1.60054 (0297) | 1.39289 (0297) | 1.482 (0297) | 1.53832 (0298) |
| Material T | 1.8876 (0297) | 1.66222 (0297) | 1.80136 (0297) | 1.66925 (0297) |
| q proxy | 4.5507 (0297) | 3.78986 (0297) | 4.93162 (0297) | 4.56497 (0297) |
| Module-peak endpoint | 2.16024 (0297) | 1.77944 (0297) | 2.0944 (0297) | 1.98692 (0297) |

The compact100 worst cases appear in the primary table; all worst values for the additional fields/states remain in [the consolidated numerical snapshot](../../diagnostics/thermal_compact_round12_20261023/consolidated_20261023_v1/reviewed_evidence.json) and their original native boards. Worst-case improvements do not cancel population mean/tail failures. The preserved original R0→original R1 stratum means are:

| VAL stratum | Fluid T | Surface T | Material T | Module peak |
| --- | --- | --- | --- | --- |
| M3 (24 cases) | 0.297285 → 0.408923 | 0.17534 → 0.356691 | 0.217944 → 0.440117 | 0.155056 → 0.397916 |
| M5 (25 cases) | 0.362135 → 0.484564 | 0.320589 → 0.537037 | 0.337867 → 0.601887 | 0.324273 → 0.555058 |
| M7 (25 cases) | 0.434747 → 0.684331 | 0.464202 → 0.843102 | 0.49974 → 0.931693 | 0.511402 → 0.933377 |
| M10 (15 cases) | 0.365167 → 0.747925 | 0.291344 → 0.844302 | 0.293698 → 0.921514 | 0.25476 → 0.92128 |

Original TRAIN24 R1/R0 priority mean ratios were 1.651/2.515/2.522/3.144 for fluid/surface/material T/peak, with higher mean/p90/worst on all nine fields. These are already exposed training diagnostics, not independent generalization evidence.

## Appendix B. Response and cache context across all stages

### B1. Original frozen references and matched zero-merge endpoints

| Response | U-H median / p90 / mixed median | U-S median / p90 / mixed median | R0 median / p90 / mixed median | R1 median / p90 / mixed median |
| --- | --- | --- | --- | --- |
| Fluid T | 0.440 / 1.093 / 1.551 | 0.493 / 1.080 / 2.069 | 0.433 / 1.083 / 1.689 | 0.411 / 1.094 / 1.589 |
| Surface T | 0.802 / 1.114 / 1.414 | 0.740 / 1.174 / 1.530 | 0.789 / 1.132 / 1.437 | 0.841 / 1.176 / 1.322 |
| q proxy | 0.837 / 1.002 / 2.938 | 0.823 / 1.165 / 3.428 | 0.811 / 0.962 / 3.723 | 0.831 / 0.979 / 3.758 |
| Material T | 0.824 / 1.449 / 1.438 | 0.720 / 1.297 / 1.540 | 0.814 / 1.377 / 1.492 | 0.786 / 1.374 / 1.363 |

Each cell contains geometry relative-L2 median/p90 and mixed-geometry median, respectively, on the 32/8 saved differences; these are separate component metrics. Original R1’s lower fluid geometry relative median 0.411 versus R0’s 0.433 did not mean lower absolute error: fluid absolute RMSE median/p90 was 0.15100/0.26398 versus R0’s 0.14722/0.26057. Parent/current coefficients are not replaced by older U10-e 5000 evidence.

### B2. Completed current-weight amended41/50/90 responses

| Checkpoint | Thermal component | Geometry relative-L2 median / p90 | Mixed relative-L2 median / p90 | Heat relative-L2 median / p90 | Geometry / mixed fraction<1 |
| --- | --- | --- | --- | --- | --- |
| Amended41 | Fluid T | 0.415232 / 0.97755 | 1.52299 / 6.48904 | 0.0771921 / 0.116817 | 90.62% / 12.50% |
| Amended41 | Surface T | 0.797705 / 1.02289 | 1.35835 / 4.57963 | 0.0508407 / 0.152496 | 84.38% / 37.50% |
| Amended41 | Material T | 0.816712 / 1.22526 | 1.25751 / 6.1165 | 0.0444899 / 0.125997 | 65.62% / 37.50% |
| Amended41 | q proxy | 0.849135 / 0.973803 | 3.14153 / 12.6864 | 0.141136 / 0.176664 | 90.62% / 0.00% |
| Amended50 | Fluid T | 0.429304 / 1.0647 | 1.64313 / 6.74688 | 0.062262 / 0.113204 | 87.50% / 0.00% |
| Amended50 | Surface T | 0.790696 / 1.03844 | 1.35269 / 5.35474 | 0.0536999 / 0.0645083 | 81.25% / 37.50% |
| Amended50 | Material T | 0.816127 / 1.23134 | 1.29627 / 7.54925 | 0.0397697 / 0.0611615 | 68.75% / 50.00% |
| Amended50 | q proxy | 0.822338 / 0.979594 | 3.47227 / 13.4202 | 0.137198 / 0.181074 | 90.62% / 0.00% |
| Amended90 | Fluid T | 0.437901 / 1.07643 | 1.70001 / 6.23995 | 0.055885 / 0.0758868 | 87.50% / 12.50% |
| Amended90 | Surface T | 0.803235 / 1.02191 | 1.3366 / 4.99105 | 0.0381415 / 0.0599335 | 75.00% / 50.00% |
| Amended90 | Material T | 0.8103 / 1.26586 | 1.36053 / 7.06601 | 0.033923 / 0.0738039 | 68.75% / 50.00% |
| Amended90 | q proxy | 0.816334 / 0.97768 | 3.10026 / 12.2408 | 0.120366 / 0.189006 | 90.62% / 0.00% |

Each checkpoint separately passes primary absolute-error preservation on 32 geometry and 8 mixed differences. The 41 native-field failure,50 interim recovery and 90 selected-best/no-feasible decision remain distinct. The original pilot’s “not evaluated” response/ranking/cache statement described its then-completed scope; the continuation subsequently filled those measurements rather than treating them as unnecessary.

### B3. Original models: actual thermal-cache parity versus physical endpoint error

| Endpoint (own native units) | U-H cache parity / physical RMSE | U-S cache parity / physical RMSE | R0 cache parity / physical RMSE | R1 cache parity / physical RMSE |
| --- | --- | --- | --- | --- |
| Fluid T | 7.47e-16 / 0.45 | 4.29e-16 / 0.493 | 7.76e-16 / 0.461 | 7.44e-16 / 0.857 |
| Surface T | 1.04e-15 / 0.361 | 5.96e-16 / 0.317 | 1.28e-15 / 0.385 | 1.1e-15 / 1.1 |
| Material T | 1.17e-15 / 0.428 | 6.39e-16 / 0.35 | 1.11e-15 / 0.392 | 1.13e-15 / 1.21 |
| q proxy | 1.94e-14 / 1.27 | 9.72e-15 / 1.24 | 2.17e-14 / 1.22 | 2.04e-14 / 1.78 |

These original-model numbers are worst per-endpoint cached/fresh FP64 RMSE and fresh/physical RMSE across eight saved heat endpoints, not the compact100 maximum-absolute statistic used in its declared 1e-10 band. Each model uses 12 fresh native absolute calls,4 promotions and 20 retained-factor applications, with zero neural-reader reentry after promotion. They demonstrate a shared reusable affine capability, not a unique compact-model gain or physical response correctness.

## Appendix C. Software, execution and storage evidence

| Stage | Software/operational validation | Historical storage scope / cleanup |
| --- | --- | --- |
| Original experiment | 26 compact/native/loss/normalizer tests;28 runner/fork/resume;22 evaluator. Actual GPU training and self-contained 100→200 terminal forks with zero additional updates. No OOM/repacking in scientific fit. | 10 GiB output/2 GiB temporary; removed 203067111 logical bytes; retained 566560559 bytes at that snapshot. |
| Diagnosis | Zero updates; actual prepare parity, no-step gradients, full-native best80 board; no unique optimization trigger claimed. | 256 MiB output/128 MiB temporary; removed 476155 bytes; retained 16754941 bytes at snapshot. |
| Amendment pilot | 18 focused+59 integration; mutation test; fixed-weight parity; actual cut/recovery. Initial one-device visibility failed before updates because checkpoint has three CUDA RNG streams; successful run preserved visibility and explicit device selection. | 2 GiB output/1 GiB temporary; removed 9033920 logical bytes (111 files/links); retained 214680147 bytes at snapshot. |
| Recovery/validation | 45 affected tests; actual same-checkpoint campaigns, independent response arithmetic, frozen-policy checks and corrected no-update cost. | 12 GiB output/2 GiB temporary; removed 2342650 bytes (30 regular files)+2 synthetic symlinks; continuation≈199.4 MB and whole historical round≈1.054GB at snapshot. |
| This consolidation | Read-only numerical reaggregation, source/checkpoint bindings, all report links, Markdown paragraph/caption structure, Python figure/PDF and visual inspection; no model training/inference. | 128 MiB new-output budget/64 MiB temporary,200 GiB minimum DATA2 headroom, zero new dataset/cache copies. Current closeout below. |

Test counts are the reported invocation scopes, not a count of mutually independent unique tests; historical retained-byte snapshots overlap and must not be summed. Old free-space snapshots describe their own closeouts, not current capacity. All original checkpoints, aliases, histories, physical references, failure evidence and figures remain protected. Only new report-owned disposable render previews/font caches are eligible for this report’s routine cleanup; generated evidence and one-time helpers remain ignored on DATA2. The live scratch run and unrelated changes are not modified or monitored.

Consolidation closeout removed 3 report-owned disposable files totaling 1,459,714 logical bytes after exact recursive allowlist, handle/mapping and dependency checks. One new PDF/PNG pair, reviewed numerical/provenance evidence, interim heat reaggregation and local QA/helpers remain; all 15 historical figure pairs and scientific evidence stay at their existing paths. DATA2 free capacity at this closeout snapshot is 3,220,682,088,448 bytes; home/root has 96,413,622,272 bytes. These shared-filesystem values are snapshots, not a claim that free-space change equals deleted bytes. The [cleanup receipt](../../diagnostics/thermal_compact_round12_20261023/consolidated_20261023_v1/cleanup_receipt.json) and [report QA](../../diagnostics/thermal_compact_round12_20261023/consolidated_20261023_v1/report_qa.json) retain the checks. Durable delivery contains this new Markdown report and a navigation pointer in the original main report; generated figures/data/helpers remain ignored.

The separate 24 October native-organizer supplement used only CPU inference on 12 checkpoint-case combinations, with 3.882 recorded extraction seconds, zero optimizer updates and no solver/CUDA job. Original checkpoint files and model/buffer states were hashed unchanged; module position/order checks pass across all 12 combinations. Its declared 128MiB output/32MiB temporary budget and 200GiB minimum DATA2 headroom were respected. After visual inspection,974,645 bytes of disposable font-cache/PDF-check rasters were removed; retained numerical/figure/helper payload was 10,902,190 bytes before the small report QA/delivery receipts, with 3,216,422,805,504 bytes DATA2 free at that closeout. The [supplement cleanup receipt](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/cleanup_receipt.json) and [report/link QA](../../diagnostics/thermal_1404_uh_hypergraph_visuals_20261024T163631Z/report_qa.json) keep this accounting separate from the original adaptation campaign.

The explanatory-figure integration reuses three existing PDF/PNG pairs without copying or regenerating them, and adds no model inference, solver work or training inspection. The count-figure round removed one disposable font-cache file of 198192 bytes; the subsequent physical-map round independently removed another 198192-byte font cache. Their [count-figure cleanup receipt](../../diagnostics/uh_entry_sparsity_explainer_20261024/cleanup_receipt.json) and [physical-map QA/cleanup receipt](../../diagnostics/uh_entry_sparsity_explainer_20261024/geometry_followup/qa_and_cleanup_receipt.json) retain the separate scopes. This report-only integration creates no disposable payload and removes 0 files/0 bytes; all existing figures, source arrays and historical evidence remain unchanged. Its preparation budgets are 1 MiB each on home and DATA2, zero temporary bytes, with minimum 20 GiB/200 GiB headroom. The [integration receipt](../../diagnostics/uh_entry_sparsity_explainer_20261024/report_integration_20261024.json) records before/after filesystem capacity, source preservation, report/link/paragraph QA and delivery. Durable upload contains only the Markdown revision; the figures, numerical evidence and one-time renderers remain local on DATA2.

The relation-reuse figure revision keeps two final PDF/PNG pairs and moves secondary in-image commentary into report captions. Original numerical evidence and source arrays remain unchanged. A 16 MiB DATA2 output/4 MiB temporary budget and 1 MiB home budget were recorded with 200 GiB/20 GiB minimum headroom. Superseded visual exports and the round-owned plotting font cache are removed only after replacement validation and exact allowlist/handle/dependency checks. The [revision/integration receipt](../../diagnostics/uh_attention_to_retrieval_explainer_20261024/revision_and_integration_20261024.json) records exact removed files/bytes, retained artifacts, before/after capacity, report QA and delivery. This is a presentation/documentation revision, with no training, new inference or physical solves; only the durable report revision is uploaded.

## Appendix D. Source map and complete original visualization archive

Read these as phase records, not four competing current verdicts. This consolidated report supplies the current cross-phase interpretation while preserving every historical report and numerical artifact. The source work plan remains [here](../../UpgradePlan/HONF_Round12_Compact_Hypergraph_Thermal_Work_Plan.md).

| Source report | Scientific scope |
| --- | --- |
| [HONF_Round12_Compact_Hypergraph_Thermal_Report.md](_bk/20261023_Round12_Compact_Hypergraph_Thermal/20261023_161941Z_HONF_Round12_Compact_Hypergraph_Thermal_Report.md) | Original common20/matched R0/R1 local 100, with zero merges and terminal field/response/decision evidence. |
| [HONF_Round12_Sibling_Policy_Diagnosis.md](_bk/20261023_Round12_Compact_Hypergraph_Thermal/20261023_171813Z_HONF_Round12_Sibling_Policy_Diagnosis.md) | Saved-run policy-gradient explanation, terminal instability and original singleton best80. |
| [HONF_Round12_Tie_Stop_Feasibility_Report.md](_bk/20261023_Round12_Compact_Hypergraph_Thermal/20261023_181429Z_HONF_Round12_Tie_Stop_Feasibility_Report.md) | Implemented tie-stop amendment and actual cut through 41; historical feasibility boundary. |
| [HONF_Round12_Compact_Recovery_and_Validation_Report.md](_bk/20261023_Round12_Compact_Hypergraph_Thermal/20261023_195951Z_HONF_Round12_Compact_Recovery_and_Validation_Report.md) | Completed 41→50→100 recovery, separate best90, current-weight fields/responses/cache/geometry/decisions and corrected cost. |

| Phase source | Original visualization | Use in this report | Retained masters / companion |
| --- | --- | --- | --- |
| Compact_Hypergraph_Thermal_Report | thermal_native_field_mean_p 90 | Additional historical visualization | [PDF](../../diagnostics/thermal_compact_round12_20261023/report_figures/thermal_native_field_mean_p90.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/report_figures/thermal_native_field_mean_p90.png) |
| Compact_Hypergraph_Thermal_Report | thermal_fluid_maps_case_0692 | Additional historical visualization | [PDF](../../diagnostics/thermal_compact_round12_20261023/report_figures/thermal_fluid_maps_case_0692.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/report_figures/thermal_fluid_maps_case_0692.png) |
| Compact_Hypergraph_Thermal_Report | 03_compact_organization | Additional historical visualization | [PDF](../../diagnostics/thermal_compact_round12_20261023/report_figures/03_compact_organization.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/report_figures/03_compact_organization.png) |
| Compact_Hypergraph_Thermal_Report | 04_query_access_and_work_v 2 | Additional historical visualization | [PDF](../../diagnostics/thermal_compact_round12_20261023/report_figures/04_query_access_and_work_v2.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/report_figures/04_query_access_and_work_v2.png) |
| Compact_Hypergraph_Thermal_Report | 05_cut_and_live_geometry_v 2 | Additional historical visualization | [PDF](../../diagnostics/thermal_compact_round12_20261023/report_figures/05_cut_and_live_geometry_v2.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/report_figures/05_cut_and_live_geometry_v2.png) |
| Compact_Hypergraph_Thermal_Report | 06_saved_responses_and_ranking_v 2 | Additional historical visualization | [PDF](../../diagnostics/thermal_compact_round12_20261023/report_figures/06_saved_responses_and_ranking_v2.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/report_figures/06_saved_responses_and_ranking_v2.png) |
| Compact_Recovery_and_Validation_Report | 01_native_fields | Additional historical visualization | [PDF](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/01_native_fields.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/01_native_fields.png) |
| Compact_Recovery_and_Validation_Report | 02_native_temperature_residuals | Embedded as Figure 4 | [PDF](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/02_native_temperature_residuals.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/02_native_temperature_residuals.png) |
| Compact_Recovery_and_Validation_Report | 03_compact_structure_and_access | Embedded as Figure 5 | [PDF](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/03_compact_structure_and_access.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/03_compact_structure_and_access.png) |
| Compact_Recovery_and_Validation_Report | 04_cut_recovery_and_live_geometry | Embedded as Figure 6 | [PDF](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/04_cut_recovery_and_live_geometry.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/04_cut_recovery_and_live_geometry.png) |
| Compact_Recovery_and_Validation_Report | 05_physical_responses_and_cache | Embedded as Figure 7 | [PDF](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/05_physical_responses_and_cache.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/05_physical_responses_and_cache.png) |
| Compact_Recovery_and_Validation_Report | 06_decisions_cost_and_quality_work | Embedded as Figure 8 | [PDF](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/06_decisions_cost_and_quality_work.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/continuation_20261023_v1/figures/06_decisions_cost_and_quality_work.png) |
| Sibling_Policy_Diagnosis | round 12_diagnosis_v 2 | Embedded as Figure 1 | [PDF](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/round12_diagnosis_v2.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/diagnosis_20261023_v1/evidence/round12_diagnosis_v2.png) |
| Tie_Stop_Feasibility_Report | tie_stop_organization | Additional historical visualization | [PDF](../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/tie_stop_organization.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/tie_stop_organization.png) |
| Tie_Stop_Feasibility_Report | tie_stop_native_fields | Embedded as Figure 2 | [PDF](../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/tie_stop_native_fields.pdf) / [PNG](../../diagnostics/thermal_compact_round12_20261023/tie_fix_20261023_v1/evidence/tie_stop_native_fields.png) |

All 15 original PDF/PNG pairs remain at their existing ignored paths; none is deleted, renamed, regenerated or uploaded by this consolidation. The new all-checkpoint comparison and [reviewed numerical snapshot](../../diagnostics/thermal_compact_round12_20261023/consolidated_20261023_v1/reviewed_evidence.json) retain exact source paths/digests locally. Source reports are evidence; their obsolete continuation commands are not executed by reading or consolidating them.
