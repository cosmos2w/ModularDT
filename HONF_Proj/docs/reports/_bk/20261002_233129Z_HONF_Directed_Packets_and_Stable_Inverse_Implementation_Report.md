# HONF directed packets and stable inverse implementation

Implemented and trained three matched comparisons: a directional versus legacy-summary Wind organizer, a broader Thermal value-weight correction, and fresh graph versus dense inverse models using bounded clean coordinates and velocity prediction. These are opt-in additions. Historical maturation runs, the epsilon sampler, checkpoints, and production policies remain available. This round used stored references and made **zero new Thermal or Wind reference solves**.

**Predictor.** Wind full-access co-adaptation lowers primary volume RMSE from .008363 to .007938 m/s (5.1%) and improves all five role means against G-u4910. W-dir is usually worse than the trained summary arm under sparse access and loses on both predeclared field planes. Thermal value8 improves all 32 calibration absolute field/role cells against value4, but wins only 23/32 finite-response cells and still misses Run1804's average accuracy in every reported channel. Increasing the value weight changes the tradeoff; it does not recover the retained predictor.

**Organizer.** Adequate sparse actions increase from 6/12 to 12/12 primary Wind layouts, but held-fold action-pair ranking is 58.5% and development action-pair ranking is 60.0%/56.3%. Conservative selection falls back on 10/12 layouts in each panel and still makes one primary false-safe choice. Learned source selection has no all-role win in 50 primary degree/weight-matched geometry comparisons. The existing packed executor rejects these direct-MM plans, so executor savings are unmeasured. Native source/receiver IDs, fractional permissions, nonredundant actions, and executor rows are saved. MM is the only newly restricted route; QE/ME/EM/QM and coarse/local paths retain full information. Learned support is computational organization, with no causal interpretation.

**Inverse.** The new representation avoids the old sigmoid corner behavior: all eight trajectories are finite, and none of their 16 generated xy coordinates needs terminal projection. Conditional quality remains weak. All six row 50 candidates fail native support; the two row 131 candidates pass the existing surrogate/support checks. Changing one observed velocity moves the generated position by only 0.000001–0.000008 D and provides no resolved sensor-fit benefit. Effective graph interventions change access, but the same-weight graph/full position difference is only 0.00017 D. These candidates have no generated-layout CFD validation.

**One next investment.** Test whether the observation representation can identify a hidden position using a small training-only conditional-regression/control experiment with fixed known geometry and paired observation interventions. The stable sampler exposes weak learned conditioning without a saturating decoder; that control would distinguish inadequate fitting from weak observation information. More trajectories or a longer maturation campaign are not justified. This is a recommendation; no further work starts automatically.

The user authorized the new plan's paired learning models within tighter bounds: **six elapsed hours, ten aggregate GPU-associated hours, eight inverse trajectories total**, with the final hour reserved for closeout. Primary targets completed: 1,000 Wind updates per arm, 240 Thermal updates per arm, and 1,000 inverse updates per arm. Extra diagnostic/discarded optimization used 115/400 updates. All GPU-associated work ends by 23:08:30Z, after 3 h 54 min 30 s elapsed. Conservative lane windows bound aggregate GPU-associated use at **6.63194 h**, including idle time and unmeasured setup; CPU closeout starts well before the reserved final hour. Exactly eight trajectories complete with no retries or automatic continuation.

## Outcome summary

| Goal | Measured gain | Measured miss | Result boundary |
|---|---|---|---|
| Wind predictor | Full-access W-dir volume .008363→.007938 m/s; all five primary full role means improve versus G | Sparse W-dir beats W-summary in all roles on only 7/48 primary matched requested cuts; both field planes worse | Stored OpenFOAM reference; all five roles; two predeclared native slices |
| Organizer | Adequate sparse actions 6/12→12/12; a same-work split improves all roles in one layout | Degree/weight-matched geometry: 0/50 primary all-role wins; 10/12 full fallbacks; packed unsupported | MM-only restriction; permission work and executor work distinguished |
| Thermal predictor | Value8 wins 32/32 absolute calibration cells versus value4; fluid-T mean relative RMSE 3.865% → 3.696% | Finite wins 23/32; Run1804 remains better on channel means; finite pressure worsens | Four exposed Re90 families and 16 historical records, with overlap |
| Inverse numerics | Eight finite persisted trails; 0/16 terminal coordinates projected; visible geometry exact | Six native-support failures; condition response ≤8.1e-6 D; no material graph advantage | Two layouts, one shared noise draw per layout; no new-layout CFD |

## Evidence and learning

The source baseline is `7221782` on `agent/honf-core-next`, following the plan's reviewed `61e9d42` with an archival-only report move. The [implementation plan](../../../UpgradePlan/ThemalChannel/20261002_191148Z_HONF_Directed_Packets_and_Stable_Inverse_Implementation_Plan.md) is governed by the user's clarification, “Train the new plan's paired models within the tighter budgets.” Its eight-/twelve-hour and 192-trajectory maxima are superseded by six-/ten-hour and eight-trajectory limits.

Physical GPU1 (`GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e`) ran Wind. GPU2 (`GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`) ran Thermal, inverse and short regression checks. Both are RTX6000 Ada devices. Scientific Python is `/home/wanglz/miniconda3/envs/ModularDT/bin/python`. GPU0 was untouched; tests requiring logical `cuda:0` used an explicit GPU2 UUID mapping, verified before execution.

Local evidence root is `/home/wanglz/Desktop/src/ModularDT/HONF_Proj/diagnostics/generated/directed_stable_20261002/`. Checkpoints, arrays, figures and one-time analysis/renderers remain ignored. Numerical tables are durable; local figures require this evidence. Existing source/checkpoint/context guards and trusted loaders are reused. No new freeze, fingerprint framework, approval store or monitoring daemon was created.

| Comparison | Actual exposure per arm | Fixed source and learning scope | Evaluation population |
|---|---|---|---|
| W-dir / W-summary | 300 scorer-only updates, 100 read-only full checks, 700 co-adaptation updates; 825 scorer and 700 physical steps | Same G-u4910 physical/scorer initialization; scorer LR1e-3, MM message/shared update LR1e-5 | 30 whole training layouts × three directions; 12 metadata-selected development layouts |
| Thermal value4 / value8 | 240 actual updates, u1300→u1540; 240 distinct history visits and all eight response families | Same G-u1300, Adam/RNG/query/history stream; full access, routes frozen; only value4 versus8 | Four Re90 calibration families plus 16 historical selectors |
| I-v-G / I-v-dense | 1,000 updates, 4,000 micro-examples, 1,000 exact t=1 examples | Fresh width96/layers3 denoisers; AdamW1e-3/weight decay.01; frozen u4910 two-packet provider | 24 training layouts, 72 direction rows, two masks each; two exposed development tasks |

Wind training layouts: `[4,8,24,28,51,52,56,70,71,79,82,88,91,92,93,114,115,119,125,132,135,143,148,150,152,173,174,176,177,199]`. Direction counts are WD270:400, WD285:300, WD300:300. Native row and phase streams match exactly between arms. Exact-action deduplication gives 39 realized cut-exposure disagreements: W-dir root/two/three-A/three-B/four counts 297/181/165/150/32; W-summary 324/179/165/149/8. Input/query streams match, but realized cut exposure is not identical. The completed update log has 100 read-only full checks; a stale W-dir phase-one manifest's99 is not substituted for it.

Development layouts: `[15,16,31,42,53,129,142,154,158,160,162,168]`. Established organizer-development layouts 81/103/107/196 remain excluded. Labels use one Q512 WD285 role panel per fit layout. Each development case has Q2048 panel1 on original native role quadrature support and panel2 rejecting the entire union of panel1 native indices. Every saved fit/development role panel uses WD285; WD270/WD300 transfer is not measured in this final panel. Intersections are zero. Panel2 estimates **remaining conditional support**, not the identical original population. The 12 primary and 12 repeat panels remain separate; directions, roles and repeats are not additional independent layouts.

Training Q512 has 384 volume and 32 queries per protected role: hub slab, downstream envelope, near turbine and background. Draws retain each role's native quadrature CDF. The objective is the maintained mean of five role MSEs with fixed Run2112 scales plus the Run2110 W-full u1500 anchor, coefficient 0.09525970142938742. Query counts are exposure, not objective weights of .75/.25. Co-adaptation contains 525 restricted and 175 full physical updates; scorer-only fitting interleaves 100 read-only full checks while physical weights are frozen.

Thermal has 600 available historical cases, but **240 visited**. Its selectors are `[0003,0027,0048,0072,0001,0310,0117,0318,0325,0162,0333,0340,0207,0348,0355,0252]`, covering M1/2/3/4/5/6/7/9/10/12. Four also occur in calibration:20 records represent **16 distinct cases**, with 12 additional distinct historical cases. This is stored/exposed evidence, not an untouched test population or maturation continuation.

Inverse training layouts: `[24,88,55,111,112,95,116,148,100,123,3,51,12,37,21,18,13,66,29,86,121,70,126,177]`, M6–30, all three directions and 144 completion tasks. None was excluded by the common domain. Main-thread reconstruction of the task/time sampling RNG matches the final checkpoint; every task was visited 15–48 times, mean 27.78. Sensor center/scale use training rows only: center`[8.239413,-.002903,.000537]`, scale`[.755474,.105470,.05]` m/s.

![Executed learning and exposure](../../../diagnostics/generated/directed_stable_20261002/figures/figure01_learning_and_exposure.png)

**Figure 1.** [PDF master](../../../diagnostics/generated/directed_stable_20261002/figures/figure01_learning_and_exposure.pdf). Completed update logs show Wind's scorer-only/co-adaptation transition at 300, Thermal's unweighted value loss, inverse's masked velocity loss and actual cut exposure. Fixed-noise inverse t=1 development clean-coordinate RMSE improves 8.965→6.179 D (graph) and 8.965→6.157 D (dense); these are two exposed review tasks. Recorded timers and conservative ownership bounds are distinguished. Training curves demonstrate fitting, not physical transfer by themselves.

## Wind predictor, grouping and action choice

The new MM hierarchy indexes active module receivers to depth 2 without fake leaves. The shared-packet distance prior uses actual receiver weights; a zero-initialized residual ramps over 100 restricted updates. W-dir retains signed downstream/crosswind/vertical offsets, component magnitudes, squared distance, extents and axis-valid masks. W-summary supplies padded legacy axis-symmetric summaries at the same width. Coordinates already use the wind-aligned frame, D=80 m/L*=1 D; no second rotation occurs. Reusable features support declared orthonormal frames and absent axes in 2-D.

Fine source states remain separate through nonlinear messages. Physical fitting is limited to MM messages and the shared module update, which also updates ME-derived context. Per-update Q512 training CUDA allocator peaks reach 447,374,336 bytes for W-dir and 451,503,104 for W-summary (426.65/430.59 MiB). These are measured training allocations, not total device memory or a packed-inference comparison. Other fine mechanisms, encoders, field head and local/coarse paths stay frozen. Independent checkpoint comparison confirms all 147 physical-model tensors in both scorer-u300 arms remain bitwise equal to G-u4910, with zero co-adaptation updates. The final pair includes its learned physical-weight differences.

Arithmetic mean of case-level, equal-component role RMSE **[m/s]**, 12 primary development layouts. Oracle minimizes eligible MM pair work among measured adequate sparse actions; it does not minimize RMSE.

| Role | Retained W-full | G-u4910 full | W-dir final full | W-summary final full | W-dir research choice | W-dir conservative | W-dir sparse oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| Volume | 0.008947 | 0.008363 | 0.007938 | 0.007940 | 0.008316 | 0.008084 | 0.008325 |
| Hub slab | 0.017149 | 0.015151 | 0.014934 | 0.014878 | 0.016224 | 0.015732 | 0.015628 |
| Downstream envelope | 0.019413 | 0.018425 | 0.017465 | 0.017399 | 0.018651 | 0.017940 | 0.018780 |
| Near turbine | 0.039312 | 0.037568 | 0.037302 | 0.037231 | 0.038216 | 0.037844 | 0.037998 |
| Background | 0.005362 | 0.005077 | 0.004742 | 0.004734 | 0.004971 | 0.004853 | 0.004973 |

The co-adapted **full** output improves before any access restriction is applied, so part of the gain is physical-weight fitting. W-dir full versus G primary mean reductions are 5.1%, 1.4%, 5.2%, 0.7% and 6.6% in the listed role order. W-summary full is slightly better in four of five primary role means. On 48 common sparse requested-cut records, W-dir wins every role in only 7 primary / 5 repeat records; only 25/48 match exact work and permission mass. Median primary W-dir/W-summary role RMSE ratios are 1.0142/1.0084/1.0205/1.0085/1.0103. These action records share 12 layouts and are not independent sample sizes.

For the conditional repeat, W-dir full role means are .007880/.012940/.017016/.033459/.004339 m/s versus retained .008809/.014737/.018921/.035388/.004802. Its research/conservative/oracle choice means, saved separately, remain conditional-population measurements. Predictor next step: retain these opt-in comparisons as evidence; the directional sparse model does not qualify for production promotion.

Archived G packet/P-u4910 actions remain in the [focused diagnosis](20261002_183337Z_HONF_Focused_Organizer_Utility_and_Predictor_Diagnosis_Report.md). They are not treated as matched predictions on new receivers. Fresh matched references are G-u4910 full, each arm's same-weight full, its geometry prior, and Run2110 W-full u1500. All five velocity roles remain distinct.

![Predeclared native Wind fields and residuals](../../../diagnostics/generated/directed_stable_20261002/figures/figure02_native_wind_fields.png)

**Figure 2.** [PDF master](../../../diagnostics/generated/directed_stable_20261002/figures/figure02_native_wind_fields.pdf). Outcome-independent WD285 slices are layout 53/row160 and largest-M layout 142/row427, near z/D.886. Both arms use the same requested four-packet cut; the saved audit distinguishes actual nonredundant K and pair work. Layout 53 uses 67,266 finite queries: component RMSE is 0.016903 m/s (W-dir), 0.015796 (W-summary), 0.016007 (G full), and 0.016799 (retained W-full). Actual pair work is 341/380 versus 339/380, with K=4 in both. Layout 142 uses 96,119 finite queries: component RMSE is 0.018584 m/s (W-dir), 0.017541 (W-summary), 0.017526 (G full), and 0.018374 (retained W-full). Actual pair work is 778/870 versus 781/870, with K=4 in both. W-dir is worse than W-summary and retained W-full on both planes; these slices show no directional-feature advantage. Residual limits use the99th percentile only for display. Numerical slice RMSE uses all finite entries without clipping and is an unweighted plane component metric, distinct from role-CDF integration above.

![Actual native directed permissions](../../../diagnostics/generated/directed_stable_20261002/figures/figure03_directed_native_interactions.png)

**Figure 3.** [PDF master](../../../diagnostics/generated/directed_stable_20261002/figures/figure03_directed_native_interactions.pdf). Real module coordinates/IDs are shown for layout 53. Arrows run source→receiver; matrices use receiver rows/source columns and retain fractional compiled weights. The prior uses final W-dir physical states with score residual disabled; W-summary has its own paired state. Prior/W-dir/W-summary retain 342/341/339 of 380 eligible pairs, with K=4 and 4/7/7 fractional positive entries. Learned support changes, while the native graph remains broadly dense. These are computational permissions, with no CFD causal interpretation.

### Availability, ranking, calibration and fallback

One ridge action model fits final W-dir evidence from 30 training layouts using maintained action-risk infrastructure. Features use actual receiver states, source inclusion/exclusion summaries/measures, signed 13-feature geometry, budgets and receiver-role geometry/CDF moments. Sparse and full-access action rows supply reference-risk labels from stored CFD. Normalization is fitted within each fold. Five whole-layout held folds give a fixed empirical 90th-percentile positive residual margin. The folds hold layouts out of the action regression; the physical predictor already trained on all 30. Ridge strength 1 and the allowance are fixed; no development tuning occurs. Reported choices are input-only minimum predicted-risk research selection, conservative acceptance/fallback, and a reference-measured oracle used only for evaluation. Synthetic controls are excluded from fit labels.

The retained-reference allowance is 10% in **every** role, with zero absolute physical allowance and the existing training-derived numerical floor. Full fallback can itself fail against the retained predictor. The residual margin and its empirical coverage pool sparse action rows from the 30 held-fold layouts; coverage is measured on the same residuals used to fit that margin. This is not an unseen-layout coverage guarantee. Unsafe research choices are not deployed; historical production policy stays unchanged.

| Panel / checkpoint | W-dir adequate sparse layouts | W-summary adequate sparse layouts | W-dir adequate full layouts |
|---|---:|---:|---:|
| Scorer-only u300, primary | 6/12 | 11/12 | 11/12 |
| Scorer-only u300, conditional repeat | 7/12 | 10/12 | 10/12 |
| Final u1000, primary | 12/12 | 12/12 | 12/12 |
| Final u1000, conditional repeat | 12/12 | 11/12 | 12/12 |

| Choice-model evaluation | Sparse availability | Pair ranking | Research sparse / inadequate choices | Conservative sparse / inadequate choices | Full fallbacks | False-safe / false-reject action rows |
|---|---:|---:|---:|---:|---:|---:|
| Training held-layout folds | 15/30 | 117/200 (58.50%) | 15 / 13 | 4 / 2 | 26 | 3 / 29 |
| Training in-fit (optimistic) | 15/30 | 166/200 (83.00%) | 11 / 7 | 0 / 0 | 30 | 0 / 31 |
| Primary development | 12/12 | 48/80 (60.00%) | 7 / 1 | 2 / 1 | 10 | 1 / 29 |
| Conditional repeat | 12/12 | 45/80 (56.25%) | 7 / 0 | 2 / 0 | 10 | 0 / 29 |

“Sparse availability” counts layouts with at least one measured action satisfying all five retained-reference limits. Ranking compares strict-sparse action pairs within the same layout; full-access and synthetic control rows are excluded. Research inadequacy includes any selected full output too, although every final development full output is adequate. False-safe/reject columns count **action rows**, while sparse choices/fallbacks count **layouts**. The in-fit result is optimistic and is kept separate from the held-layout folds.

The fixed role residual margins are .4545/.5912/.5111/.4029/.4636 in log-risk units, much larger than the approximate log(1.10)=.09531 limit. Pooled joint empirical coverage is 76%, measured on the same held-fold sparse residuals used to set the per-role margins. Zero-margin selection uses sparse actions in 11/12 layouts per development panel, but only 8/11 choices are adequate; it marks 11 primary / 12 repeat action rows falsely safe. The conservative margin rejects 29 adequate action rows per panel while still accepting an inadequate primary action at layout168. Margins do not repair ranking or extrapolation.

At layout168/M9, the research/conservative action cut_3_4_2 has predicted worst-role log risk **−75.2808** versus measured **+.117505** on the primary panel; the same input-only prediction is used on the disjoint repeat, whose measured risk is +.033299. Read-only feature inspection identifies a source-empty MM receiver packet: three descriptor rows correspond to effective K2; an empty row contributes zero to an unmasked pooled maximum while nonempty rows are about −.169718. One pooled-max column moves about **1,345 training standard deviations** (training mean −.169651, scale .0001261). This is severe feature-support extrapolation, with no detected label leakage or full-row sentinel collision. Development action-pair role log-risk-difference MAE is about 3.02, versus .27–.36 in held training folds. The overconfident prediction remains a selector failure in the denominator. No development-driven clipping or refit hides it.

Input-only research choices vary with native geometry: layout15/M10 chooses K1, layout31/M12 K3, layout154/M18 K4, layout142/M30 K3, and layout168/M9 K2 (primary inadequate). Five layouts choose full access. This is variation in the chosen computations, not a demonstrated conditional-quality guarantee. Mean eligible-MM permission work fractions are **.9380 research / .9819 conservative / .8799 oracle** on the primary panel; repeat oracle .8785. These are permission-work reductions of 6.20% / 1.81% / 12.01%, with all other paths and dense executor work counted separately.

### Group utility and executed work

Direct geometry preserves each receiver's out-degree and exact positive permission-weight multiset, while choosing nearest eligible sources and retaining native source measures. Float64 mass checks avoid rejecting a permutation of float32 values through summation rounding. Effective rewire preserves total pair work/global weight multiset, can change degrees, and saves the changed native ID pairs. Source-union collapse gives every receiver the union at unit permission: its extra pairs and mass are explicit.

| W-dir compared against | Primary: all-role learned wins / paired records | Conditional repeat | Work / permission-mass comparison |
|---|---:|---:|---|
| Frozen geometry prior | 4/46 | 2/46 | 24/46 primary records match work and mass |
| Root, split at exact work | 1/3 | 1/3 | 3/3 primary records match work and mass |
| Degree-matched direct geometry | 0/50 | 1/50 | Every receiver degree and exact positive-weight multiset preserved |
| Effective native-pair rewire | 13/50 | 12/50 | Total work and global weight multiset preserved; degrees can change |
| Source-union collapse | 0/50 | 0/50 | Mean extra 30.36 pairs and 30.8858 permission mass; 12/50 unchanged |

Every comparison uses final W-dir physical weights, matched native receivers, and saved permissions. Frozen-prior comparisons share requested cuts, but only 24/46 match work/mass; they are not a universal same-work ablation. In contrast, all 50 direct-geometry records match each receiver degree and positive-weight multiset exactly. Median learned/direct RMSE ratios are **1.0694/1.0794/1.0874/1.0287/1.0366** for the five primary roles; learned selection is worse in every median and has zero all-role wins. Rewiring changes actual native pairs and gives some local wins, but all five mean MSE benefits are negative. The learned graph has no broad source-selection advantage.

A localized useful-grouping witness exists at layout168/M9: cut_1_2/K2 and root/K1 each use **64 pairs with equal permission mass**. Split/root primary role RMSE ratios are **.7082/.6799/.8206/.9726/.8164**, improving every role. Across all exact-work split/root records, only 1/3 wins all roles on each panel. Mean primary near-turbine MSE benefit is negative (−1.02488e-4 m²/s²); conditional-repeat background benefit is negative (−4.21658e-6). A useful partition in one case does not establish useful learned source choice or safe input-only selection. Source-union collapse wins more often while using up to 93 extra pairs and 98.9823 extra permission mass, so its gains cannot be assigned to grouping alone.

Organizer next step: keep the new scorer and hierarchy opt-in, retain full fallback and the source-control evidence, and do not promote this ridge selector. Any further learning needs a separately authorized training-only control; no executor refactor or continuation starts here.

![Grouping utility and full fallback](../../../diagnostics/generated/directed_stable_20261002/figures/figure04_grouping_utility_and_fallback.png)

**Figure 4.** [PDF master](../../../diagnostics/generated/directed_stable_20261002/figures/figure04_grouping_utility_and_fallback.pdf). Primary/conditional repeats remain separate. Open work1 markers show same-weight full outputs. Source-selection utility uses degree/weight-matched geometry; policy bars separate availability, research failures and full fallback. Selected K counts nonredundant computations. In layout15/row46/K1, only 81/90 eligible MM pairs are permitted, while dense preparation executes **100 rows**, including 19 masked/self/padded rows. ME and EM each execute 5,120 rows; QM 20,480 and QE 1,048,576 remain full. Synchronized dense prepare+decode takes .538431 s. Packed support, packed latency and prediction parity are unavailable because the existing executor rejects the plan. Timing is one synchronized prepare/decode pass, excludes encoder/scorer/selection, retains diagnostics and has no repeated-trial uncertainty estimate. Remaining full/global paths preclude a whole-model speedup claim.

### Physical-weight drift and access error

For aligned vectors, `e=G_full−CFD`, `d=F_new_full−G_full`, `h=F_new_action−F_new_full`, the saved audit computes exactly:

`MSE(action)=MSE(G_full)+2⟨e,d⟩+||d||²+2⟨e+d,h⟩+||h||²`.

Signed terms account for output differences, not causal training attribution; they add in MSE, not RMSE. Earlier Run2110→G full drift is kept separately. For layout53's primary volume/root action, G MSE is 7.5352261e-5 m²/s². Physical co-adaptation contributes cross **−1.4865543e-5** plus squared **+7.0748422e-6**, giving full MSE 6.7561560e-5. Access then contributes cross **+2.2600287e-6** plus squared **+1.3123989e-5**, giving root-action MSE 8.2945578e-5: access loses more than physical fitting gains in this role. The adequate three-packet cut_1_5_6 instead adds only 2.0335092e-6 access MSE. These examples keep action availability separate from drift and access error.

Mean final W-dir access RMS over its 50 strict-sparse primary action records is .004858/.009295/.011968/.013836/.002080 m/s in role order. The corresponding disjoint values are .005006/.009197/.011039/.013738/.002049; neither is a CFD error score by itself. All 27,141,120 saved prediction entries are finite. Independent role RMSE reconstruction agrees within 6.844e-9 m/s, and maximum signed-MSE closure error is 3.470e-18. Control mass/work is recomputed from saved matrices; 153 stale rewire integer-work metadata entries are identified without altering the underlying predictions. Per-role terms and checks are in `wind/native_evaluation/main_signed_error_attribution.json` and `main_native_summary.json`.

## Thermal predictor: value recovery remains incomplete

Each temporary arm starts from G-u1300 with identical Adam/RNG state, all routes full, 40 routing parameters frozen and the same 80-physical-tensor scope as the focused pilot. Organizer shadow optimization is disabled. Only the value coefficient differs; anchor coefficient is .11492634546278409. Run1804 e4738 remains the retained reference. Reviews at 120 and 240 do not extend training. Native wrapper calls total 6,933 successful, zero failed, including student training, no-grad teacher-loss calls, setup and evaluation; evaluation contributes 114 including setup. No reference solver is called. Absolute metrics retain dataset-native, role-specific units; no SI conversion is documented for this stored benchmark.

Additional calibration families 0325/0340 join previously exposed 0310/0355. Value8 wins 32/32 absolute cells versus value4, 25/32 versus G and 2/32 versus Run1804. Finite-response counts are 23/32, 19/32 and 6/32. Fluid temperature, interface heat-flux proxy and solid temperature improve versus value4 in all 16 historical selector records. None beats Run1804 in any of its 16 absolute historical records for these three channels.

| Quantity | Run1804 | Value4 | Value8 | Measure/population |
|---|---:|---:|---:|---|
| Fluid temperature absolute | 2.450% | 3.865% | 3.696% | Mean case-relative RMSE, 20 records / 16 distinct cases |
| Interface q_normal absolute | 7.899% | 10.660% | 10.202% | Same; native heat-flux proxy before normalization |
| Solid temperature absolute | 2.257% | 2.955% | 2.732% | Same material receivers |
| Fluid temperature finite | 71.874% | 74.279% | 74.053% | Four Re90 families; raw i_plus−baseline |
| Interface q_normal finite | 99.689% | 102.452% | 101.894% | Four Re90 families |
| Solid temperature finite | 87.473% | 94.196% | 93.708% | Four Re90 families |
| Module-peak absolute error | .261088 | .340598 | .319412 | Mean absolute error, 112 module-case records; dataset temperature units |
| Module-peak finite error | .115124 | .119542 | .119408 | Mean absolute error, 25 calibration module records |
| Pressure-drop absolute error | .00123110 | .00175030 | .00169290 | Mean absolute error, 20 case records; dataset pressure units |
| Pressure-drop finite error | .00015663 | .00013789 | .00014364 | Mean absolute error, four families; value8 worsens versus value4 |

Relative RMSE entries are arithmetic means of case-level errors, not pooled mixed-unit MSE. All eight absolute and finite field/interface/material channel means remain worse than Run1804. Worst baseline module-peak error is .826355 for Run1804, 1.262417 for value4 and 1.210457 for value8; finite worst peaks are .866435/.981530/.974301. Small gains over the weaker arm do not establish recovery.

![Thermal physical, interface and material comparisons](../../../diagnostics/generated/directed_stable_20261002/figures/figure05_thermal_recovery.png)

**Figure 5.** [PDF master](../../../diagnostics/generated/directed_stable_20261002/figures/figure05_thermal_recovery.pdf). Stored family 0325, M5/Re90, uses all 7,823 valid temperature receivers of 8,192 and every active module's same-material-receiver peak. Motion is **module0, +.1500001 in y**, not x. Raw interface responses are not divided by displacement. Value8 fluid-T weighted RMSE is .243480 versus .267193 (value4), .281521 (G) and .143505 (Run1804), in dataset temperature units. The paired improvement is measured, while the retained-reference miss remains large. Display clipping does not affect metrics. Receiver noise floors are absent in all four families, so 128 finite field metric resolution statuses remain unknown; solver residuals are not receiver floors.

In that example, Run1804 MSE is .02059359535; value8's drift cross term is +.006805203495 and squared term +.03188371156, adding .03868891506 to final MSE .05928251040. Access terms are zero because all routes are full; closure error is below 1e-17. Independent recomputation over 6,440,252 finite saved prediction entries matches recorded RMSE within 1.281e-9. This separates full-model physical drift from sparse access without identifying a single causal training term. Thermal next step: retain Run1804 as the physical reference and keep the value8 result as an incomplete recovery; no maturation continuation is justified by these measurements.

## Inverse: stable sampling, weak conditional quality

The new model affinely encodes generated xy in the supplied common box, with x and y each ranging from −15 D to +15 D, as `z0∈[-1,1]²`. Bounds/input descriptors do not use the hidden clean position. Noisy states remain unbounded; only the frozen provider's candidate proxy clips generated coordinates. Visible states remain exact, with zero generated noise on visible/invalid slots. Native-row bounds are evaluation information only.

With exact endpoints `a=cos(πt/2), b=sin(πt/2)`, the target is `v=aε−bz0`. Recovery is `z0_hat=a zt−b v_hat`, `ε_hat=b zt+a v_hat`. The deterministic 20-step angular DDIM update never divides by tiny `a`; raw/box-projected terminals are saved separately. There is no guidance, forward-gradient optimizer, surrogate observation training loss or relabelled epsilon checkpoint. The new format rejects target/state mismatches while historical loaders remain unchanged.

The historical clamped epsilon schedule has reverse-mean amplification 24.57584 and clean-estimate amplification 316.22617; new algebra closes to 2.22e-16. These CPU checks motivate the repair without proving the cause of every old sample. A 100-update training-only eight-task diagnostic reduces masked loss 1.4236→.4307 and t=1 clean MSE .5064→.2452, while normalized sensor discrepancy worsens .01097→.01245. Its weights are discarded, and both primary arms restart fresh. Fitting capability does not imply sensor-consistent designs.

Primary arms share initialization, task, noise and time streams, training-only sensor scaling and frozen candidate embeddings. Dense access is applied after the same G-u4910 provider. Forward/scorer weights receive no inverse gradient; candidate-dependent states and links are recomputed. The provider retains the two-packet MM.90/QE.95 research action and its limited forward adequacy. Inverse fitting cannot remove that inherited limitation. Post-persistence sensor scoring uses G-u4910 physical weights in **full access** for every candidate/control; this is a common surrogate audit, distinct from access supplied to the denoiser.

| Fixed-noise development clean MSE | t=.25 | t=.50 | t=.75 | t=1 |
|---|---:|---:|---:|---:|
| I-v-G update1 | .148674 | .271353 | .215309 | .357167 |
| I-v-G update1000 | .180782 | .277382 | .117784 | .169699 |
| I-v-dense update1000 | .181013 | .278073 | .118048 | .168503 |

These are normalized xy MSEs on two exposed development tasks. High-time reconstruction improves; low-time development errors worsen. Update901–1000 mean masked velocity loss is .4210 in both arms; a final noisy batch is not substituted for it. Neither arm extends beyond 1,000.

### Exactly eight native trajectories

Task A is row 50/layout16/M10/WD300, hiding slot7. Original/changed observations are compared under I-v-G graph, I-v-dense full and the **same I-v-G weights** forced to full access (six draws). Task B is row 131/layout43/M13/WD300, hiding slot10, with original graph/dense controls (two draws). One initial-noise realization is shared per task. All eight attempts complete, with zero reverse failures/retries. Each draw persists 21 states, 20 raw clean estimates and 20 provider/link records **before target scoring**.

| Original-observation task/control | Observed RMSE [m/s] | Held RMSE [m/s] | Native support | Projection |
|---|---:|---:|---|---|
| Row50 I-v-G graph | .339370 | .191238 | Fail | 0/2 xy |
| Row50 I-v-dense full | .339514 | .191420 | Fail | 0/2 xy |
| Row50 same I-v-G forced-full | .339371 | .191239 | Fail | 0/2 xy |
| Row131 I-v-G graph | .0335233 | .0219309 | Pass | 0/2 xy |
| Row131 I-v-dense full | .0335788 | .0219373 | Pass | 0/2 xy |

All eight pass public generated-coordinate support and rotor clearance. Native support passes **2/8**: all six row 50 original/changed variants fail its native x lower bound. Graph terminal`[-12.7159,7.8090] D` lies beyond native lower x−11.1698 D; stored hidden target is`[2.1062,4.5911] D`. Row131 terminal`[2.5176,-7.0602] D` is supported but far from target`[6.8793,10.1011] D`. Hidden-target Euclidean errors 15.1674/17.7069 D are descriptive, not uniqueness criteria. Row131's clean-layout observed surrogate RMSE .0282058 m/s remains slightly better than its candidate .0335233. Passing the existing clean+.5 m/s surrogate criterion does not validate a generated layout physically.

The condition intervention replaces observed-array slot 14, raw sensor 22, Ux 8.29953098→5.53794909 m/s using the farther training-only endpoint at the same sensor. Intended coordinate`[12.2,5,.875] D` snaps to`[12.210355,5.002931,.885874] D`. Other coordinates/values stay fixed. The changed observation has **no independent physical realization**.

Both original/changed samples are scored against the same changed observations: approximately .591418 m/s (graph),.591598 (dense),.591419 (same-weight full). Differences are at float32 precision (≤6e-8 m/s), with no resolved conditioning benefit. Condition terminal displacements are 3.815e-6/8.092e-6/9.537e-7 D: 0/3 are bitwise identical, but 3/3 lie within 1e-5 D. Original graph/dense separations are .0218185 D on row 50 and .0191439 D on row 131; row 50 same-weight graph/full separation is .000169797 D. Only one noise realization exists per task, so posterior sample spread is **untested**; intervention spread is not sampler diversity.

Effective first-step access changes are row 50 MM 78→90 and sensor-QE 7640→8192, with ME/EM 5120 and QM 160 unchanged; row 131 MM 136→156 and QE 7752→8192. Initial candidates/noise/frozen embeddings match exactly. These are active link interventions whose small quality/output differences establish no material graph advantage at this checkpoint.

![Stable inverse estimates, trails, residuals and failures](../../../diagnostics/generated/directed_stable_20261002/figures/figure06_inverse_repair.png)

**Figure 6.** [PDF master](../../../diagnostics/generated/directed_stable_20261002/figures/figure06_inverse_repair.pdf). All eight raw clean-estimate paths are finite; 0/16 terminal xy values need projection. Six row 50 support failures stay in the denominator. Observed/held residuals are surrogate measurements without generated-layout CFD. The inset retains an old u64 trail; all four old draws decode to the same`[-15,+15] D` corner. Sensor panels align exactly, but **old hidden slot2 differs from new slot7**, so old/new quality is not a controlled same-task comparison. Old per-time raw clean estimates were not saved and are not reconstructed. Denoising paths are reverse sampling, not optimization progress.

Independent saved-array audit verifies finite states/estimates/links, exact visible-state/provider-coordinate preservation and absence of padded module/invalid environment rows in these tasks. Recomputed surrogate RMSE agrees within 1e-7 m/s (`inverse/post_sample_numerical_audit.json`). No redraw hides failed geometry. Inverse next step: use the single training-only conditioning-control recommendation on the first page if separately authorized. No further trajectories are drawn, and surrogate-supported samples are not promoted as physically valid designs.

## Resource use, remedies and repository closeout

First substantive inspection: **2026-10-02 19:14:00 UTC (15:14 EDT)**. Science stops by 00:14Z and delivery by 01:14Z on October 3 (20:14/21:14 EDT on October 2). Last GPU1 process ends 23:08:29Z (bound rounded up to :30); last GPU2 regression ends 21:57:25Z. No GPU work follows. All science completes before the five-hour cutoff, leaving the final hour fully available for closeout. The aggregate cap is verified through conservative lane ownership rather than incomplete per-process sums.

| Lane / component | Recorded interval or timer | Accounting boundary |
|---|---:|---|
| GPU1 ownership bound, Wind | 14,070 s / 3.90833 h; 19:14→23:08:30Z | Entire lane window, including idle and untimed setup |
| GPU2 ownership bound, Thermal/inverse/checks | 9,805 s / 2.72361 h; 19:14→21:57:25Z | Entire lane window; serial ownership on GPU2 |
| Aggregate upper bound | **23,875 s / 6.63194 h**, cap 10 h | Sum the two bounds once; component timers below are nested inside |
| Wind final evaluator | 4,496.18 s | GNU whole process, including both slices |
| Wind failed evaluators | 165.34 + 411.51 = 576.85 s | Whole processes; all three evaluations total 5,073.03 s |
| Wind corrected paired training window | 1,635.612 s observed | W-dir 700 co-adaptation and W-summary 300+700; startup/exit boundaries incomplete |
| Wind earlier valid W-dir scorer300 | Unknown process wall | Completed before the corrected pair; not substituted with a guessed timer |
| Wind discarded co-adaptation attempt | ≥326 s observed | Fifteen discarded updates; startup excluded from this lower bound |
| Thermal measured GPU-associated processes | 2,715.6016 s | Nonoverlapping whole processes, including measured failure |
| Thermal estimates / pre-CUDA failure | 2 s estimated failure; 1.2 s measured pre-CUDA | 2,718.8016 s including both; do not add per-arm update timers |
| Inverse diagnostic | 125.3005 s | Runner timer; CLI/provider/task setup excluded |
| Inverse primary pair | 1,184.6900 s | Cumulative runner across resumes; optimizer/review breakdowns not added again |
| Inverse final eight trails | 44.06 s | GNU whole process; 40.6951 s per-trail sum nested inside |
| Inverse failed preflight | 60 s estimate only | Zero trajectories; not a measured timer or rigorous setup bound |
| Final GPU2 regression | 5.30 s whole pytest process | 4.07 s test-suite duration nested inside |

Wind evaluation failures are retained/count once: 165.34 s for a missing tree-depth metadata attribute and 411.51 s for a float32 reduction-tolerance failure. They consume no optimizer updates. The original 54 native panels are restored and validated on retries; failed-attempt partial outputs remain separately archived. Scientific summaries use the final successful pass and recompute control work/mass from saved permission arrays where older runtime metadata is stale.

During initial Wind co-adaptation, main review caught a missing scorer-gradient reset. All 15 attempted co-adaptation updates were discarded; nine restricted steps had accumulated gradients. Valid scorer-u300 weights/optimizer were restored. The final pair contains 700 clean co-adaptation updates per arm. These 15 discarded steps plus 100 inverse diagnostic updates consume 115/400 extra updates.

An inverse preflight initially confused observed-array slot 14 with raw sensor 14. Its guard stopped before any reverse step/attempt record, consuming **zero trajectories**. Mapping to raw sensor 22 was corrected and tested. Its unmeasured time remains an estimate. Thermal startup/evaluation failures are included in its accounting. Completed measurements, rather than startup parity, support the results.

Final regression: **108/108 tests pass in 4.07 s**, whole pytest process 5.30 s, on verified physical GPU2. Tests cover directional frame/permutation behavior, zero-residual geometry, local tree/typed-route separation, source IDs/measures/work, weighted controls, saved-panel restoration, resume-history archival, velocity endpoints/oracle conversion, masked noise/visible states, target-free bounds/provider, sensor mapping and historical spatial/checkpoint behavior. A subsequent 17/17 CPU directional test run covers the final metadata and optional UTC-deadline changes. Full configured Ruff and `git diff --check` pass. Initial CPU-only execution's 12 required-GPU availability assertions were rerun on the authorized device and are resolved.

Durable changes are the opt-in directional scorer/depth-limited module index; maintained Wind pair trainer/evaluator; Thermal pair trainer/evaluator, case selector/configuration and tests; and the separate bounded-velocity model, Wind adapter/runner and tests. Future Wind resumes archive executed post-checkpoint rows before rebuilding the active history. The trainer accepts an optional explicit UTC stop time, so a permanently expired round date does not block reuse. Formatting the 12 new Python files preserves their parsed syntax trees exactly; existing modified files are not reformatted wholesale. Historical defaults and trust policies are preserved; no production promotion occurs.

This clone uses `.githooks` for the existing pre-push artifact gate. The entire outgoing range is audited, including paths added and later removed. Only durable source/tests/configuration/report changes are uploaded; generated arrays, checkpoints, figures and one-time analyses/renderers stay ignored. Commit/push synchronization is verified in task closeout; the containing Git commit identifies this revision. No persistent learning job, reference solve or automatic follow-on remains.

## Selected local figures and source index

The short local [figure index](../../../diagnostics/generated/directed_stable_20261002/figures/INDEX.md) retains six PDF masters plus small raster companions for these direct embeds. All are visually inspected. Main numerical summaries are `wind/native_evaluation/main_native_summary.json`, `main_signed_error_attribution.json`, `wind/w_dir_selector_audit.json`, `thermal/value_recovery_pair/evaluation_final_u1540/main_saved_array_audit.json`, and `inverse/post_sample_numerical_audit.json`. Local links are checked before closeout. Generated files and one-time renderers are not uploaded.

Velocity parameterization follows [Salimans and Ho](https://arxiv.org/abs/2202.00512), deterministic reverse construction follows [Song, Meng and Ermon](https://arxiv.org/abs/2010.02502), and endpoint consistency is motivated by [Lin et al.](https://arxiv.org/abs/2305.08891). This implementation uses its own sign convention/exact endpoints. Published image quality, speedups and distillation results are not HONF evidence; experimental claims here come from saved native arrays.
