# HONF Round13-6: faithful integration and useful organization

Round13-6 restores the original ME role amplitudes while reusing live scene-normalized keys, factored scorer affines and a shared paired query plan. The fixed-state native replay preserves the original joint95 candidates and connected gradients. Its measured prepared 48-case native forward/loss/backward call is faster and uses less peak memory; complete native inference is slightly slower. Typed organization-conditioned competition is a separate change that preserves the original role-local strength statistics and original-member access.

The bounded investigation completed fixed-state measurements, 15 qualification epochs and one scratch TRAIN600 lineage through 100 epochs. H100 has seven lower all-VAL89 mean errors than matched-age ME100, with fluid/surface temperature regressions and unresolved local errors. Completed measurements are distinct from any startup-only continuation. The original R13-5 `thermal_r13_5_me_joint95_train600_formal_v1` epochs 100 and 500 remain the scientific anchors. No inverse fitting, solver campaign or pristine TEST evaluation is included.

## What changed

O preserves the original equation, `c_t = rho_t sum_i p_i^t V_i`, with the original prior-weighted score calibration, candidate-conditioned role statistics, four applied MM→EM→EE→ME relation blocks, two contextual cycles, original IDs and values, one joint physical head and one joint95 cutoff. The execution changes prepare normalized member keys and affine key terms once per live scene and share only the query/entrance work whose dependencies agree between full and restricted reads. Full and restricted likelihood calibration remains separate. There is no detached persistent feature cache or pooled replacement for original members.

H changes member competition within each role: every organization row can reach M or E members, and each row's selected members compete before paths coalesce onto original IDs. Typed normalization slices member columns, while retaining all eligible address rows. The strength `rho_t` still uses O's marginal role statistics. K is an independent optional retained-prior fraction, disabled in the initial O/H comparison; it does not create a second cutoff or a per-role quota.

For role t, O uses `p_i = P_i exp(ell_i)/Z_t`. H instead uses `p_i = sum_a [(w_a d_at/L_t) (R_ai exp(ell_i)/Z_at)]`, with `d_at = sum_selected R_ai`, `L_t = sum_a w_a d_at` and `Z_at = sum_selected R_ai exp(ell_i)`. Empty row-role terms and empty selected roles return exact zero. Both laws multiply their coalesced original-member value read by the same O-derived `rho_t`. The implementation uses three contractions across the role-split member axes and avoids a batch×query×address×member allocation. Legal M-only or E-only query sets remain allowed.

## Figure index

| Figure | Evidence |
| --- | --- |
| 1 | Same-state role attenuation and head sensitivity |
| 2 | Original-law complete inference and paired training-macro cost |
| 3 | Finite-dose organization response at fixed original ME500 weights |
| 4 | Nine-field all-VAL89 mean, p90 and worst |
| 5a–b | Common-scale u/temperature fields and signed residuals |
| 6 | Per-original-module errors and native interface profiles |
| 7a–b | Complete sampled training history and actual validation review |
| 8 | Native input dependence and live learning paths |
| 9 | Applied four-block relations and three full-read spatial views |
| 10 | Query entrance, row competition and coalesced member trace |
| 11 | Trained H100 finite-dose grouping effect |
| 12 | Fidelity versus executed work and complete cost |

## Original-law execution replay

The replay uses the literal original ME100 and ME500 checkpoints on fixed TRAIN0006/M1 and TRAIN0231/M12, with Q1024 and the complete native query union, including the maintained finite-difference observations. Candidate masks match exactly in both read modes. At ME100, separate-versus-paired O fields differ by at most `7.16e-7`; old-versus-O parameter-group relative gradient differences are `5.54e-7`–`4.55e-6`, with no gradient-presence mismatch. At ME500 the largest tested group difference is `2.04e-5` for typed strength, with maximum absolute gradient difference `1.26e-7` in that group. These are FP32 execution differences at fixed weights, not trained endpoint gains. Same-layout old/old gradient repeats and every native observable comparison are retained with the replay.

The native field/stencil differences are measured independently of the core field differences, because differentiation and physical units can amplify reduction variation. [ME100 replay and gradient evidence](../../../../diagnostics/round13_6_20261027/faithful_replay_e100.json) and [ME500 replay and gradient evidence](../../../../diagnostics/round13_6_20261027/faithful_replay_e500.json) retain the absolute differences, native loss terms, candidate equality, paired shared counts and all parameter groups. No checkpoint was overwritten.

The largest native output relative L2 difference is `1.70e-6`; the largest absolute interface-array difference is `2.27e-4` in packed native units. Ordinary old/old gradient repeats reach `9.49e-6` relative difference in a parameter group, while old/O differences reach `2.04e-5`. Candidate-dependent native calculations and their real gradients therefore agree at the measured FP32 scale, with repeat variation documented rather than a bitwise-equality assertion.

## Role amplitude diagnosis and isolated K control

At the same original weights, context, candidates and calibration, the A law satisfies `c_A,t = d_t c_ME,t`, where `d_t = Z_t/(Z_M+Z_E)`. On TRAIN0003 and VAL0678 at ME100/ME500, the identity's maximum relative L2 residual is `1.30e-6`. Across this first-512-native-union-row panel, mean E factors range from `0.397%` to `12.075%`; they represent an added amplitude attenuation, separate from the original typed gate. The same-state joint-field changes range from `0.846%` to `4.950%` in relative L2. A one-percent independent context attenuation also measures downstream head sensitivity. These equation controls do not attribute all differences between independently trained A/B and ME endpoints to that factor.

![Same-state E attenuation, identity residual, joint-field effect and head response for original ME100 and ME500 on two fixed scenes.](../../../../diagnostics/round13_6_20261027/figures/01_role_amplitude.png)

Figure 1. Each panel uses the original checkpoint at identical inputs; F/R mean full/restricted. The relative identity residual is FP32 scale while the changed equation has a resolved field effect. These first 512 native-union rows differ from the separate historical 512-query spatial organization panel. [PDF master](../../../../diagnostics/round13_6_20261027/figures/01_role_amplitude.pdf). [Numerical role, value and context evidence](../../../../diagnostics/round13_6_20261027/amplitude_diagnostic.json).

K was evaluated independently on the same native stencil before any proposed use in H. It preserves the full reader at fixed weights and scales each restricted role only by its own selected/legal prior fraction. The ME100 M1 case's weighted native objective worsens by approximately 8%, while the M12 case has a small mixed change. This panel does not justify bundling K into typed competition. Its exact full-reader identity, legal-empty behavior and singleton limitation are also covered by focused tests; a singleton retained role can still jump when selection changes from empty to present.

The native gradient diagnosis also covers the original ME100/ME500 and independently trained A100/B100 endpoints on the fixed M1/M12 TRAIN probe. A same-state A equation switch changes gradients in every tested parameter group, including the joint head, contextual relations, member values, member scorers, address scorers and typed strength. The largest relative change occurs in the ME500 M1 address scorer, whose original gradient norm is only `6.51e-4`; absolute norms and native objective terms are retained to avoid interpreting a small denominator as a large physical effect. The independently trained A/B norms describe adapted states, while the same-state switches isolate the equation. [Complete native parameter-group evidence](../../../../diagnostics/round13_6_20261027/amplitude_gradient_diagnostic.json).

## Measured execution

The ordinary synchronized ME100 comparison uses physical GPU 1, identical weights, native scenes and three alternating repeats after warmup. Complete host-load-through-native-extraction medians increase from `144.29→152.49 ms` and `148.45→159.76 ms` at M1 full/restricted, and `148.34→157.13 ms` and `151.85→160.93 ms` at M12. There is no measured complete inference speedup.

The prepared 48-case native forward/loss/backward median falls from `3.174 s` to `1.729 s` (45.5%), with peak allocated memory falling from `15.51` to `10.40 GiB` (32.9%). The panel is the first 48 ordered TRAIN600 members (45 M1, one M2 and two M3 cases), divided into two 24-case microbatches; it is not a stratified whole-training-population estimate. This measurement includes context preparation, both physical reads and non-reentrant replay/backward, while excluding fresh sampling and parameter application. Complete epoch cost is reviewed separately. Instrumented spans are diagnostic, overlap and are never added together; the instrumented O call still records 1,946 stream synchronizations versus 2,363 for the original call. Address scoring, support composition, prefix sorting/indexing and reporting counters remain real costs. [Exact fixed cost-panel identity](../../../../diagnostics/round13_6_20261027/cost_panel_identity.json).

![Measured complete native inference, prepared paired training-macro latency and peak allocated memory at fixed original ME100 weights.](../../../../diagnostics/round13_6_20261027/figures/02_equivalent_execution_cost.png)

Figure 2. All three repeats are retained. Sparse candidate/value work and complete elapsed time are separate measurements. The inference result includes preparation and native extraction; the paired training result excludes fresh sampling and parameter application. [PDF master](../../../../diagnostics/round13_6_20261027/figures/02_equivalent_execution_cost.pdf). [Ordinary timings and separate instrumented evidence](../../../../diagnostics/round13_6_20261027/faithful_replay_e100.json).

The matched fixed-TRAIN150 timing control completes three epochs at `19.91/20.17/20.50 s`, versus `12.08/12.10/12.20 s` for O's first three epochs, a 40.0% median reduction in complete training time. The maintained training measurement includes data transfer, native preparation, full/restricted loss, backward/replay and parameter application; monitoring is separate. Case ordering and query draws match exactly. Original ME used physical GPU 1 and O used physical GPU 2, both RTX 6000 Ada; this is a real complete-epoch comparison across two devices, separate from the isolated fixed-weight single-device macro trial. Full three-epoch engine process times, including ordinary boundary work, were `76.39` and `52.77 s`. [Completed qualification and cost review](../../../../diagnostics/round13_6_20261027/qualification_review.json).

## Useful organization at fixed weights

The ME500 transplant control prepares the original scene once and changes only the O/H read hook. Proposal weights, P, candidates, calibrated likelihoods and O-derived gate statistics agree. H performs row-conditioned competition over original members, with a separate unit conditional distribution for each nonempty role; it does not allocate a joint M/E amplitude budget. This is a fixed-weight mechanism control, not a trained H500 predictor result.

On the historical 512-query TRAIN0003/M1 and VAL0678/M10 panel, support/mass-selected four-cycles use 25%, 50% and 90% of their local feasible capacity. Fifteen nondegenerate point/role/read groups give 90 H/O arm-dose outputs, with actual moved Q mass from `1.55e-5` to `0.11948`. P and retrieval-row residuals stay within `1.22e-7` and `5.96e-8`, and support/candidates remain exact. Maximum H normalized joint-field changes are `4.94e-5` at M1 and `0.01113` at M10, versus O changes below `9.54e-7`; same-layout repeat variation is exactly zero in this CPU layout. These resolved H changes establish a row-composition effect beyond the nearly unchanged member marginal, without establishing native stencil accuracy or physical causality.

![Finite-dose H and O pointwise field responses, numerical marginal residuals and exact empty/singleton native role degeneracies at original ME500 weights.](../../../../diagnostics/round13_6_20261027/organization/typed_organization_ME500_finite_dose_v5.png)

Figure 3. The directory perturbation is query-specific, so all responses are pointwise outputs at the existing native query coordinates; no derivative is assembled from different perturbed directories. Nine remaining point/role/read combinations have zero or one selected member and admit no nontrivial candidate-preserving cycle or row transfer. Separate synthetic positive controls are labeled in the metadata and never enter native outputs. [PDF master](../../../../diagnostics/round13_6_20261027/organization/typed_organization_ME500_finite_dose_v5.pdf). [Compact numerical summary](../../../../diagnostics/round13_6_20261027/organization/typed_organization_ME500_finite_dose_v5_summary.json).

## Qualification and formal candidate selection

The completed qualification cost is 15 epochs: three original-ME timing epochs, six O epochs and six H epochs. O and H both stop at epoch 3 and recover the same lineage through epoch 6, retaining complete history, optimizer/RNG/sampler/provider state and loss curves. The actual O checkpoint loader reconstructs all 194 saved tensors exactly. Native H separate-versus-paired replay has no gradient-presence mismatch, with parameter-group relative differences at most `1.64e-6`. [O checkpoint recovery](../../../../diagnostics/round13_6_20261027/O_e3_checkpoint_load.json) and [H native paired/backward replay](../../../../diagnostics/round13_6_20261027/H6_native_replay_profile.json).

At H epoch 6 all eight sampled native terms are below their epoch-1 values. M/E value gradients remain active on fixed TRAIN M1/M2/M5/M12 cases in both read modes, and the smallest material-temperature response-to-repeat ratio is 362 for the prescribed +10% heat-input probe. This checks native input dependence; it is not a solver-response accuracy claim. No collapse, forced role winner or diversity objective is needed. Qualification establishes execution and initial learning health only.

One new formal lineage is selected: scratch seed-0 H on TRAIN600 through epoch 100. C remains exactly O, so original ME100/ME500 supplies the scientific control after the original-law replay and actual recovery checks; a redundant C100 fit is omitted. The actual formal builders produce the same 194 initial tensors, TRAIN-only normalizer, ordered TRAIN600/VAL89 membership and query sampling arm as original ME. K remains disabled. [Selection rationale](../../../../diagnostics/round13_6_20261027/formal_selection.json) and [actual initialization comparison](../../../../diagnostics/round13_6_20261027/formal_initialization_comparison.json).

## Completed H100 predictor review

Scratch H completed all 100 epochs on TRAIN600, retaining the epoch-100 state, complete sampled training history and actual all-VAL89 validation. Every sampled native training term falls between epoch 1 and 100; the final-to-initial ratios range from 0.0259 to 0.1084. The final 20-epoch mean terms remain 34–53% below the preceding 20-epoch means. These are sampled training trajectories, not validation improvement or convergence evidence. The protocol saves one full VAL89 validation review at epoch 100; no intermediate validation points are invented.

Against matched-age original ME100, seven of nine mean errors are lower in both full and restricted reads. The table reports the deployed restricted reader on all 89 exposed canonical VAL cases, with equal-case native RMSE for the first eight observables and the maintained module-peak endpoint RMSE. Units remain the packed dataset/native temperature or maintained interface-target units. Negative changes mean lower error; the mature ME500 reference appears as training-age context in the figure, not as a matched-age control.

| Observable | ME100 mean | H100 mean | H100 p90 | H100 worst | Mean change | p90 change | Worst change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| u | 0.11495 | 0.10808 | 0.13132 | 0.15374 | -5.98% | -3.68% | -9.89% |
| v | 0.01861 | 0.017398 | 0.020421 | 0.023569 | -6.51% | -2.42% | +0.48% |
| p | 0.064694 | 0.060319 | 0.069762 | 0.082815 | -6.76% | -5.06% | -4.33% |
| omega | 0.27085 | 0.26617 | 0.29592 | 0.32244 | -1.73% | -0.36% | -5.85% |
| Fluid temperature | 2.2222 | 2.3443 | 2.7538 | 4.5306 | +5.50% | +4.98% | +25.48% |
| Surface temperature | 1.9319 | 2.0141 | 2.8212 | 4.9288 | +4.26% | +3.70% | +24.28% |
| Material temperature | 1.7962 | 1.7557 | 2.6094 | 4.4502 | -2.25% | +1.39% | +0.79% |
| q-proxy | 4.5039 | 4.2413 | 5.1986 | 6.325 | -5.83% | -7.35% | -7.69% |
| Module peak endpoint RMSE | 1.9187 | 1.7323 | 2.7867 | 4.3245 | -9.72% | -0.61% | -12.50% |

VAL89 covers M3 (24 cases), M5 (25), M7 (25) and M10 (15). Other declared module-count strata have no VAL cases and are not zero-error results. Flow u/v/p/omega means and q-proxy improve in every represented stratum. Fluid-temperature means worsen by 9.71/5.63/5.00/0.17% at M3/M5/M7/M10; surface-temperature changes are −0.62/+6.38/+3.99/+7.34%. Material-temperature mean worsens 1.28% at M7 while improving in the other represented strata. The fixed 21-module detail panel improves material RMSE on 14 modules and absolute module-peak error on 12; gains concentrate at VAL0293 while VAL0678 remains mixed. All nine H100 mean errors remain above the mature ME500 anchor, whose longer trajectory is not an age-matched comparison. These cuts and every full/restricted mean, p90 and worst are retained in the [endpoint metric CSV](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/endpoint_metrics.csv).

Temperature misses are substantive: restricted fluid/surface means rise 5.50/4.26%, and their worst errors rise 25.48/24.28%. H100's worst fluid, surface and material temperatures occur at VAL0300; this is a reported tail case, not an added error-selected plotting panel. Restricted v and material worst errors also rise slightly (0.48/0.79%). Own-full→restricted mean penalties range from 2.47% to 9.26%; pressure, omega and v penalties are 9.26/8.47/7.77%. Restriction is therefore not free, and part of the temperature miss already exists in full prediction. No extra amplitude correction is introduced at epoch 100.

![Nine native observables on the same exposed VAL89 cases, comparing full and restricted ME100 and H100 with mature ME500 as training-age context.](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/01_all_val89_metrics.png)

Figure 4. Every field shows equal-case mean, p90 and worst native error; ME500 has a longer training history. H100 improves seven means while fluid/surface temperature and several tails remain worse. [PDF master](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/01_all_val89_metrics.pdf). [Full all-VAL89 evaluation](../../../../diagnostics/round13_6_20261027/evaluations/H_e100/result.json).

![Target, original ME100 and H100 velocity u fields and their signed residuals on the fixed TRAIN0003 and VAL0274, VAL0293, VAL0678 panel.](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/02_physical_fields_u.png)

Figure 5a. Deployed restricted u fields use one absolute scale and one shared signed prediction-minus-target residual scale across all four fixed input-selected scenes. Original module IDs and native coordinates are retained; local wake residuals persist. [Two-page PDF master](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/02_physical_fields.pdf).

![Target, original ME100 and H100 fluid and material temperature fields with common scales and signed residuals on the same fixed four scenes.](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/02_physical_fields_temperature.png)

Figure 5b. Fluid and original-member material temperature share the native temperature scale; both residual columns use the same symmetric scale. H100 does not uniformly repair local thermal plumes. Rows are TRAIN0003/M1, VAL0274/M3, VAL0293/M7 and VAL0678/M10. The material coordinates reuse the protected native cache. [PDF master, page 2](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/02_physical_fields.pdf).

![Per-original-module material-temperature RMSE on four fixed scenes and surface-temperature and q-proxy profiles for input-selected original module zero at VAL0293.](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/03_fixed_interfaces.png)

Figure 6. All present original modules are shown, with full and restricted ME100/H100 readings; no module is chosen by error. VAL0293 module 0 has improved material/surface fit, while other modules and VAL0678 show mixed changes. The q-proxy profile retains large differences from saved native targets; this maintained proxy is not a verified physical flux. [PDF master](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/03_fixed_interfaces.pdf). [All fixed-panel local errors](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/fixed_panel_local_errors.csv).

![Complete H sampled native training contributions over epochs one through one hundred, separately plotted for eight physical objective terms.](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/04a_training_terms.png)

Figure 7a. The eight weighted normalized sampled training contributions retain their complete scratch history. Their sustained decline supports further learning; no artificial scalar organizer objective is constructed. [Two-page PDF master](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/04_training_history.pdf).

![Actual H all-VAL89 restricted validation contributions at the single saved review epoch one hundred.](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/04b_validation_terms.png)

Figure 7b. Validation pools valid native elements and is distinct from sampled training macro means. Only the actual epoch-100 review is plotted; this does not establish a validation trend. The complete source history and normal in-process loss curves remain retained. [PDF master, page 2](../../../../diagnostics/round13_6_20261027/figures/endpoint_review/04_training_history.pdf).

## H100 native dependence and learning paths

The fixed TRAIN0006/M1, TRAIN0051/M2, TRAIN0322/M5 and TRAIN0231/M12 panel retains live M/E member-value gradients in both read modes. The smallest +10% prescribed-heat native response divided by same-input repeatability across its eight observables, four cases and two modes is 8,545. Native separate/paired backward has no gradient-presence mismatch and maximum tested parameter-group relative difference 3.23e-6. All four applied relation blocks remain recorded; M1 has no legal distinct MM pair and consequently zero positive-edge MM adjoint. Tiny first-eight-query proposal-score gradients or an empty selected role are interpreted with actual support and preserved native dependence, not treated as collapse. [Native H100 replay and instrumented cost](../../../../diagnostics/round13_6_20261027/H100_native_replay_profile.json).

![H100 native field responses to prescribed heat versus repeatability, M and E value and proposal gradients, and actual-positive-edge adjoints for four applied relation blocks.](../../../../diagnostics/round13_6_20261027/health_epoch_cost_review/response_live_path.png)

Figure 8. The response is TRAIN-standard-deviation-normalized RMS prediction change from +10% prescribed heat; it is not accuracy against a changed-input solver solution. Gradients and applied-edge adjoints establish model learning paths, not physical causality. M1's zero MM entry is structurally legal. [PDF master](../../../../diagnostics/round13_6_20261027/health_epoch_cost_review/response_live_path.pdf). [Health and epoch-cost review](../../../../diagnostics/round13_6_20261027/health_epoch_cost_review/health_epoch_cost_summary.json).

## Trained H100 organization review

The completed trained H100 archive reuses the historical 512-query TRAIN0003/M1 and VAL0678/M10 inputs and actual original IDs. At identical H100 weights, the O/H read-hook comparison preserves proposals, P, candidates, calibrated likelihoods and original-ME gate statistics. The three-view full-read spatial panel shows original ME100 M, H100 M and H100 E on the same scene and fixed native query. Separate heatmaps retain applied MM/EM/EE/ME blocks and their saved cycle/stage provenance; the directory is not inferred from a transposed index or an attention picture.

![Actual applied four-block relations at ME100 and H100 plus original ME100 M, H100 M and H100 E full-read spatial views at one fixed native query in VAL0678.](../../../../diagnostics/round13_6_20261027/organization/organization_H100_routes_spatial.png)

Figure 9. All three spatial views use VAL0678 q256/id4224 and the same 0–12 × 0–6 native scene coordinates. Colors and sizes encode post-strength reads on original members. These are full reads; the separate restricted trace below legitimately has no E at q256. Applied block heatmaps retain receiver/donor IDs and actual stage provenance. [PDF master](../../../../diagnostics/round13_6_20261027/organization/organization_H100_routes_spatial.pdf).

H100 does not learn disjoint physical groups. On VAL0678, mean retrieval-row cosine/Jaccard overlap is 0.996/0.933 for EE, 0.942/0.939 for EM, 0.710/0.342 for ME and 0.463/0.812 for MM, versus ME100 values 0.982/0.712, 0.916/0.827, 0.719/0.575 and 0.234/0.538. Strong EE/EM overlap remains a concrete limitation; the operator can still use different row compositions, and no target group count or diversity penalty is imposed. Full H100 VAL0678 has 10 M and 148 eligible E candidates, with median selected likelihood-score ranges 2.349 and 3.311. Restricted E is empty in 147/512 queries and singleton in 80, leaving 285 multi-candidate queries. TRAIN0003 has a singleton M role throughout; its restricted E is empty in 231 queries and singleton in 20. These exact degeneracies explain some weak pointwise effects without establishing scene-wide collapse.

![Restricted H100 query entrance weights, top-row member competition and coalesced original-member contributions at three fixed queries in two scenes.](../../../../diagnostics/round13_6_20261027/organization/organization_H100_query_trace.png)

Figure 10. Queries q0/q256/q511 are fixed by historical panel position. Each role traces entrance α through a selected row's conditional competition to coalesced original-member probabilities. Empty E at q256 is displayed explicitly; no role is forced into the joint95 prefix. Overlapping paths coalesce before the value read, and contextual members remain individually accessible. [PDF master](../../../../diagnostics/round13_6_20261027/organization/organization_H100_query_trace.pdf).

Fifteen support/mass-selected local cycles at 25/50/90% feasible dose yield 90 actual H/O pointwise outputs. Actual moved Q mass spans 1.23e-6–0.03895; candidate masks and retrieval support remain exact. Q row/column residuals are zero in the stored control, while reconstructed FP32 P and retrieval-row residuals reach 1.15e-7 and 1.19e-7. H's maximum normalized joint-field response is 0.05035 at VAL0678, versus O's maximum 1.25e-6 and zero same-layout CPU repeat noise. The fixed-marginal row-composition effect is resolved at meaningful finite dose. It establishes an operational grouping effect; the matched-age field review supplies observed prediction gains and misses, rather than assigning all accuracy differences to that effect or claiming physical many-body causality.

![Actual trained H100 finite-dose row-composition response versus O, with marginal residuals and exact native support degeneracies.](../../../../diagnostics/round13_6_20261027/organization/typed_organization_H100_finite_dose.png)

Figure 11. The query-specific control changes membership pairing at approximately fixed member marginal. Responses are pointwise joint fields, not native derivatives from separately perturbed stencil points. Nine sampled review entries admit no nontrivial candidate-preserving intervention; separately labeled synthetic row-transfer controls are not native field evidence. [PDF master](../../../../diagnostics/round13_6_20261027/organization/typed_organization_H100_finite_dose.pdf). [Organization, overlap, likelihood, trace and finite-dose review](../../../../diagnostics/round13_6_20261027/organization/organization_H100_review.json).

## H100 fidelity, work and complete cost

Across all VAL89 primary native queries, restricted execution computes 5,289,544 of 50,421,760 full member-score/value rows (10.49% retained; 89.51% skipped). M rows retain 4,148,180/4,276,224 and E rows 1,141,364/46,145,536. The maintained counts show selected original members are chosen before their expensive query-conditioned scores and values. Address preparation, relation/support composition, prefix sorting and H's three contractions remain real work: each H contraction executes 28,548,382,720 address/member pairs per full or restricted evaluation, for 85,645,148,160 across the three. The pair calculations do not disappear with the smaller candidate set. Full reading is a separately measured control/fallback with its own accuracy and cost.

A same-H100-state O/H read-hook timing control isolates the extra competition without changing context, candidates, calibration, gate or learned tensors. On physical GPU 1 after warmup and three alternating repeats, complete native medians rise O→H from 155.14→169.52 ms and 160.44→173.34 ms at M1 full/restricted, and 162.08→171.93 ms and 166.54→177.59 ms at M12 (6.1–9.3% slower). The prepared 48-case paired native forward/loss/backward medians are 1.4601 and 1.4579 s with overlapping repeat spreads; this does not resolve a training-time overhead or saving from H's contractions. Peak allocated memory rises from 7.431 to 7.639 GiB (2.8%). The panel is the same biased first 48 TRAIN600 members described above, and excludes fresh sampling and parameter application. Figures 2 and 12 use different trained states and selected supports; only each within-state timing comparison isolates its stated execution change.

![Complete native latency, paired native training-macro cost, peak allocated memory and all-nine-field fidelity versus actual executed member-score work for H100.](../../../../diagnostics/round13_6_20261027/figures/07_H100_cost_fidelity.png)

Figure 12. O and H are read-hook controls at the same completed H100 state. Dots retain all timing repeats. The fidelity panel joins only the two measured full/restricted modes; the lines are not an interpolated budget sweep. All nine restricted mean errors rise relative to H's own full read while 89.51% of member scores are skipped. Sparse work and lower memory are not complete inference speedups. [PDF master](../../../../diagnostics/round13_6_20261027/figures/07_H100_cost_fidelity.pdf). [Same-state timing evidence](../../../../diagnostics/round13_6_20261027/H100_same_state_H_vs_O_cost.json) and [work/fidelity summary](../../../../diagnostics/round13_6_20261027/H100_cost_fidelity_summary.json).

H100 and original ME100 match case order and query draws in every one of their first 100 epochs. Mean complete training-loop times are 31.317 versus 58.587 s per epoch (46.5% lower for H100), with sums 3,131.66 versus 5,858.73 s. This historical whole-trajectory observation includes different learned supports and is separate from O's frozen-state equivalence trial. H100's training-loop median is 30.41 s per epoch; the engine process takes 3,325.76 s including ordinary boundary/monitoring work. These whole-run observations are distinct from the prepared macro trial and its frozen-state execution comparison. Instrumented stage/profiler spans are retained separately and are nonadditive. GPU 1 was exclusive to the measured call; unrelated GPU 0 work was preserved and CPU-only figure/organization analysis ran concurrently.

## Final candidate decision and next evidence

H is selected as the round's one final formal candidate for same-lineage continuation from 100 to 500 total epochs. This commits 400 additional epochs and 500 total new full-data epochs, within the 600-epoch cap; qualification remains 15/72 and no second scratch arm is introduced. The decision uses the complete field, organization, path-health, trajectory and cost review above. Seven restricted means, six p90 errors and five worst errors improve against matched-age ME100, all eight sampled training terms continue falling, native input dependence is intact and finite-dose row composition has a resolved effect. Temperature regressions, increased own-full→restricted penalties and inference overhead remain explicit misses. A 100-epoch endpoint is preliminary and does not settle mature architectural quality. [Recorded evidence and continuation decision](../../../../diagnostics/round13_6_20261027/final_candidate_decision.json).

For the predictor, the gain is observed matched-age flow, material, q-proxy and module-peak accuracy; the miss is fluid/surface temperature and several tails. The next scientific assessment is the same lineage's mature fieldwise and local-error review against retained original ME500, with the same fixed panels and native targets. For the organizer, the gain is actual relation reuse, original-member access and a finite-dose row-composition effect; EE/EM overlap and the link from row composition to improved native accuracy remain uncertain. Mature analysis should retain this separation rather than imposing physical groups or an artificial diversity reward. For execution, O has measured unchanged-law training gains; H preserves sparse member scoring but adds measured inference overhead and full-address contractions. Future optimization should target those measured preparation/index/contraction costs and preserve the established equation. Inverse fitting, valid-design claims and new solver/TEST campaigns remain outside this round.

The authorized continuation preserves weights, Adam state, RNG, provider/sampler state, TRAIN-only normalization, ordered membership, query draws, native laws/objectives and the existing constant schedule. The epoch-100 checkpoint remains retained; ordinary output keeps latest, one best-field alias, 100/200/300/400/500 milestones and curves every 10 epochs. Substantive tests, report/figure review and cleanup are completed before launch. Successful startup will be confirmed once; the final continuation has no Luna monitor, observer, polling, completion wait or follow-up automation. Completion at 500 is unobserved at this handoff.

## Experiment scope and storage

The fixed TRAIN150/VAL22 manifest, original TRAIN600 membership/case order, TRAIN-only normalization, seed 0, physical objectives, FP32 precision and constant optimizer schedule are retained. The round ceiling is 72 qualification epochs and 600 new full-data epochs, with at most two initial scratch 100-epoch lineages and at most one same-lineage continuation to 500. Historical controls are reused when equivalent. Qualification is execution evidence and no short endpoint is treated as convergence or an architectural verdict.

The three mandatory workspace storage links resolve to their declared DATA2 targets on `/dev/nvme0n1p1`. Initial free space was `3,209,786,130,432` bytes on DATA2 and `77,002,145,792` bytes on home. The hard round budgets are 24 GiB retained and 2 GiB temporary, with an 8 GiB estimated new peak and at least 100 GiB DATA2/20 GiB home headroom. Dataset and reusable caches remain in place. [Storage preparation](../../../../diagnostics/round13_6_20261027/storage_preparation.json) and [epoch budget](../../../../diagnostics/round13_6_20261027/epoch_budget.json) record the execution scope.

Round cleanup removed 183 regular files and 15 temporary symlinks, totaling 213,619,348 logical payload bytes. This includes superseded organization archives, validated fixture renders/test payloads, redundant font caches and bytecode. Exact recursive allowlists retain ownership, inode/link information and duplication checks; active handles/mappings and downstream dependencies were checked before removal, and absence was verified afterward. Compact revision metadata, convention/failure evidence and receipts remain. Original ME100/ME500/A100/B100 checkpoint size, inode and modification time are unchanged. The initial user document diffs, runtime budget functions, tests and live work plan remain preserved; unrelated jobs were not stopped. [Aggregated cleanup receipt](../../../../diagnostics/round13_6_20261027/round_cleanup_summary.json), [protected evidence and final temporary cleanup](../../../../diagnostics/round13_6_20261027/temporary_cleanup_receipt.json) and [user-work preservation](../../../../diagnostics/round13_6_20261027/user_edits_preservation.json).

The pre-continuation closeout retains approximately 470 MB of round-local evidence: qualification and H100 checkpoints/histories, selected native/organization arrays, metrics/provenance, 12 PDF masters with 14 PNG pages, inspected rendered-report evidence and local diagnostic helpers. Small reusable output caches remain for the active continuation; datasets and protected migration backups are untouched. Measured remaining free capacity at that cleanup boundary is `3,209,314,877,440` bytes on DATA2 and `76,885,651,456` bytes on home, above the declared headroom. Shared-filesystem activity prevents treating free-space deltas as exactly equal to removed logical bytes. Generated numerical/visual/helper artifacts remain local and ignored; durable code/tests/configs and this report follow the upload rule.

All local report links resolve; all 14 PNG pages decoded and all 12 PDF masters were checked for their complete 14 pages. Every included figure was visually inspected, the repaired endpoint/spatial figures were reinspected, and the rendered Markdown displayed all 14 images without browser errors or horizontal overflow. Prose and captions remain single continuous source lines. [Rendered-report validation](../../../../diagnostics/round13_6_20261027/report_validation/browser_receipt.json).

## Verification and implementation

Focused tests cover original-law output and live-gradient equivalence, shared paired replay, empty/padded/pure-role behavior, K full-reader identity, H normalization and exact degeneracies, cross-role address access, finite-dose controls, strict checkpoint reconstruction, native task/evaluator dispatch and maintained horizon extension. The final integration/regression run passed 72 tests; a separate historical regression selection passed 83 tests, and the later bounded row-transfer helper and faithful-law selection passed 14 tests. These suites overlap and their counts are not added. Actual qualification recovery and complete native backward checks provide execution evidence beyond mocked resume tests. [Focused integration receipt](../../../../diagnostics/round13_6_20261027/focused_test_result.json), [final implementation test log](../../../../diagnostics/round13_6_20261027/final_implementation_tests.log) and [historical regression log](../../../../diagnostics/round13_6_20261027/historical_regression_tests.log).

Durable changes stay inside the existing member-routing core, Thermal adapter, task provider, checkpoint loader, evaluator and CLI. Historical models retain their original schemas and loaders. The new faithful and typed cores have no additional learned tensors; the checked 194-tensor scratch initialization agrees with original ME. No new solver, plugin registry, monitoring engine, admission layer, group quota, response loss or physics-specific prediction branch is introduced. Code and focused tests were pushed in `c48ab3d` and `3501de6`; the report closeout is recorded separately after the completed formal review.

The final integration audit found that the shared engine's existing explicit horizon-amendment path lacked the two R13-6 workflow IDs. The runner-only test had mocked this boundary. The narrow fix adds those IDs to the unchanged strict member-routing amendment; an actual CPU engine test now saves epoch 100 and executes epoch 101 under the 500-epoch horizon for both O and H. It verifies RNG, provider, sampler and Adam-state recovery, complete history, curve cadence and rejection of scientific schedule/objective changes. Both cases pass. This repair is pushed as `4b8ec2b`; it enables R13-6 continuation through the existing engine and preserves the active first-100 prediction equation and training trajectory. [Full outgoing-range audit and remote-tip receipt](../../../../diagnostics/round13_6_20261027/horizon_fix_outgoing_audit.json).

## Run locations and exercised commands

The formal lineage is discoverable through its [canonical Thermal forward-run alias](../../../../Trained_Results/ThermalChannel/HONF_Forward_Runs/_bk/15_20261029_Intermediate_Runs/thermal_r13_6h_typed_organization_train600_formal_v1). Its physical payload is `/data2/wanglz/ModularDT/round13_6_20261027/thermal_r13_6h_typed_organization_train600_formal_v1`. Qualification and diagnostics stay under the [round diagnostics alias](../../../../diagnostics/round13_6_20261027). Physical GPU 1 maps to logical `cuda:0` through UUID `GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e`; independent GPU 2 work uses UUID `GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`. Device ownership was checked before each launch and unrelated jobs were preserved.

The following environment and commands were exercised from `/home/wanglz/Desktop/src/ModularDT`. The initial training command names the physical payload; the canonical alias resolves to that same lineage. The qualification resume recovers epoch 3 through 6 and does not extend its declared horizon.

```bash
R136_ROOT=/data2/wanglz/ModularDT/round13_6_20261027
R136_PY=/home/wanglz/miniconda3/envs/ModularDT/bin/python
export CUDA_VISIBLE_DEVICES=GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e
export TMPDIR="$R136_ROOT/tmp" XDG_CACHE_HOME="$R136_ROOT/tmp/cache"
export MPLCONFIGDIR="$R136_ROOT/tmp/mpl" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$R136_PY" -B HONF_Proj/tools/honf_research.py train-member-routing \
  --config HONF_Proj/configs/research/thermal_r13_6h_typed_organization_train600_formal_v1.json \
  --output "$R136_ROOT/thermal_r13_6h_typed_organization_train600_formal_v1" \
  --epochs 100 --device cuda:0

"$R136_PY" -B HONF_Proj/tools/honf_research.py resume-member-routing \
  --run "$R136_ROOT/qualification/H_v1" --device cuda:0 --devices cuda:0 \
  --stop-after-epoch 6
```

The completed all-VAL89 evaluation used the retained epoch-100 checkpoint through the canonical alias. The final selected command uses the maintained same-lineage horizon extension; its startup receipt is recorded at handoff.

```bash
R136_RUN=/home/wanglz/Desktop/src/ModularDT/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_r13_6h_typed_organization_train600_formal_v1
"$R136_PY" -B HONF_Proj/tools/honf_research.py evaluate-r13-6-member-routing \
  --checkpoint "$R136_RUN/checkpoints/epoch_0100_model.pt" --device cuda:0 \
  --output "$R136_ROOT/evaluations/H_e100"

"$R136_PY" -B HONF_Proj/tools/honf_research.py resume-member-routing \
  --run "$R136_RUN" --device cuda:0 --stop-after-epoch 500 --extend-horizon
```

## Continuation handoff

The single startup confirmation at `2026-10-27T21:02:46.917626+00:00` succeeded: the maintained command restored source epoch 100 and executed epoch 101 on physical GPU 1 (logical `cuda:0`), with absolute stop 500 and milestones 100/200/300/400/500. The epoch-100 checkpoint is unchanged; the same weights, Adam/RNG/provider/sampler state and constant schedule were recovered. The owned process is PID `3450268` in tmux `r13_6_H_formal500`; ordinary in-process log, progress, curves and checkpoints remain under the canonical lineage. [Single startup receipt](../../../../diagnostics/round13_6_20261027/continuation_startup_receipt.json).

No monitor, observer, polling loop, completion wait or follow-up automation is attached to this final continuation. Epoch-500 completion is unobserved at handoff. The completed H100 review is pushed in `9c27c56`; this final handoff update follows the same full outgoing-range audit and remote-tip verification. [Final upload receipt](../../../../diagnostics/round13_6_20261027/final_outgoing_audit.json).
