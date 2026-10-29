# HONF Round13: Thermal spatial group functions and paired utility learning

Round13 replaces per-stage MST grouping and M-first query postings with one balanced spatial hierarchy, a context-conditioned cut shared across both interaction stages, and reusable group functions read directly by physical receivers. F13-0 stores group means; F13-1 also stores two spatial slopes. Both prescribed-control joint physics, E→E, and the early group→M relay remain active. Both requested adaptations completed 100 additional development epochs, totaling 800 main updates, without extra arms or extensions.

**Predictor:** F13-1 lowers eight of nine DEV22 mean errors relative to F13-0, including fluid T by 20.0%, surface T by 11.0%, and demeaned DEV4 surface-angular error by 16.8%; pressure worsens slightly. These exposed-development gains do not recover the literal balanced-singleton parent's previously reported thermal quality, and individual M1/M2 material/peak results regress. The next user-controlled comparison should keep F13-0 as the matched control and measure whether F13-1's detail gains survive full-TRAIN adaptation.

**Organizer:** F13-0 uses intermediate cuts on 19/24 input-selected TRAIN scenes, whereas F13-1 stops at the root on 22/24. Fresh paired-probe sign agreement is 11/24 for both arms, with small measured gains. First-order value modes carry useful information, but reliable task-adaptive scale choice and relation-sensitive subdivision are not established. Root stopping is retained as measured; no target K, extra work price, or replacement loss was introduced.

**Inverse relevance:** saved heat responses improve in several thermal components with F13-1, while geometry responses remain mixed and the homogeneous zero-heat limit fails in both arms. There is no validated inverse-design recommendation. No inverse optimizer, new solver, extra physical data, or optional candidate ranking was run. The next inverse-facing decision needs stronger geometry-response and physical-limit evidence; this study does not authorize that follow-up automatically.

The complete M10 native call is 38.37 ms for F13-0 and 40.22 ms for F13-1; the measured six-case optimizer macro including paired and response exposures is 0.529 versus 0.578 s. F13-1 has lower endpoint coefficient storage for its near-root cuts, but neither complete inference nor complete training is faster in this pair. These conclusions concern the two new endpoints, not a same-hardware replay of historical rounds.

## Evidence and figure index

This report implements the [reviewed Round13 plan](../../UpgradePlan/HONF_Round13_Spatial_Group_Functions_Work_Plan.md) from branch `agent/honf-core-next` at revision `3627a2c`; `75f7af9` is historical context. The existing [R12-3 report](HONF_Round12-3_Interaction_Group_Reuse_Development_Report.md) supplies historical recovery evidence. All newly generated evidence lives behind the existing DATA2-backed [round directory](../../diagnostics/round13_spatial_20261025/), including [analysis](../../diagnostics/round13_spatial_20261025/analysis.json), [F13-0 results](../../diagnostics/round13_spatial_20261025/f13_0_evaluation/result.json), and [F13-1 results](../../diagnostics/round13_spatial_20261025/f13_1_evaluation/result.json).

The six selected figure pages are: [1. learning/native fidelity](#predictor-and-physical-detail), [2. physical fields](#physical-field-panels), [3. surface detail](#surface-bias-and-angular-detail), [4. organization](#organization-and-useful-scale), [5. saved responses](#saved-physical-responses-and-inverse-relevance), and [6. complete cost/physical limits](#complete-execution-cost). Each inline PNG has a retained PDF master; underlying numerical arrays, histories and provenance remain local and ignored by Git.

## Completed protocol and adaptation identity

Both arms copy 151 compatible core tensors identically from the literal `thermal_interaction_groups_balanced_dev_v2/checkpoints/epoch_0100_model.pt` G-EE endpoint, initialize eight new core state entries identically, and start fresh AdamW states. Each has 1,252,101 parameters; the mean control has no unused three-mode value projection. The [initial native receipt](../../diagnostics/round13_spatial_20261025/initial_forward.json) checks real M1/M12 forward calls, identical common tensors, the exact source normalizer and heat scale 1.3164895030821728. These are new-model adaptations with local epochs 1–100, not exact optimizer/RNG resumes of the source.

Both inherit the same fixed TRAIN150-fitted normalization and development manifest `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044` without refitting. All 150 selected TRAIN cases are visited in each development epoch. Native primary Q1024, effective batch 48, microbatch 24, AdamW LR 3e-4, weight decay 1e-5 and gradient clip 1 are identical. The final macro has six actual cases. Existing case-order and query-sampling identity strings match for every epoch between arms; no new hash or snapshot was created. DEV22 is exposed validation from the maintained quarter-data protocol, not a fresh independent TEST result.

| Exposure per scientific arm | Completed count |
|---|---:|
| Additional development epochs | 100 |
| Primary case visits | 15,000 |
| Main optimizer updates | 400 |
| Primary sampled fluid-query presentations | 15,360,000 |
| Paired native utility probes / extra case-forward presentations | 100 / 200 |
| Physical response pairs / endpoint-scene presentations | 100 / 200 |
| Primary scale-exploration / learned-cut presentations | 3,691 / 11,309 |

Scale exploration remains active from epoch 1, keyed identically across arms, with a quarter-probability draw over uniform available tree depths. The realized 24.61% fraction is not enforced by changing cases or K. Response fitting uses only TRAIN0348 every fourth update, alternating heat and single geometry, with the literal saved coefficient 23.47364739373233 and existing 10% shared-gradient downward cap. The same cadence supplies one paired TRAIN probe, alternating children and grandchildren. Its two physical losses are averaged with weight 0.5/6 each on this final six-case macro; the detached signed normalized gain trains only the utility head with Huber delta/coefficient 1. There is no score-function term or new work incentive.

Both fits completed with finite histories and exit status 0. External process wall times were 541.51 s and 645.34 s; the latter includes approximately 22 s overlapping F13-0 terminal evaluation. Steady epoch medians after the first ten epochs were 5.208 and 6.286 s. Luna confirmed finite startup and handled the necessary 25/50 and near-completion observations; the parent did independent implementation/report work instead of duplicating polling. The original 100-epoch validation/checkpoint cadence, 10-epoch loss plots, tqdm, full histories, and best/latest aliases remain intact. Both epoch-100 milestones, latest and best-field aliases are retained; no 25/50 checkpoints were created.

## Implemented information path and bounded diagnosis

The 24×8 environment catalogue produces a cached, longest-physical-extent balanced tree with 192 leaves, 383 total nodes, 191 internal candidates and maximum depth 8. Prefix moments in canonical leaf order replace per-node membership scans. Node features retain original-module-axis relation association through vector scatter and child discrepancy before scalar reductions. The context-conditioned signed utility head selects a legal top-down cut once, then both stages use that cut over their evolving states.

The executed path is MM/ME/EE → pointwise inherited group transform → weighted ridge spatial coefficients → early group→M relay → MM/ME/EE → new coefficients on the same cut → tiled direct group read → one joint field trunk. Mean keys and bias-free value coefficients use stage-specific inherited KV tensors. First-order coefficients use a ridge 1e-6 linear solve and a bounded C1 coordinate extension that is linear on members and saturates toward ±2 outside the group. Missing axes and singleton slopes are zero; zero-mass groups remain finite. No raw M/E neural-value bypass, M proposals/postings, final unused M update, dense ancestry matrix, or Q×K×H×D value expansion is used. Every positive-mass group is read; this is functional value restriction, not a sparse query-selected executor. E→E remains dense and leaf transforms still run on all original E records.

The [bounded legacy diagnostic](../../diagnostics/round13_spatial_20261025/legacy_adaptive_diagnostic/result.json) reused four existing adaptive cases (0006, 0051, 0348, 0631), sampled frontiers and one native parent/descendant comparison each without optimizer updates. First-order fits reduced same-state latent residuals, but signed physical gains included both signs and did not establish a decoder fix. A position–state reassignment test preserving both marginals distinguishes spatial covariance from separate position/state scatter. Native qualification then ran local epoch 1 and ordinary resume to epoch 2 with 300 primary visits, eight updates, two paired probes and two response pairs; [the receipt](../../diagnostics/round13_spatial_20261025/qualification_receipt.json) records the single visible CUDA RNG stream and named Adam state. Qualification was disposable and did not initialize either scientific child.

## Predictor and physical detail

The table reports equal-case native RMSE in the packed dataset's existing units, with mean/p90 over all 22 exposed DEV cases. Worst IDs refer to each arm's maximum case error. The baseline is the same selected-TRAIN mean predictor, not an additional trained arm. Per-case masks, M strata, maxima, baseline errors, and all nine observables remain in the retained terminal results. No physical SI conversion is asserted.

| Observable | F13-0 mean / p90 | Worst ID | F13-1 mean / p90 | Worst ID | TRAIN-mean baseline mean / p90 | F13-1 / F13-0 mean / p90 |
|---|---|---|---|---|---|---|
| u | 0.22601 / 0.26136 | 0288 | 0.22237 / 0.24163 | 0677 | 0.45437 / 0.4932 | 0.98389 / 0.92449 |
| v | 0.030347 / 0.033708 | 0665 | 0.029004 / 0.032487 | 0686 | 0.025876 / 0.03276 | 0.95574 / 0.96376 |
| p | 0.10823 / 0.12089 | 0677 | 0.10903 / 0.1249 | 0687 | 0.099458 / 0.12195 | 1.0074 / 1.0332 |
| omega | 0.41701 / 0.46756 | 0686 | 0.37851 / 0.42779 | 0688 | 0.69132 / 0.75085 | 0.90768 / 0.91494 |
| Fluid T | 3.9894 / 4.5628 | 0684 | 3.1912 / 4.0269 | 0294 | 6.4489 / 8.4102 | 0.79991 / 0.88255 |
| Surface T | 4.5449 / 5.5926 | 0663 | 4.0439 / 4.8774 | 0663 | 7.1019 / 8.4183 | 0.88977 / 0.87212 |
| Material T | 4.5519 / 5.432 | 0663 | 4.0829 / 5.0744 | 0663 | 7.0225 / 9.0676 | 0.89697 / 0.93416 |
| Normal heat proxy | 11.571 / 15.665 | 0655 | 9.9939 / 12.597 | 0655 | 7.5305 / 8.9048 | 0.86374 / 0.80413 |
| Module-peak T | 4.4563 / 5.8582 | 0663 | 4.0203 / 5.3875 | 0663 | 6.5792 / 9.4778 | 0.90216 / 0.91965 |

![Training history and all-nine DEV22 errors](../../diagnostics/round13_spatial_20261025/figures/01_native_quality.png)

Figure 1. Both complete 100-epoch histories lower the primary native TRAIN objective; response, paired-native and utility auxiliary terms are excluded from this displayed scalar. F13-1 improves eight DEV22 mean/p90 quantities, while pressure rises by 0.74%/3.32%. Both still exceed the TRAIN-mean baseline on several flow/heat-proxy summaries; relative improvement alone does not establish physical adequacy. [PDF master](../../diagnostics/round13_spatial_20261025/figures/01_native_quality.pdf).

For historical context, the already measured literal balanced-singleton parent's DEV22 mean fluid/surface/material/peak errors were 2.6288/3.1486/2.7304/2.5618. F13-1's 3.1912/4.0439/4.0829/4.0203 remain worse, despite its 100 additional adaptation epochs. Those historical numbers were reused from R12-3, not remeasured as a third newly trained arm. The present principal contrast is F13-1 versus its identically initialized, equally exposed F13-0 control.

The maintained low-module-count TRAIN risk panel shows why the DEV average is insufficient. These are exposed TRAIN checks, with detailed worst IDs and baselines retained in the numerical results.

| Observable | M1: F13-0 mean / p90 | M1: F13-1 mean / p90 | M2: F13-0 mean / p90 | M2: F13-1 mean / p90 |
|---|---|---|---|---|
| u | 0.22194 / 0.25492 | 0.15666 / 0.17793 | 0.2059 / 0.22935 | 0.18244 / 0.22555 |
| v | 0.028846 / 0.032255 | 0.02192 / 0.023518 | 0.025574 / 0.027228 | 0.024057 / 0.026447 |
| p | 0.078181 / 0.088233 | 0.067174 / 0.076896 | 0.087399 / 0.094224 | 0.083998 / 0.097545 |
| omega | 0.37678 / 0.47001 | 0.25877 / 0.27632 | 0.32806 / 0.36171 | 0.29365 / 0.31573 |
| Fluid T | 3.3885 / 4.7761 | 2.0732 / 2.2177 | 3.247 / 3.5371 | 2.6481 / 3.0998 |
| Surface T | 2.7924 / 4.0463 | 1.9031 / 3.2342 | 3.188 / 4.0645 | 3.2534 / 3.9026 |
| Material T | 1.9162 / 2.8117 | 2.2665 / 3.5244 | 2.4646 / 3.0309 | 3.038 / 3.1903 |
| Normal heat proxy | 4.2087 / 5.8999 | 4.8367 / 6.6429 | 5.7957 / 7.1338 | 5.982 / 6.4677 |
| Module-peak T | 1.6149 / 2.9668 | 2.0946 / 4.2429 | 1.6219 / 2.2675 | 3.1997 / 3.9704 |

F13-1 improves M1 fluid/surface errors but worsens its material, normal heat proxy and peak errors; M2 material/peak errors also worsen. There is no claim of uniform fieldwise retention or of independent-test generalization.

### Physical field panels

![Native physical reference fields, both predictions, and signed residuals](../../diagnostics/round13_spatial_20261025/figures/02_physical_fields.png)

Figure 2. The saved DEV0287/M5 epoch-100 panel shows fluid T, u, omega and material T with common scales across reference/F13-0/F13-1 and a separate common signed-error scale. Fluid values use the existing 8,192 native receiver rows and masks; each module retains its 3,096 material samples in physical coordinates. F13-1 reduces several broad errors but still misses wakes, interface detail and module temperature structure. Circles mark the original module geometry; these are field predictions, not inferred causal links. [PDF master](../../diagnostics/round13_spatial_20261025/figures/02_physical_fields.pdf).

### Surface bias and angular detail

TRAIN4 is 0303/0322/0337/0348; DEV4 is 0277/0287/0294/0684. Each panel has 25 modules and 1,600 valid surface-angle samples. Bias is the equal-module RMS of signed module means; angular and total RMS pool their actual valid-angle counts. These different reductions are stated explicitly rather than equating bias improvement with recovered angular information.

| Panel | Arm | Module-mean bias RMS | Demeaned angular RMS | Total surface residual RMS |
|---|---|---|---|---|
| TRAIN4 | F13-0 | 2.6461 | 3.3048 | 4.2336 |
| TRAIN4 | F13-1 | 2.6732 | 2.971 | 3.9966 |
| DEV4 | F13-0 | 3.6926 | 3.4905 | 5.0813 |
| DEV4 | F13-1 | 3.0806 | 2.9049 | 4.2342 |

![Surface bias, demeaned angular errors, and signed error curves](../../diagnostics/round13_spatial_20261025/figures/03_surface_detail.png)

Figure 3. Demeaned angular RMS falls 3.3048→2.9710 on TRAIN4 and 3.4905→2.9049 on DEV4 (10.1% and 16.8%). TRAIN module bias slightly rises 2.6461→2.6732, while DEV bias falls 3.6926→3.0806. Signed 64-angle curves for original module slots 0/1 of TRAIN0348 and DEV0287 expose remaining nonuniform errors, including regressions on a fitted TRAIN module. Lower aggregate surface error does not imply correct local heat transfer. [PDF master](../../diagnostics/round13_spatial_20261025/figures/03_surface_detail.pdf).

## Organization and useful scale

Both stages share exactly the same selected node IDs per scene. F13-0's 24-scene K spans 1–118 (median 35), with five roots and 19 intermediate cuts; F13-1 has 22 roots and two K2 cuts. P is 1 versus 3, so actual K×P spans 1–118 versus 3–6. There are no all-leaf deterministic endpoints in this panel. In these one-scene evaluations, allocated, positive-mass and live K agree; training batch allocation and actual work remain separately recorded. All 191 candidate rows are scored even after root stopping. Cut-level ancestry sizes and selected/inactive node statistics are preserved in the terminal results and support arrays.

| Panel | Arm | Sign agreement | Normalized prediction MAE | Raw native mean regret | Max absolute raw gain | Gains within ±0.05 normalized |
|---|---|---|---|---|---|---|
| TRAIN24 | F13-0 | 0.45833 | 0.032434 | 0.00021632 | 0.0046329 | 24/24 |
| TRAIN24 | F13-1 | 0.45833 | 0.076552 | 0.00083556 | 0.0066375 | 24/24 |
| DEV4 | F13-0 | 1 | 0.037839 | 0 | 0.001748 | 4/4 |
| DEV4 | F13-1 | 0.5 | 0.04568 | 0.00013579 | 0.00043431 | 4/4 |

Each fresh comparison uses the same targets, query set and off-path choices for parent and descendants, forcing ancestor paths as needed and rebuilding the real early relay and dependent second stage. Candidate selection uses input-only depth/size ranks, not errors. Eighteen F13-0 and 23 F13-1 candidates lie below a stopped selected ancestor. All 24 TRAIN and four DEV normalized measured gains lie within ±0.05; the maximum raw gains are below 0.00664 in maintained normalized-loss units. Thus 11/24 sign agreement and tiny raw regret do not establish a reliable utility estimator. DEV4 is diagnostic exposed validation with only four probes.

Signed native advantages vary with candidate size; positive numbers mean descendants reduce loss. These strata have small, unequal counts, and the full [stratification evidence](../../diagnostics/round13_spatial_20261025/organizer_strata.json) retains depth and M1–M12 breakdowns. No subgroup result is a significance claim.

| Candidate leaf count | n | F13-0 mean signed native gain | F13-1 mean signed native gain |
|---|---|---|---|
| 192 | 1 | -0.0046329 | 0.0020773 |
| 24 | 1 | -0.003116 | 0.00012695 |
| 12 | 2 | 0.0013064 | 0.0031234 |
| 6 | 4 | 0.00027214 | 6.225e-05 |
| 4 | 4 | 0.0003849 | -0.0028335 |
| 2 | 12 | 7.6825e-05 | -0.00027599 |

Receiver-role components also disagree. These are unweighted native standardized component-loss differences, not physical RMSE or a sum of quantities with different units.

| Paired native component | F13-0 mean raw parent−child gain | F13-1 mean raw parent−child gain |
|---|---|---|
| flow/omega | -0.00013439 | 0.0003938 |
| flow/p | -0.00011006 | -0.0010052 |
| flow/u | 0.00022398 | 0.0006265 |
| flow/v | -0.0005539 | 0.00061918 |
| q_proxy | 0.0042453 | -0.0055063 |
| temperature/fluid | 0.00030233 | -0.00018021 |
| temperature/material | -0.00025686 | -0.00035752 |
| temperature/surface | -0.0012847 | 0.00022737 |

![Actual spatial group supports, signed utility probes and common covers](../../diagnostics/round13_spatial_20261025/figures/04_organization.png)

Figure 4. TRAIN0348 uses K58/P1 versus K1/P3; environment colors encode selected-node depth 0–8, rectangles show actual spatial support, black squares retain original module IDs, and only the two largest mean ME relation-prior links per module are drawn for readability. These links are learned information associations, not causal interactions or a claim that only those links executed. All 24 actual cuts, fresh signed utility probes and same-state latent fits are shown. First-order latent fits improve within each arm/state/cut, which does not prove improved physical fields or compare identical latent states between trained arms. [PDF master](../../diagnostics/round13_spatial_20261025/figures/04_organization.pdf).

The five fixed geometry covers are diagnostics at unchanged endpoint weights on only 0006/0051/0348/0631. They are neither fitted arms nor an inference depth selected for deployment.

| Forced cover K | F13-0 native loss | F13-1 native loss | F13-0 surface RMSE | F13-1 surface RMSE |
|---|---|---|---|---|
| 1 | 0.31613 | 0.33445 | 3.1824 | 3.2338 |
| 8 | 0.31861 | 0.26092 | 3.3974 | 2.7457 |
| 32 | 0.32837 | 0.23696 | 3.7083 | 2.8837 |
| 128 | 0.32374 | 0.2435 | 3.5853 | 2.7464 |
| 192 | 0.32402 | 0.25201 | 3.5901 | 2.7981 |

F13-1's common intermediate covers lower its four-case native loss from 0.33445 at the root to 0.23696 at K32, whereas its learned endpoint remains near-root. This supports a selector/representation mismatch diagnosis at these weights, not forced K or a new compression incentive. F13-0's native loss is nearly flat across these covers. Common-cover comparisons support a capacity contribution beyond the differing learned cuts, but do not isolate a unique physical mechanism.

A fixed-cut final-read slope ablation preserves keys, the full early relay and the selected cut; its four-case mean native loss rises from 0.33445 to 0.47929 for F13-1. Zeroing slopes in both relay and final read gives 0.45858. Bypassing the early group→M relay and recomputing all dependent outputs gives mean native loss 0.38115 for F13-1 versus 0.31959 for F13-0 (baselines 0.33445 and 0.31550). The changes have mixed signs across individual cases, so the relay carries information but is not uniformly beneficial. Removing only organizer relation features leaves all four cuts and predictions unchanged in each arm; relation-sensitive subdivision is unestablished, even though ordinary MM/ME states and group values retain module information. All-nine ablation metrics and exact cut IDs remain in each terminal result.

## Saved physical responses and inverse relevance

The evaluator reuses all eight existing physical-reference families. Only 0348 fits responses; 0001/0318/0333 are TRAIN-reference-only, and 0304/0320/0335/0350 are response-validation-only. Every family originates in the original source TRAIN partition; “response validation” does not mean an independent source TEST cohort. The same saved controls, physical reference arrays and valid masks are used for both endpoints. There were zero new solver attempts. Heat, single-geometry and mixed-central response errors are reported separately below across families/endpoints; no flow observable is removed or replaced by a heat-to-flow invariance target.

![All-nine physical response error ratios and finite-change maps](../../diagnostics/round13_spatial_20261025/figures/05_physical_responses.png)

Figure 5. Filled/open points compare F13-1/F13-0 median/p90 absolute reference errors for all nine observables and three intervention types. Heat thermal errors improve, while geometry changes remain mixed. Lower panels show actual saved heat_transfer_plus fluid-T finite changes on fitted TRAIN0348 and response-validation-only0304, using one common signed scale within each family. Both models underrepresent the localized 0348 reference response. Finite-response agreement is a separate measurement from static field fidelity and learned support. [PDF master](../../diagnostics/round13_spatial_20261025/figures/05_physical_responses.pdf).

Absolute errors use each reference's native finite-change or derivative units; relative errors divide by that response's valid reference RMS and exclude near-null responses using the maintained native floors. The valid-relative counts below are identical across arms. Reference RMS values, floors, excluded IDs, per-family components and masks remain in the retained result JSON and `response_delta_panels.npz`; errors with different physical units are never pooled into one scalar.

### Heat perturbations

| Observable | F13-0 absolute median / p90 | F13-1 absolute median / p90 | F13-0 relative median / p90 | F13-1 relative median / p90 | Valid relative / all |
|---|---|---|---|---|---|
| u | 0.0030208 / 0.013334 | 0.0035793 / 0.0074005 | — / — | — / — | 0/16 |
| v | 0.00048117 / 0.0021158 | 0.00054897 / 0.001064 | — / — | — / — | 0/16 |
| p | 0.001323 / 0.0065502 | 0.001811 / 0.0038223 | — / — | — / — | 0/16 |
| omega | 0.0063028 / 0.027078 | 0.0090867 / 0.014996 | — / — | — / — | 0/16 |
| Fluid T | 0.24208 / 0.27222 | 0.18176 / 0.21322 | 0.84201 / 0.95552 | 0.69073 / 0.93992 | 16/16 |
| Surface T | 0.31573 / 0.35946 | 0.28084 / 0.3225 | 0.76519 / 0.9901 | 0.68268 / 0.90185 | 16/16 |
| Material T | 0.4139 / 0.49066 | 0.36027 / 0.41364 | 0.77768 / 0.95429 | 0.66041 / 0.91693 | 16/16 |
| Normal heat proxy | 0.83042 / 1.0682 | 0.72605 / 0.89185 | 0.95505 / 0.99876 | 0.80223 / 0.96239 | 16/16 |
| Module-peak T | 0.57561 / 0.67588 | 0.52989 / 0.57216 | 0.83733 / 0.97611 | 0.7383 / 0.93631 | 16/16 |
### Single-geometry perturbations

| Observable | F13-0 absolute median / p90 | F13-1 absolute median / p90 | F13-0 relative median / p90 | F13-1 relative median / p90 | Valid relative / all |
|---|---|---|---|---|---|
| u | 0.02061 / 0.028181 | 0.019188 / 0.03208 | 1.0224 / 1.3024 | 1.0213 / 1.3166 | 32/32 |
| v | 0.0035475 / 0.0055244 | 0.0032947 / 0.0071091 | 1.0301 / 1.5408 | 1.0525 / 2.4865 | 32/32 |
| p | 0.011766 / 0.015612 | 0.012244 / 0.01828 | 1.0266 / 1.5089 | 1.0923 / 1.7801 | 32/32 |
| omega | 0.05723 / 0.14504 | 0.051545 / 0.14907 | 1.0162 / 1.8504 | 0.99564 / 1.6387 | 32/32 |
| Fluid T | 0.18139 / 0.79346 | 0.22972 / 0.7417 | 1.4023 / 3.2172 | 1.7408 / 4.118 | 32/32 |
| Surface T | 0.39706 / 0.90081 | 0.3957 / 1.0633 | 2.0495 / 3.9813 | 1.9798 / 4.6388 | 32/32 |
| Material T | 0.35404 / 0.9484 | 0.37769 / 1.14 | 1.8536 / 4.874 | 1.9738 / 5.2606 | 32/32 |
| Normal heat proxy | 1.2001 / 3.1916 | 1.3633 / 2.7689 | 1.247 / 1.7065 | 1.2662 / 1.9455 | 32/32 |
| Module-peak T | 0.25837 / 0.96482 | 0.33528 / 1.1298 | 1.522 / 4.3047 | 2.2774 / 6.2096 | 31/32 |
### Mixed central derivatives

| Observable | F13-0 absolute median / p90 | F13-1 absolute median / p90 | F13-0 relative median / p90 | F13-1 relative median / p90 | Valid relative / all |
|---|---|---|---|---|---|
| u | 0.050046 / 0.12284 | 0.033041 / 0.057114 | 2.6962 / 11.728 | 3.038 / 6.5557 | 4/8 |
| v | 0.0053603 / 0.022551 | 0.0057742 / 0.0067538 | — / — | — / — | 0/8 |
| p | 0.010637 / 0.064636 | 0.010947 / 0.036042 | — / — | — / — | 0/8 |
| omega | 0.13498 / 0.43756 | 0.12765 / 0.45224 | 2.5345 / 8.9496 | 2.8695 / 12.103 | 4/8 |
| Fluid T | 1.1184 / 2.7854 | 0.92437 / 1.9928 | 2.8256 / 27.604 | 3.9682 / 16.615 | 5/8 |
| Surface T | 3.0309 / 4.4967 | 2.1655 / 3.603 | 14.986 / 74.275 | 12.591 / 41.135 | 7/8 |
| Material T | 2.6597 / 4.7609 | 1.9342 / 3.3812 | 19.291 / 96.457 | 15.164 / 46.534 | 7/8 |
| Normal heat proxy | 6.911 / 9.7557 | 4.0638 / 8.5169 | 15.89 / 67.395 | 11.441 / 75.494 | 7/8 |
| Module-peak T | 2.8692 / 4.8501 | 1.566 / 5.3251 | 17.592 / 57.092 | 26.323 / 52.849 | 6/8 |

For mixed raw four-corner contrasts, which are distinct from central derivatives, median absolute fluid/surface/material/heat-proxy errors and all-nine details are retained in [analysis](../../diagnostics/round13_spatial_20261025/analysis.json). For example surface absolute median/p90 improves 0.18468/0.39767→0.083982/0.30443, but four of eight surface contrasts are near-null for relative scoring. Mixed-derivative surface relative median remains 14.986 versus 12.591, excluding one of eight references. These small-reference ratios expose unresolved physical-response errors; they are not launch rejection bands.

Surface-response errors split by their original cohorts are shown to prevent fitted-family performance from standing in for held-out response behavior.

| Response cohort | Kind | F13-0 surface absolute median / p90 | F13-1 surface absolute median / p90 |
|---|---|---|---|
| fit_train_0348 | heat | 0.25418 / 0.2544 | 0.25369 / 0.25419 |
| fit_train_0348 | single_geometry | 0.19761 / 0.28983 | 0.32 / 0.35263 |
| fit_train_0348 | mixed_central_derivative | 3.5984 / 3.5984 | 3.539 / 3.539 |
| train_reference_only | heat | 0.34935 / 0.35946 | 0.29998 / 0.33409 |
| train_reference_only | single_geometry | 0.39007 / 0.73896 | 0.56328 / 1.2204 |
| train_reference_only | mixed_central_derivative | 2.4634 / 4.0655 | 3.4594 / 3.6937 |
| response_validation_only | heat | 0.31573 / 0.40479 | 0.26718 / 0.31232 |
| response_validation_only | single_geometry | 0.43627 / 1.273 | 0.42368 / 0.84347 |
| response_validation_only | mixed_central_derivative | 2.6009 / 4.504 | 0.65642 / 0.88468 |

A bounded original-control check varies only module `0348:module:5` heat, retains all original heat columns and fully rebuilds finite-difference endpoints. Fixed-cut JVP versus central FD absolute discrepancies are at most 0.000708 for F13-0 and 0.001144 for F13-1 across eight readouts; no cut crossing occurs at the tested step 0.0013164895. This is differentiation/self-consistency evidence, not validation against a physical-gradient reference. Saved single-geometry/mixed interventions also rebuild physical preparation. The bounded saved-input check moved only module `0348:module:4` by +0.0749998 in x, and separately transferred +0.125/−0.125 heat between original modules 0/1. Neither arm changed its cut (K58 versus K1), but the module-relay state changed by 2.09%/2.09% for geometry and 1.22%/1.06% for heat. Common-node final coefficient relative changes were 2.18%/1.37% and 0.185%/0.104%. Saved fluid-T response RMS was 0.07998/0.09807 for geometry against physical-reference RMS 0.18079, and 0.00315/0.00699 for heat against 0.12777. These are real input/state/read dependencies with weak localized heat response, not evidence of useful cut adaptation. [F13-0 trace/control evidence](../../diagnostics/round13_spatial_20261025/cost_trace_only_f13_0.json) and [F13-1 trace/control evidence](../../diagnostics/round13_spatial_20261025/cost_trace_only_f13_1.json) retain exact IDs and all readout components.

With all prescribed heat and homogeneous inlet/wall temperature set to zero, the analytic temperature rise is zero, yet F13-0/F13-1 RMS fluid T is 5.7282/6.1208, surface T 6.9431/7.8128 and material T 7.0491/7.8957; heat-proxy RMS is 4.4520/5.6596. This clear physical-limit failure is preserved rather than corrected with a new output branch. Optional six-candidate ranking was omitted; there is no candidate-selection claim, valid-design diversity claim or feasible inverse recommendation.

## Complete execution cost

GPU0 used the ModularDT environment and one visible logical `cuda:0`; unrelated active jobs on GPU1/GPU2 were left alone. After both fits finished, one small warmup and three timed repetitions per endpoint measured preparation, cached core decode and complete native calls on M1 case0006 and M10 case0348. Complete calls include fresh preparation, native union/stencils and extraction. Q32/Q1024 refer only to primary fluid rows; the accompanying surface/material/stencil queries remain active, explaining the larger union below. Full native union is 8,192 unique rows in these calls. These endpoint measurements are uncontended by another owned GPU0 process, not extrapolated from K or matrix shapes.

| Case | Primary Q | Native union Q | Prepare 0 ms | Prepare 1 ms | Cached decode 0 ms | Cached decode 1 ms | Complete 0 ms | Complete 1 ms |
|---|---|---|---|---|---|---|---|---|
| 0006 | 32 | 206 | 20.994 | 21.164 | 1.561 | 1.6338 | 23.68 | 23.867 |
| 0006 | 1024 | 1293 | 20.659 | 20.59 | 3.1868 | 3.4813 | 25.396 | 25.798 |
| 0006 | full_native | 8192 | 20.636 | 20.731 | 13.422 | 15.105 | 35.82 | 37.871 |
| 0348 | 32 | 1469 | 23.307 | 23.659 | 3.219 | 3.4815 | 27.768 | 28.202 |
| 0348 | 1024 | 2556 | 22.978 | 23.233 | 4.7944 | 5.2676 | 29.432 | 30.142 |
| 0348 | full_native | 8192 | 23.859 | 24.013 | 13.863 | 15.127 | 38.374 | 40.221 |

![Complete native/optimizer costs, memory and zero-heat failure](../../diagnostics/round13_spatial_20261025/figures/06_complete_cost.png)

Figure 6. All measured complete M10 native calls are slightly slower for F13-1 despite its smaller learned K. The exact final six-case macro includes Q1024 primary work, the paired native comparison, response pair, utility regression, cap, clipping and Adam. Full-epoch maximum allocated memory is 5.87/6.73 GiB; the separate endpoint six-case macro reserves 1.045/0.805 GiB. These are different workloads and allocated/reserved measurements, not interchangeable peak claims. The zero-heat panel shows a failed physical limit, not solver-reference validation. [PDF master](../../diagnostics/round13_spatial_20261025/figures/06_complete_cost.pdf).

The disposable macro restores endpoint model, named AdamW and provider state before one warmup and three measured steps; its final six IDs are 0407/0186/0262/0267/0152/0142. Median times are 0.52926/0.57800 s, including all scheduled probe/response work. Those four temporary optimizer steps per arm occur in RAM and never advance the saved scientific checkpoints or reported 800 main updates. The live 100-epoch learning peaks are 6,299,917,312/7,225,046,016 allocated bytes; the endpoint macro peaks are 1,006,384,640/767,027,200 allocated bytes as recorded exactly in [F13-0 cost](../../diagnostics/round13_spatial_20261025/cost_f13_0.json) and [F13-1 cost](../../diagnostics/round13_spatial_20261025/cost_f13_1.json). The warm instrumented full-native trace aggregates MM/ME/EE at 4.23/4.28 ms across both stages, all-node moments at 4.11/4.17 ms, selected coefficient formation at 0.420/0.489 ms, and shared relay plus query reads at 9.86/11.38 ms; embeddings, query projection/trunk/head, union/stencils and extraction are retained separately in the cost JSON. One additional, non-repeated trace separated the two formations and relay from final query tiles:

| Instrumented boundary | F13-0 ms | F13-1 ms |
|---|---:|---:|
| Stage 1 all-node moments (cold first trace) | 67.8698 | 2.3268 |
| Stage 2 all-node moments | 2.5549 | 2.2160 |
| Utility feature formation | 11.7381 | 0.6043 |
| Stage 1 selected coefficients | 0.5345 | 0.5550 |
| Early group→M read | 0.9370 | 0.7506 |
| Stage 2 final coefficients | 0.2182 | 0.2438 |
| Final 16 query-tile reads | 9.6976 | 10.7488 |
| CPU cut host interval | 0.0837 | 0.0302 |

These CUDA intervals are nested and must not be summed into total cost. The additional trace has no warmup: F13-0’s first moment pass is 67.87 ms and feature pass 11.74 ms, far above the warmed measurements, so this trace cannot establish between-arm substage speed differences. Cold initialization is a possible contributor, not a separately measured causal attribution. Complete-cost conclusions use the earlier uninstrumented one-warmup/three-repeat medians; no benchmark suite or repeated device search followed.

Preparation still computes all 192 transformed E values and dense E→E work; final query work uses only group coefficients, with zero original-E or module neural-value query reads. No query-level sparse retrieval executor or full-access fallback was activated. Full-precision weighted moments and all-mode F13-1 contraction cost are charged. For M10/full-native, persistent adapter-prepared tensor storage is 5,202,934/4,949,054 bytes, including the native union and retained metadata used in this cost call; compact selected neural coefficient capacity must not be equated with the entire prepared object or process memory. Lower coefficient storage does not yield a complete speedup here.

## Validation, handoff and closeout

The maintained train/resume commands were exercised on real native CUDA data in the disposable epoch-1→2 qualification; both scientific fits then used the same maintained runner to epoch100 and terminal native/response evaluation. A first F13-0 terminal evaluation failed on missing helper imports before finishing; imports were repaired and the successful retry completed in 22.25 s. The 6.88 s failure and its log remain retained and charged. F13-1 terminal evaluation completed in 19.96 s. A final review corrected metadata accumulation for `group_value_modes_per_group`: old per-tile records may show 16/48 instead of 1/3; field outputs, gradients, timings, actual cut-level K/P and additive value-read counts are unaffected. Historical records were preserved. The final focused regression suite passed 69 tests in 25.82 s, covering changed spatial equations/tiling, utility gradients/weighting, config lineage, maintained training/evaluation dispatch and legacy interaction groups. The first collection attempt required the existing tests directory on PYTHONPATH; its error log is retained. Six saved-data figure pages were visually inspected, and Markdown rendering/link validation is recorded in the local closeout evidence.

The [tested launch/resume guide](../guides/_bk/20261029_Guide_Consolidation/03_R12_R13_Predecessor_Recipes/Thermal_Spatial_Group_Functions_Recipe.md) includes both separately labelled full-TRAIN600/VAL89 configurations, fresh AdamW warm starts from their matching F13 epoch100, the exact inherited TRAIN150 normalization and heat scale, and a manual first 200-epoch segment. The 1000-epoch absolute configuration horizon launches nothing automatically. Both configs, maintained CLI entrypoints, actual saved endpoint strict loads, source-normalizer equality, empty fresh optimizer states and input-only 600/89 membership were checked on CPU; the [handoff receipt](../../diagnostics/round13_spatial_20261025/fulltrain_handoff_check.json) states the limits. Full-TRAIN optimization, full VAL89 target evaluation and a formal full-TRAIN resume were not performed. The epoch-1→2 development replay validates ordinary resume machinery under one visible CUDA stream, not a completed formal run.

For a field/detail-focused next experiment, F13-1 is the promising candidate and F13-0 its required matched control. For combined useful organization, geometry responses and complete runtime, there is no established overall winner. The handoff is supplied for the user's decision despite ratio misses; no formal training, extra seed, portfolio or extension was launched. Future 25/50 reviews inspect compact history without extra checkpoints; the existing 100-epoch selection/checkpoint and 10-epoch plotting cadence stays in the configs.

The round's conservative compute charge is 1978.19 s (0.550 h), comprising 1378.19 s of recorded per-process wall times plus a 600 s allowance for unmetered CPU preflight/tests/analysis/handoff work. The learning/qualification charge is 1,216.37 s and diagnostic/CPU charge 761.82 s, below the respective 8 h/2 h allowances and 10 h total ceiling. Concurrent process wall times are added; this is not measured device utilization. [The compute receipt](../../diagnostics/round13_spatial_20261025/compute_budget_receipt.json) includes qualification, failed evaluation, collection/render retries and disposable cost steps. No new training follows this closeout.

Before output work, DATA2 had 3,217,729,355,776 bytes free, with a hard 4 GiB retained round budget, 512 MiB temporary ceiling and 200 GiB headroom; home allowed at most 1 MiB durable source and zero scientific payload. Existing DATA2 workspace/shared-cache symlinks were verified and preserved; no dataset or per-arm cache was copied. Cleanup removed 52 disposable files, 6 temporary links and 37 directories totaling 96,322,229 apparent bytes: expired engineering qualification checkpoints/plots, unit fixtures and duplicate terminal stdout JSON. Compact qualification histories, validation records and failure receipts remain. Parsed equality against retained results was checked after saving the CLI output-size/path/budget fields; no new hashes were created. [The exact allowlist](../../diagnostics/round13_spatial_20261025/cleanup_allowlist.json) and [cleanup receipt](../../diagnostics/round13_spatial_20261025/cleanup_receipt.json) record ownership, sizes, inodes/link counts, accessible process descriptors/mappings, report dependencies and absence after removal.

At cleanup closeout the retained round payload was 197,179,236 bytes (about 188 MiB), including both epoch100/latest/best scientific states, full histories, native/response arrays, six selected PDF/PNG pages, current engine loss plots and ignored helpers/receipts. DATA2 free space changed from 3,217,307,013,120 to 3,217,403,621,376 bytes (about 2.93 TiB); home had 84,038,389,760 bytes free. Filesystem free space may also change as unrelated active jobs progress. All 41 protected file metadata records matched across removal, all three required symlink bindings remained exact, and both unrelated GPU job identities remained active and unchanged. Protected migration backups and user-authored pending edits were left untouched. Durable code/tests/configuration/guide/report changes are audited across the entire outgoing Git history, while numerical evidence, figures and one-time renderers remain local and ignored.
