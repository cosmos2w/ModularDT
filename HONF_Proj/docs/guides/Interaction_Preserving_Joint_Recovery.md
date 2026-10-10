# Interaction-preserving joint recovery: technical protocol

This protocol describes the opt-in P-family comparison. Its [evidence report](../reports/HONF_Interaction_Preserving_Recovery_20261010.md) records completed measurements and decisions; constructibility, inference parity, fitted field recovery, and organizer value are separate claims. The older [J-family protocol](Joint_Regional_Field_Development.md) and completed checkpoints retain their original identities.

## Executed model

P uses the maintained nonlinear source-conditioned core with two H128/message128 pair rounds. Each MM, ME, and EM message is formed from destination state, donor state, and signed relative geometry before reduction. Donor weights use physical measures; MM excludes self donors and renormalizes. The final nonlinear field read still visits individual physical sources before pooling, with the inherited global path retained.

P-G and P-H insert one width 64 collective block between the two pair rounds. Source anchors remain tied to physical source IDs, alongside 16 registered Thermal or 32 registered Wind regional anchors. Typed source and environment content is nonlinear before pooling into an edge. The normalized transpose return shares context with source and environment nodes; it is not a physical reciprocity law. Receiver collective content enters the source-read hidden preactivation additively.

Both organized arms start with the same Gaussian geometry prior of strength 1. P-H adds learned score residuals whose final projections start at zero. Source, environment, and receiver output projections also start at zero; ordinary upstream content parameters remain trainable. This initialization makes P-G and P-H recover the common P function exactly. Zero initialization alone does not establish fitted recovery or useful organization.

Thermal shares the pair context but uses separate nonlinear source reads for the four flow outputs and the affine temperature coefficients. Heat is absent from context, memberships, flow prediction, and coefficient preparation; only the final physical heat contraction changes with heating. Wind retains the inherited source-conditioned nonlinear velocity read and the TRAIN-fitted height-profile output transform. No learned historical checkpoint initializes these fits.

| Contract | Thermal | Wind |
|---|---|---|
| Source/context/environment feature widths | 8/14/8 | 2/11/7 |
| Physical source capacity | 12 modules | 30 turbines |
| Geometry-only environment inventory | 24×8, E192 | 2×2×2, E8 |
| Coordinate frame | Native physical x-y | Rotor diameters D |
| Output law | Nonlinear u/v/p/omega and source-resolved affine temperature | TRAIN profile in m/s plus standardized nonlinear residual × Uref × safe std |
| Primary TRAIN query budget | Q1024 plus native roles and operator128 | Q4096 across five roles |
| Effective case batch | 48 | 24 |
| Cohort | fixed25_v1: TRAIN150/DEV22 | wind_shared_fixed24_v1: TRAIN72/DEV24 rows, 24/8 layouts |
| Initialization/query seed | 0 | 42 |
| AdamW weight decay | 1e-4 | 1e-5 |

Wind source features are rotor radius and hub height in D. Its context contains the three direction categories, module count, reference speed, support origin, and support extent. The seven environment and receiver support features are three lower-face distances, three upper-face distances, and absolute height, scaled by [50,38,6.25] D. Environment tokens contain geometry only. Wind's selected TRAIN cohort omits M18 and covers 17.1% of source TRAIN; its direction rows remain clustered by layout.

## Calibration and exact identity

Thermal retains equal flow/temperature family weights: each standardized flow loss has effective weight 1/8, each standardized fluid/surface/material temperature loss 1/6, and native q-proxy weight 0.05 lies outside those families. The stored-velocity operator uses 128 rows/case as supervision only. The four response families 0001/0318/0333/0348 are an explicit original-TRAIN auxiliary addendum; only 0348 belongs to the primary 150-case cohort.

Measure fresh-P response/operator gradient ratios once on the fixed TRAIN-only calibration scenes and those four TRAIN response families, without optimizer construction or DEV target materialization. The receipt binds initialization, direct-state tensors, selected membership, normalization, query/native budgets, response-file hashes, coefficient values, and training-source hashes. Every sibling uses that same sealed receipt. Missing or changed calibration is an error, not a fallback to historical 0.1 or 1.0 coefficients.

Wind fits and selects with the same corrected five-role component-balanced physical objective. Each role has weight 0.2 and its own three TRAIN-profile residual scales. The environment-independent scale calibration reuses the verified historical E64 cache, while the fitted model uses E8. Historical scalar-per-role selectors remain archival and cannot be divided by the new component objective to claim a performance ratio.

The durable recipes under `src/config_core/forward/joint_regional/*interaction_preserving*.json` declare every effective scientific setting. Thermal templates require the ignored fresh-P receipt at `diagnostics/generated/interaction_recovery_20261010/thermal_fresh_P_calibration.json`; the runner resolves and hashes it before a start. Generated resolved recipes and numerical receipts remain local. Microbatch packing is sealed only after measuring the two candidates 8/16, checking full-objective query/denominator invariance, and retaining 20% of currently available GPU memory.

Training-source identity consists of literal file hashes and the last commit touching those files. A documentation-only commit does not change the model recipe. Changed runtime code, mathematical settings, membership, normalization, calibration, or schedule cannot silently resume a saved fit. Exact resumes preserve named optimizer moments, RNG state, sampling identity, selection policy, and absolute optimizer age.

## Commands and review boundaries

Run from the HONF project directory in the ModularDT environment. The following commands describe the runner interface; every GPU invocation in this bounded campaign must additionally use the ignored resource governor so staging, failures, calibration, fitting, and evaluation count toward the budget.

```bash
python tools/joint_regional_train.py calibrate-thermal \
  --recipe-json src/config_core/forward/joint_regional/thermal_interaction_preserving_p.json \
  --device cuda:0 \
  --output-dir diagnostics/generated/interaction_recovery_20261010/calibration

python tools/joint_regional_train.py prepare \
  --recipe-json src/config_core/forward/joint_regional/wind_interaction_preserving_p.json

python tools/joint_regional_train.py start \
  --recipe-json PATH_TO_SEALED_RECIPE --device cuda:0 \
  --output-dir PATH_TO_FRESH_RUN --stop-after 100

python tools/joint_regional_train.py resume \
  --recipe-json PATH_TO_SAME_RECIPE --device cuda:0 \
  --output-dir PATH_TO_SAME_RUN --stop-after 500
```

The full development horizon is 2500. Use 20 warmup epochs from 3e-5 to 3e-4, hold through 1000, then cosine decay to 3e-6 at 2500; gradient clipping is 1. Every selected case is visited once per epoch. Checkpoints, latest, best-field selection, native DEV monitoring, and curves use a 100-epoch cadence. Review 100/500/1000 before extensions; a healthy P and decisive organized comparators must receive the same mature 2500 horizon when the measured budget permits. Stops are absolute epochs, not extra epochs.

The goal begins 2026-10-10 19:14:23Z. Training must end by 2026-10-11 09:14:23Z, leaving the final two elapsed hours for delivery before 11:14:23Z. The total cap is 24 aggregate GPU-associated hours; the governor conservatively limits training to 22 and accounts process-group lifetimes on physical GPUs 1/2. An epoch-boundary clean stop requires both its acknowledgement and a verified resumable checkpoint. A crash or safety signal is reported separately.

## Measurements and decision

At 100, 500, and selected final checkpoints, compare fixed-weight TRAIN probes with the same DEV metric panel. Report each Thermal physical channel, temperature role and peak/tail, and all 15 Wind role/component errors and tails. Frozen-weight gradients by physical loss and parameter family, their norms/cosines, and actual residual magnitudes distinguish optimization reach from useful learned content. Native curl/omega consistency is valid for the Thermal reference; an incompressibility constraint is not part of this dataset.

Compare independently trained P, P-G, and P-H. On a small fixed panel, supplement that comparison with same-weight collective removal, learned-to-geometry replacement, and physical-ID matched-mass donor reassignment. Count actual changed donors and singleton buckets, retain the direct/global/own-anchor paths explicitly, and verify restored outputs. Intervention effects show reliance; they do not by themselves show an advantage over P or physical causality.

Complete native inference cost includes staging, context, all physical heads, conversion, and CPU output. Report active physical records and padded executed MM/ME/EM/source-read/collective-read work separately, alongside synchronized latency, training epochs/updates/case visits/queries, allocated/reserved peaks, process VRAM, and CPU catalogue residency. Dense full access is the default. Any later retained-mass experiment is an approximation and requires separate same-function fidelity/work evidence; no executor savings follows from logical concentration.

Recovery precedes organizer promotion. A sustained approximately twofold Wind wake or Thermal flow miss relative to the strong references is a recovery failure, with unmatched historical data/objective/query exposure disclosed. Against mature matched P, the initial organizer guards are at most 10% degradation in any priority mean and 15% in its corresponding tail, with roughly 5% useful difficult-field improvement or a measured accuracy/work benefit beyond the geometric control. Single-seed gains remain provisional.

Allow at most one recorded, evidence-guided structural correction with a new function lineage and a matched comparator. A credible segmented recovery can trigger the user's separately authorized fresh full-data 1000 comparison only within the remaining resource budget; it cannot trigger a 5000 extension. Faithful manual 5000 preparation requires a qualified preferred recipe, original TRAIN transforms and fresh calibration, an explicit development-to-formal diff, and an inert native check with optimizer construction blocked. Readiness and a completed formal fit remain separate statuses.

Generated checkpoints, arrays, figures, profiles, one-time evaluators, and renderers remain ignored. Preserve the inherited 647 bindings, newer 3905/3906/2203/2204 formal artifacts, user-owned files, WindTEST lock, and 326/326 solver ledger. Audit every outgoing commit object, run the pre-push artifact hook, push durable source/tests/configuration/report changes only, and verify remote/local branch tips.
