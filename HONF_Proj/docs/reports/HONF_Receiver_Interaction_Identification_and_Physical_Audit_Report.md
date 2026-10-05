# Receiver interaction identification on frozen G-fast

## What changed

Two new query interfaces, H-add and H-joint, attach to the same development G-fast epoch-1000 physical model. All inherited weights, buffers, adapters and Stage-A parameters remain frozen. H-add uses source-only and receiver-only control preactivations; H-joint adds their explicit receiver–source contrast. The two interfaces have identical architecture, initialization and optimizer budgets. A single input-only source plan is reused through P0/P1/P2; original fine module and environment values remain distinct through dense physical readers.

**Predictor:** both interfaces completed 500 fit epochs. At matched 500, H-joint improves u/v/pressure means6.41%/1.48%/1.16% versus H-add, but fluid/material temperature slightly worsen; selected H-joint fit 100 also loses all three thermal means to selected H-add fit 300. H-joint retains9.81%/11.28%/14.17% thermal gains versus G-fast, with pressure 4.86% worse. Keep the measured tradeoffs; no uniform superiority or formal promotion follows.

**Organizer:** the joint term has real local utility: removing I worsens thermal errors on three representatives and improves them on 0291. Actual plans/components/phase ancestry are exported faithfully, while fine physical work stays dense. All-new-zero recovery is exact on three cases but misses the original interface tolerance at three 0687 values. Retain this numerical qualification and both matched research fits; do not force more groups.

**Inverse:** after explicit ceiling 326 authorization, twelve counted attempts produced eleven converged local benchmark states; 0277's failed baseline remains unavailable. All four retained models read the same eleven states. On six primary transfers, selected H-joint improves fluid/surface/material response error6.59%/14.85%/15.06% over selected H-add, but loses to retained Tensor-H and predicts the wrong mean fluid-temperature direction on both 0291 transfers. AD/FD agreement near 1% does not repair this physical miss. No inverse search or validated design was produced; module-position truth remains absent.

**A—added learning value:** modest equal-exposure flow gains, no consistent aggregate thermal joint gain. **B—faithfulness:** actual joint controls and dense/global dependencies are measured; no physical executor savings. **C—transfer:** some local thermal response improvement, with wrong direction, amplitude and null-flow failures preventing inverse-readiness claims. Both stopped formal histories and mature G-fast/Tensor-H remain intact.

## Experiment identity and limits

This executes [the receiver interaction plan](../../UpgradePlan/HONF_Receiver_Interaction_Identification_and_Physical_Audit_Plan.md), following the retained [Lean reset results](HONF_Lean_Interaction_Reset_Development_Report.md). Implementation revision: `6704b2076cfccfa0f6929945de42f2378ed56bb3`. The fitting guide is [Thermal receiver interface fitting](../guides/Thermal_Receiver_Interface_Fitting.md).

The frozen physical source is development Run3601 G-fast `epoch_1000_model.pt`, not formal Global3502 or old Tensor-H. New native identities are Run3701 H-add and Run3702 H-joint. Checkpoint epoch and selection epoch mean **new interface-fit age**; the physical backbone stays at epoch 1000. Mature physical objective scheduling uses `1000 + fit_age`, while the matched case/query sampling uses fit age. Fitting never updates the inherited physical backbone. Final state audits verify identical dataset/loss, all global/local normalizers and response calibration. Initial/final RNG states, pinned seed/fit-age sampling configuration and actual per-epoch counts match between arms. The run does not separately retain every sampled query array; sampler state is not a separate serialized object. The [RNG/sampling receipt](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/continuation/final_rng_sampler_pair_audit.json) states this evidence boundary.

Both arms reuse `fixed25_v1`: 150 selected training cases and the same 22 repeatedly exposed development-validation cases, with normalization fitted only on selected training cases. Module-count strata have six M3, six M5, six M7 and four M10 validation cases. Detailed fields, graphs and interventions use only fixed representatives 0277/0291/0294/0687. No population-test interpretation follows from this exposed panel.

Both new interfaces contain 38 trainable tensors /120,557 scalars, initialized with seed 0. The optimizer contains only these new tensors: fresh AdamW moments, constant learning rate `3e-4`, weight decay `1e-5`, no scheduler. Every development epoch visits all 150 selected cases, uses 1024 primary fluid queries per case, effective batch 48 and microbatch 8, and performs four optimizer updates. The existing train-only response callback runs **once per epoch**, at the last native batch, with two physical wrappers; it is not run four times per epoch. Its calibration is inherited unchanged. The heat-null loss coefficient remains zero.

Monitoring, latest state and best-field alias follow the native 100-epoch cadence. Both arms use the same saved-monitoring field-MSE selector. The e100 measured review approved continuation to 500; any extension of both arms to 1000 must be decided from late 400/500 fitting evidence and remaining budget before new physical-reference outcomes. Exact matched endpoints are reported separately from selected checkpoints if selection ages differ.

The resource envelope is12 aggregate GPU-associated hours and 8 elapsed hours, starting 2026-10-05 21:04:05UTC. GPUs1/2 are reserved for this work; unrelated GPU0 training remains untouched. Failed starts, engineering execution, training and model evaluations count in the GPU envelope. Local physical-reference attempts additionally have a1200-second CPU wall ceiling, including failures and imports. Final measured totals appear below.

## Physical and control operator

For each nonempty source type M/E, measure-normalized donor density `b` satisfies the measured `sum(mu*b)=1`. The complete input receiver catalogue supplies normalized reference measure `nu`; the implementation uses measured donor means rather than assuming exact floating-point unity. With `abar=sum(nu*a)` and `bbar=sum(mu*b)`, actual pre-tanh terms are:

```text
C      = sum_e abar[e] * bbar[e] * gamma[e]
S(s)   = sum_e abar[e] * (b[e,s] - bbar[e]) * gamma[e]
R(q)   = sum_e (a[q,e] - abar[e]) * bbar[e] * gamma[e]
I(q,s) = sum_e (a[q,e] - abar[e]) * (b[e,s] - bbar[e]) * gamma[e]
```

The raw action reconstructs as `U=C+S+R+I`. H-add executes `base+tanh(S+R)`; H-joint executes `base+tanh(S+R+I)`, followed by the native outer gain `1+tanh(action)`. QE retains separate score/head-gain channels and unique physical-source softmax. This is a separable **preactivation** competitor: downstream nonlinearities and the physical reader still permit interaction. The joint term's weighted-zero marginals refer to the declared receiver/source product measure before tanh, not arbitrary masks or the fluid-grid measure.

The full receiver catalogue has 1036 original rows:192 environment anchors,12×64 port anchors,64 inlet/outlet anchors and 12 module centers. Zero-weight padded rows remain identifiable. Environment, ports, inlet, outlet and centers each contribute one fifth of normalized reference mass. Original row indices and roles are exported. M7 has 711 positive reference rows and M10 has 906; this reference is distinct from the8192-row physical fluid grid and its mask.

Signed normalized relative geometry augments the shared access keys. The weak distance prior is `-0.25*log(1+distance²/0.25²)` in adapter-normalized coordinates; the environmental-background proposal uses zero distance bias. The new small relative score has a zero final layer at initialization. This prior is a modeling choice, not a fitted physical law. Access remains positive to all admitted groups: no independently sparse receiver subgraph is claimed.

There is one proposal per present module plus background, with allocated capacity 13. Fit1–100 uses soft admission/donors;101–200 uses the declared blend;201 onward uses exact sparse projections. Valid proposal count, admitted K, donor support and physical executor work are different quantities. Zero joint interaction or K=1 is an admissible outcome. No forced K, entropy/diversity penalty, recursive Tree or hard/soft shadow is present.

Group summaries pool nonlinear per-source content under fixed memberships and refresh live phase content. MM/ME/EM keep original G-fast actions and native reads **given their current phase states**. Query corrections can still alter later states through predicted ports and Stage-A. Planning, global calibration, reference centering and coarse/local context remain separate full-information paths. A signed centered contrast at a donor with zero direct membership is reference compensation, not a new admitted donor. Learned support is not physical causality.

The fixed-weight I-only intervention retains all weights and the input-only source plan. It removes I at each query phase, then runs ordinary state refresh; later S/R values may consequently change with altered upstream states. Its field effect is an end-to-end operational intervention, not a linear decomposition of final P2 output. Removing one input-selected group's I does not renormalize admission.

## Native attachment, learning and numerical qualification

At attachment, all 344 inherited state tensors are exactly equal to the source checkpoint, with unchanged dataset/loss/normalization/calibration. Both new gamma heads are zero. The whole physical forward and Stage-A remain differentiable; only parameter updates are frozen. Native low/high-module wrappers preserve input VJPs and demonstrate real new-head gradients and updates.

An initial harness attempt failed on sorted versus unsorted parameter-name inventory before any optimizer update. A second attempt passed low-module output/VJP checks, then missed the unchanged pointwise high-module output tolerance on H-add's interface field. The failed comparisons remain saved. One bounded arithmetic remedy uses the tanh addition identity for the new query contrast:

```text
tanh(base + residual) - tanh(base)
  = tanh(residual) * (1 - tanh(base)^2)
    / (1 + tanh(base) * tanh(residual))
residual = tanh(S + R [+ I])
```

The compact parent reduction order remains present, with an exactly zero contrast and live derivative at zero gamma. The replay passed the original pointwise `atol=2e-5, rtol=2e-5`, without widening tolerance. Across saved output roles, low-module maximum differences were0; high-module maxima were1.8835e-5 for H-add and 2.6703e-5 for H-joint. A maximum absolute difference alone is not the pointwise acceptance rule. High-module input-VJP maxima were1.49e-8 and 2.98e-8. Whole-wrapper identity is therefore within native tolerance, not universally bitwise.

The original numerical cause remains unproven: a direct legacy wide-minus-compact tanh probe on the same wrappers was exactly zero. The stable identity gives a structural zero/live-gradient property; it does not establish that a particular tanh kernel caused the earlier miss. Old Tensor-H semantics were not changed.

Four real effective48 native updates passed, with every inherited tensor unchanged. Actual first scientific optimizer boundaries had projection-gradient norms0.004423/0.006058 (H-add/H-joint); second-boundary content, admission, donor and receiver gradients were nonzero in both arms. These are genuine task gradients, before the last-batch response callback. Constructed algebra tests supplement this execution evidence.

At the completed e100 review, both arms had15,000 case visits,400 optimizer updates,15.36million primary training queries,2.2528million sampled validation queries,888,940 auxiliary all-role query positions and 200 response wrappers. Mean complete train+validation epochs were10.0040s/9.93769s. H-joint versus G-fast improved fluid/surface/material temperature RMSE means9.81%/11.28%/14.17%, while pressure worsened4.86%. H-joint versus equally fitted H-add improved those temperature means2.96%/2.48%/1.38%, with pressure mean 0.256% worse. Pressure paired wins were15/22 despite the worse mean, illustrating why tails and paired counts must remain visible. These are review measurements, not final results.

## What the old Tensor-H query action represented

Six saved-weight native wrappers decomposed old Tensor-H on the four fixed cases and removed only QM/QE joint I on 0277/0687. All384 state tensors remained exact; zero optimizer or solver calls occurred. The20.303-second GPU1 process was charged once. Its original eight allocated receiver-reference slots, five positive in these cases, remain distinct from the new full catalogue.

On that original reference, P2 joint-I RMS relative to the old centered-action RMS ranged0.0422–0.2619% for QM and 0.0132–0.0665% for QE. On the positive fluid grid, the corresponding ranges were0.0838–3.8821% and 0.0636–0.1951%. Source-only S dominated. Wider arithmetic reconstructs the decomposition of recorded coefficients to 1.11e-16, while comparison with the actual FP32 raw action differs by up to 4.21e-8; this is not a whole-model FP64 conversion.

I-only removal changed fluid temperature by RMS9.962e-5/1.022e-4 on 0277/0687, while truth RMSE **improved**0.003845%/0.001870%. The physical work stayed unchanged. This localizes a small, unhelpful old joint contribution on two cases, without deciding the new fit's coefficients or treating the old grand-mean implementation as incorrect. Bounded logits alone do not force constant access. A saved truth-attachment bug for the two nonintervened cases was corrected from each case's own H5 data; the interventions and predicted/control arrays were unaffected.

## Final learning, cost and selection

Both arms completed 500 fit epochs normally, without updating their344 inherited state tensors. Each performed75,000 training-case visits and 2000 new-interface optimizer updates, with 76.8million primary training queries,11.264million sampled validation queries,500 auxiliary callbacks,1000 response wrappers and 4,444,700 auxiliary all-role query positions. These are new-interface updates on a fixed physical model, not additional physical-backbone training.

The last50-epoch windows351–400 versus 451–500 show training field MSE decreasing0.245% for H-add and 1.081% for H-joint, while exposed-validation field MSE **increases0.496% and 0.0776%**. Validation temperature MSE increases1.121%/0.0444%. Neither arm improves its previously best saved field checkpoint. This does not leave substantive validation underfitting evidence for a matched 1000 extension; both stop at 500. The decision and checkpoint selection were sealed before opening any new physical-response outcomes. Spare compute budget alone did not justify extending.

| Fit milestone | H-add validation field MSE | H-joint validation field MSE |
|---|---:|---:|
|100|0.02067556|**0.02010768**|
|200|0.02050025|0.02019268|
|300|**0.02030265**|0.02025435|
|400|0.02041288|0.02015296|
|500|0.02038756|0.02025837|

The identical saved-monitoring selector chooses **H-add fit 300, exact controls**, and **H-joint fit 100, soft controls**. Both best-field aliases match the selected milestone's age and every model-state tensor. The selected comparison therefore has different fit ages and stages; the matched exact-stage fit 500 comparison is reported separately below. H-joint's selected soft positive-group count is never labelled exact sparse K.

![Matched interface learning curves and complete-epoch time and memory](../../diagnostics/generated/receiver_interaction_20261005/figures/01_learning_cost.png)
Figure1. Saved native CSVs show500 complete epochs per arm, with mean train+validation time10.2676s for H-add and 10.2454s for H-joint; summed epoch work is5133.806s/5122.712s. Peak allocated training memory is5.1890/5.1916GiB. The curves show early validation gains followed by a plateau while training continues to improve. Epoch sums exclude startup, checkpoint writing and independent evaluation; unique process envelopes separately account for the resource ceiling. The inherited1000-epoch physical backbone is frozen throughout, and physical value reads remain dense.

The backend-returned executed-row ledger records33,319,224,000 primary fine-MLP leading rows and 795,922,800 response rows per arm, identical between arms. These counts exclude backward/recomputation and are not hardware utilization. The response callback uses 279.30s/278.83s, already included in training time (about 5.92%); it is not charged twice.

Evidence: [fit500 work ledger](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/query_work_ledger_fit500.json), [late400/500 review](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/continuation/late_review_fit500.json), [stop decision](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/continuation/review_fit500_decision.json), [sealed selection](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/physical_audit/selection_decision.json), [selector-alias identity](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/continuation/selector_alias_identity_fit500.json), and [learning/cost figure receipt](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/figures/01_learning_cost_receipt.json).

## Predictor: all 22 native roles and physical fields

Role statistics use denormalized dataset-native benchmark scales, with no asserted SI/CFD calibration. Near receivers are within two native module radii; far is their fluid complement. Material peaks compare per-module maxima, while initial/refined ports compare outside-temperature and h tokens against stored port targets. The evaluator's edge-pressure difference is minimum-x fluid-edge mean p minus maximum-x edge mean p; the physical audit's8%-band pressure functional is separately defined. q-normal is a proxy.

The common selector gives H-joint fit 100 versus H-add fit 300: H-joint improves u/v/omega means5.644%/0.165%/0.114%, but worsens fluid/surface/material temperature0.579%/0.423%/1.682%, pressure 0.659% and q-proxy 0.159%. Only10 of 24 native role means improve. Both interfaces retain thermal gains over frozen G-fast; H-joint's corresponding gains are9.814%/11.277%/14.171%, with pressure 4.857% worse. This is a thermal calibration result with broad-field tradeoffs, not evidence of uniformly added joint-control value.

Selected H-joint's near-pressure mean improves1.014% versus G-fast while far-pressure worsens6.664% and the separately measured edge-pressure functional worsens11.681%. Refined outside-port temperature improves14.198% versus G-fast, whereas initial outside-port temperature worsens2.088%. H-joint's thermal p90 and maxima generally improve over G-fast, but its worst pressure error is0.0195582 on 0291, versus 0.0181317 on 0663 for G-fast. The material-temperature maximum remains2.1564 on 0663, worse than H-add's2.07562. No flow or tail limitation is hidden by normalized aggregate MSE.

![Four physical temperature fields and signed temperature and velocity residuals](../../diagnostics/generated/receiver_interaction_20261005/figures/02a_physical_fields.png)
Figure2a. Saved truth and selected predictions use identical coordinates and positive fluid masks for0277/0291/0294/0687. H-joint/H-add fluid-temperature RMSE is0.596/0.628,1.16/1.15,1.21/1.27 and 0.925/0.918 respectively; the last case visibly retains a weak joint result. Velocity-u residuals expose simultaneous flow errors. Temperature, velocity and coordinates retain dataset-native units; the analytic-wake/shared-grid benchmark is not CFD or mesh-converged truth. No new solve was used to render these fields.

![Surface temperature and heat-proxy residuals with per-module material temperature errors](../../diagnostics/generated/receiver_interaction_20261005/figures/02b_interface_material.png)
Figure2b. Actual angular surface rows and present material modules show selected H-joint/H-add surface-temperature RMSE0.673/0.663,1.13/1.05,1.19/1.29 and 0.890/0.890. The q-proxy errors remain about 2.84–3.61 native units, and the module curves preserve errors that case averages can conceal. Module slot is not a spatial y coordinate; q-normal is a benchmark proxy. Shared residual scales contain the recorded extremes without clipping.

### Selected all 22 role, stratum and tail tables

Error change is `100 × (candidate/control − 1)`: negative means lower RMSE. Means and tails weight each of the same 22 exposed development cases equally. The selected fit ages are separate from the frozen e1000 backbone.

| Native role | G-fast e1000 | Tensor-H e1000 | H-add fit300 | H-joint fit100 | Joint/add Δ% | Joint/G-fast Δ% | Joint/add wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| fluid/u | 0.0306896 | 0.0317718 | 0.0320716 | 0.0302616 | -5.644 | -1.394 | 19/22 |
| fluid/v | 0.00377759 | 0.00435925 | 0.00386756 | 0.00386119 | -0.165 | +2.213 | 12/22 |
| fluid/p | 0.0143447 | 0.017395 | 0.014943 | 0.0150415 | +0.659 | +4.857 | 11/22 |
| fluid/omega | 0.122856 | 0.133456 | 0.122044 | 0.121905 | -0.114 | -0.774 | 12/22 |
| fluid/temperature | 1.17451 | 0.940162 | 1.05314 | 1.05924 | +0.579 | -9.814 | 11/22 |
| surface_temperature | 1.23966 | 1.0903 | 1.09523 | 1.09986 | +0.423 | -11.277 | 13/22 |
| material_temperature | 1.04896 | 0.93089 | 0.885417 | 0.90031 | +1.682 | -14.171 | 11/22 |
| q_normal_proxy | 3.72215 | 3.54808 | 3.73734 | 3.74326 | +0.159 | +0.567 | 9/22 |
| far/omega | 0.0660679 | 0.0804763 | 0.0675177 | 0.067864 | +0.513 | +2.718 | 10/22 |
| far/p | 0.0134785 | 0.0165687 | 0.0141638 | 0.0143767 | +1.503 | +6.664 | 9/22 |
| far/temperature | 1.16265 | 0.938012 | 1.04867 | 1.05473 | +0.578 | -9.282 | 11/22 |
| far/u | 0.0290492 | 0.0296384 | 0.0309603 | 0.0287304 | -7.203 | -1.097 | 19/22 |
| far/v | 0.00346979 | 0.00405206 | 0.00356008 | 0.00355922 | -0.024 | +2.577 | 10/22 |
| final_port/h_effective | 0.89437 | 0.793863 | 0.875613 | 0.876623 | +0.115 | -1.984 | 8/22 |
| final_port/outside_temperature | 1.77781 | 0.953183 | 1.50521 | 1.52539 | +1.341 | -14.198 | 7/22 |
| initial_port/h_effective | 11.8021 | 11.8285 | 11.8015 | 11.8024 | +0.008 | +0.003 | 3/22 |
| initial_port/outside_temperature | 1.13386 | 1.82412 | 1.17133 | 1.15754 | -1.178 | +2.088 | 12/22 |
| inlet_outlet_pressure_difference | 0.00841499 | 0.00574119 | 0.00931276 | 0.00939797 | +0.915 | +11.681 | 12/22 |
| module_material_peak | 1.15867 | 0.984887 | 0.923779 | 0.94858 | +2.685 | -18.132 | 9/22 |
| near/omega | 0.278226 | 0.285592 | 0.272783 | 0.271912 | -0.319 | -2.269 | 12/22 |
| near/p | 0.0182509 | 0.0210678 | 0.0184675 | 0.0180659 | -2.175 | -1.014 | 16/22 |
| near/temperature | 1.24004 | 0.918134 | 1.05916 | 1.06902 | +0.931 | -13.791 | 10/22 |
| near/u | 0.0373988 | 0.0417406 | 0.0370148 | 0.0366158 | -1.078 | -2.094 | 16/22 |
| near/v | 0.00500943 | 0.00567524 | 0.00506638 | 0.00503646 | -0.591 | +0.539 | 13/22 |

Per-module-count strata retain their original membership; no error-based resampling.

| M | n | Native role | G-fast | H-add | H-joint | Joint/add Δ% | Joint/G-fast Δ% | Joint/add wins |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | 6 | fluid/u | 0.0342782 | 0.0337529 | 0.0322424 | -4.475 | -5.939 | 6/6 |
| 3 | 6 | fluid/v | 0.00305785 | 0.00322748 | 0.00317683 | -1.569 | +3.891 | 5/6 |
| 3 | 6 | fluid/p | 0.0127172 | 0.0133239 | 0.0135269 | +1.523 | +6.367 | 3/6 |
| 3 | 6 | fluid/omega | 0.0994517 | 0.0998198 | 0.0995207 | -0.300 | +0.069 | 4/6 |
| 3 | 6 | fluid/temperature | 0.911449 | 0.857305 | 0.852197 | -0.596 | -6.501 | 5/6 |
| 3 | 6 | surface_temperature | 0.911349 | 0.7225 | 0.734584 | +1.672 | -19.396 | 2/6 |
| 3 | 6 | material_temperature | 0.794998 | 0.592014 | 0.616948 | +4.212 | -22.396 | 1/6 |
| 3 | 6 | q_normal_proxy | 3.38795 | 3.41132 | 3.43019 | +0.553 | +1.247 | 2/6 |
| 5 | 6 | fluid/u | 0.0291972 | 0.0312055 | 0.0302511 | -3.058 | +3.610 | 4/6 |
| 5 | 6 | fluid/v | 0.00351882 | 0.0035883 | 0.00360154 | +0.369 | +2.351 | 2/6 |
| 5 | 6 | fluid/p | 0.0141422 | 0.0149202 | 0.0152799 | +2.411 | +8.045 | 1/6 |
| 5 | 6 | fluid/omega | 0.123682 | 0.120933 | 0.121662 | +0.603 | -1.634 | 1/6 |
| 5 | 6 | fluid/temperature | 1.08724 | 1.012 | 0.976473 | -3.511 | -10.188 | 4/6 |
| 5 | 6 | surface_temperature | 1.2647 | 1.18516 | 1.16564 | -1.647 | -7.832 | 5/6 |
| 5 | 6 | material_temperature | 1.11326 | 0.999018 | 0.998432 | -0.059 | -10.315 | 4/6 |
| 5 | 6 | q_normal_proxy | 3.84763 | 3.88615 | 3.89186 | +0.147 | +1.150 | 2/6 |
| 7 | 6 | fluid/u | 0.0295038 | 0.0321047 | 0.029216 | -8.998 | -0.975 | 5/6 |
| 7 | 6 | fluid/v | 0.00404869 | 0.00411124 | 0.00411209 | +0.021 | +1.566 | 4/6 |
| 7 | 6 | fluid/p | 0.0151529 | 0.0154443 | 0.0153791 | -0.422 | +1.493 | 5/6 |
| 7 | 6 | fluid/omega | 0.130186 | 0.13074 | 0.130549 | -0.145 | +0.279 | 4/6 |
| 7 | 6 | fluid/temperature | 1.40127 | 1.22493 | 1.27008 | +3.686 | -9.362 | 1/6 |
| 7 | 6 | surface_temperature | 1.50499 | 1.33181 | 1.35815 | +1.978 | -9.757 | 3/6 |
| 7 | 6 | material_temperature | 1.28092 | 1.07964 | 1.11256 | +3.050 | -13.143 | 3/6 |
| 7 | 6 | q_normal_proxy | 3.97728 | 3.96075 | 3.96907 | +0.210 | -0.206 | 3/6 |
| 10 | 4 | fluid/u | 0.029324 | 0.0307994 | 0.0288744 | -6.250 | -1.533 | 4/4 |
| 10 | 4 | fluid/v | 0.00483868 | 0.00488107 | 0.00490087 | +0.406 | +1.285 | 1/4 |
| 10 | 4 | fluid/p | 0.0158775 | 0.0166536 | 0.0164492 | -1.228 | +3.601 | 2/4 |
| 10 | 4 | fluid/omega | 0.14573 | 0.144003 | 0.142879 | -0.781 | -1.956 | 3/4 |
| 10 | 4 | fluid/temperature | 1.35988 | 1.15091 | 1.17769 | +2.327 | -13.397 | 1/4 |
| 10 | 4 | surface_temperature | 1.29655 | 1.16456 | 1.16167 | -0.248 | -10.403 | 3/4 |
| 10 | 4 | material_temperature | 0.985526 | 0.863786 | 0.859785 | -0.463 | -12.759 | 3/4 |
| 10 | 4 | q_normal_proxy | 3.65252 | 3.66802 | 3.65125 | -0.457 | -0.035 | 2/4 |

The p90 is the 90th percentile of case RMSE; maxima retain their actual case IDs.

| Native role | G-fast p90 / max (case) | H-add p90 / max (case) | H-joint p90 / max (case) |
|---|---:|---:|---:|
| fluid/u | 0.0348687 / 0.0381129 (0279) | 0.0369561 / 0.0404354 (0279) | 0.0349177 / 0.0385954 (0279) |
| fluid/v | 0.00476269 / 0.00531493 (0687) | 0.00478643 / 0.00536984 (0687) | 0.00483193 / 0.00538229 (0687) |
| fluid/p | 0.0174392 / 0.0181317 (0663) | 0.017518 / 0.0191221 (0291) | 0.0173755 / 0.0195582 (0291) |
| fluid/omega | 0.14507 / 0.167664 (0686) | 0.142005 / 0.166758 (0686) | 0.142175 / 0.166189 (0686) |
| fluid/temperature | 1.74552 / 1.86501 (0299) | 1.48166 / 1.63874 (0684) | 1.58128 / 1.71238 (0684) |
| surface_temperature | 1.84401 / 2.67311 (0663) | 1.68599 / 2.41284 (0663) | 1.66861 / 2.48484 (0663) |
| material_temperature | 1.51399 / 2.33637 (0663) | 1.32908 / 2.07562 (0663) | 1.39201 / 2.1564 (0663) |
| q_normal_proxy | 4.8297 / 5.21872 (0663) | 4.85817 / 5.04451 (0663) | 4.87951 / 5.06323 (0663) |


### Matched exact-stage fit 500 comparison

At the identical fit 500 exact-stage endpoint, H-joint improves u/v/p means6.406%/1.479%/1.164% versus H-add, with 18/17/17 paired case wins. Surface temperature improves0.641% (14/22 wins), while fluid and material temperature means worsen0.125%/0.155%, despite14/16 paired wins. Omega worsens0.0787%; q-proxy improves0.1697%. Both interfaces improve all three thermal means versus G-fast, while pressure remains3.716% worse for H-joint and 4.937% worse for H-add. The joint term earns modest broad-field value at equal fit exposure, without a consistent aggregate thermal advantage. These modest differences from one exposed seed do not establish formal noninferiority. Matched500 thermal tails are less favorable than the means: H-joint fluid-T p90/maximum worsen4.981%/9.565% versus H-add; surface-T p90 worsens6.961% despite its better mean, and material-T p90 worsens7.922% despite16/22 paired wins. Surface/material maxima improve5.642%/4.979%. Matched500 u improves in all four M strata, but v worsens at M10 and pressure worsens slightly at M7. Surface/material means worsen at M3/M5 and improve at M7/M10; the unchanged stratum tables retain these mixed effects.

| Core role | H-add fit500 | H-joint fit500 | Joint/add error change % | Joint/G-fast error change % | Joint/add wins |
|---|---:|---:|---:|---:|---:|
| fluid/u | 0.0327612 | 0.0306624 | -6.406 | -0.088 | 18/22 |
| fluid/v | 0.00384234 | 0.00378549 | -1.479 | +0.209 | 17/22 |
| fluid/p | 0.015053 | 0.0148778 | -1.164 | +3.716 | 17/22 |
| fluid/omega | 0.121752 | 0.121847 | +0.079 | -0.821 | 9/22 |
| fluid/temperature | 1.06697 | 1.06831 | +0.125 | -9.042 | 14/22 |
| surface_temperature | 1.09685 | 1.08982 | -0.641 | -12.087 | 14/22 |
| material_temperature | 0.889273 | 0.890651 | +0.155 | -15.092 | 16/22 |
| q_normal_proxy | 3.74147 | 3.73512 | -0.170 | +0.349 | 14/22 |

Full matched 50024-role means, per-M strata and p90/worst-case figures are retained in the [matched endpoint tables](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/report_inputs/role_tables_matched500.md) and [numerical source](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/report_inputs/role_tables_matched500.json). The full selected tables above remain a distinct selector-based comparison.

Evidence: [selected comparison](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/evaluation/comparison_selected.json), [selected role/table source](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/report_inputs/role_tables_selected.json), and [physical figure receipt](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/figures/02_physical_receipt.json).


## Organizer: actual components, support and intervention utility

At matched fit 500, both arms use the actual exact sparse projection. H-add K spans3–9, mean 5.136; H-joint K spans4–10, mean 6.364. H-add varies within M5/M7/M10; H-joint varies within M7/M10, while M3/M5 retain all valid groups. These are measured input-dependent admissions, not forced group budgets or independently sparse query subgraphs. Allocated capacity remains13. The selected H-joint fit 100 is soft: its four positive-group counts4/6/8/11 equal the valid M+1 proposals; no exact K is asserted for that selected model.

| M | Cases | H-add positive K (min / mean / max) | H-joint positive K (min / mean / max) | Valid proposals | Allocated |
|---:|---:|---:|---:|---:|---:|
| 3 | 6 | 4 / 4.000 / 4 | 4 / 4.000 / 4 | 4 | 13 |
| 5 | 6 | 3 / 4.333 / 5 | 6 / 6.000 / 6 | 6 | 13 |
| 7 | 6 | 4 / 5.167 / 6 | 6 / 7.500 / 8 | 8 | 13 |
| 10 | 4 | 7 / 8.000 / 9 | 8 / 8.750 / 10 | 11 | 13 |

| Case | M | H-add K | H-joint K | H-add M / E donor pairs | H-joint M / E donor pairs |
|---|---:|---:|---:|---:|---:|
| 0277 | 3 | 4 | 4 | 9 / 336 | 8 / 342 |
| 0279 | 3 | 4 | 4 | 10 / 377 | 10 / 356 |
| 0287 | 5 | 3 | 6 | 12 / 298 | 23 / 526 |
| 0288 | 5 | 5 | 6 | 15 / 438 | 20 / 537 |
| 0291 | 5 | 5 | 6 | 19 / 499 | 23 / 565 |
| 0294 | 7 | 5 | 8 | 25 / 426 | 36 / 673 |
| 0299 | 7 | 6 | 7 | 32 / 577 | 34 / 655 |
| 0637 | 3 | 4 | 4 | 9 / 335 | 8 / 342 |
| 0641 | 3 | 4 | 4 | 5 / 343 | 6 / 313 |
| 0644 | 3 | 4 | 4 | 11 / 387 | 11 / 351 |
| 0645 | 3 | 4 | 4 | 9 / 389 | 9 / 351 |
| 0648 | 5 | 5 | 6 | 15 / 437 | 21 / 519 |
| 0650 | 5 | 3 | 6 | 11 / 311 | 21 / 518 |
| 0655 | 5 | 5 | 6 | 18 / 494 | 22 / 520 |
| 0663 | 7 | 6 | 8 | 26 / 529 | 32 / 701 |
| 0665 | 7 | 5 | 8 | 25 / 451 | 37 / 696 |
| 0674 | 7 | 4 | 6 | 23 / 388 | 33 / 551 |
| 0677 | 7 | 5 | 8 | 26 / 468 | 36 / 730 |
| 0684 | 10 | 8 | 9 | 52 / 751 | 55 / 810 |
| 0686 | 10 | 9 | 8 | 57 / 789 | 45 / 689 |
| 0687 | 10 | 7 | 10 | 52 / 642 | 67 / 884 |
| 0688 | 10 | 8 | 8 | 51 / 711 | 46 / 700 |


Donor pairs count positive memberships only among admitted groups, separately for original module and environmental control sources. They are neither unique physical-source counts nor executed MLP rows. Every all 22 evaluation—selected H-add fit 300, selected H-joint fit 100 and both exact500 endpoints—executes43,972,896 padded fine-MLP leading rows in3542 successful physical calls, matching retained G-fast and Tensor-H. Native M capacity 12 and E192 remain distinct; the stored H5 environmental catalogue is not silently substituted. There is **no physical executor saving**, allocated control compaction or hardware sparsity claim.

![Actual selected H-joint source, receiver, joint and executed gain controls on physical coordinates](../../diagnostics/generated/receiver_interaction_20261005/figures/03_controls_page1.png)
Figure3a. Actual selected H-joint fit 100 P2 QM controls are joined across 64 saved native chunks to 8192 original grid rows, with 7976/7837/7691/7463 positive fluid rows under the corresponding masks. Across fixed4, source-only S RMS is0.1002–0.1525, receiver-only R0.00107–0.00678, and joint I0.03526–0.06973; these signed pre-tanh controls have no energy or probability interpretation. The final column shows the actually executed outer-gain contrast, rather than a gamma norm. Each column shares an unclipped scale across cases. Stars identify the first original valid module source, chosen before errors; nonzero joint structure alone does not establish improved truth.

![Full receiver reference, actual control donor memberships and query access with environmental gain and score](../../diagnostics/generated/receiver_interaction_20261005/figures/03_controls_page2.png)
Figure3b. Predeclared M7 case0294 uses 711/1036 positive reference rows, with 0.2 mass on each of the five roles. Nonnegative donor mass and actual receiver access to all eight soft positive groups are distinct from signed reference compensation. The group is selected by largest input admission, with no error-based choice; the displayed receiver is native grid row4031. QE score0 and gain0 show actual executed channels, with all eight channel norms retained in the receipt. Physical M/E values, global planning/calibration/reference means, ports and Stage-A remain live; the picture is control provenance, not a physical-causality graph.

### Fixed-weight joint utility and recovery qualification

Twelve additional native wrappers reused the four selected normal archives and removed I only, all new corrections, or one input-admission-argmax group's I. Every wrapper built exactly one plan and carried the same object through P0/P1/P2; every intervention retained the original fine-work ledger. Removing I has a substantial field effect and improves or harms truth separately:

| Case | I-removal fluid / surface / material T field RMS | I-removal fluid / surface / material truth-RMSE change % |
|---|---:|---:|
| 0277 | 0.114835 / 0.063583 / 0.050143 | +8.694 / +2.295 / +1.256 |
| 0291 | 0.087045 / 0.132733 / 0.117787 | -0.310 / -1.146 / -1.624 |
| 0294 | 0.342024 / 0.275318 / 0.237832 | +9.458 / +12.519 / +12.724 |
| 0687 | 0.100213 / 0.141054 / 0.130351 | +3.762 / +1.999 / +1.436 |

Positive error change means I was useful before removal. I improves all three thermal errors on 0277/0294/0687, but **hurts all three on 0291**. Removing the single preselected group's I changes fluid T by RMS0.04119/0.04052/0.25397/0.02640 and changes its truth RMSE by+0.01449/−0.01180/+0.06279/+0.004160 native units. These operational interventions retain memberships and admissions without renormalization; phase-state feedback remains live. They establish a nontrivial local joint contribution, not a linear share of the all 22 model gap or a universally useful causal group.

![Matched exact admission and reconstruction with selected joint-removal field and truth-error effects](../../diagnostics/generated/receiver_interaction_20261005/figures/04_organization_utility.png)
Figure4. Above: all 22 exact-stage fit 500 K and matched reconstruction, with mean K5.136 for H-add and 6.364 for H-joint; u improves6.406%, while fluid and material T means slightly worsen. Below: selected soft H-joint fit 100 I-removal fluid-T RMS0.1148/0.08705/0.3420/0.1002 and truth-error change+8.694%/−0.310%/+9.458%/+3.762%. Surface/material effects and truth changes retain the unfavorable0291 case. Dense physical work is unchanged; positive error change supports local joint utility, while a visible field change alone does not.

All-new-zero outputs are bitwise identical to cached G-fast on 0277/0291/0294 for all five saved output roles. On0687, fluid, material and both port outputs pass the original pointwise tolerance; **three interface-array values fail** `2e-5 + 2e-5*abs(G-fast)`, with full-array maximum absolute difference1.78337e-4. All344 inherited tensors, normalization, dataset/grid identity and fine work remain exact. Thus the intended unchanged-base operator is supported by state identity and three exact replays, but a universal native numerical recovery claim fails on this high-module case. The failure is retained without another remedy, retry, tolerance change or post-fit code modification. The interface relative L2 discrepancy is9.389e-7; all three failing values are q-proxy channel1 in module slot5. Their saved values and original limits are retained in the [explicit miss receipt](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/organizer/selected_joint_utility/all_new_zero_0687_interface_tolerance_miss.json). A small global norm does not turn those pointwise failures into passes.

Actual native control-only audits over four cases ×three phases ×two routes reconstruct U=C+S+R+I with maximum FP32 absolute discrepancy7.62939e-06; I's full-reference weighted mean is at most1.07654e-06 and source-weighted mean at most1.14809e-08. Reference access exactly matches the saved plan. These24 component and 24 raw-action calls run no additional physical decoder. The fluid-grid mean is not required to vanish under the different declared reference measure, nor are post-tanh means.

The profiler-off native cost comparison uses identical selected-training B8/Q1024 inputs: eight M1 cases and eight M12 cases, sampled at epoch 1 with physical port/loss flags at 1001. Each arm/panel has one warm wrapper, three synchronized FP32 inference samples and one common heat/query input VJP; all input gradients are present and finite. There are30 wrappers and six VJPs, with no optimizer steps. Selected H-add300 and H-joint100 are identified explicitly.

| Arm | Native panel | Inference median s | Input forward + VJP s | Inference peak allocated / reserved GiB | Input-gradient extra peak GiB |
|---|---|---:|---:|---:|---:|
| G-fast | low | 0.114451 | 0.315366 | 0.2757 / 0.4102 | 1.5193 |
| G-fast | high | 0.210556 | 0.455120 | 0.5550 / 0.8203 | 4.8774 |
| H-add | low | 0.141016 | 0.329400 | 0.3190 / 0.4395 | 1.7418 |
| H-add | high | 0.253108 | 0.565980 | 0.5583 / 0.9004 | 5.7653 |
| H-joint | low | 0.137248 | 0.319376 | 0.3131 / 0.4395 | 1.7399 |
| H-joint | high | 0.253645 | 0.552279 | 0.5583 / 0.9004 | 5.7669 |

H-add/H-joint inference costs are1.232/1.199× frozen G-fast at low M and 1.202/1.205× at high M. Complete input forward+VJP costs are1.044/1.013× and 1.244/1.213× respectively. The new interfaces are affordable within the1.5× target on these measured panels, but they **add inference work** rather than saving physical execution. Three inference samples and one VJP per panel do not establish a population latency distribution or a full-data epoch ETA. Reserved inference memory is explicitly measured; input-VJP extra allocation subtracts the resident allocated baseline, and VJP reserved memory was not separately recorded.

Evidence: [native cost samples and gradients](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/engineering/selected_native_cost/native_cost_summary.json) and [native cost process receipt](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/engineering/selected_native_cost/receipt.json).

Evidence: [joint utility and recovery audit](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/organizer/selected_joint_utility/joint_utility_summary.json), [utility process receipt](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/organizer/selected_joint_utility/receipt.json), and [organization/utility figure receipt](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/figures/04_organization_utility_receipt.json).

Evidence: [all22 exact-stage organization](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/report_inputs/organization_matched500.json), [actual control figure and channel norms](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/figures/03_controls.json), and [fit200 CPU exact-projection witness](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/organizer/probe_fit200_0687.json). The last witness is one real native CPU wrapper, not all 22 statistics or an executor-saving measurement.


## Inverse reuse: numerical checks and physical audit

The existing geometry-only Q14 probe was executed on all four representatives, using the mean fluid temperature at six fixed observed sensors. The directions are balanced first/last-module heat transfer and first-module x displacement, with centered FD steps0.001/0.0005. H-add300 exact and H-joint100 soft each execute44 wrappers and eight input VJPs; total88 wrappers/16 VJPs, no parameter updates or solver calls. Full field differences are reduced in wider arithmetic from saved FP32 arrays, while the native six-sensor scalar objective remains FP32. These directions are distinct from the held twelve-attempt heat request.

| Arm | Case | Direction | AD sensitivity | FD 0.001 | FD 0.0005 | AD/FD relative discrepancy % (large / small step) |
|---|---|---|---:|---:|---:|---:|
| H-add | 0277 | heat_transfer | -0.0259031 | -0.0257492 | -0.0257492 | 0.5941 / 0.5941 |
| H-add | 0277 | first_module_x | 0.793445 | 0.79298 | 0.793934 | 0.0586 / 0.0616 |
| H-add | 0291 | heat_transfer | 0.397916 | 0.397682 | 0.397682 | 0.0587 / 0.0587 |
| H-add | 0291 | first_module_x | 0.145679 | 0.145435 | 0.145912 | 0.1675 / 0.1595 |
| H-add | 0294 | heat_transfer | -0.191507 | -0.191689 | -0.192165 | 0.0947 / 0.3426 |
| H-add | 0294 | first_module_x | 0.340452 | 0.339985 | 0.339985 | 0.1373 / 0.1373 |
| H-add | 0687 | heat_transfer | 0.890813 | 0.890732 | 0.890732 | 0.0091 / 0.0091 |
| H-add | 0687 | first_module_x | -0.15836 | -0.157356 | -0.15831 | 0.6341 / 0.0318 |
| H-joint | 0277 | heat_transfer | 0.0873761 | 0.0870228 | 0.087738 | 0.4044 / 0.4125 |
| H-joint | 0277 | first_module_x | 0.870197 | 0.869989 | 0.870705 | 0.0238 / 0.0583 |
| H-joint | 0291 | heat_transfer | 0.267585 | 0.267506 | 0.267982 | 0.0296 / 0.1484 |
| H-joint | 0291 | first_module_x | 0.123374 | 0.123262 | 0.123024 | 0.0903 / 0.2835 |
| H-joint | 0294 | heat_transfer | -0.147708 | -0.147581 | -0.148296 | 0.0857 / 0.3969 |
| H-joint | 0294 | first_module_x | 0.370818 | 0.370502 | 0.370979 | 0.0850 / 0.0436 |
| H-joint | 0687 | heat_transfer | 1.04006 | 1.03951 | 1.03855 | 0.0535 / 0.1452 |
| H-joint | 0687 | first_module_x | -0.117532 | -0.117302 | -0.116348 | 0.1957 / 1.0071 |

Maximum relative scalar AD/FD discrepancy is0.6341% for H-add and 1.0071% for H-joint (0687 x, smaller step). Smaller FD steps do not consistently improve agreement, consistent with FP32 cancellation; there is no silently widened acceptance threshold or whole-model FP64 conversion. All32 direction/step comparisons record zero admission/M/E support changes across P0/P1/P2. They are **same-support local checks**, not tested support crossings or proof of global smoothness.

Heat-independent analytic flow provides an exact known-null target without a new solve. Across the saved Q14 heat trials, maximum RMS increments in u/v/p/omega are4.564e-5/3.725e-6/2.484e-5/3.050e-4 for H-add and 3.856e-5/3.464e-6/2.060e-5/2.853e-4 for H-joint. The maximum increment divided by its selected-training channel standard deviation is2.993e-4/2.800e-4. Leakage is small on this sampled panel but **nonzero**; relative error against a zero physical increment is undefined. The later finite heat audit below supplies different-direction physical truth; module-position signs remain unvalidated.

Evidence: [AD/FD, all-role responses and actual support checks](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/response/selected_input_adfd/derivative_summary.json) and [differentiation process receipt](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/response/selected_input_adfd/receipt.json). No inverse generator, search, design trail or validated inverse design is produced.

Retained TRAIN0348 response measurements are exposed training evidence. The earlier G-fast/Tensor-H Q14 sensitivities disagree in sign on0291 heat transfer (+0.10975/−0.07750) and0687 first-module x (−0.22955/+0.18506). AD/FD self-consistency alone supplies no physical-response truth; the later finite audit tests its own specified heat directions. The twelve-call request changes heat only, so it cannot resolve the module-position sign disagreement. [Retained numerical response evidence](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/report_inputs/old_response_reference_summary.json) is reused without another old-arm evaluation.

The exact requested local analytic-wake/shared-grid audit has 12 attempts: a fresh baseline and two opposite feasible fixed-total heat transfers on each of 0277/0291/0294/0687. The request uses slots0/2,4/1,2/1 and 0/5, with respective amplitudes0.0291021,0.0575153,0.0244189 and 0.00333354. These differ from the existing Q14 first/last-module derivative directions and were not silently substituted. Exact embedded inputs, physical module IDs, fluid/material/interface joins and local generator availability were verified without a solver call.

### Authorized physical execution, failure and alignment

After the neural selector and stop500 decision were sealed, the user explicitly raised the cumulative benchmark ceiling from320 to326 on2026-10-05. The unchanged prior count is314:300 reconciled historical attempts plus six later coverage and eight later inverse-reference attempts. The older78 calls were already executed and counted once; neither they nor the separate M0 allowance create extra capacity. The new exact request consumed **twelve counted attempts**, ending at **326/326, zero remaining**.

The first0277 baseline worker failed before invoking the solver because its one-time preparation helper supplied the unsupported evidence split `test`. The failed start counts, and its log and ledger entry remain preserved. After that process exited, the ignored helper was corrected to the supported `development` split; the remaining eleven unique requested states ran once. There was no baseline retry or thirteenth attempt. Consequently0277 has two converged trial fields but **no fresh baseline**, and its two baseline-to-trial responses are unavailable. No stored H5 baseline or model-only baseline was substituted.

The first actual solve,0277 transfer-minus, converged in2.338s of solver work and3.971s of full worker time. Its final step was9300, with maximum final temperature update5.795e-5 and relative update4.690e-6. The main-thread review inspected this saved outcome before releasing the remaining ten attempts. All eleven invoked solves converged in9300–14900 steps; final maximum updates were5.795e-5–6.589e-5 and relative updates1.184e-6–4.690e-6, below the declared1e-4/1e-5 stopping criteria. Actual solver time sums45.076s; summed successful worker time is64.348s. The complete audit clock, including failed startup, imports, correction and review wait, is303.617s of the1200-second CPU wall limit. These NumPy CPU solves consume no GPU allowance.

Fluid coordinates, masks, query IDs, quadrature, interface angles, material coordinates, physical module IDs, context and solver controls match within each family. References contain8192 fluid-grid rows, with7976/7837/7691/7463 positive fluid masks for0277/0291/0294/0687, plus M×64 interface rows and M×3096 material rows. The adapter widens saved FP32 values to float64; this is **not** an FP64 physical solve. Model inputs use checkpoint normalization and native FP32 physical design encoding; the largest requested heat rounding is5.364e-8 on0291,2.384e-8 on0294/0687 and exactly zero on0277. These representational differences are disclosed rather than changing reference inputs after selection.

Fresh-baseline versus stored-case replay discrepancies were measured separately for0291/0294/0687. Fluid-T RMS is2.877e-7/3.802e-7/4.847e-7, surface-T1.602e-6/1.781e-6/1.733e-6, material-T1.683e-6/1.926e-6/1.785e-6 and q-proxy1.206e-5/1.331e-5/9.971e-6. Replay similarity never replaces the missing0277 baseline. Output ULP, final-step convergence and replay differences are numerical warning indicators, **not certified grid-error bounds**. Tiny module responses comparable to them have unresolved physical signs.

Evidence: [explicit326 authorization](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/physical_audit/authorization.json), [unchanged prior allowance reconciliation](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/reference/preparation/current_allowance_reconciliation.json), [twelve-attempt ledger](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/reference/reference_attempts.jsonl), [first actual solve review](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/reference/first_attempt_review.json), [reference outcome summary](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/reference/reference_outcome_summary.json) and [independent reference quality audit](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/report_inputs/reference_cpu_quality_audit.json).

### Matched finite-response evidence

Each retained model makes exactly eleven ordinary native wrappers on the eleven actual solved inputs: G-fast physical1000, retained Tensor-H physical1000, selected H-add fit300 exact and selected H-joint fit100 soft. All344/384/382/382 loaded state tensors respectively remain bitwise unchanged. Inputs contain design, prescribed context and role geometry; solved values are evaluation targets only. Each model executes21,986,448 padded fine-MLP leading rows in1771 successful physical calls: identical dense value work, no executor savings. No optimizer, input VJP, inverse search, reference retry or post-outcome reselection occurs.

The primary response denominator is **six** baseline-to-minus/plus comparisons from0291/0294/0687. Each metric below is the unweighted mean of six per-state quadrature-weighted RMSEs, in native benchmark units. The zero-change control predicts no finite change; its error is the measured reference response RMS. These are repeatedly exposed validation layouts with new heat responses unused in fitting, not untouched layouts or a broad population test.

| Primary response error | Zero change | G-fast | Retained Tensor-H | H-add300 | H-joint100 |
|---|---:|---:|---:|---:|---:|
| Fluid temperature |0.0511852|0.0431030|**0.0278715**|0.0467881|0.0437068|
| Surface temperature |0.0977555|0.0543307|**0.0332353**|0.0684079|0.0582474|
| Material temperature |0.117894|0.0536341|**0.0343091**|0.0677365|0.0575351|
| q-normal proxy |0.195847|0.0879251|**0.0842230**|0.0888057|0.0893501|

Selected H-joint improves fluid/surface/material response error6.586%/14.853%/15.060% against selected H-add, and improves14.610%/40.415%/51.197% against zero change. Its q-proxy error is0.613% worse than H-add. Each thermal role wins only3/6 paired transfers. Averaging squared per-state errors before taking a root gives a different declared aggregation: H-joint fluid/surface/material RMSE0.058237/0.077093/0.076009, with6.856%/17.816%/17.712% gains over H-add; q-proxy then improves0.471%. The corresponding H-add fluid RMS0.062524 slightly exceeds zero-change0.062345, so even its apparent macro-mean benefit is aggregation-dependent. H-joint remains worse than G-fast for all four macro response means, and retained Tensor-H has the lowest means. Since the selected fit ages/stages differ, these physical results do not establish a matched-age causal joint advantage. The matched500 reconstruction comparison and fixed-weight I interventions remain the complementary evidence above.

Absolute reference-state errors are different quantities: over the eleven available states, macro fluid/surface/material T RMSE is1.20563/1.25158/1.08748 for G-fast,0.844857/0.883203/0.784862 for retained Tensor-H,1.02376/1.00102/0.855164 for H-add, and1.00643/0.993892/0.853796 for H-joint. H-joint slightly improves this eleven-state reconstruction versus H-add while retaining meaningful temperature bias. No absent0277 baseline enters these absolute means.

![Fresh local reference before and after heat transfer with measured and selected-model finite responses](../../diagnostics/generated/receiver_interaction_20261005/figures/05a_reference_response_fields.png)
Figure5a. Actual saved reference fields use baseline→plus on0291/0294/0687; measured fluid-T response RMS is0.094454/0.051829/0.007273 native temperature units. The0277 row is explicitly the **secondary minus→plus span**, twice the one-sided amplitude, with RMS0.157425; it does not supply the failed baseline or enter the six primary means. Reference temperatures and measured/predicted signed finite responses show both useful local structure and unresolved amplitude/direction failures; panel error labels give response RMSE rather than a residual color map. Native coordinates/temperature units and the analytic-wake/shared-grid reference limit are retained; no CFD claim follows.

The0291 failure is resolved at a substantial fluid-temperature scale. Its minus/plus reference mean changes are−0.0289189/+0.0289138. G-fast predicts+0.0285575/−0.0312943, H-add+0.0373588/−0.0384973 and H-joint+0.0332576/−0.0354399: **both directions have the wrong mean sign** for those three models. Retained Tensor-H predicts−0.00270472/−0.0119571, retaining the wrong plus direction and an underestimated minus amplitude. H-joint's0291 plus response error0.0999655 exceeds the zero-change error0.094454. The correctly signed0294/0687 mean fluid-T responses do not rescue this miss; signed means and spatial response error must be reported together.

Unchanged-own-heat receivers demonstrate actual nonlocal thermal dependence in the benchmark. For example,0291 module2 peak changes−0.263251/+0.263251 despite unchanged own heat; H-joint predicts−0.163203/+0.137327, underestimating the amplitude.0294 module4 reference magnitude is0.074878 and0687 module7 is0.017497. The near-zero0294 modules5/6 changes2.86e-6–7.63e-6 and0687 module2 change7.82e-5 remain numerically qualified; tiny predicted/reference sign agreement is not positive physical identification.

Across32 unchanged-own-heat module-direction responses, pooled peak RMSE is0.059703/0.041874/0.077315/0.065056 for G-fast/Tensor-H/H-add/H-joint, versus zero-change0.094318. H-joint beats zero change on only15/32 entries. Of22 reference peak changes above the summed-final-iterate warning scale,17 have the same predicted sign; ten smaller entries are not asserted resolved. On0291 module0 minus it predicts−0.002809 versus reference−0.158561; on0294 module3 minus it overpredicts−0.10208 versus−0.02279, while0687 module7 minus is comparatively close at−0.018886 versus−0.017497. These favorable and unfavorable amplitudes prevent a blanket nonlocal-transfer claim.

The independently aligned0277 minus→plus span is secondary only: reference fluid-T signed mean+0.040241 and RMS0.157425, with unchanged-own-heat module1 peak+0.092199. H-joint fluid response RMSE is0.069629 versus H-add0.073336, G-fast0.062683 and Tensor-H0.060818. H-joint almost erases the mean increment (−3.65e-5) and predicts module1 peak only+0.006757. Two valid endpoints support this finite comparison without another solve or model call. Its2× amplitude, absent baseline and exclusion from primary aggregates remain explicit.

![Four retained models versus zero-change and H-add controls for measured finite thermal response and unchanged-flow leakage](../../diagnostics/generated/receiver_interaction_20261005/figures/05b_response_comparison.png)
Figure5b. Six primary finite responses compare all four retained models and the zero-change control. H-joint fluid/surface/material response errors0.043707/0.058247/0.057535 improve6.586%/14.853%/15.060% against H-add, but retained Tensor-H remains better on all three means. The wrong0291 direction and amplitude remain visible alongside nonlocal module-peak responses. Reference flow and8%-band pressure changes are exactly zero. The flow bars show mean response RMS divided by the common selected-training standard deviation, not maxima; pressure bars show mean absolute increments,0.000182494 for H-add and0.000201389 for H-joint. These finite model changes show remaining null-control failure. These measurements establish local successes and misses, not inverse-design validity or learned-group physical causality.

All six primary references have bitwise-zero u/v/p/omega responses and zero8%-band pressure increment, as required by this heat-independent analytic flow generator. Maximum per-transfer RMS leakage in u/v/p/omega is0.002466/0.0002704/0.0009893/0.006687 for H-add and0.002825/0.0003014/0.0008081/0.007431 for H-joint. Maximum absolute8%-band pressure response is0.000353974/0.000349005 respectively, versus reference0. Relative-to-zero error is undefined; these larger finite-amplitude requests are distinct from the epsilon Q14 probes. Joint control has not enforced the known heat-null flow constraint.

**Predictor gain/miss:** some reconstruction and response calibration over H-add, but retained Tensor-H leads the response means and0291 direction fails. **Organizer gain/miss:** actual query I can be useful locally, yet its groups do not establish correct causal thermal transfer and physical value execution remains dense. **Inverse gain/miss:** differentiability and local AD/FD consistency are measured; directional/amplitude and null-flow failures prevent reliable inverse reuse. **A/B/C:** A remains mixed learning value, B is faithful control provenance without executor savings, and C now has actual bounded physical evidence with consequential failures. Preserve these models and outcomes; any targeted response-supervision or module-position truth experiment requires a separate reviewed scope. No automatic formal or inverse run follows.

Evidence: [matched four-model comparison](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/physical_audit/model_evaluation/comparison.json), [checkpoint normalization binding](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/physical_audit/model_evaluation/normalization_binding_audit.json), [independent secondary0277 span](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/report_inputs/secondary_0277_span_review.json), [final326 attempt reconciliation](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/reference/final_attempt_reconciliation.json), [physical-response figure receipt](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/figures/05_physical_response_receipt.json) and [final cumulative physical-audit status](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/physical_audit/final_status.json).

## Retained histories and limitations

G-fast and old Tensor-H scientific checkpoints and artifacts remain intact. Formal Tree3501 was already interrupted after 97 completed epochs and has no saved checkpoint; scalar logs do not recover its weights. Formal Global3502 was already stopped with a resumable e400 checkpoint. Neither was gracefully paused by this round, restarted or used as the new physical source.

The completed Lean reset measured 144 Tree index builds for eight cases (18 per case), versus once-wrapper tensor planning. G-fast's same-operator specialization saved 2.24%/5.11% on representative low/high effective updates. Historical complete development epochs were9.78937s for G-fast and 12.39667s for Tensor-H (+26.63%); these are subset epochs, not a measured full-data ETA. New frozen-host fitting has a different optimizer workload and is not a direct physical-backbone training acceleration comparison.

The older generic-Global/G-fast comparison retains five strict high-module VJP misses at its unchanged `rtol=2e-5, atol=1e-6`, alongside passing native output and saved-update checks. Both new fits share the same G-fast operator, so their attachment checks do not reclassify that older numerical result. The saved Tensor-H release-reader parity remains a separate passed comparison.

Old Tensor-H versus G-fast improved fluid/surface/material temperature means19.95%/12.05%/11.26%, while pressure worsened21.26% and all four flow-channel means worsened. Initial port outside temperature worsened on all 22 while refined final port outside temperature improved on all 22. Material temperature at M10 worsened despite the overall gain, and q-proxy maximum worsened despite its improved mean. Both retained references executed43,972,896 padded fine rows in3542 physical calls over all 22. The report's edge-pressure difference, full pressure-field RMSE and 8%-band pressure functional remain separately named quantities.

## Figure and evidence index

All eight embedded pages were inspected against saved numerical evidence. Five PDF masters are the retained primary exports; the eight raster companions exist specifically for Markdown display.

| Group | Retained PDF | Saved evidence |
|---|---|---|
|1 Learning and cost|[Learning/cost](../../diagnostics/generated/receiver_interaction_20261005/figures/01_learning_cost.pdf)|Native500-epoch CSVs and measured time/memory|
|2 Physical fields/residuals|[Two physical pages](../../diagnostics/generated/receiver_interaction_20261005/figures/02_physical.pdf)|Selected H-add300 / H-joint100 fixed4 arrays|
|3 Actual controls/support|[Two control pages](../../diagnostics/generated/receiver_interaction_20261005/figures/03_controls.pdf)|Selected H-joint100 native phase graphs and full receiver reference|
|4 Organization utility|[Admission and intervention](../../diagnostics/generated/receiver_interaction_20261005/figures/04_organization_utility.pdf)|Matched500 all22 and selected I-removal arrays|
|5 Physical responses|[Two measured-response pages](../../diagnostics/generated/receiver_interaction_20261005/figures/05_physical_response.pdf)|Eleven actual reference states,44 matched model wrappers, six primary transfers and secondary0277 span|

Generated numerical evidence, one-time renderers, PDF masters and necessary raster companions remain local in ignored paths. These figure links intentionally depend on the retained local scientific artifact tree; Git contains durable source, tests, guide and report only. No inverse trail is fabricated.

## Validation and delivery

The combined focused CPU regression run passed77 tests, including new interface/attachment/stop tests and historical Tensor/export/work/resume cases. Ruff passed the seven new Python source/test/tool files. Implementation commit6704b20 passed the configured pre-push artifact gate; the entire outgoing object range contained durable source, tests and the fitting guide only. Local and remote branch tips matched after push. The native stop-request implementation uses the existing atomic latest-checkpoint writer after a complete epoch boundary. Its disposable AdamW/dropout/accumulation test stops, resumes and reproduces the next uninterrupted update bitwise; a failed save preserves the old latest and leaves the request unacknowledged. This is an implemented durability improvement, not retroactive recovery of Tree3501.

Twenty unique GPU process envelopes account for **3.034311 aggregate GPU-hours**: fitting2.892332, endpoint evaluation0.074910, engineering0.015949, selected neural audits0.043836 and final physical-reference model reads0.007285. Four scientific trainer and five all 22 evaluator envelopes completed normally. Two failed pre-update engineering starts are included (3.939s and 7.158s); the passing disposable engineering check performs four updates. All130 selected neural-audit wrappers and22 input VJPs completed, followed by44 physical-reference model wrappers with zero optimizer updates or extra solver calls. CPU tests/preparation/rendering add no GPU charge. Callback, per-wrapper and VJP timings are nested within their processes and never charged twice. No same-GPU overlaps occurred. GPU1/2 are idle at closeout; unrelated GPU0 work remains untouched.

The H-joint outer PTY launcher returned143 when polled, while its actual native child PID, [done] log, final 500 checkpoint and process receipt establish exit0. This launcher observation is retained separately rather than relabelling the scientific fit as failed or hiding the observation.

The original neural-work-finished receipt was2.1176elapsed hours; the continued physical audit and final report/delivery time are recorded from the original21:04:05UTC start in the ordinary closeout receipt. Total GPU work remains below12hours, elapsed time below8hours and the CPU audit below20minutes. Exactly twelve new counted local benchmark attempts occurred, eleven converging and one failing before solver invocation. No1000 extension, formal restart, CFD solve, training-atlas expansion or inverse search occurred.

Evidence: [unique-process budget and work](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/accounting/final_gpu_budget.json), [native endpoint inventory](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/evaluation/final_endpoint_inventory.json), [outer launcher observation](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/accounting/outer_launcher_h_joint_observation.json), and [final delivery/elapsed receipt](/data/wanglz/ModularDT/thermal_development/receiver_interaction_20261005/accounting/delivery_receipt.json). The final round adds this durable Markdown report; scientific checkpoints, arrays, generated figures and one-time helpers remain outside the Git payload.
