# HONF next development: inexpensive multiview native routing and a stable comparison protocol

**Status:** proposed Goal-mode development plan. No experiment in this plan has been performed by the reviewer.  
**Reviewed source:** `cosmos2w/ModularDT`, `agent/honf-core-next`, `0004ab6567ed1e2a908012d9b4690cee2f2b83be`.  
**Primary task:** retain the source-preserving predictor, remove avoidable routing overhead, and test whether two feature-block-specific donor distributions improve the accuracy/robustness tradeoff of the present scalar organizer. Establish fair, reusable development and mature-reference comparisons at the same time.  
**Not authorized:** automatic full-TRAIN/5,000-epoch fitting, new physical solves, inverse generation/search, a recursive Tree, a new collective latent bank, a frozen D-sep replacement, or a broad architecture/loss sweep.

## 0. Objective and scope

The project objective remains:

**Learn from multi-field reconstruction → learn a reusable organization of module–environment interactions → reuse that organization in modular inverse design.**

A better scalar loss, more concentrated attention, a larger test count, and fewer logical links are not substitutes for that objective. Conversely, a useful development direction need not beat every mature classic checkpoint or already deliver a sparse executor. Keep scientific direction, practical forward quality, numerical contracts, execution price, and inverse validity separate.

This phase has three concrete deliverables:

1. A maintained, inexpensive one-/two-view native-routing implementation, with the original P computation and old P-R/P-C checkpoints preserved.
2. A fresh matched P/R1/R2 comparison on each existing segmented dataset, trained long enough to judge the proposed representation, with at most one conditional second-seed confirmation pair.
3. A stable evaluator and compact report that distinguish matched-development effects from the gap to mature reference systems, and show actual physical fields, native gathering, and a small saved-response replay.

The intended new mechanism is deliberately narrow: **different message-feature blocks may gather different donors, without evaluating two value networks or introducing extra latent group slots.** This is a hypothesis about an inductive bias, not an established explanation of the current Thermal failures.

## 1. Read and preserve the current evidence

Paths below are relative to `HONF_Proj/` unless stated otherwise. Read current `AGENTS.md`, dataset protocols, and:

- `docs/reports/_bk/20261012_053211Z_HONF_Native_Interaction_Routing_Development_Report.md` and its maintained companion evidence index;
- `docs/reports/_bk/20261011_224701Z_HONF_Reference_Anchored_Consolidation_20261011.md` and `20261011_224701Z_HONF_Reference_Anchored_Evidence_Index_20261011.md`;
- `docs/reports/_bk/20261010_195650Z_HONF_Interaction_Preserving_Recovery_20261010.md` and its appendix;
- `docs/reports/_bk/20261010_154719Z_HONF_Joint_Formal_Comparison_20261010.md`;
- `docs/guides/Thermal_Model_Development_Protocol.md`, the native-routing guide, and the existing shared TrainingEngine instructions;
- the preceding `20261011_235532Z_HONF_Native_Interaction_Routing_Development_Plan.md`.

### 1.1 What the last round established

| Area | Observed result at the declared scope | Interpretation to retain |
|---|---|---|
| Wind learning | Warm P-R e2500 priority ratio 0.946386 versus P and 0.942879 versus P-C; all declared priority mean/tail guards pass on DEV24 | Conditional donor-selective learning benefit, one parent/seed and eight correlated-layout clusters; not fresh-start or full-data superiority |
| Thermal learning | Fluid-T RMSE P/R/C = 0.608297 / 0.574835 / 0.591745; v = 0.00331994 / 0.00337088 / 0.00333517 | Real temperature improvement, but no common priority field improves 5% against both controls; not a broad reconstruction collapse |
| Thermal robustness | M5 material/peak misses; R worst fluid-T 0.996270 versus 0.906601 P and 0.871385 C | Averages do not remove local risk; these cases remain visible, not excluded or reweighted retrospectively |
| Organizer | Full rebuild/reset/reassignment changes physical predictions and restores exactly; Wind MM changes dominate inspected examples | Operational reliance, not physical causality or proof of higher-order expressivity |
| Cost | Wind Q8192 P ≈15.3–15.6 ms, R ≈18.8–19.1 ms; smaller-Q overhead 51–56%; full-objective update overhead ≈11% | Real overhead, not a return to the old Tree's execution regime; its source must be measured |
| Interface | Tested heat independence/affinity, finite input VJPs, source IDs and stale-state rejection remain | Software contracts, not physical response or inverse-design qualification |

Preserve the original result: the old 20% Q8192 target was missed, so the old plan correctly blocked its optional fresh pair. This new request authorizes a new bounded fresh-learning study. Do not rewrite the old gate or label its missing optional tests as completed failures.

The first 100 new Wind R epochs lack observed query hashes. The matched suffix and local parent replays do not establish a complete historical query-provenance pass. Do not spend this round trying to manufacture the missing prefix. Fresh matched streams will resolve this limitation prospectively.

The corrected 0291 provenance is exposed DEV/audit, not TRAIN. The protected historical inventory includes 67 pre-existing absent bindings. Preserve those absences as observations; do not report 953 present files. Keep all prior checkpoints, numerical arrays, reference artifacts and migration provenance unchanged.

## 2. Diagnosis: what is known and what is proposed

### 2.1 Known from the implementation

`NativeInteractionRouting.score` produces one scalar per destination–donor pair and route type. `InteractionContextCore.prepare_context` multiplies that scalar weight across the entire message vector. The scalar scorer is shared across the two rounds, separately for MM, ME and EM. The existing nonlinear values are still computed before donor reduction, and individual physical-source reads remain in the final field decoder.

Thus current P-R can choose different donors at different native destinations, but it uses **one donor distribution for every feature coordinate of a given message**. Thermal's shared context feeds both nonlinear flow and affine-temperature readers. Wind's context feeds its nonlinear velocity reader. These are real code facts; they do not establish that a shared distribution caused the measured Thermal tradeoff.

`score`, `gain`, and `_native_route_weights` also perform repeated Python `bool(torch.isfinite(...).all())` checks. Native ID validation performs per-batch masked indexing/uniqueness checks. R and C both enter parts of this validation path; P does not enter all of them. These operations are candidates for host/device synchronization overhead. Their elapsed contribution is **not yet measured**. Do not attribute the entire penalty to them merely from inspection.

### 2.2 Proposed scientific hypothesis

One destination may need more than one way to gather information: for example, one feature subspace may encode near-body variation while another encodes broader context. We will permit two distributions over the same donors, each acting on half of the existing message coordinates. The views are learned, not labelled as “flow” and “temperature,” and both remain available to every applicable output head.

This could reduce an unnecessary shared-weight restriction while retaining the cheap architecture. It could also fail because the real limit is data diversity, the value representation, the objective, or missing response excitation. The experiment must distinguish those outcomes rather than force a two-view success.

### 2.3 Things not changed

Do not simultaneously change fine-reader width, number of context rounds, environment resolution, source features, physical output laws, primary/auxiliary loss coefficients, query allocation, normalization, or effective batch size. Do not add a third view, gradient surgery, loss rebalancing, extra local field heads, forced sparsity, or a new geometry prior.

No analytic wake or physical solver becomes part of inference. The new Thermal model remains self-contained and jointly trained, not a composition with frozen D-sep. Generality means one implementation and training workflow with explicit dataset capabilities, not identical physical laws, weights, or every hyperparameter across datasets.

## 3. Fair comparison: three boards, three questions

### 3.1 Board A — controlled development effect

**Question:** under the same available information and training protocol, what does the organizing mechanism add?

Use fresh P, R1 and R2 on identical segmented TRAIN/DEV memberships within each dataset. Match common initial value tensors, case order, sampled receiver IDs, primary and auxiliary supervision, normalization/calibration, schedule, effective batch, precision, and checkpoint-selection opportunities. Routing-specific random initialization must not advance common-path or sampling RNG streams.

- P: original measure-weighted source-preserving core, no native scorer.
- R1: current single-distribution routing, trained from scratch rather than attached at e1000.
- R2: proposed two-view routing; all other settings equal to R1.

P versus R1 is a mechanism-plus-small-capacity comparison. The previous warm P-C comparison is supplementary evidence, not a fresh P-C ablation. R2 versus R1 changes only 96 scalar parameters at the proposed widths, so it is the principal near-capacity-matched test of the two-view hypothesis. State that distinction explicitly.

Historical warm T4301/T4302 and W2601/W2602 remain separate curves/markers. They are not the fresh controls and do not add training seeds to the new study.

### 3.2 Board B — practical distance to mature references

**Question:** how useful is this development predictor today compared with the best retained systems?

Evaluate the following preserved checkpoints on the **same authorized evaluation cases, receivers, masks, units and output definitions used for the new models**:

| Reference | Recorded identity to verify before evaluation | Role |
|---|---|---|
| Thermal1404 | Mature literal e5000, native routing-only pairwise implementation | Requested classic organizing reference |
| Thermal1804 | Mature literal e5000, native dense implementation | Requested strong accuracy reference |
| Wind2102 | Current reports bind native E512/K6 selected e2440 from a 2500-epoch history | Requested classic organizing reference |
| Wind2103 | Current reports bind native E512 dense selected e2475 from a 2500-epoch history | Requested strong accuracy reference |
| Migrated Wind2202 | Qualified self-contained P/direct e5000 with full TRAIN420 exposure | Retained efficient mature reference |
| Thermal3904 | Native thermal endpoint plus explicitly frozen3901 flow | Optional efficient historical system reference; not a new joint-model control |

Do not assert all Wind classics trained 5000 epochs. Resolve the actual paths, stored epoch, state hash, selected/terminal policy, full history and available later checkpoints. A genuinely available newer state can be listed separately, never silently substituted. Preserve the known 2440/2475 bindings.

Each classic uses its own saved normalizer, native context inventory, field/local-surrogate path and source features. In particular, do not pass E8 context to an E512 classic. Denormalize before physical error comparison. A classic's frozen local model or other historical pretraining is extra information and must be listed, not hidden by the common primary-data count.

If cached classic predictions have exactly matching semantic case/query/mask/target bindings, reuse them. Otherwise perform one read-only evaluation on the authorized DEV22/DEV24 panels. Do not average the old canonical89 metric beside a new DEV22 metric or the old fullVALID90 metric beside a new DEV24 metric and compute a ratio. Mark any unavailable comparison NA after a bounded lookup; never fabricate or reconstruct missing outputs from aggregate statistics.

This is a legitimate **trained-system benchmark**, not an equal-training architecture ablation. A subset model need not beat a full-data mature reference to justify further research. Conversely, beating weak P by 5% does not mean the absolute classic gap has closed.

### 3.3 Board C — formal study readiness

**Question:** is there a stable fresh recipe worth a user-launched full-TRAIN study?

Prepare, but do not run, a paired full-TRAIN recipe for the preferred candidate and P only when the development results support it. The selected recipe must preserve the tested mathematical model and objective; only the declared data membership, TRAIN-fitted statistics, run identity and explicitly reviewed full-horizon schedule change.

A formal candidate-versus-P comparison tests organizing value at full exposure. Existing classics remain practical references. A claim isolating superiority over a classic architecture would require a separately budgeted matched classic refit, including its input/pretraining advantages. Do not retrain all four classics in every development round or weaken them to fit the new model's context budget.

**Research launch readiness is not classic replacement, response certification, independent-test success, or inverse readiness.** Report these as separate decisions. A mature organizer is not required to prepare a sensible research experiment.

### 3.4 Work and statistical units

Every comparison table includes a compact identity record: primary and auxiliary training families, inherited/fresh state, saved age, optimizer updates, case visits, primary/auxiliary query counts, objective version, and input context. Epochs alone are not comparable work.

For illustration, 2500 epochs over Thermal150 supply 375,000 primary case visits and ordinarily 10,000 updates; 5000 epochs over Thermal600 supply 3,000,000 visits and ordinarily 65,000 updates. Wind's historical effective batches and Q differ as well. Verify actual counters; never extrapolate unavailable historical timing or equate case visits with unique information.

Compute per-case/role errors first. Thermal uses equal-case summaries; Wind uses equal-layout paired summaries with the three stored directions inside each layout. Keep row quantiles as descriptive quantities. Do not bootstrap cells or direction rows as independent experimental samples. Optional paired cluster resampling describes this exposed panel, not training-seed uncertainty or a blind population test. Report every seed separately; never report the best seed as the method.

## 4. Mathematical design: one value computation, two gathering views

### 4.1 Retain the original pair values

For round `l`, typed route `tau`, native destination `r` and donor `j`, retain

\[
 m_{rj}^{(l,\tau)}=\Phi_\theta^{(l,\tau)}
 [z_r^{(l)},z_j^{(l)},\gamma((x_r-x_j)/L)]\in\mathbb R^{128}.
\]

The legacy route names have explicit directions: MM is physical source → physical source; ME is environment → physical source; EM is physical source → environment. MM self-exclusion, absent slots, physical measures and simultaneous state-update order remain unchanged.

Let `w0[r,j]` be the actual legacy row weights, already incorporating the applicable measure and masks; let `s_r=sum_j w0[r,j]`. An empty valid donor row has zero aggregate. Do not silently assume the floating-point row sum is exactly one.

### 4.2 R1 and R2 score networks

For each route type, use the existing shared-across-rounds width-32 hidden score trunk. R1 has one output; R2 has two. The raw score input remains destination state, donor state and signed relative geometry. For view `h`,

\[
 \ell_{rj,h}=2\tanh(f_{\eta,\tau,h}(z_r,z_j,\gamma(\Delta/L))/2),
 \qquad
 w_{rj,h}=s_r\frac{w^0_{rj}\exp(\ell_{rj,h})}
                         {\sum_k w^0_{rk}\exp(\ell_{rk,h})}.
\]

Use the maintained `expm1` zero-start implementation, with its explicit head axis, rather than switching to a numerically different softmax for convenience. The formula above defines the mathematics; the existing zero-score contract controls the implementation.

For nonempty rows a stable identity-start form is

\[
 d_{rj,h}=\operatorname{expm1}(\ell_{rj,h}),\quad
 \bar d_{r,h}=\frac{\sum_jw^0_{rj}d_{rj,h}}{s_r},\quad
 \rho_{rj,h}=1+\frac{d_{rj,h}-\bar d_{r,h}}{1+\bar d_{r,h}},\quad
 w_{rj,h}=w^0_{rj}\rho_{rj,h}.
\]

Handle zero rows explicitly without undefined divisions. All scores remain finite and bounded; no clipping/tolerance change is authorized after results.

### 4.3 Blockwise gather

Partition the existing message vector into two fixed coordinate blocks, `m=(m^(1),m^(2))`, each width64. R2 gathers

\[
 a_r=\operatorname{Concat}\left[
       \sum_jw_{rj,1}m_{rj}^{(1)},\;
       \sum_jw_{rj,2}m_{rj}^{(2)}\right].
\]

The existing update network consumes the same width-128 aggregate. The value MLP is evaluated once per pair, not once per view. There is no extra concatenated field branch, new regional bank, new residual predictor, or new output projection. The value coordinates themselves are learned, so the fixed split does not assign a hand-coded physical role to a head.

Both final score projections initialize to zero. The common value tensors are identical to P at initialization; R1/R2 score-trunk tensors are identical within a seed. Initial ordinary fields recover the P function under the declared arithmetic criteria, and new final-layer gradients remain live. Do not force diversity, specialize one view to flow and the other to heat, or add random source-ID offsets to make plots look different.

### 4.4 Parameter and work expectations

With H/message128 and width32 scores, the reviewed single-view counts are 26,496 Thermal and 27,360 Wind additional parameters. Replacing three output vectors of length32 by three matrices of shape2×32 adds exactly **96 parameters**:

- R2 Thermal: expected 26,592 added parameters;
- R2 Wind: expected 27,456 added parameters.

Verify instantiated counts before training. Views double score-output/normalization arrays, not pair-value MLPs or total message width. Do not claim runtime parity from this parameter calculation. Count actual value rows, score rows, reduction work and complete time.

### 4.5 What the organizer represents

The trace identity is `(round, route type, native destination ID, view ID)`, with native donor IDs, coordinates, physical measure, weights and weighted message content. It describes a directed weighted gather into an actual source/environment state. It can be drawn as a destination-anchored hyperedge, but it is also message-passing attention; no claim of genuinely higher-order expressivity follows.

Views are latent feature partitions, not energy, probabilities of physical causation, or independently validated response columns. Source-conditioned states still carry upstream global/other-donor information. The final arbitrary field query still reads physical sources directly; do not rename a native context destination as a new query-packet center.

There is no entropy target, fixed small K, learned merge controller, compulsory source pruning, or requirement that the two views differ. Identical learned views are a valid negative finding.

## 5. Narrow execution improvement before model selection

### 5.1 Bounded attribution and repair

Reserve at most **60 minutes of engineering elapsed time and 0.5 aggregate GPU-hours** for this stage, including failures. Take at most two short profiler traces on preselected TRAIN low/high-M scenes. Time profiler-off calls separately. Inspect:

- repeated scalar reads of CUDA finite flags in `score`, `gain`, and `_native_route_weights`;
- per-call native ID uniqueness checks that could be verified when immutable CPU input catalogs are bound;
- receipt-only tensors or diagnostics constructed when tracing is off;
- repeated allocation of constant masks and shape-only metadata;
- ordinary context preparation versus unchanged final query reads.

Prefer consolidating device finite flags and checking them at the public preparation/forward or optimizer boundary. Invalid inputs/intermediates must still be rejected before results escape or any optimizer step is committed. Tests must cover one bad score in one route/round; an invalid flag cannot be overwritten by a later finite route.

CPU validation reuse is permitted only for genuinely immutable, identity-bound catalog metadata. Geometry/control/parameter changes and tensor mutations must still invalidate prepared state. Do not cache stale numerical context, reuse detached geometry gradients, remove safety checks globally, or accept invalid input to save milliseconds.

Keep all scientific arithmetic and physical value reductions unchanged. Do not revive unqualified affine factoring, enable TF32/mixed precision, alter batch shapes, or introduce a custom CUDA/Triton project in this phase.

### 5.2 Qualification and fallback

At retained P/R/C weights, compare fields, source/query input VJPs, parameter gradients and a cloned AdamW boundary before/after the scheduling repair on the existing low/high-M native cases. Apply the repository's pre-existing criteria without widening them. Keep failing historical derivative/curl/flux cases distinct from this comparison.

If the fast path does not qualify or save meaningful time, retain checked execution and continue the scientific fresh comparison within the resource cap. This stage must not consume the whole task or become another same-function factoring campaign.

Record cold setup, fresh-scene complete calls and prepared fixed-scene calls separately. Existing context reuse is a baseline capability; amortizing it is not a new organizer speedup. A repeated-query result is valid only for a stated repeated-query workload and its full formation price.

## 6. Implementation map

| Maintained file | Required narrow change |
|---|---|
| `src/honf_forward_core/interface_fields/native_interaction_routing.py` | Versioned one-/two-view score output; head-aware measure-preserving normalization; maintain exact scalar-v1 loader behavior; optional consolidated validation bookkeeping |
| `src/honf_forward_core/interface_fields/interaction_core.py` | Reuse existing pair features/value tensors; apply head weights to feature blocks before donor reduction; preserve original measure path and two-round order; export head-resolved traces only on request |
| `src/honf_forward_core/interface_fields/interaction_preserving_joint.py` | Serialize view count/partition/config; preserve P/P-R/P-C/P-G/P-H historical states; include new semantics in freshness guards |
| `Case_ThermalChannel/src/channelthermal/joint_regional.py` | Fresh routed-model schema and strict loading; preserve heat-free preparation, native curl, all output extraction and physical forcing law |
| `Case_WindFarm/src/windfarm/interaction_preserving.py` | Fresh routed-model schema; preserve E8 native P inputs, profile/normalizer, nonlinear output and native coordinates |
| `Case_ThermalChannel/src/channelthermal/training/joint_task.py`, `Case_WindFarm/src/windfarm/training/joint_task.py` and reused Wind provider | Exact current objectives, TRAIN-only calibration and task sampling; no copy of the engine |
| `src/honf_runtime/unified_training.py`, `tools/joint_regional_train.py` | Fresh model initialization and explicit fresh identity through the maintained engine; deterministic per-case query streams, saved100 review and resume |
| Existing native-routing tests/evaluators | Extend—not replace—normalization, identity, interference, native metrics, state checks, runtime and reference manifests |

The present routed Thermal engine loader expects an architecture-child receipt tied to P/e1000. A fresh run requires a **new explicit initialization schema**. Do not invent an e1000 parent or loosen the existing child/migration/400-update-diagnostic guards. A fresh receipt must record no inherited model/optimizer history, its common initialization digest, active model configuration, provider/scales and complete sampler namespace.

Do not implement a second routing core under each dataset. Different spatial dimensions, environmental inventory, output laws and normalization belong in adapters/configuration. Both datasets must call the same tested gather routine and the same TrainingEngine.

## 7. Mandatory software preflight

Keep the test matrix small and targeted. Cover:

1. P, R1 and R2 common initial tensors; zero-score native predictions; live new gradients; exact compatible shape/state inventories. R1 loader must reproduce old scalar checkpoints without reinterpretation.
2. Nonuniform physical measures, zero rows, no environments, M1 self-exclusion, padding, donor permutation and consistent environment atom refinement. Apply measure exactly once per view.
3. Tied-head R2 versus R1 mathematical/numerical equivalence; a constructed two-view example in which destination-dependent donor ordering differs; weighted-content sums reproduce executed aggregates.
4. Source/query/geometry VJPs, strict reload, stop/save/resume and one cloned ordinary optimizer step. Fixed input/parameter mutations, policy changes, cross-owner objects and disable/reenable cycles invalidate stale states.
5. Thermal actual-input heat-null flow, affine heat increments and native-curl grid restrictions; no current heat or observations in coefficient preparation. Wind nonlinear capability must reject exact affine increment requests.
6. Invalid metadata and single-route nonfinite flags still raise under any reduced-synchronization path. Public safety and loss/gradient checks cannot be disabled to pass performance.
7. One short fresh P/R1/R2 stream audit checks identical **observed** receiver indices, cases, masks, normalization and effective denominators from epoch1. Retain compact hashes/counters, not every enormous query array.

A software pass permits the bounded learning experiment; it does not certify physical response or sparse execution.

## 8. Primary fresh development runs

### 8.1 Immutable datasets and recipes

Reuse `fixed25_v1`: 150 original TRAIN cases, 22 exposed DEV cases, all available primary TRAIN module-count categories. Reuse `wind_shared_fixed24_v1`: 24 TRAIN layouts/72 direction rows and eight DEV layouts/24 direction rows. Preserve every stored direction in a layout; do not split correlated directions across TRAIN and DEV.

The Wind TRAIN subset's missing M18 remains a disclosed coverage limit. Do not silently replace cases based on the new errors. A future balanced v2 subset would be a separate experiment; this round keeps continuity and isolates the proposed change. DEV M18 results are extrapolative in module-count category, not a newly independent test set.

Use current P configuration: two rounds, hidden/message128, Thermal E192 and Wind E8, Thermal Q1024 primary fluid points plus all established auxiliaries, Wind Q4096 total across the five roles. Effective batches remain Thermal48 and Wind24, with established microbatch8 unless a purely computational packing change is independently qualified and then applied to every live sibling before fitting.

Use fresh common initialization, Thermal seed0 and Wind seed42, and entirely fresh optimizer states. No T4111/W2401 weights, full-data migration, D-sep teacher, classic weights, or pilot weights initialize these fits. Those are read-only references.

Retain current P physical losses, query ratios, clipping, dataset-specific weight decay and TRAIN-only calibration algorithms. Fit normalization and calibration only on each declared selected TRAIN inventory; response addenda remain their explicitly listed original-TRAIN families. An auxiliary family outside primary150 must remain visible in total exposure. Pin the same resulting coefficients/scales across siblings; do not calibrate a different objective for each model.

Use the existing 2500-development-epoch schedule, including its declared warmup/hold/decay, resolved and saved before fitting. Do not reset the scheduler at reviews. An epoch visits every selected primary case exactly once; count partial final macro batches correctly.

### 8.2 Run matrix and mandatory horizon

| Dataset | Arms | Horizon per arm | Updates per arm | Primary visits per arm |
|---|---|---:|---:|---:|
| Thermal | Fresh P, fresh R1, fresh R2 | 2500 epochs | 10,000 | 375,000 |
| Wind | Fresh P, fresh R1, fresh R2 | 2500 epochs | 7,500 | 180,000 direction-row visits |

Primary total: **52,500 updates**. Assign new run identifiers through the existing allocator; proposed labels are not permission to overwrite old run IDs.

Start with a 100-epoch review. Continue healthy finite models through 500, 1000, 1500 and 2500 inside the cap. Save checkpoints, latest/best updates and monitoring plots every100 epochs. Full statistical evaluation uses every selected DEV case/row. Detailed spatial exports default to four representatives per dataset, not every checkpoint/case.

Do not stop a healthy candidate at100 because it has not yet beaten the mature full-data classic or achieved a 5% final gain. Early stops are for implementation/physical-law failure, divergence, prohibitive projected cost, or persistent severe damage to protected fields. Preserve and report the exact reason and completed horizon; never rename an interrupted fit as a mature negative result.

The primary matched comparison is literal e2500. Also report the predeclared saved100 field-only selection for each arm; use the same 25 eligible ages and same selector weights within a dataset. Report selected age explicitly. Do not choose different per-field best checkpoints, or use selected graphs with unlabeled terminal-field numbers.

### 8.3 One bounded pre-seal amendment

Codex may revise the proposed implementation once based on source inspection, numerical preflight or **TRAIN-only** pilot evidence. The allowance is at most1000 additional scientific updates total across disposable pilot models. It may address an implementation defect, score parameterization instability, or a cheaper equivalent placement within the specified two-view gather. It may not add a third view, new value branch, new loss/data selection, forced K, or a different dataset-specific backbone.

Record the hypothesis, affected code/config, parameter/work counts and all pilot outcomes. Do not initialize scientific runs from the pilot. Seal the final R1/R2 definitions before opening new DEV outcomes. Once sealed, no silent model update is permitted inside a scientific history; correctness repairs require explicit invalidation/branching and leave the failed evidence intact.

## 9. Physical evaluation and intuitive error diagnosis

### 9.1 Common physical metrics

Thermal: full native fluid u/v/p/omega/T on a **common externally defined** valid stencil/fluid mask; surface/material/q/peak on identical requested native supports across methods. Do not let model-dependent validity hide difficult cells. Preserve the distinction between the all-grid field board and sampled monitor auxiliaries. Include per-case means, p90, worst case, M strata, fluid-T bias and sampled/module peak errors. The M5 stratum and previously identified worst-T case are permanent challenge annotations, not additional fitting weights.

Wind: fixed matched native Q8192 TOTAL role-balanced receivers per DEV row, Ux/Uy/Uz and vector RMSE for all five roles. Vector RMSE has no division by three. Show near/downstream transverse means, row-p95 and layout-level paired differences, alongside Ux and absolute m/s errors. Relative L2 on weak transverse components needs the raw reference amplitude beside it. No double wind-coordinate rotation.

Use the new P's selected-TRAIN field scales only for a common dimensionless plotting/selection score where required. Physical headlines remain per-field native units; never sum temperature, pressure and vorticity errors into one purported physical quantity.

### 9.2 Fixed-weight fitting evidence

At e1000 and e2500 evaluate a small input-selected TRAIN panel with the same fixed weights, native queries and metrics used for DEV. Growing-weight training loss is not a fixed-weight generalization-gap estimator. Keep panel size and representativeness visible. This check decides whether a bad field is already badly fitted, not whether an unmeasured full-data fit would succeed.

Do not respond to an isolated train/DEV gap with another unplanned architecture or loss sweep. The primary deliverable is the completed controlled comparison.

### 9.3 Prospective development decisions

These thresholds are research decision rules proposed now, not physical standards or retrospective changes to the prior report.

- **Wind organizing direction:** at e2500, candidate priority geomean (near/downstream Uy/Uz) at most0.95 versus fresh P; every component/vector mean no worse than1.05 P and every protected row-p95 no worse than1.10 P. Report layout exceptions explicitly. To favor R2 over R1, require at least3% better priority score, or clearly better physical tails at comparable means and price.
- **Thermal organizing direction:** at least5% improvement in fluid-T or v mean versus fresh P, with all other primary/auxiliary role means within1.03 P and corresponding p90s within1.10 P. Every M-stratum mean and worst fluid-T/peak comparison stays within1.10 P. To favor R2 over R1, require at least3% improvement in a priority mean or a resolved repair of the M5/tail miss, without purchasing an equivalent loss in another priority field.
- **No synthetic victory by averaging:** a slightly better common selector does not override wrong response direction, severe local peak damage, or weak transverse fields.
- **Engineering target:** aim for complete Q8192 cost at most1.20 fresh P; cost1.20–1.30 remains a priced research direction, not an automatic cancellation of learning. Above1.30, do not recommend formal scale-up without an unusually useful, explicitly quantified physical tradeoff and a separate user decision. Q1024 and preparation-heavy use cases retain their own results.

Always publish raw effects and the accuracy–cost frontier, even when a binary gate fails. A 4.9% gain is not physically distinct from5.1%; the gate selects the next bounded action, not the truth of a universal scientific claim. The previous23% overhead failure remains unchanged in its original report.

Do not require identical preferred view count across datasets to claim shared implementation. A Wind R1 / Thermal R2 preference is acceptable only when obtained under the declared selection rules, and is not a common-model-across-datasets accuracy win.

## 10. Organizer evidence: distinguish weight patterns from useful gathering

Use the same selected checkpoint for a method's graph, interventions and selected physical fields. Retain literal e2500 as the primary matched accuracy endpoint.

For each of four input-selected representatives, export two rounds of typed native destinations/donors, view IDs, base measures, bounded scores, densities, actual normalized weights, and feature-block weighted-content norms. Keep direct final source-query reads and global/environment ancestry on the diagram. Do not display just an attention heatmap and call it a physical interaction map.

For R2, measure on fixed input-selected TRAIN and DEV scenes:

- reset each view's density to its legacy measure weights, one view at a time and jointly;
- collapse the two view densities to their average while retaining both value blocks and row mass;
- one deterministic equal-measure donor reassignment per route, with full two-round downstream rebuilding;
- exact restoration of all fields and traces.

Report physical field-error changes, not only output-change norms. Weights and source identities stay unchanged. A large reset effect establishes reliance; fresh R2 versus R1 establishes whether extra views help learning. They answer different questions.

Compute view similarity and weight concentration as descriptions, not losses. Distinct views with no advantage over R1 do not qualify. Identical views are not repaired with a diversity penalty.

Expose an explicit native-route trace object with source/destination IDs, measures, view/round/type, configuration/weight freshness binding and output-law metadata. Reuse existing ownership machinery. No new global manifest framework or duplicate approval daemon is required.

## 11. Bounded saved physical responses and downstream interface

Unlike the previous routing round, this phase should attempt one **small read-only physical-response replay** after checkpoint selection. This is not an inverse campaign.

Reuse only already authorized, correctly bound stored baseline/perturbed states. Separate original-TRAIN response-fitting families from exposed response-development/audit families. Start with the three complete fixed audit layouts0291/0294/0687 and their two signed heat changes; include the known failed0277 baseline only as unavailable unless its actual artifact is found. Give provenance recovery at most30 minutes; do not relabel an unrelated nominal H5 state as the missing baseline.

Evaluate P/R1/R2 on exactly the same state pairs, common query supports, masks, heat/source IDs and native scales. Report fluid/surface/material finite-response RMSE, signed spatial-mean changes, unchanged-own-heat module-peak changes, and exact software heat-null flow. Distinguish `max(T1)-max(T0)` from `max(T1-T0)`.

The archived opposite heat transfers supply one independent direction per layout. Individual learned K columns are not independently validated Green functions. A zero predicted response is a diagnostic comparator, not the physical truth. Numerical stopping indicators do not become certified physical error bounds.

If compatible existing geometry-change records can be reused within the same evaluation budget, report one input-selected pair as a secondary challenge; no broad geometry-response qualification follows. For Wind, do not invent turbine-displacement truth: software VJPs/stale-state tests remain separate from absent independent finite-response labels.

No response-audit outcome alters the selected checkpoint or training recipe. It can prevent an inverse-readiness recommendation or identify the next research issue. No optimizer, sampler, generated design, new physical solve or solver-ledger increment is authorized here. Ledger remains326/326.

## 12. Efficiency measurements and restrained sparsification

Measure P/R1/R2 and the available mature reference systems on the same device and native output workload, with the same precision and semantically matched receivers. Retain each classic's native context and internal computation. Use alternating/reversed model order, two warmups and seven measured trials at low/high M and Q1024/Q8192. Profiler must be off for reported latency. Source binding, geometry/context preparation, native curl/stencil work, output conversion, synchronization and CPU return belong in complete-call timing.

Report inference latency, complete-objective training-update time, cumulative fit time, total model residency, incremental allocated workspace and peak allocated/reserved training memory separately. Do not describe cached zero-reservation deltas as zero memory. Label external GPU contention and shared model residency; rerun only the small decisive timing panel under quieter conditions when feasible, not entire fits.

The optional sparsification check is secondary and must not block the primary report. After useful fresh learning and adequate time remain, test a single **99%-mass context cut** at selected weights on input-fixed scenes. For R2, retain the union of needed donors across both views; keep uncut weights without falsely describing a renormalized operator as mere omission. Report physical error as well as same-model distortion. Excluded/padded rows and all final-query reads remain separately counted.

A mask after dense value evaluation is only a logical omission study. Implement actual selected-value evaluation only if profiling and union counts show meaningful total-work headroom and it fits the resource reserve. Do not promise net speedup for sparse small M/E context work. No cut is deployed by default; no forced-small-K fitting sequence is authorized.

For fixed geometry/control-compatible queries, demonstrate correct prepared-object reuse against fresh reconstruction with stale-state rejection. Compare against P's own prepared reuse, not against repeated unnecessary P preparation. Approximation, execution savings and interface reuse remain separate claims.

## 13. Conditional second-seed confirmation

After all six primary endpoints and mandatory physical/cost evaluations are complete, one dataset may receive **one additional fresh P versus preferred-routing pair**, each2500 epochs, within the remaining cap. This is not a model search.

Eligibility: primary candidate meets its physical direction/robustness rules, costs no more than1.30 P at the declared Q8192 workload, and no decisive implementation failure is unresolved. Select Wind by default if both datasets qualify, because its current positive routing evidence is only one parent/seed. Select Thermal only when Wind does not qualify and Thermal does. Freeze that choice before any second-seed results.

Use a predeclared second initialization/sampling seed (Thermal1 or Wind43), matched within the pair; preserve TRAIN membership and the same normalization/calibration policy. Prefer retaining the exact primary TRAIN-fitted scale/coefficient values to isolate run variability, with independent common-path initialization and case/query draws. Report both seeds, their paired effects, and any reversal. Two seeds remain limited evidence; do not select a favorable seed or average away a failure.

Extra update ceiling:20,000, sufficient for the Thermal pair; Wind uses15,000. No extra C, third view, classic refit or second confirmation dataset is authorized in this phase.

## 14. Resources, process ownership and stopping

Proposed ceiling: **10 aggregate GPU-associated hours and12 elapsed hours**, with the final two elapsed hours reserved for evaluation/report delivery. Use only the currently authorized physical GPUs1 and2 after verifying UUIDs and unrelated process ownership. These are ceilings, not runtime promises or a requirement to exhaust the allocation.

Scientific update accounting:

- primary fresh matrix:52,500;
- pre-seal TRAIN-only pilots:at most1000;
- conditional second-seed pair:at most20,000;
- total scientific ceiling:73,500;
- disposable numerical/packing/update checks:at most192, separately charged.

Every failed start, retry, validation, physical replay, profiler run and owned GPU process lifetime counts. Charge each owned GPU-process lifetime once and sum those lifetimes conservatively across devices and concurrent owned processes; do not add nested timing scopes to that same charge. Record measured fit scopes, conservative associated lifetimes, elapsed time and any physical-device occupancy union separately. A conservative process-hour sum is not active GPU utilization; simultaneous use of two GPUs is not one GPU-hour.

Before fitting, forecast the six primary runs from a short measured complete epoch, including scheduled objectives and validation. If the forecast cannot fit, prioritize complete matched P/R1/R2 endpoints on both datasets over optional second-seed and sparsification work. Reduce optional work first. Do not silently reduce primary case coverage or query counts.

When an external process occupies a GPU, continue useful work on the other authorized device or perform CPU implementation/reference binding. At most brief bounded availability checks; no hours of idle waiting. Do not kill, pause, move or modify another user's process. A changed microbatch/precision/effective batch is not a silent contention workaround.

Use ordinary maintained checkpoint/resume mechanisms and isolated worktrees where needed. No hot editing of an active formal run, no relabelling a killed process as a clean pause, no automatic restart of old formal jobs.

## 15. Required reports and visualizations

Deliver a main report readable without its appendix. Open with three short sections: **Predictor, Organizer, Inverse system**. In each, state what changed, one or two key gains, the most important miss, what is and is not established, and one next decision. Then give A/B/C:

- A: value beyond an equally trained simpler model, distinguishing R1/P and R2/R1;
- B: exported organization agrees with actual information flow and successful restoration;
- C: physical response transfer, separating held-from-this-fit evidence, historical exposure and missing labels.

A compact reference table must make the two main comparison classes unmistakable: fresh segmented controls and mature full-data systems. Do not put a single “FAIL strong reference” banner over a useful controlled development result, nor “Goal complete” over an unqualified deployment claim.

Use at most six figure families:

1. **Learning and comparison identity:** P/R1/R2 physical priority curves versus epochs/updates and measured training time; clear fresh/segmented labels; mature reference values measured on the same DEV panel shown as contextual horizontal markers, not equal-age rivals.
2. **Thermal fields and tails:** reference/P/R1/R2 native T and v maps with signed residuals, raw RMSE and relative L2; representative M3/M5/M7/M10 cases, not TRAIN-only success examples. Identify preselected known-M5 challenge maps as challenges. Include a material/peak or boundary-flux view for the local miss.
3. **Wind wakes:** native hub plane plus a transverse cut for input-selected low/high-M DEV layouts, all velocity components or the decisive transverse components with Ux context. Show actual wake shape and amplitude, not only an attention plot or scatter cloud.
4. **Actual organization:** same native destination and physical donors in R1/R2, both views, weighted content, direct downstream source reads; one full-rebuild removal/collapse/restoration physical field change. Head names must not imply physical meanings not established by evidence.
5. **Accuracy and cost together:** per-field/role error versus complete ms, with separately styled matched controls and mature references; train/update/memory scopes clearly separated.
6. **Saved response and interface:** measured versus predicted delta-field and signed peak/mean responses, plus a compact software/physical-evidence status strip. No invented inverse trail when no generator ran.

Every contour caption states dataset/partition, physical case/direction, model/checkpoint, query count/mask, units, reference provenance, display clipping and metric denominator. All numerical metrics use unclipped arrays. No interpolation should masquerade as native solution detail. Use filled native cells/triangles when justified by stored topology; avoid smoothing away physical misses.

Keep full per-case/stratum tables, query/moment/protection receipts, configuration details and failures in a linked appendix/index. Retain only necessary figure masters/companions, inspect rendered figures, and verify internal links. Do not delete scientific evidence or clean unrelated directories.

## 16. What can be called ready at the end

Produce separate conclusions:

1. **Shared implementation ready:** both datasets genuinely use the same one-/two-view gather and TrainingEngine, with strict dataset-owned contracts.
2. **Development direction supported:** matched fresh learning/robustness and practical cost support R1 or R2. A second-seed pass strengthens, but does not universalize, this conclusion.
3. **Manual full-data research recipe ready:** the preferred fresh candidate has useful absolute physical fields, passes the agreed development safeguards, and has a credible measured cost forecast. Prepare P and preferred-candidate paired recipes with a visible method/config diff. No optimizer/run directory/actual training launch occurs during preparation.
4. **Classic replacement established:** only after the appropriate completed full-data system comparison; not inferred from this phase's segmented curves.
5. **Inverse validity established:** not inferred from differentiation or a handful of stored responses. State exact remaining physical/excitation/geometry gaps.

When the candidate fails, retain the competent P and any useful R1 result rather than discarding the shared core. When R1 wins and R2 fails, keep R1; do not enlarge the organizer. When the two views merely duplicate each other, report that the added organizing degree of freedom was not used. A common framework with modest dataset-specific configuration remains valuable even without one winner everywhere.

Formal5000 preparation must not continue a2500 cosine schedule at a nearly zero terminal LR and call it a fresh formal recipe. Explicitly declare fresh full-TRAIN initialization, full-TRAIN fitting of statistics, the reviewed5000 horizon/hold/decay, optimizer/update/query budget and checkpoint selection. Preserve the actual final development objective; avoid the earlier silent locality/coefficient mismatch between development and formal studies.

## 17. Closeout and durable deliverables

Commit only maintained source, focused tests, versioned configurations and documentation. Keep checkpoints, solver artifacts, arrays, large figures and one-time renderers under the established ignored artifact boundary. Audit all outgoing objects, not just the working tree.

Deliver:

- implemented and tested one-/two-view core and any qualified same-function synchronization repair;
- six completed fresh run histories/endpoints, or exact bounded stop reasons, with all primary counters;
- one fresh comparison manifest and one native-reference evaluation manifest;
- one compact main report, full numerical appendix/index and inspected visualizations;
- bounded saved physical-response results or explicitly documented missing bindings;
- separate readiness decisions and conditional manual recipes, never an automatic launch;
- preserved historical observations, including absent bindings; unchanged mature Wind migration, all classics and completed formal runs;
- WindTEST/unexposed Thermal protection and unchanged solver326/326;
- resource/process closeout, focused regression results, outgoing-object audit and verified remote/local tips.

Do not let inventory bookkeeping replace the model experiment. Do not let a small engineering-target miss turn every subsequent useful learning test into a forbidden action. The scientific stopping point is a clear comparison of one-view and two-view information gathering, with trustworthy benchmarks and prices—not a compulsory positive result.

## 18. Source and methodological notes

The source-derived facts in this plan come from the native-routing report at0004ab6, the preceding reference-anchored report atb851acd, and the named formal/recovery reports. The two-view mechanism, new resource allocation, and prospective thresholds are proposals, not findings in those sources.

The use of different feature subspaces for different attention distributions is standard multiview/multi-head reasoning; it is not claimed as a novel primitive here. See Vaswani et al., *Attention Is All You Need*, section3.2.2, arXiv:1706.03762. Our proposed implementation retains destination-conditioned nonlinear pair values and does not replace the core with a Transformer.

The emphasis on initialization/data variability and matched comparisons is consistent with Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, MLSys2021. This motivates transparent paired seeds, not a large sweep.

For execution diagnostics, use the official PyTorch Performance Tuning Guide's CPU–GPU synchronization guidance and the installed profiler documentation. A source-level synchronization opportunity is a hypothesis until profiled and validated on the native workload.
