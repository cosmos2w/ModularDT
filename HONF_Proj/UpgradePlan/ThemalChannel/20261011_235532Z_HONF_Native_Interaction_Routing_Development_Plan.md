# HONF next development: native interaction routing without another backbone reset

**Status:** proposed, executable Goal-mode work plan; no experiments in this document have run.  
**Reviewed repository:** `cosmos2w/ModularDT`, branch `agent/honf-core-next`, commit `b851acd4402c6172f69c727937b820f6c0631e01`.  
**Primary decision:** test whether inexpensive, explicit donor selection inside the existing nonlinear context aggregations improves joint physical fitting without paying for a second collective network.  
**Scientific scope:** two dataset-specific matched development pairs, using one shared implementation, existing segmented TRAIN populations and all existing physical objectives. No full-TRAIN campaign, new solver call or inverse-design campaign.

## 0. What the preceding round completed

Read the following before editing:

- `docs/reports/_bk/20261011_224701Z_HONF_Reference_Anchored_Consolidation_20261011.md`;
- its companion `docs/reports/_bk/20261011_224701Z_HONF_Reference_Anchored_Evidence_Index_20261011.md`;
- `docs/guides/HONF_Reference_Anchored_Development.md`;
- the main and appendix of `HONF_Interaction_Preserving_Recovery_20261010`;
- the prior reference-anchored work plan, the current repository `AGENTS.md`, and applicable dataset development protocols.

The preceding task is complete as a **consolidation/diagnosis experiment**, not as a successful new accuracy or acceleration result. Do not repeat it wholesale.

| Previous requirement | Supported completion | Remaining boundary |
|---|---|---|
| Retain the mature Wind function | Strict 52-tensor migration, self-contained reload, exact TRAIN transforms/profile, and bitwise original/migrated fullVALID90 outputs | It inherits Run2202 TRAIN420/e5000 exposure. Its retained optimizer moments are archival, not an automatically valid P resume state. |
| Separate age from architectural differences | Run2202 and W2501 compared at e1000/18,000 updates on the same Q8192 panel | Objectives and histories still differ. Near-turbine vector means are essentially tied; transverse Uy is worse in W2501. No prediction of its untrained e5000 result follows. |
| Implement one same-function execution experiment | Optional affine factoring, native complete-cost measurements and CPU/GPU tests | GPU gradient/update and Thermal output criteria fail; gains are small/mixed. Keep legacy execution. |
| Diagnose learning on actual optimizer updates | Fixed-weight TRAIN panels; source-moment counterfactuals; a matched 400-update Thermal pair with an exact ordinary-joint historical replay | Flow-only improves final v relative to joint but damages flux/tails. This does not identify a uniquely harmful term or qualify a joint replacement. |
| Run the conditional Wind pair only when justified | No pair: measured base-message auxiliary effect was too small to support that investment at the inspected state | This does not prove the auxiliary never matters at another age. Do not restore it speculatively. |
| Reassess organizer value and optional reuse | Existing Thermal P-H benefit and cost were separated; no unsupported promotion | The optional frozen reuse probe was not run. No reusable approximation or sparse speedup has been demonstrated. |

Two small reporting corrections belong in the new handoff. The preceding report's Figure 4 calls **0291 a TRAIN response case**. Prior cohort records identify 0291 as an exposed development/audit pool, not one of the four response-fitting families. Trace the existing source membership and correct the caption/index consistently; do not admit it to TRAIN. Also distinguish **unchanged observations** from **all artifacts present**: 67 of the historical 953 bindings were already absent. Preserve the initial absent set, and do not claim a clean 953-present pass. Missing response receipts remain missing until their actual provenance is recovered; raw frame-byte equality alone is not the missing receipt.

## 1. The development decision

### 1.1 Keep what works

Retain the mature Wind migration and all original reference checkpoints. Retain the direct P readout, its individual physical source columns/messages, the two nonlinear source/environment rounds, and the shared TrainingEngine. Retain the self-contained, jointly trained Thermal model: **no frozen D-sep, no historical flow teacher at inference, and no second complete thermal predictor**.

Keep `factorized_affine_execution=False` for this campaign. Fixing that executor's last few floating-point differences is not this round's research objective, particularly given its small complete-cost changes. A failed strict parity criterion is not proof that the physical model is catastrophically inaccurate; neither should a small physical difference be relabelled a parity pass.

Do not add PCGrad/FAMO, loss-weight searches, more environment records, larger fine networks, a new curl law, local PDE solves, or a new field-correction head. The preceding evidence does not isolate any of those as the needed fix.

### 1.2 One hypothesis, with one real capacity control

**Hypothesis:** destination- and donor-conditioned weighting of the existing nonlinear context messages may organize information more effectively than fixed measure-only aggregation. Implement that weighting where the messages are already gathered. Do not append a separate latent hyperedge predictor and then retain every original computation beside it.

The primary candidate is **P-R: native routed aggregation**. Its paired control is **P-C: donor-blind aggregate modulation**, matched in added trainable parameter count. Existing P supplies an archival, same-age direct control after its replay contract is verified.

The hypothesis is an inductive-bias/learning claim, not an assertion that P has no interactions or lacks information. P's message values already depend nonlinearly on destination, donor and signed geometry. A sufficiently capable value network can absorb weighting. Explicit routing is worth testing because it exposes selection separately at low cost; it is not guaranteed to improve the function class or accuracy.

This is a minimal **directed, destination-anchored organizer**. A weighted star can be represented as a hyperedge gathering a donor set into a destination state, but it is also standard directed message passing/attention. Do not claim novel higher-order hypergraph expressivity, physical causal edges, adaptive final-query packet counts or a new hypergraph theorem. The immediate question is whether this explicit organization earns useful physical accuracy at approximately direct-P cost.

### 1.3 Why this differs from previous unsuccessful paths

| Previous mechanism | This experiment |
|---|---|
| J replaces much source-conditioned computation with a collective representation | Preserve every original nonlinear value path and source-resolved final read. |
| P-H adds a collective block and a receiver-to-edge read alongside direct work | Replace only the weights of existing native context reductions; no extra collective state bank or receiver-to-regional-edge read. |
| Adaptive detail adds base/fine/router work over final source-query pairs | No base/fine duplication or final-query router. Context routes are prepared before final queries. |
| Fixed latent K or forced merger/sparsity objectives | No latent K, entropy target, merger rule, gate schedule or mandatory support reduction. |
| Flow-only diagnostic sacrifices thermal objectives | All Thermal physical and auxiliary objectives remain active with their original weights. |
| A routing-versus-plain model difference can be extra-capacity benefit | Compare P-R with P-C as well as the independently continued historical P path. |

The primary target is **learning value with low overhead**, not a speedup caused by sparsity. Query-local computation reuse remains a separate future mechanism; it is not silently substituted for this experiment.

## 2. Source findings and implementation map

All paths below are relative to `HONF_Proj/` and were inspected directly or identified in the reviewed source/documentation. Inspect current functions before making the narrow change.

| File / entry point | Current behavior and required use |
|---|---|
| `src/honf_forward_core/interface_fields/interaction_core.py` — `InteractionContextCore.prepare_context` | Two rounds form MM/ME/EM nonlinear messages before measure-weighted sums. MM removes self donors and renormalizes. ME/EM use normalized environmental/source measure. These three reductions are the insertion point. Keep the existing simultaneous source/environment update order. |
| Same file — `_geometry`, `_normalized_measure`, `NonlinearFieldReadout` | Preserve existing relative features, measures, individual source-query messages and nonlinear field head. Do not change the final source reduction in this experiment. |
| `src/honf_forward_core/interface_fields/interaction_preserving_joint.py` | P retains the inherited direct context/read path; P-G/P-H add separate collective processing. New R/C use the P path with no collective block. Preserve historical modes/loaders. |
| `Case_ThermalChannel/src/channelthermal/joint_regional.py` | Native Thermal extraction, independent flow and affine-temperature source reads, source identity, normalization, native-curl wrapper and prepared-state guards. Preserve those laws. |
| `Case_ThermalChannel/src/channelthermal/training/joint_task.py` | Joint objective, native role denominators, response addendum and training-side operator. Reuse all terms and sampling; do not inherit the diagnostic flow-only overrides. |
| `Case_ThermalChannel/src/channelthermal/flow_curl.py` | Discrete native cell-centred curl; not a general off-grid derivative. Keep the mask and neighbor-query accounting unchanged. |
| `Case_WindFarm/src/windfarm/interaction_preserving.py`, `joint_regional.py`, `training/joint_task.py` | E8 P scene, TRAIN profile/normalization, component-role loss, native coordinates and nonlinear velocity. Use the P provider's corrected train/validation scale convention. |
| `Case_WindFarm/src/windfarm/migration.py` | Strict Run2202 compatibility artifact. Preserve as an inference reference; do not weaken it into a generic training initializer. |
| `src/honf_runtime/unified_training.py` | Existing fitting, named optimizer state, sampling and loss reduction. Use the same training engine. |
| `tools/diagnostics/reference_anchored_diagnose.py` | Useful native comparisons and zero-step audits. Its diagnostic branch is intentionally limited to identical model structure and at most 400 updates; do not relax it to smuggle in a new architecture. |
| `tools/joint_regional_train.py` and versioned configuration dispatch | Add explicit new-family preparation/branch support through existing infrastructure. Keep old recovery/formal launch caps intact. |

Prefer a small shared `native_interaction_routing.py` plus a narrow reduction hook in `interaction_core.py`. A new helper/family name is proposed, not an existing API. Default/no-policy calls must execute the original reduction path. Do not copy the whole joint trainer, create another registry framework, or add a new monitoring/approval daemon.

## 3. Mathematical contract

### 3.1 Explicit directions, not ambiguous route names

Let `r` be a **native context destination**, and `j` a donor. It is not the final arbitrary field query `q`.

- MM: destination physical source receives from other physical sources;
- ME in the existing code: destination physical source receives from environment records;
- EM in the existing code: destination environment record receives from physical sources.

Export both the legacy route name and the explicit donor/destination types. Do not rely on the letter order alone.

For round `l` and type `tau`, retain the existing nonlinear message:

\[
 m^{(l,\tau)}_{rj}
 =\Phi^{(l,\tau)}_\theta
   \big[z_r^{(l)},z_j^{(l)},\gamma(\Delta_{rj}/L)\big].
\]

Write the actual old donor weights as `w0[r,j]`, including physical measure, presence, self-exclusion and its existing normalization. The old reduction is

\[
 a^{(l,\tau)}_{r,0}=\sum_j w^0_{rj}m^{(l,\tau)}_{rj}.
\]

The existing `source_updates`, `environment_updates`, environmental pooled context, and final `global_encoder` remain otherwise unchanged. The pooled environmental/global paths are disclosed bypasses, not newly learned incidence.

### 3.2 P-R: learned density over the existing measure

Use **three small score MLPs**, one per MM/ME/EM type, shared across the two rounds. Each receives the same destination/donor states and signed relative geometry used by its value message, but has hidden width **32** and one scalar output. Its final linear layer has no bias and is initialized to zero. Hidden layers use a private initialization RNG so existing P parameters and sampling RNG are not disturbed.

The initial version is fixed before training:

\[
 s_{rj}^{(l,\tau)}
 =2\tanh\left(f^\tau_\eta
    [z_r^{(l)},z_j^{(l)},\gamma(\Delta_{rj}/L)]/2\right).
\]

The bound is a declared numerical/modeling choice, not a physical law or fitted locality scale. Do not tune it on DEV or add temperature annealing. The joint nonlinear scorer must genuinely condition on both endpoints, rather than reduce to a donor-only ordering plus a row constant that cancels in normalization.

Let `S[r] = sum_j w0[r,j]`. Preserve the old floating-point row mass rather than silently assuming it is exactly one:

\[
 w_{rj}
 = S_r\frac{w^0_{rj}\exp(s_{rj})}
               {\sum_k w^0_{rk}\exp(s_{rk})},
 \qquad a_r=\sum_j w_{rj}m_{rj}.
\]

Zero-mass rows return zero. In real arithmetic, ordinary nonempty normalized rows have `S=1`. The explicit `S` only preserves the original numerical normalization contract.

For an exact zero-score start with a live score gradient, a stable implementation is

\[
 e_j=\operatorname{expm1}(s_j),\quad
 \bar e=\frac{\sum_jw^0_je_j}{S},\quad
 \rho_j=1+\frac{e_j-\bar e}{1+\bar e},\quad
 w_j=w^0_j\rho_j.
\]

At zero scores, `e=0`, `rho=1`, and the original weights and original weighted reduction can be retained exactly. Handle `S=0` with a finite safe denominator and explicit zero masking. The score bound keeps the nonempty-row denominator away from zero. Verify the implementation against a high-precision exponential normalization reference.

**Do not** use a data-dependent `if scores == 0: return old_output` shortcut that removes the new parameters from autograd. Do not add physical measure twice. Do not normalize heat powers as donor masses. Do not overwrite the prepared source measure used by the final readout.

The score must be available **before expensive value evaluation**. Reuse already constructed pair features where practical, but do not define the score from the expensive message value itself. This leaves a legal path to future selected-value execution; it does not itself implement or establish that speedup.

### 3.3 P-C: added capacity without donor-selective incidence

The control retains `w0` and adds a small, destination-only vector modulation to each already reduced aggregate:

\[
 a^{(l,\tau)}_{r,C}
 =a^{(l,\tau)}_{r,0}\odot
   \left(1+\tanh g^\tau_\xi[z_r^{(l)},c]\right).
\]

Use three MLPs shared across rounds. Each produces `message` components, with its final weights and bias initialized to zero. Every added parameter must be active/trainable after the expected zero-output initialization transient. This control must not receive individual donor states, donor-specific geometry, source IDs, or a hidden attention reduction.

Choose its hidden width **once from integer parameter counts**, not from field errors, to match the added R parameters within 5%. With the reviewed H128/message128 settings, the nominal counts are:

- Thermal P-R: `3 * (274*32 + 32 + 32) = 26,496` added parameters; P-C width32: `26,400`;
- Wind P-R: `3 * (283*32 + 32 + 32) = 27,360`; P-C width34: `27,720`.

Recompute actual counts from the instantiated models. Width selection is a parameter-accounting rule, not a tuning portfolio. Record parameter and measured cost differences: parameter matching is not perfect compute matching.

Both R and C recover the same P function at attachment. Both update all inherited parameters throughout scientific fitting. A gain over P alone can be generic extra capacity; a gain over C is stronger evidence for donor-selective organization. Neither comparison proves superiority over every alternative attention or geometric prior.

### 3.4 Preserve the output laws

The final source-query readers are unchanged:

\[
 \widehat u(q)=\Psi_\theta\left(q,g,
   \sum_i\bar\mu_i\,\phi_\theta(z_i,g,q-x_i)\right)
\]

for a nonlinear field block, and

\[
 \widehat T(q,h)=\sum_i K_{\theta,i}(q;\mathcal X)h_i
\]

for the existing zero-offset Thermal response block. The new routing changes learned contextual states, not these laws.

Thermal heat must remain absent from the context, route scores, flow path and coefficient preparation. Wind remains nonlinear with the exact native profile/normalizer; no affine turbine-displacement/yaw/thrust capability is added. All source identities and all final physical source reads remain present.

## 4. What makes the organizer operational

Each native destination owns a typed donor gather. Its trace has a persistent native destination identity and ordered donor identities, round, measures, masks, scores, density `rho`, actual mass `w`, and actual reduced value. No arbitrary latent label substitutes for a physical source ID.

The prepared trace can be shown as a directed weighted star/hyperedge:

`typed donor records -> existing nonlinear messages -> destination-anchored gather -> updated native state`.

The final field-query trace must show its separate individual-source reads and then the relevant upstream two-round gathers. Do not draw final `q -> regional latent edge` links that this model does not execute. Query-specific source messages differ even though this round's learned routing is at native context destinations, not at final queries.

Expose enough provenance to replay a declared route intervention and reject stale state. Do not serialize whole autograd graphs, store all training message rectangles, or turn logging on by default in benchmark calls. Scores and values must stay live during training; only exported diagnostic copies are detached.

The source/environment record counts determine the available gathers. There is no global six- or twelve-slot bottleneck. Dense positive learned weights mean **adaptive weighting**, not sparse support. An entropy-effective donor count is not a count of eliminated reads.

## 5. Preparation, compatibility and initial qualification

### 5.1 Reuse verified evidence; keep preflight small

Confirm current branch, applicable instructions, source hashes, the existing protection receipts and source checkpoints. Use the corrected metadata panels from the previous round; do not rebuild an extensive reference inventory or re-evaluate all mature checkpoints. Any pilot fitting must remain inside the exact parent's primary TRAIN membership. A broader fixed-weight probe outside that subset is an exposure-labelled audit, not permission to expand its training set.

Use large output/cache locations under the supported `/data` arrangement. Verify current free disk space, available authorized GPUs and file-descriptor limits. GPU0 and unrelated processes remain untouched. Do not assume yesterday's free device or FD setting is still current.

Repair the 0291 label and retain the absent-binding condition in an additive correction. These are documentation/provenance tasks, not grounds for another full diagnostic campaign.

### 5.2 Baseline and candidate identities

Implement explicit new R/C family/configuration identities. Do not reinterpret a P, P-G, P-H or migration checkpoint under another model law. Legacy constructors/defaults/loaders must remain strict and unchanged unless an optional new field is deliberately versioned without changing historical meaning.

Do not bypass the current `diagnostic_age_branch` restrictions. Add a narrow **architecture-child initialization** route using the existing TrainingEngine: exact old state/moments plus an explicitly enumerated new parameter set. It is not an exact resume of a different architecture.

The branch receipt must contain parent checksum, parent cohort/normalizer/calibration, inherited age and updates, old/new parameter inventories, new-module initialization, named optimizer mapping, schedule/sampler age, and the new family/configuration identity. Every inherited tensor and every retained named moment must match. Initialize new parameters with fresh moments; archive no fake earlier Adam age for them.

Keep original groups/betas/epsilon/weight decay for inherited parameters. New parameters use the same contemporaneous learning rate and declared decay as the inherited group. Use the ordinary complete-model clipping rule. Additional router gradients can change the clipping factor even at a zero-output start; that is a real architecture-training difference, not an optimizer parity bug. Report it without adding a clipping-policy search.

### 5.3 Focused tests before scientific fitting

Required tests include:

1. Zero-score/zero-gain output and physical-input VJP recovery of P on native low/high-M examples; baseline/off-policy historical calls remain unchanged.
2. Stable routing normalization, finite empty rows, M1 MM self-exclusion, absent/padded records, nonuniform measures, and no accidental double weighting.
3. Whole-rebuild source/environment permutation equivariance, using IDs only for provenance; quadrature refinement checks for duplicate **environment** atoms with split measure. Do not claim invariance to duplicating a physical turbine/obstacle.
4. An explicit score test where changing the destination can reverse donor ordering; actual route-weight changes, not just different row offsets.
5. Live router final-projection gradient at initialization, and upstream score-network gradients after a disposable update on nondegenerate native data. Zero upstream gradients at the first zero-output boundary are expected.
6. Complete native objective finite forward/backward/AdamW step through all existing Thermal terms and Wind component-role terms. No omitted auxiliary denominator or silently flow-only configuration.
7. Self-contained reload and exact configuration/normalizer/source-ID round trip; changed geometry, weights, policy or model owner rejects prepared state.
8. Thermal heat-null flow/routing, affine heat increments and source-column order; Wind rejection of exact affine increments; native-curl and proxy definitions unchanged.

Test the new normalization in FP64 for algebra and in the authoritative FP32 runtime for native behavior. Preserve existing tolerances for compatibility claims. Do not widen them to admit a failed comparison. Nonzero learned R/C outputs after fitting are intentionally different functions; they are evaluated against physical targets, not required to equal P.

Use existing core tests and a small new routing test module. No universal broad-suite pass may be claimed unless actually run.

## 6. The concrete learning experiment

### 6.1 Parents and scientific populations

Use the intact **segmented P checkpoints at e1000**, not the mature full-TRAIN Wind migration and not the flow-only diagnostic child.

| Dataset | Parent | Primary population / native recipe | New comparison |
|---|---|---|---|
| Thermal | T4111/P e1000, native-curl joint model | fixed25_v1 TRAIN150 / exposed DEV22; H128/message128, E192, Q1024 plus the same native surface/material/response/operator work, effective48/micro8, seed0 | P-R and P-C, both jointly trained on all original terms |
| Wind | W2401/P e1000 | fixed24 TRAIN72 rows / 24 layouts and DEV24 rows / eight layouts; H128/message128, E8, Q4096 total across the five roles, effective24/micro8, seed42 | P-R and P-C, nonlinear velocity and corrected component-role objective |

The reviewed Thermal e1000 parent is recorded in `Case_ThermalChannel/configs/reference_anchored_thermal_flow_pair_20261011.yaml`; verify the actual file/hash rather than copying an unverified path. Recover the Wind parent from its own saved run receipt. Keep normalization and calibration from the exact parent without refitting on DEV.

The Wind subset omits M18. Its directions are correlated by layout. Thermal DEV22 is an exposed subset of the source H5 partition named `test`. Report actual exposed DEV reads and distinguish them from unexposed TEST access; do not use the misleading blanket statement that no source-`test` targets were read.

### 6.2 Same start, meaningful maturity

Both children attach at the same e1000 P function, inherit the same old tensors/moments, and use identical case/query streams and all existing loss coefficients. New R/C networks are initialized privately, with zero final projections. All old and new parameters are trainable from the first scientific update.

Run each pair through **1,500 new development epochs**, ending at absolute age **e2500**:

| Pair | New epochs per arm | New updates per arm | New visits per arm | Primary label queries per arm |
|---|---:|---:|---:|---:|
| Thermal | 1,500 | 6,000 | 225,000 | 230,400,000 at Q1024 |
| Wind | 1,500 | 4,500 | 108,000 | 442,368,000 at Q4096 |

Primary total: **21,000 new optimizer updates** across four arms. Additional native Thermal neighbor/stencil/interface/material/response/operator queries are not included in those primary-label counts; record them separately.

Use the parent's absolute hold/decay schedule toward e2500, its sampling namespace and RNG conventions. Do not restart warmup, silently reset optimizer age, refit normalization, increase Q, enlarge the fine hidden width or change the environment inventory. Checkpoints/physical monitoring remain every 100 epochs. Review after +100, +500, +1000 and +1500 new epochs.

Early review checks health, gradient reach, field trajectories and resource forecasts. It is not a requirement to beat e5000 classics at e1100. A healthy learning curve with finite physical fields normally receives its declared mature horizon. Stop for resource/safety failure, irreproducible semantics, invalid targets, or sustained severe divergence—not merely an unfavorable first 100-epoch comparison. A quality stop must report its observed age rather than a mature architecture verdict.

### 6.3 Reuse the historical P continuation responsibly

T4111/P and W2401/P already have saved e1000-to-e2500 trajectories. They can supply a same-age direct comparison without a fifth/sixth primary run, but only after a bounded old-path replay verifies parent loading, sampling/denominators, schedule and original computation under the new source.

The previous Thermal ordinary-joint child already matched T4111 e1100 bitwise. Reuse that evidence and perform a small default-path regression after editing; do not spend 100 repeated epochs just to reconstruct it again. For Wind verify its corresponding old-path checkpoint/update boundary with its exact moments and provider.

Where archival sampling hashes or output arrays are unavailable, state what is verified and what is not. Do not invent missing digests. If full trajectory equivalence cannot be established, retain P as a contextual same-age reference; **R versus C remains the primary live matched experiment**. Do not silently substitute mature Run2202 for a segmented control or launch another large P campaign to make a table look complete.

### 6.4 Bounded self-refinement before sealing

Codex may use up to **2,000 total exploratory training updates** across both datasets before the primary recipes are sealed. These are charged, labelled pilots, not free optimization. Use the existing TRAIN-only panels and disjoint TRAIN queries; do not select changes using 0291/DEV response outcomes.

At most **one evidence-led scientific amendment** to the native-routing mechanism is allowed in these pilots. It may refine the small scorer's parameterization or placement within the existing MM/ME/EM rounds; it may not add a final-query router, P-H residual, output head, second predictor, loss-weight schedule, external teacher, or changed data population. Explain the observed failure, changed computation, cost and paired evidence. Do not run a width/temperature/entropy sweep.

Discard pilot-trained weights. Final primary R/C fits always begin from the exact e1000 P parent with newly sealed modules. Recompute the capacity-control count if the one amendment changes parameters. Fix ordinary implementation defects as needed, preserving attempts and invalidating affected evidence explicitly.

After primary sealing, do not hot-edit the scientific computation. An implementation defect may require a clearly identified restart; its discarded work still consumes the cap. A disappointing valid endpoint is not permission for a second architecture portfolio.

### 6.5 Conditional fresh-start check, not a formal run

Warm-start success does not prove that a fresh full-TRAIN recipe works. If a pair establishes a useful routing direction and the budget permits, choose **one** dataset by the predeclared rule: Thermal first if it meets its criteria; otherwise Wind if it meets its criteria. Run a fresh segmented R/C pair for at most **500 epochs per arm**, with a full intended e2500 schedule rather than a compressed e500 decay. This spends at most 4,000 extra updates for Thermal or 3,000 for Wind.

This is a birth/learning-health check with fresh common P initialization, not a mature comparison. Preserve the selected TRAIN transforms and dataset-owned calibration procedure; all weights are trainable and no full-data weights enter. Disclose that only the chosen dataset has this additional check.

No full-TRAIN5000 run is authorized. Manual configurations may be prepared and dry-run tested for a genuinely supported recipe, labelled with the exact warm-start or fresh-start evidence available. Do not call both datasets formally ready when one lacks its fresh-start check, and do not fill a missing parent/calibration with a historical fallback.

## 7. Physical evaluation and decisions

### 7.1 Fixed metrics, not a new favorable selector

Use the exact inherited field selector within each dataset at the existing 100-epoch cadence. Report both selected and absolute-e2500 endpoints; same-age endpoints are the primary R/C comparison. Do not replace a poorly ranked endpoint with a different channel-specific best state.

Thermal: report u/v/p/omega, fluid/surface/material T, normal-flux proxy, sampled per-module peak and tails, plus low/high-M strata. The full-native fluid panel uses Q8192 and the existing valid mask; the Q1024 monitor remains a separate statistic. Keep the selected physical units as benchmark-native, not asserted SI.

Wind: all five role-vector means and all 15 component means, with row-p95 and worst rows, and module-count/direction breakdowns. Keep the role-balanced panel, vector-RMSE definition and corrected TRAIN component scales fixed. Layouts, not direction rows or points, are the resampling unit for any descriptive uncertainty interval. Exposed DEV is not independent TEST.

Do not average heterogeneous Thermal units into a physical headline. Record fixed-weight TRAIN and DEV metrics from the same checkpoint, separately from moving-weight training logs.

### 7.2 Prospective direction criteria

These are **development decision thresholds**, not physical standards, statistical guarantees or required wins over every mature classic.

**Thermal routing direction:** versus both same-age P-C and the qualified historical P comparison, improve either v mean or fluid-T mean by at least 5%, keep the other within +3%, keep u/p/omega and other temperature/flux/peak means within +5%, and priority p90 values within +10%. Inspect every per-M stratum and explain any material local miss. If P comparison cannot be qualified as matched, mark the result only as an R-versus-C finding.

**Wind routing direction:** near/downstream transverse components Uy and Uz are primary. The geometric mean of their four mean-RMSE ratios should be at most 0.95 versus each qualified comparator, with no individual one above 1.05. Near/downstream vector means should remain within +3%; other role/component means within +5%, and priority row-p95 values within +10%. Show all components so a good Ux cannot conceal a transverse miss.

If P-C produces the useful physical/cost improvement while P-R does not, retain that predictor result and reject the routing-benefit claim. Do not discard a useful simpler model merely because it lacks the desired organizer, or relabel its gain as evidence for routing.

A tiny change near numerical/replay variation is unresolved, not a win. One parent/seed per dataset establishes provisional conditional evidence only. Do not retrofit thresholds after seeing results. Missing a guard can still be a useful, specifically described tradeoff; it is not a global promotion.

### 7.3 Cost guard

The nominal complete Q8192 inference overhead target is **at most 20% relative to P**, with training-boundary/epoch overhead targeted below 30%. Record complete Q1024 and Q8192 scopes, memory and cold setup separately. At equal parameter count C may be cheaper because its modulator is destination-only; that difference remains visible.

A measured overhead above 35% at the primary inference workload makes R noncompetitive for this round's intended low-cost direction unless it yields a clearly larger, explicitly qualified accuracy gain. Do not hide that decision behind the fact that it is still faster than an old dense model on another device. Such a tradeoff remains research-only, not automatic promotion.

These are targets to measure, not promised speedups. The new model is expected to keep the direct query cost, not remove it.

## 8. Organizer evidence beyond a graph drawing

### 8.1 Actual trace and intervention requirements

On fixed input-chosen low/high-M scenes for each dataset, export the two-round native gathers, physical donors and one or two actual field-query paths. Use one common native coordinate system and retain Wind's existing frame without a second rotation.

At selected weights, compare:

- ordinary learned routing;
- density reset to one on a single typed route, and on all routed reductions;
- equal-measure/presence donor reassignment using original physical IDs;
- restored ordinary execution.

Recompute the complete context and second-round scores after first-round interventions. Do not freeze downstream state and describe the result as a full model intervention. Count changed donors and skip singleton buckets honestly. A simultaneous permutation of values, identities and weights is an invariance test, not an intervention.

Report physical target errors and output changes separately. Exact restoration checks unchanged parameters/default execution. These interventions establish reliance and mechanism; the independently trained R/C comparison establishes the conditional added learning value. Neither supplies a physical causal graph.

Do not claim unique superiority over geometry-only organization without an independently trained appropriate comparator. A small fixed geometry intervention may be included as a labelled diagnostic, but it is not that training comparison.

### 8.2 Functional content, not just weight concentration

A large attention mass can multiply a small message, and a small mass can multiply a large message. Retain a sampled weighted-message norm and actual aggregate/output change beside each shown route. Do not use attention ranks as output attributions or source heat sensitivities.

Compare structure on the same scene under an input-only geometry/context change, rebuilding state, and on two different native destinations. Report whether learned ordering and resulting states actually adapt. Such model probes use no new physical labels and cannot validate finite physical geometry responses.

### 8.3 Small sparsifiability check; no sparsity campaign

After a useful dense R fit, a **single 99% retained-mass context-cut policy** may be inspected on at most four TRAIN and four exposed DEV scenes per dataset. It preserves every physical source record and every final source-query read. The cut is an approximate context computation, not the same mathematical model and not a source-causality claim.

Initially report mass retention, weighted-value distortion and physical field mean/tail changes. No general error certificate follows from mass alone. Route scores are legal pre-value quantities, but score formation and all upstream dependencies still cost time.

Only implement selected-value execution if measured context work is a meaningful part of the complete call and the proposed cuts remove enough of it to plausibly pay for indexing. Recompute scores/values through both modified rounds. Compare selected and dense-masked execution of the **same cut**, including gathers, scatters, fallback, context preparation and CPU output. Keep the existing numerical parity criteria for that executor comparison; keep physical approximation allowances separate.

If few rows can be removed or complete calls are slower, preserve dense R and report no sparse benefit. No additional cut levels, learned budget chooser, entropy penalty, top-K schedule or second router may be trained. The primary research outcome is not contingent on a small donor-count histogram.

## 9. Efficiency, memory and numerical scope

Routing adds preparation work of the order

\[
 O\!\left(2(M^2+2ME)\,d_{\rm pair}\,32\right),
\]

with small score/mass arrays and no new Q-by-regional-edge stack. Constants and kernel launches matter; this expression is not a latency prediction. Fine source/environment value messages and final source-query reads still execute densely in the authoritative model.

Start with the verified microbatch8 and tile512 scientific protocol. Do not change the main experiment's effective batch, query stream or precision to fill VRAM. Use free memory for cached target-free geometry/catalogues and reliable execution only where reuse is valid. A larger microbatch is an optional engineering measurement, not evidence of more model capacity or supervision. Preserve the known native-curl batch-shape limitations; do not fold a numerical reassociation experiment into the primary learning comparison.

For timing use one verified available authorized GPU, identical native input/output scopes, two warmups and seven interleaved synchronized trials, with diagnostic exports off. Include input staging, all preparation, all learned outputs, native conversion/curl/proxies and CPU return. Keep setup-only, prepared-read and complete-call measurements separate; medians of overlapping scopes are not additive.

Native Thermal curl still requires the existing cell-centered domain and solid mask. Its interface and q proxy amplify small field differences. This does not justify changing their definitions or deleting them from comparisons. Off-grid requests retain their existing rejection. Do not impose zero divergence on the analytic Thermal benchmark.

## 10. Response and inverse interfaces

Preserve exact source-ID ordering, physical heat units, affine finite heat application, the native response interpolation, and current stale-state rejection. A geometry/context change invalidates all routed context. New queries may reuse a still-valid prepared context but must create their own receiver bindings. Model moves/dtype or execution-policy changes must be handled by existing owner/version semantics.

Run bounded software checks of heat-null flow, affine contraction/increments and geometry/input derivatives, with the native-curl derivative domain stated. No inverse generator or design search is part of this task.

A single final replay of already saved Thermal response pools is permitted after model selection, only where original reference provenance and masks can be bound. It does not select the model or calibrate a loss. In particular, 0291 remains exposed DEV/audit, and an absent counted-response receipt must not be replaced with an invented one. Report NA where binding is insufficient.

Separate `max(T_endpoint) - max(T_baseline)` from `max(T_endpoint - T_baseline)`. Opposite balanced heat transfers excite one independent direction. Learned source columns are not independently validated Green's functions, and accuracy of the affine arithmetic is not physical response accuracy.

## 11. Budget, autonomy and stopping

By adopting this plan, the new campaign supersedes only the **previous round's** 1,600-update diagnostic allowance. It does not change the protected old runner caps or authorize full-data fitting.

**Proposed envelope:** 8 aggregate GPU-associated hours, 10 elapsed hours, final two elapsed hours reserved for delivery. Use elapsed time relative to the actual Goal start, not a previous report's timestamp. Charge staging, failures, cached/cold work, profiling and evaluation. A ceiling is not a target to spend.

| Work | Maximum new optimizer updates |
|---|---:|
| Four primary R/C children to absolute e2500 | 21,000 |
| TRAIN-only pilot/self-refinement before seal | 2,000 total |
| One conditional fresh-start R/C pair | 4,000 maximum |
| Disposable calibration/optimizer/compatibility checks | 128 total, separate from scientific evidence |

Scientific total is at most **27,000 updates**; all invalidated/pilot attempts count. The primary pair comparison and final reserve take priority over the optional fresh-start and sparsifiability checks. Forecast remaining time from complete measured native epochs, not an isolated MLP benchmark. Stop/save owned jobs cleanly before the reserve. Do not kill or repurpose unrelated processes.

No new physical solve is authorized: solver ledger remains **326/326**. WindTEST and unexposed Thermal targets remain locked. The user owns later formal5000 launches.

Preserve all historical model/data/configuration artifacts, the original 953-observation baseline including its 67 absent entries, the recorded 764 additional protected files and the completed migration/diagnosis outputs. Reconcile the current inventory; do not assume every missing path is a new deletion. Do not move/delete inherited data to recover disk space automatically.

Keep generated checkpoints, arrays, profiles, large tables and figures local/ignored. Use compact metrics and selected native planes rather than full Wind volumes. No repeated report rendering should dominate the training budget.

## 12. Deliverables

The task must deliver **new model code and measured fitting**, not just a proposal or another optimizer-cosine report, unless a concrete reliability/resource blocker prevents the scoped experiment.

Required durable outputs:

1. One shared native routing primitive and its P-R integration, plus the donor-blind P-C control and strict versioned identities.
2. Architecture-child preparation through the existing TrainingEngine, with explicit state/moment/age mapping and immutable native sampling/normalization/loss settings.
3. Focused compatibility, numerical, dependency, identity, autograd and native learning tests.
4. Completed primary child checkpoints/histories and selected native evaluation, or a clearly documented partial age/blocker; old references remain intact.
5. A compact report with separate Predictor, Organizer, Execution and Interface conclusions; one evidence index and small machine-readable physical/cost tables.
6. A decision on whether the next user-controlled formal run is supported, and exactly which warm/fresh recipe evidence exists. An inert command check is not a completed formal experiment.

Suggested report name: `docs/reports/_bk/20261012_053211Z_HONF_Native_Interaction_Routing_Development_Report.md`. Suggested guide: `docs/guides/HONF_Native_Interaction_Routing.md`. These are proposed names. Preserve all prior reports except the explicit additive caption/provenance corrections.

Use at most four main figure families:

- paired mature physical learning/fixed-weight fit, with same-age P context;
- actual native field/residual maps and Wind transverse wake cuts;
- executed native gathers, weighted values and full-rebuild interventions;
- matched complete cost/work/memory, with an optional single sparsifiability panel.

Wind horizontal plane plots should use one actual native z plane and componentwise Ux/Uy/Uz. A hub-slab sample projected into XY is not a planar contour and cannot establish wake shape; label any sampled projection honestly. Annotate RMSE and relative L2 from unclipped native valid arrays. Keep common scales within a component and separate zero/weak-signal denominators.

Do not regenerate 21 old reference evaluations. Reuse verified historical metrics at their exact scope and add only the new matched comparisons.

### Terminal decisions

| Decision | What is sufficient |
|---|---|
| Better predictor direction? | Mature R/C native physical comparisons, explicit tails, and matched-age P context where verified; not a favorable normalized total alone. |
| Useful organizer direction? | R exceeds the added-capacity control with acceptable cost and the trace/intervention matches actual computation. A graph drawing or a changed weight alone is insufficient. |
| Efficient enough? | Complete matched calls and full-objective training costs; no inference from parameter count, unused memory or nominal pair counts. |
| Safe interface preserved? | Source identity, input dependency, affine/nonlinear laws, native numerical domain and stale-state rejection retain their measured scope. No physical inverse-readiness claim. |
| Next formal action? | A specific supported recipe and its warm/fresh evidence, or no recommendation. No automatic full-data or 5000-epoch launch. |

Finish with focused affected tests, default-path legacy checks, stopped owned processes, saved states, current protection reconciliation, the existing outgoing-object/artifact audit, push of durable changes only, and verification of remote/local tips.

## 13. Evidence and external context

The new mechanism and thresholds above are **proposals**, not conclusions derived from the preceding results. The preceding results support retaining P, rejecting the factored executor for authoritative use, preserving all Thermal objectives, and testing an organizer at a cheaper insertion point. They do not prove that fixed-measure aggregation caused the physical errors.

Repository evidence at `b851acd`:

- consolidation report, sections “Reference identity and maturity,” “Self-contained Wind migration,” “Fixed-weight fit and controlled learning,” and “Organizer decision and physical limits”;
- linked evidence index and development guide;
- `interaction_core.py`: existing MM/ME/EM reductions and final source read;
- `interaction_preserving_joint.py`: P versus P-G/P-H paths and affine response semantics;
- `windfarm/migration.py`: strict inference migration, preserved provenance and reload;
- `reference_anchored_thermal_flow_pair_20261011.yaml`: actual T4111 e1000 parent and six zeroed thermal terms in the completed diagnostic.

Two primary-source methodological references inform the proposal without validating HONF performance:

- Shaked Brody, Uri Alon and Eran Yahav, **How Attentive are Graph Attention Networks?**, ICLR 2022, [arXiv:2105.14491](https://arxiv.org/abs/2105.14491). The relevant point is genuinely destination-conditioned attention ordering. This does not mean the current nonlinear P value network has the paper's static-GAT limitation.
- Zongyi Li and colleagues, **Neural Operator: Graph Kernel Network for Partial Differential Equations**, 2020, [arXiv:2003.03485](https://arxiv.org/abs/2003.03485). The relevant point is nonlinear operator/message computation with explicit integration measures. It does not certify discretization invariance or physical conservation of this proposed normalized routing rule.

The intended advance is modest but substantive: **make the existing information gathering explicit and trainable, keep the inexpensive physical readout, and test the result against a real non-routing capacity control.** Do not exchange that test for a new global bottleneck, another frozen-flow composition, or an unmeasured sparse-execution claim.
