# HONF Round9 — joint native M–E retrieval, physical-response evidence, and sparse memory

**Status:** prospective implementation and scientific work plan, not completed results.  
**Repository / branch:** `cosmos2w/ModularDT`, `agent/honf-core-next`.  
**Reviewed remote tip:** `6ecc1086454705dc791d9ea782dba0f68b459e55` (`6ecc108`, Round8). Recheck the tip and local changes before editing.  
**Primary evidence:** the complete uploaded *HONF_Learned_Budgeted_Native_Hypergraph_Round8_Report.md*, including its field/cost appendices, and the pinned source inventory in §3. The uploaded report has SHA256 `06172fdec80216cc0ba01ce446c2c6e24641de3974a9bc8ff6b71abb4db600e3`.  
**Review boundary:** report and source review; private checkpoint/array bytes, current GPU occupancy and uncommitted workstation changes were not independently inspected in this review. Do not describe the proposed mechanisms or targets below as measured results.

## 0. Scientific decision and order of priorities

The project thesis remains:

> Learn from multi-field reconstruction → discover a reusable organization of module–environment interactions → reuse that organization to generate and evaluate modular designs.

Round9 is **not** another attempt to win a forward-surrogate latency benchmark. Its priorities, in order, are:

1. Implement an actual, jointly trained, query-dependent **M-and-E value consumer**, with original native record identities and a physically coherent readout.
2. Learn a **shared typed hypergraph** that selects both kinds of records together, rather than attaching two unrelated selectors or displaying unconsumed E metadata.
3. Retain useful native-field accuracy against both an equally matured A2-D control and an equally matured dense M+E control. Modest losses are acceptable within declared bands; the Round8 surface/material/peak and streamwise-wake regressions are not acceptable evidence of preservation.
4. Demonstrate **actual sparse execution and lower memory relative to dense A2-DE**, with fair tiling, checkpointing, microbatch and output scopes. Complete latency is measured, not required to improve.
5. Separate operational routing, original-input dependence and physical-reference response. Determine how much organizer value the existing data support; do not manufacture physical causality or independent environmental excitation.

The concrete primary model is a **native-seeded joint typed directory plus an E-conditioned native-module reader**. Each group contains original M and E IDs. One query dispatch selects a joint group; E records are individually processed at that query, and their resulting query context conditions the original module-specific nonlinear interactions. This provides a real path from selected E values to Thermal source coefficients, not merely to its flow head.

The primary executor is **flat ragged selected values**, not Round8's globally padded group rectangles. Group lists are a representation and a reuse opportunity; the implementation need not execute giant padded blocks to deserve the name hypergraph.

Do not begin a new dataset-generation campaign, full-TRAIN run, inverse generator, solver campaign, or broad historical refactor. A bounded saved-design response/ranking replay is conditional on the endpoint qualifications in §13. All new numerical choices in this document are prospective research choices, not amendments to Round8.

## 1. What Round8 actually established, and what must change

### 1.1 Retain the achievements and negative results

All four D8/G8 continuations reached literal e2500, with 36,000 new scientific updates and matched case/query hashes. High-M **flat** execution retained about 60% of original query–module rows. Lists were input-conditioned and variable: Thermal TRAIN0631/M12 had degrees 3–12; Wind TRAIN533/M30 had degrees 12–26 at e2500. The original individual nonlinear values were actually consumed. This is a real learned native-retrieval result.

However, Thermal DEV surface/material RMSE rose 27.8%/29.8%, and within-case module-peak RMS rose 58.2%. Wind Ux/vector priority means rose roughly 16–26%. High-M B8/Q8192 flat latency was 7.77%/3.29% slower. Same-count routing controls had mixed outcomes, not consistent learned-group superiority. Preserve all of this evidence. [R8: Predictor, Organizer, Execution]

The membership diagrams are **directed seed–member incidence and query routing**, not physical-causality graphs. A native seed is a group identifier, not an artificial averaged physical object. Different degrees in two pictured scenes are not by themselves a controlled demonstration of adaptation to a physical perturbation.

### 1.2 Five findings that change the next experiment

**A. Thermal degradation is not only omission.** TRAIN0006/M1 has no removable module, yet G8 surface/material errors worsened markedly. The all-selected TRAIN table also exposes particularly poor small-M thermal roles. Shared-weight adaptation, objective balance and router-gradient exposure need scrutiny; simply restoring more M values is not a proven repair. Do not assert a single cause without measurement. [R8: lines 45–75, TRAIN stratum appendix]

**B. Router-gradient cadence was uneven.** The literal `every fourth global macro` rule always selected Thermal's final six-case macro after the initial phase because there are four updates per epoch. Across the continuation, dense steps covered 13.6% of primary case visits, not 25%. This is an exposure confound to correct prospectively, not proof that it caused the thermal failure. [R8: Execution; `budgeted_core.py::_router_mode`, `budget_state.py`]

**C. Wind closed late and abruptly.** Deterministic retention remained about 0.996 at e1800 and fell to about 0.598 at e1900. A native-loss spike accompanied the transition; the dual hit its cap at 11/20 saved milestones. A good terminal relaxed-work count did not imply a gentle or fidelity-preserving training path. Track hard membership changes and physical roles during the new fit, and do not repeat the same aggressive M target by habit. [R8: Learning]

**D. Grouped padding was algorithmic, not evidence that native lists are inherently useless.** `_grouped_native` takes maxima of query occupancy and source width across groups, unions protection exceptions at group level, and evaluates the rectangular product for all seeds. The measured execution was 1.56× Thermal and 2.95× Wind dense rows. Do not reuse this as the primary sparse backend. [R8: costs; `budgeted_reader.py::_grouped_native`]

**E. The existing exact-tie check does not scale to joint seeds.** `seed_dispatch.py::dispatch_native_seed` constructs a `B×Q×S×S` Boolean `tied_pairs` tensor, even in ordinary scenes without indistinguishable geometry. With S=M+E this can become a major allocation independent of selected-value sparsity. Replace this in the new family with scene-level duplicate-key grouping and query-linear tie checks; do not carry the quadratic query-seed-pair rectangle into Round9. This is a source-level scaling diagnosis, not a measured Round9 memory result. [C5]

### 1.3 Mandatory software inheritance

Use the postfit loader/clock, weak-reference lifetime and counter fixes present at `6ecc108`. Do not train from the earlier leaking implementation. Preserve Round8's original fitting-memory histories, old strict FP32 misses and separate repaired measurements. A postfit test is not a retroactive repair of scientific evidence.

The new study starts from **D8**, not G8: preserve the useful original M field function rather than inheriting G8's damaged shared weights and budget state. The exact D8 e2500 bytes must be resolved from Round8's `scientific_closeout.json` and checked, not guessed from an alias or filename.

## 2. Questions, comparison logic, and limits

### 2.1 Four terminal arms per dataset

| Arm | Name | Reader | What the comparison tests |
|---|---|---|---|
| `D9-M` | continued A2-D | Original dense M reader; original upstream M/E contextualization | Matched maturity and practical accuracy reference |
| `D9-ME` | A2-DE | Dense E query values followed by E-conditioned dense M values | Incremental usefulness of direct E access, with its extra capacity disclosed |
| `H9-ME` | Joint-HONF | One learned native-seeded joint M/E list plus protections; selected E and M values | The main joint-organizer hypothesis |
| `S9-ME` | direct sparse control | Independently learned query–M and query–E supports; the same value function | Whether reusable joint groups add value beyond simpler learned sparsity |

`H9-ME` and `S9-ME` inherit the **same D9-ME warm-up checkpoint**, including optimizer moments. They use the same physical reader, original loss, query streams, typed work targets and corrected router-learning cadence. No frozen teacher or second predictor is present in a deployed model.

The trained sparse control is important. Same-weight posthoc replacements are still required, but they do not replace an equally trained alternative. If H and S end with appreciably different actual work, present the difference; do not call it a matched-work architecture result or force inference cutoffs after looking at errors.

### 2.2 “Independent E information” must be stated precisely

Current Thermal E192 and Wind E64 records are adapter-owned, target-free geometric descriptors. Much of E is determined by the module/domain inputs. If `E=f(X)`, adding E does not add independent physical information to an ideal predictor that already knows X. It can add a useful representation and a direct computational path.

Therefore the main data-supported question is:

> Does direct, query-dependent access to original E records improve the learned predictor or preserve useful information when M/E access is organized sparsely?

Do not convert a positive answer into evidence for independently varying terrain, heterogeneous conductivity, inflow fields or boundary forcing absent from the data. Those remain future dataset questions. Conversely, weak E gains on these datasets do not prohibit implementing or learning the joint architecture.

### 2.3 Three levels of interaction evidence

- **Executed organization:** selected native IDs reach actual nonlinear values; modifying membership changes the field; the graph earns accuracy/work utility against controls.
- **Original-input dependence:** geometry/prescribed input changes affect predictions and routes after rebuilding the full scene, including upstream contextualization.
- **Physical-response agreement:** predictions of actual saved perturbations agree with their corresponding reference endpoints, changes and rankings.

Report these separately. Dense contextualization remains in this round: a selected contextualized record can carry information from omitted raw records. There is no claim of globally sparse physical dependence or a causally complete hypergraph.

## 3. Pinned code review and implementation boundaries

Paths below are relative to `HONF_Proj/`. The table maps the inspected core implementations and their required existing integration boundaries. It is not a claim that every legacy helper was exhaustively audited. New filenames proposed later are not existing-source claims.

| Existing source at 6ecc108 | Finding / use |
|---|---|
| `src/honf_forward_core/content_hypergraph/model_v2.py` | Shared preparation, local/readout orchestration and affine APIs. Direct temperature is assembled separately from the nonlinear flow field. |
| `src/honf_forward_core/content_hypergraph/readers_v2.py` | Original query-specific SiLU module values; thermal coefficients do not use information-attention weights. Preserve its parameter identities/formula at migration. |
| `src/honf_forward_core/content_hypergraph/staged_attention_v2.py` | Ordered MM→EM→EE→ME. Native capture currently covers MM/ME; EM/EE use the older score-snapshot interface. All-native seed support requires an explicit extension, not relabelling snapshots. |
| `src/honf_forward_core/native_retrieval/budgeted_core.py` | Hardcoded M-only capture, one-seed dispatch, heat-free graph, old macro cadence; postfit weak-reference hook and recursive clock restoration. |
| `src/honf_forward_core/native_retrieval/budgeted_graph.py` | Shared calibrated memberships, protected union, analytical expected work; graph dimensions assume seeds=M. Reuse algebra concepts, not this square-M-only contract unchanged. |
| `src/honf_forward_core/native_retrieval/seed_dispatch.py` | Native categorical routing, geometric tie semantics and ID/RNG separation. Its query-by-seed-pair tie tensor must not be inherited at joint M+E seed counts. |
| `src/honf_forward_core/native_retrieval/budgeted_reader.py` | Independent multiplicative zero-sensitive dense reduction and true flat reads. Grouped padding and per-request telemetry have substantial costs. |
| `src/honf_forward_core/native_retrieval/contracts.py`, `route_capture.py` | Native stores and ownership/version contracts. Extend with explicit type-qualified IDs and selected M/E read records. |
| `src/honf_runtime/native_budgeted_{protocol,factory,runtime}.py` | Strict no-E, D0-e500, e2500 contracts and named optimizer inheritance. Add a new family; do not mutate Round8's meanings. The current archival loader also checks external source/admission files. |
| `tools/native_budgeted_train.py` | Reusable managed-launch pattern; no startup speed gate. New family needs its own stages, budget and horizon. |
| `Case_ThermalChannel/src/channelthermal/{content_hypergraph.py,native_observations.py}` | Geometry-only E records, zero-boundary heat-affine capability, native union/interpolation/curl/flux extraction. |
| `Case_ThermalChannel/src/channelthermal/training/content_hypergraph_task_v2.py` | Current native objective/provider and source-read task lineage. |
| `Case_WindFarm/src/windfarm/{content_hypergraph.py,geometry.py}` | Target-free E64 input catalogue, aligned frame and stored TRAIN profile/residual transform. E records are not the CFD target velocity cells. |
| `Case_WindFarm/src/windfarm/training/content_hypergraph_task.py` | Distinct original TRAIN and DEV query/monitor semantics; reuse the fixed provider, not an improvised five-role training objective. |
| `src/honf_runtime/unified_training.py` and its actual current helper modules | Reuse TrainingEngine and implement narrowly scoped capability hooks. Inspect the live module layout before editing; do not copy a training loop. |
| `tools/content_hypergraph_evaluate.py` and registered loaders | Keep TRAIN/source monitor and five-role physical boards distinct. Decoder traces belong on decode, not preparation. |

Read the Round8 report, guide, evidence index and current source before implementation. Do not rebuild enormous historical inventories repeatedly or spend the round repairing unrelated families. Pin relevant parents and outgoing changes once; use bounded incremental checks thereafter.

## 4. Model: a native joint typed directory

### 4.1 Records and seeds

The scene contains separately retained records

\[
\mathcal V=\mathcal M\sqcup\mathcal E,\qquad
r_v=(\mathrm{type}_v,\mathrm{id}_v,x_v,\ell_v,\mu_v,f_v,z_v^\star).
\]

The disjoint union is important: `M:0` and `E:0` are different records. IDs are correspondence/RNG labels, never numeric learned features. Preserve presence, input units, characteristic lengths, native measures, feature provenance and the distinction between raw features and contextualized states.

Use actual present **M and E records as possible seeds**, not six latent slots or a fixed learned anchor bank. For each native seed s, the group stores two lists:

\[
\mathcal H_s=(s,\mathcal M_s,\mathcal E_s),\qquad
B^{M}_{si},B^{E}_{sj}\in\{0,1\}.
\]

S follows the actual native inventory. A current E catalogue can be a regular input quadrature; this does not make a learned pooled atlas. Do not infer irregular-environment physical generalization from changing a synthetic tensor shape.

The generic core must support variable M/E counts, padding, irregular coordinates and either empty type in data-free tests. Existing native training adapters retain their exact E192/E64 records and target meanings in the primary experiment. Do not change native E resolution and reader architecture simultaneously during training.

### 4.2 Typed membership scores

Keep the two-cycle contextualizer and its parameters. Extend opt-in native score capture to all four final-cycle routes with exact stage/owner/identity binding:

- M seed / M member uses MM.
- M seed / E member uses ME.
- E seed / M member uses EM.
- E seed / E member uses EE.

Names are **destination then donor**. Native capture must not change the actual attention update. Preserve old return signatures when capture is disabled and preserve old EM/EE snapshots for historical loaders.

For each route a←b, use the existing head-calibrated construction:

\[
\bar L^{ab}_{sv}=\sum_h\omega_h^{ab}
\left(L^{ab}_{hsv}-\max_{u\in\mathcal V_b^{valid}}L^{ab}_{hsu}\right),
\quad
\ell^{ab}_{sv}=\kappa^{ab}\bar L^{ab}_{sv}+b^{ab},
\quad p^{ab}_{sv}=\sigma(\ell^{ab}_{sv}).
\]

Use a distinct convex head mixture for each typed route; do not identify head labels between different attention blocks. Retain bounded positive calibration scales, initialized at 1 in [0.25,4], and uniform head mixtures. Calibrate route biases once on at most eight fixed TRAIN batches, aiming for about .98 valid nonmandatory membership probability. This is an initialization condition, not evidence of deterministic sparsity.

MM's contextual self exclusion remains unchanged. Retrieval retains each seed's own record. For a mixed scene, preserve **one cross-type member per seed**, chosen by the maximum calibrated cross-type score with the declared geometric tie rule. This ensures that both type-specific readers have an eligible record without using a field target. This mandatory cross-type member is an explicit architectural floor, not proof of a learned physical interaction. Other members are learned binary choices. Track self, mandatory cross-type, learned members and query protections separately.

This is one specific joint-neighborhood hypothesis. Do not force every learned member to be local, impose a wake cone, or add learned per-layout tables. Do not require all seeds to be occupied or all groups to be different.

### 4.3 Sampling and one shared query dispatch

Sample the binary membership matrices once per prepared scene on stochastic steps. Reuse that sample for all receiver tiles and auxiliary calls associated with the same objective scene. Keep independent type/route RNG namespaces and exact resume semantics. Thermal baseline and changed-heat requests at the same scene use the same graph realization.

Compute **pre-E** query-to-seed scores from the original query state `q0`, original M scoring and the new E reader's scoring. Calibrate their head mixtures separately by seed type, with a small learned dimensionless type offset. No physical M/E measures are added together to create a common dimensional score. These scores select one seed from the available typed native inventory. The final value weighting still uses its own type-specific attention heads.

Dispatch must precede the E read. Do not condition this dispatch on an already sampled/read E list: that creates a circular dependency and invalidates the simple expected-work calculation in §7.

Primary deterministic dispatch is one argmax native seed; stochastic training uses the declared categorical/Gumbel estimator. This is a bounded execution restriction, **not** a claim that one group is universally sufficient. Record the known discontinuity limitation. Do not add top-two blending, sparsemax, extra prototypes or another dispatcher during the main fit. Those are separate future alternatives if the measured joint model warrants them.

Ties use canonical input geometry, characteristic lengths and declared type, not native ID. Truly indistinguishable tied seeds follow an explicit rejected/indeterminate diagnostic path unless a symmetric tied-union reference is separately implemented. Never silently choose by source-array position. Primary native training must not contain unresolved indeterminate ties.

Precompute canonical seed order and duplicate-key group IDs once per scene. Determine whether a query has multiple tied maxima in an indistinguishable group using segmented counts or an equivalent `O(BQS)` operation; a scene-level `O(BS²)` check is acceptable at current sizes, but a `B×Q×S×S` tensor is not. Test equivalence to the literal small tie reference, including cross-type ID collisions, duplicate geometry, padded records and source permutation. This optimization must preserve the declared tie policy, not replace it with ID-based preference. Report dispatch mass and occupancy by seed type: more E records must not be mistaken for evidence that E is intrinsically more influential.

### 4.4 Final supports and floors

For t∈{M,E},

\[
g^t_{qv}=P^t_{qv}\lor B^t_{S(q),v}.
\]

`P^M` is the **unchanged** original compact local protection; it selects direct nonlocal values in addition to preserving the separate old local branch.

`P^E` contains the nearest positive-measure original E record under the adapter's length-normalized Euclidean metric, with the declared geometric tie handling. This supplies a minimal local environmental read and a nonempty type denominator; it is not a ReceiverCover, interpolation atlas or pooled value. E's mandatory member already suffices for nonemptiness, but P^E makes its local-read floor explicit. Do not enlarge either protection radius/set to repair errors mid-fit.

Self and mandatory cross-type memberships are applied before this union. Deduplicate original IDs; compute a native query/type/source value once. Log the exact union and incremental exception rows. Count physically present zero-information-measure M heaters correctly: their Thermal response path is not removed by an information-attention mask. E has no heating-source role; its positive quadrature eligibility is separately declared.

For per-query lower bounds, distinguish `max(1,|P^t_q|)` from the **actual mandatory/self/cross-type union for the selected seed**. Use the latter when declaring a floor-adjusted hard-work target; use its corresponding seed-weighted expectation during stochastic calculation. A floor controlled by a learned discrete anchor is not described as an immutable physical law.

## 5. Reader: E values must influence module response, not only flow

### 5.1 Query-specific E values first

Prepare tokenwise E keys/value projections once, keeping every record separate. At query q, evaluate only the selected original E values:

\[
m^E_{qj}=W_o^E\,\mathrm{SiLU}
\left(W_q^Eq_0+W_Ez_j^\star+W_\gamma^E\gamma(q-x_j)\right)+b_o^E,
\]

\[
\alpha^E_{qhj}=
\frac{g^E_{qj}\,\widetilde\mu_j^E\,\exp l^E_{qhj}}
{\sum_k g^E_{qk}\,\widetilde\mu_k^E\,\exp l^E_{qhk}},
\qquad c_E(q)=\mathrm{headwise\ reduction}(\alpha^E,m^E).
\]

The raw-score stabilization and empty-type behavior must be explicit. `mu_tilde` is the original within-type measure divided by its declared reference. Do not combine module counts/areas and environmental areas/volumes in one dimensional denominator.

This reduction occurs **after original E-specific query nonlinearities**. It is not a prepared mean/moment/hyperedge embedding substituted before query interaction. No `Q×M×E×width` tensor is required.

### 5.2 Condition the native M query, then retain individual M values

Define

\[
q_E=q_0+A_Ec_E(q).
\]

Then use the existing M reader formula with q_E in the query-state position:

\[
m^M_{qi}=W_o^M\,\mathrm{SiLU}
\left(W_q^Mq_E+W_Mz_i^\star+W_\gamma^M\gamma(q-x_i)\right)+b_o^M.
\]

The native M value still depends on the individual original module, not its group's centroid or average. Final M information weights are recomputed using q_E. Pre-E scores used for dispatch and post-E scores used for final M information are distinct, explicitly counted operations; do not silently reuse the former as if they were equal.

Pass q_E to the existing local reader and nonlinear field head too. Keep the local branch's geometry, native masks and smooth weights unchanged. Thus direct E information can affect local and nonlocal source coefficients and the flow/residual field.

The cross-type coupling is operational: changing selected E values at fixed M records changes the nonlinear M messages and their physical coefficients. This is stronger than merely adding an unrelated E output head. It does **not** prove physical many-body causality or an unrestricted three-body kernel.

### 5.3 Exact parent boundary and learnable E path

Initialize the new E reader with nondegenerate parameters; compatible M projections may be copied as a declared initialization, or use the original name-based initializer. Use a **bias-free A_E projection and initialize only its weight to zero**. At the migration boundary, q_E=q0 and original M/local/field outputs must recover the parent. Do not zero both the E values and A_E, which can create a dead path.

Check that A_E has a useful task gradient at the first boundary and that E value/key parameters acquire task gradients after A_E moves. A zero-gain initial boundary is not a claim that the E reader is already trained.

Exact forward recovery and exact copied optimizer state do not imply that the first active-E optimizer step equals D9-M: the new A_E gradient can change global gradient clipping and intentionally starts a different training trajectory. An original-step parity check must use an explicit E-disabled disposable mode with the same clipping set; do not freeze or discard the real E gradient merely to manufacture optimizer-step parity.

All D9-ME/H9-ME/S9-ME parameter names and initial value tensors agree at their shared branch boundary. H/S add their respective router parameters only. D9-M has no unused trainable E path and is timed without artificial E work.

### 5.4 Thermal physical capability

For the current Thermal benchmark,

\[
K_i(X,E,q)=g^M_{qi}\,k_i(q_E,z_i^\star,\Delta x_{qi})
+K_i^{local}(q_E,X,q),
\qquad T(X,E,q,h)=\sum_iK_i(X,E,q)h_i.
\]

Retained M coefficients are full-amplitude signed values. Binary support does not become fractional scalar attenuation, and information-attention weights do not become physical heating coefficients. E has **no fictitious heater index**. E affects K through q_E and existing contextualization.

Routing, q_E and K remain independent of actual h in this benchmark. Zero heat produces zero temperature; heat cannot alter flow. Preserve native surface/material interpolation, flux-proxy extraction, curl and independently recomputed maxima.

The affine restriction is a dataset-owned capability, not the definition of Joint-HONF. Nonzero boundary/background response requires an explicit `T0 + K h` capability and appropriate data. Genuinely nonlinear heating requires a different data-supported response capability. Neither is silently added in this round.

A concrete failure to avoid: appending c_E only to `FieldHeadV2` would affect Thermal flow while leaving its separate direct/local thermal coefficients unchanged. That is **not** an acceptable implementation of this plan.

### 5.5 Wind physical capability

Use the same typed joint reader, with no Thermal source sum. Preserve the actual stored OpenFOAM velocity targets, original native cell IDs/roles and m/s conversion. The final field remains the maintained TRAIN-fitted height-profile plus scaled residual in the already aligned frame. Do not add the absolute normalizer mean, rotate twice, or introduce an inference wake equation.

### 5.6 Reusable inference interface

Expose a new family, suggested schema `joint_native_hypergraph_v1`, with explicit value mode, routing mode and physical output capability. Do not turn Round8's strict `environment_value_reader=False` flag into a permissive escape hatch.

Prefer a narrow new core derived from `ContentHypergraphCoreV2`, or small default-preserving hooks, rather than a chain of special-case wrappers. E must be read **before** the current local-reader call, so `_decode_native_retrieval` alone is not a sufficient insertion point. Share the new decode primitive among ordinary prediction, prepared queries, affine source export, increments and JVP/VJP interfaces.

Public prepared objects bind scene/weights/normalizer/type catalogue and query identity. Changing backend or intervention must not bypass stale-state checks. Provide a separate explicit counterfactual-route interface that applies validated stored IDs to **freshly rebuilt states**; it must not reuse stale prepared tensors.

An exported inference artifact must be self-contained in its model, normalizer, adapter schema and capability metadata. It must not require ancestor checkpoint files, TRAIN datasets or an old research-admission receipt to predict a supplied scene. Training/resume still verifies full source lineage. Do not alter historical loaders to achieve this.

## 6. The trained direct sparse control

`S9-ME` uses the same E-conditioned reader, contextualizer, protections, local branch and output laws. It replaces the shared seed/member directory with per-query typed binary gates calibrated from the pre-E query–M/E raw scores:

\[
p^t_{qv}=\sigma\!\left(\kappa_t\sum_h\omega_h^t\bar l^t_{qhv}+b_t\right).
\]

Retain a declared best-scoring valid donor per available type in addition to P so denominators are nonempty. Count that floor separately. Use no new utility MLP, graph bank or target-informed selector. Its support can vary independently per query; there is no reusable seed-list restriction.

Use the same hard-forward estimator, typed work targets and training requests as H. The models differ in organizer restrictions and small router parameter inventories, which must be reported. Same expected budgets do not guarantee the same deterministic final work; endpoint comparisons must disclose both.

Train S, do not merely apply direct-score pruning after H has adapted. Additional same-weight count-matched controls remain useful for localization, not as substitutes for this trained control.

## 7. Training: native fidelity first, gentle typed work pressure

### 7.1 Preserve the native task

Retain every original supervised term, scale, role weighting, auxiliary query and scheduled response call. Thermal must retain its original field, surface/material, q-proxy and response terms; do not assume that every reported endpoint statistic is separately supervised. Wind retains the original 3:1 volume/hub primary draw, not the posthoc five-role board.

At startup, audit per-term numerator, denominator and macro accumulation on one fixed native update for each dataset. Check gradients reach intended M/E/local parameters. Verify that tiling/checkpointing/microbatch changes preserve the same objective. Do not replace physical targets with parent outputs or add a hidden teacher-distillation loss.

A smaller scalar weighted native loss is not a field-preservation gate. Persist every native term and fixed-panel physical role separately throughout learning.

### 7.2 Two typed work constraints, not a universal M=60% rule

For unit-noise membership Bernoulli samples and categorical one-seed dispatch independent conditional on the prepared scene,

\[
\mathbb E[g^t_{qv}]=P^t_{qv}+(1-P^t_{qv})\sum_s\pi_{qs}p^t_{sv}.
\]

Override mandatory membership probabilities to one. Compute expected mandatory/protected floor work with the same seed probabilities. Count rows, not physical measure or attention mass. With deterministic/scaled-noise routing this becomes a labelled relaxation, not an exact expectation.

Aggregate all actually requested query tiles, native union roles and response requests within each case, then average case fractions. Preserve separate role/stratum statistics. Count both expected/relaxed and actual unique M/E support. Never hide an E-dense path behind a small aggregate M+E ratio.

Primary terminal **removable-row retention targets** are:

- M: **0.85**, substantially gentler than Round8's 0.60.
- E: **0.60**, where the larger native inventory offers the principal memory opportunity.

These are objectives, not hard inference cuts or requirements that every scene must be sparse. Fully protected or genuinely dense scenes are legal and reported. Self/mandatory/protection floors cannot earn a fictitious sparsity reward.

Use two small serialized dual states, one per consumed type, with the same form as the existing augmented positive-residual penalty. Calibrate initial coefficients from at most eight fixed TRAIN batches to approximately 5% of that router family's native task-gradient norm **per type** (about 10% combined when both are active). Set zeta=lambda0, lambda_max=10*lambda0 and update every ten successful macro-steps using the window's observed residual. Handle zero/unsupported task-gradient norms explicitly; do not divide by a numerical zero. In the unit-noise early phase use the exact expectation for the controller; subsequently use observed deterministic work, with the differentiable penalty labelled a relaxation.

This reuses a known estimator rather than inventing a new optimizer. Its limitations remain: biased gradients, possible dual saturation and possible hard-work cliffs. No DEV-tuned work-weight sweep or hidden per-field reweighting is allowed.

### 7.3 Correct dense-router exposure

Use dense **hard-selected-forward** router learning during the first 200 sparse-branch epochs. Subsequent dense router steps use a balanced four-epoch pattern:

```text
is_dense_router_step = ((sparse_phase_epoch_zero_based + macro_index_zero_based) % 4 == 0)
```

For Thermal, each of its four macro positions is dense once in four epochs. For Wind, each of its five positions is dense once in four epochs. This covers 25% of primary case visits over the cycle, including each final partial-macro position, rather than always selecting Thermal's six-case batch. Persist the phase anchor and use epoch/macro indices restored exactly on resume. Do not derive this from a mutable unsaved Python counter.

The scheduled Thermal response still follows its original native cadence. Record dense coverage separately for primary cases, primary query rows, auxiliary rows and response calls. Sharing one dense/non-dense decision across all microbatches of a macro is mandatory.

On dense-router steps, use multiplicative gate normalization so omitted E and M values can have restorative gate gradients. On selected steps, use real selected values and fixed-support task gradients plus the differentiable work penalty. Do not label selected-only zero-gate gradients as complete router learning.

### 7.4 Sparse-phase schedule

The shared dense E warm-up and exact ages are defined in §8. Relative sparse-branch age r starts at 1 after that warm-up.

| Sparse age | Policy | Removable targets M / E |
|---|---|---|
| r1–200 | Binary stochastic forward; dense hard-forward router estimator every macro; relaxation temperature 1→0.5 | .98→.95 / .98→.90 |
| r201–1000 | Binary stochastic forward; balanced dense-router cycle, selected other steps; noise 1→0 | .95→.85 / .90→.60 |
| r1001–2000 | Deterministic hard forward; balanced dense-router estimator, selected other steps | Hold .85 / .60 |

The terminal half of the sparse fit is deterministic. Plot deterministic and stochastic supports separately while noise is active. Do not borrow a dense earlier checkpoint's accuracy and a later sparse checkpoint's memory.

At 25-epoch TRAIN-only reviews, report per-type deterministic support, fraction of memberships near threshold, changed hard IDs, dual cap occupancy and original task-role changes. A >0.15 absolute retention change across one review interval triggers a diagnostic annotation and a fixed-batch replay, not a hidden reset or posthoc cutoff. Mild error regressions do not justify stopping healthy learning before maturity.

There is no automatic scientific architecture amendment. A genuine semantic bug is repaired with a recorded restart/resume decision and correctly labelled source version; do not silently mix model formulas within one history.

## 8. Matched continuations and meaningful maturation

### 8.1 Parents and stages

Use the exact **segmented D8 e2500** sources for Thermal and Wind, resolved by Round8's scientific closeout/evidence index. Do not use G8, selected best-field aliases, full-TRAIN A2-D or a mature classic as the parent.

The source D0 e500 ancestry hashes are retained for verification only:

- Thermal: `3ea129d26b890f989ae330369914fcf75b20eba13bb1bd1660eee79f126a5daf`.
- Wind: `ca7713328f3d4c018ecf91e7cdf2acf6fc5334ae3196c73b54482a5eb028c2d7`.

They are **not** the new e2500 source hashes. Obtain the latter from actual local receipts and checkpoint bytes.

For each dataset:

1. Continue D9-M from D8 e2500 to e5000: 2,500 new epochs.
2. Create D9-ME at the exact D8 boundary with zero A_E and nondegenerate E parameters. Train all parameters jointly for e2501–3000: **500 genuine dense-E warm-up epochs**.
3. At literal e3000, fork three exact children: continued D9-ME, H9-ME and S9-ME. Each runs e3001–5000: **2,000 new epochs**, with H/S following §7.4.

All terminal arms therefore share the original D8 ancestry and the same 2,500 new case/query epochs. H/S share the additional 500-epoch dense-ME preparation. Warm-up compute is charged once, but its training exposure is disclosed in each child lineage. D9-M is a separately trained same-age control, not the unchanged e2500 source.

The main campaign is 9,000 actually executed new epochs per dataset: 2,500 D-M + 500 common D-ME + 3×2,000 children. That gives **36,000 Thermal and 45,000 Wind main optimizer updates: 81,000 total**. This is a prospective count, to be reconciled with actual histories.

### 8.2 Data and optimizer identity

Thermal remains TRAIN150 / exposed DEV22, Q1024 primary fluid plus unchanged auxiliaries/native union and saved response, effective batch 48. Wind remains TRAIN72 direction rows from 24 layouts / exposed DEV24 from eight layouts, primary Q8192 with 3:1 volume/hub, effective batch 16. Updates per epoch remain four and five, including partial final batches of six and eight cases.

Use exact parent TRAIN normalizers, source membership, query namespace, sampler seed (including Wind's distinction between initialization and sampler seed), native loss and adapter transforms. All arms use common absolute-epoch case/query streams. A new experiment namespace must not randomize one arm differently. Compare hashes at every epoch.

Restore original named AdamW moments. New E parameters receive fresh moments at e2500; H/S routers receive fresh moments at e3000. Extend the source optimizer schedule explicitly to e5000 at its verified terminal learning-rate policy; do not silently reset Adam or invent a warm restart. If the source schedule is constant, retain its exact constant value. Serialize any necessary horizon extension and verify all arms use it.

Dense E values may require smaller microbatches and/or checkpointing. Choose one feasible common microbatch configuration in TRAIN-only engineering for D-ME/H/S; D-M uses it too for the main matched study. Retain effective batch and partial-update handling. Gradients and per-term numerators/denominators must agree with an unchunked small fixture. A microbatch change must not change which response family is applied or repeat/omit its loss.

### 8.3 Reviews, checkpoints and stopping

Review e2750/e3000 for the E warm-up; review e3200/e3500/e4000/e4500/e5000 for the children. Save resumable checkpoints every 100 epochs, plus literal shared branch and terminal states, with latest and original-selector best aliases. Plot losses and deterministic work every 10 epochs.

Do not stop E development merely because it fails to improve a few early TRAIN samples. Lack of a current-dataset E gain is a legitimate result, not a reason to manufacture an E auxiliary target. Do not stop H/S solely for slower training or inference.

A healthy admitted sparse branch should receive at least **1,000 new epochs**, normally all 2,000. Allowed stops are explicit user stop, unavailable required source, unresolved semantic/nonfinite failure, persistent catastrophic native divergence (e.g. >5× matched dense-control loss over two 100-epoch windows), or actual resource exhaustion. Resource stops preserve matched milestones and an exact continuation handoff. An unfinished e4000 result is not a completed e5000 study.

Run paired arms in blocks so evaluation compares the same age and no model consumes the entire resource allowance before its controls. Reserve endpoint reporting and the trained S control before optional kernel work.

## 9. Sparse execution and honest memory claims

### 9.1 Primary backends

Implement:

- `dense_typed_reference`: the same ME function on all dense native values with the chosen hard masks; independent correctness/router-learning reference.
- `flat_typed_selected`: packed original query/type/source rows; main sparse training and inference backend.
- `dense_ME`: all-open A2-DE; same physical/value function, no routing machinery.

A future occupancy-bucketed executor may be an optional engineering comparison after the main scientific fits. It is **not** a required new compilation campaign. Do not reuse Round8's all-seed maximum-Q × maximum-width rectangle. Do not silently switch a selected prediction to all-open A2-DE.

Gather before wide Fourier/value features and SiLU. Keep only selected wide activations. Cheap dense query scores, small binary/logit tensors and dense contextualization remain allowed and counted. Never materialize a dense `B×Q×E×width` tensor and then call it sparse because it is subsequently masked.

Process E first, reduce to c_E for each receiver, then process M using q_E. E and M native value rows are individually attributable. Keep protection exceptions inside the final type-specific attention normalization, not separately normalized then added. There is no duplicated value evaluation for an ID reached through multiple permissions.

Trace messages, physical-ID lists and ancestry metadata are opt-in bounded outputs. Do not allocate full trace rectangles in routine training or timing. Move aggregate counters to host at logged boundaries, not inside every tile. Count score reconstruction and capture explicitly.

Bound routing memory as well as value memory. Inspect actual intermediate shapes at S=M+E, particularly tie detection and support expansion. A small synthetic M-only fixture cannot qualify the joint planner. Avoid any query-by-seed-by-seed or query-by-seed-by-donor rectangle; compute one dispatched list directly and calculate expected work by tiled matrix contractions without materializing the product paths.

### 9.2 Tiling and activation checkpointing

A Python loop over query tiles does not by itself bound training activation memory: autograd can retain every tile until backward. If dense A2-DE requires recomputation, implement it deliberately around pure value/readout primitives with `torch.utils.checkpoint(..., use_reentrant=False)` on the installed stack. Keep the same capability available to selected and dense controls. [E2]

Freeze sampled supports/RNG inputs for recomputation. Do not redraw membership, update duals, increment work counters twice or re-enter public stale-state validation inside a checkpointed primitive. Native extraction/curl dependencies and the E→M gradient must remain intact. Never detach the contextualizer or c_E to create a memory gain.

The independent dense literal reference may be small and slow; the **performance denominator** must use a fair tiled/checkpointed A2-DE implementation. Do not benchmark sparse against an intentionally untiled/OOM-prone dense rectangle.

### 9.3 Memory measurements

Measure one model per fresh process, identical precision/TF32 policy, real input/output scope and initial optimizer state. Report allocated and reserved baselines/peaks and whether gradients are resident. Do not retain four control models on the device during the memory comparison.

Required scopes:

1. Same-weight same-mask ME dense versus flat value execution on fixed low/high-M native requests, with E and M row hooks.
2. Full native inference, B1/B8 and Q1024/Q4096/Q8192, including preparation and native output extraction.
3. Real mixed-scene native forward/backward/optimizer steps with all original losses and auxiliaries, at a common effective/microbatch setting.
4. A complete four-epoch sparse-training cadence cycle, separating selected-step peaks, dense-router-step peaks, cycle maximum and total elapsed time.
5. A bounded repeated-lifetime check of at least 20 complete H9 native epochs per dataset, with gradients cleared consistently and references released, not three target-free forwards. Count these as engineering updates and discard their optimizer changes.

The primary desired result is **at least 20% lower peak allocated memory on real selected physical forward/backward scopes versus fair A2-DE**, plus actual omission of E and M rows where learned support allows it. Also report whether the full cadence-cycle maximum is lower. If dense router steps set the same maximum as dense A2-DE, say so: selected-step savings do not establish a lower end-to-end training capacity requirement.

Keep the scientifically useful and engineering outcomes separate. A measured sparse-memory benefit with somewhat slower training is acceptable. No memory benefit under fair settings is a miss, not something to hide behind logical support counts. Do not reduce Q, native roles, batch or training gradient scope for only H.

### 9.4 Runtime and work

Complete host-input/native-CPU-output latency is descriptive, not a promotion gate. Retain prepared-scene/new-query and Thermal fixed-query/new-heat reuse as separate scopes. Use warmups and alternating paired repeats; disclose fixed-scene timing intervals rather than population confidence.

Report per type: available present pairs, mandatory/protected floor, learned selected pairs, unique consumed IDs, nonlinear forward rows, projection rows, padding, dense estimator rows and checkpoint recomputation. Invalid mixed-batch padding reduction is not learned sparsity. Report how much total cost still belongs to dense contextual preparation and cheap scores.

## 10. Predictor evaluation and endpoint decisions

### 10.1 Native boards

Use unchanged complete Thermal DEV22 and Wind DEV24 boards at e3000/e4000/e5000, with literal terminal e5000 primary. Include fixed-weight all-selected TRAIN audits and all observed module-count strata. Preserve individual components, mean/p90/worst, case IDs and Wind layout-level summaries.

Thermal priorities include fluid/surface/material temperature, q-proxy and module peaks; show u/v/p/curl separately. M1 and M2 strata must be explicitly visible, because the Round8 failure was especially revealing there.

Wind priorities include every five-role Ux and vector error; retain Uy/Uz secondary components and original wake cuts. The five-role physical board does not change the inherited two-region training loss or the original DEV monitor quotas.

Use the same saved receiver IDs, masks, frames, units and original TRAIN normalizers across arms. Do not interpolate Wind targets. Do not compare different warm-start/full-data histories as causal ablations.

### 10.2 Prospective reporting bands

For H9-ME versus **both** same-age D9-ME and D9-M, target:

- Each priority mean RMSE within 10% and priority p90 within 15%.
- Each secondary mean within 10%, with all tails visible.
- No unexplained >25% regression in a populated module-count stratum's priority mean, and no repetition of the Round8 multi-fold small-M surface/material/peak failure.

These are new, deliberately modest Round9 bands. They do not relabel Round8's stricter criteria. A tiny/unresolved baseline denominator requires an absolute-scale analysis using the frozen TRAIN role scale, not an infinite ratio or a hidden exclusion. Report the resolved absolute numbers.

A promising but partially missed band is a qualified limitation, not fully preserved fidelity. Do not average a failed physical role into a favorable scalar. D9-ME may be worse than D9-M; H must not earn qualification merely by matching a newly degraded dense denominator.

### 10.3 E utility and router comparison

Report D9-ME−D9-M changes by role and stratum and parameter/cost differences. This is incremental architectural utility of a direct E path, not independent environmental information. E gradients or nonzero A_E alone do not establish useful information.

Report H9-ME versus trained S9-ME alongside actual type-specific row fractions, memory and parameter counts. If work differs by >5 percentage points in a key scope, label it unmatched work and supplement with the fixed-weight final-count controls; do not claim causal group superiority.

Best-field aliases are secondary. A best H checkpoint can only use its own graph, memory and physical outputs. Historical full-TRAIN A2-D and classics remain practical context on common receivers, not required to be beaten and not new full campaigns.

## 11. Organizer and physical interaction evidence

### 11.1 Actual typed value consumption

On a fixed input-selected low/high-M panel for both datasets, capture a small set of original receiver IDs with:

- dispatched native seed type/ID;
- M/E group membership, mandatory members and protection exceptions;
- actual packed M/E value-hook rows and original IDs;
- E nonlinear values, c_E, q_E−q0 and individual M messages;
- Thermal signed K contributions and unchanged source-column correspondence.

Require exact integer support/ID accounting. Each consumed E record must reach the E value function and have a live path to the field; a membership mask alone is insufficient. In a controlled nondegenerate numerical fixture, changing a selected E value with M states held fixed must change an M message and its Thermal coefficient through q_E.

At trained weights, report actual effect sizes and native-reference error changes, including null or harmful E-path effects. Do not set an artificial lower bound on A_E or force E use to obtain an attractive graph.

### 11.2 Same-weight controls with final-work matching

Use fixed inputs selected before control errors are read. Include:

1. Learned joint support.
2. Direct-score and geometry supports matched **per query and per type** to the final union counts, including protections.
3. Typed incidence rewiring that preserves mandatory/self records and valid per-type degrees; retain exact final-query work deltas if final-union equality cannot be achieved.
4. Fixed M list/dispatch with E membership changed, and fixed E list/dispatch with M membership changed.
5. E-value removal/replacement after preparation, with support held fixed, and exact restoration.

A graph-degree-preserving rewire is not automatically equal work after query protections. Reject invalid swaps rather than silently changing denominators or selecting a nicer query after seeing results. Use more than one predeclared editable seed where available so one fully occupied/busiest seed does not make the whole test unavailable.

Track interaction between M and E values: does changing the E part of a joint group alter which M records are useful, or only an unrelated final offset? Report vector/field changes and changes in physical-reference error separately. Restoration integer identities must be exact; floating-point restoration is assessed within its declared numerical envelope, not automatically called bitwise.

### 11.3 Layout adaptation, not two unrelated screenshots

On a bounded input-only panel of legal module position perturbations (for example 4 scenes per dataset and 2 small admissible shifts per scene), rebuild all geometry-derived M/E features, contextualization and routes. Keep original IDs corresponded. Query fixed receiver locations that remain valid; module-relative receiver roles require explicit coordinate/mask re-extraction.

Compare the same scene before/after perturbation: typed membership changes, query dispatch changes, consumed M/E IDs and field changes. Check source permutation and receiver-alone/chunk/batch independence separately.

These unlabelled geometry perturbations demonstrate **model adaptation**, not physical accuracy. Do not score them against the old unmodified target field. No solver is called for this panel.

### 11.4 Separate final reads from contextual ancestry

Conduct two distinct interventions:

- **Read-path intervention:** hold prepared token states fixed and alter final E/M values or list membership.
- **Original-input intervention:** change a legitimate prescribed input, recompute all derived features and preparation, then compare output and routing.

Dense M/E contextualization and scene-summary reinjection remain. Report that omitted final records can still influence selected contextualized records. A final native read list is not a causal blanket over raw physical inputs.

For a small fixed TRAIN panel, compare original-input sensitivities under normal rebuilt routing and fixed validated route IDs applied to newly rebuilt values. Use same-unit/standardized-per-feature changes and original native masks. Do not obtain “frozen-route” results by circumventing stale-state guards or detaching the preparation graph.

Raw changes to geometry-derived E features with X unchanged are off-manifold model interventions, unless a valid independent environmental input is actually available. Label them as such; they are not solved environmental counterfactuals.

### 11.5 Existing physical-reference response inventory and replay

Before any claim of physical interaction, inventory the **existing saved** response/design records. Identify baseline, endpoint, perturbation variable, original IDs, split exposure, native query support, units and whether the generator/provider really matches the current dataset. The current report explicitly verifies a limited Thermal heat-response replay; additional geometry or combined-action records must be found and verified, not assumed to exist.

Replay the available valid Thermal response families on D-M, D-ME, H and S at the same literal age. Preserve fluid, surface/interface, material, peaks and response signs/magnitudes separately. Record the rank of independent heating directions; opposite signed endpoints are not independent directions.

Where a genuine saved geometry perturbation exists, compare rebuilt-route and fixed-route model responses against that perturbed reference. Do not pool unmatched geometry/heat families or synthesize missing targets by interpolation.

For two interventions a,b, the finite interaction diagnostic is

\[
I_{ab}(q)=F(X+a+b,q)-F(X+a,q)-F(X+b,q)+F(X,q).
\]

Evaluate this against physics only when all four compatible saved reference states exist. A missing combined endpoint means this test is **unavailable**, not zero. For fixed-geometry Thermal heat amplitudes, I_ab should be approximately zero under the current generator; do not seek nonexistent heat–heat nonlinearity. Geometry–heat and genuinely varying environmental interventions are different questions.

If no independently varying environmental reference exists, explicitly conclude that **independent physical M–E interaction identification remains untested**, even if the typed value path and layout adaptation succeed. Produce a small inert future-data specification listing the missing factorial configurations; do not create that dataset in this round.

## 12. Spatial/geometry seams and numerical qualification

Round8 measured a Wind hard-dispatch jump of about 0.00381 m/s over 2.77e-5 D on one fixed switch. It did not resolve a literal Thermal native-grid crossing. Preserve those distinctions. [R8: Inverse]

The primary Round9 dispatcher remains hard. Direct E access does not make its support globally differentiable. Check input-selected seed-score crossings and geometry changes using D-ME/H/S at the same coordinates. Use multiple shrinking step sizes where the public API permits them, and separate native-grid sampling effects from core continuous-coordinate behavior. Test membership switches as well as query dispatch. No blanket smooth-inverse claim is allowed.

Numerical layers:

1. Exact parent state/output boundary when A_E=0 and all M access is original.
2. Independent small FP64 literal M/E loops and source-affine contraction, with forward and fixed-support VJPs.
3. Same trained weights, exact hard supports, dense versus flat typed execution on native roles.
4. Training-only relaxed/ST gradient law and omitted-M/omitted-E restorative direction.
5. Real physical-reference finite response, which is not implied by 1–4.

Use the existing TRAIN-scale research envelope as a starting engineering contract: normalized primitive RMS≤1e-5, derived curl/interface RMS≤1e-4, and numerical RMS≤1% of the same-fixture baseline physical RMSE where resolved. Retain max/relative errors and unchanged strict legacy comparisons separately. Seal any new independent-fixture conditioning tolerance before scientific fitting, never after seeing a trained miss. Do not demand bitwise equality across packed reduction orders. [E3]

Gradcheck the continuous primitives and fixed-support paths, not the discontinuous hard threshold against the biased ST derivative. Dense normalization must provide a derivative at a zero learned gate. Prove both omitted E and omitted M can receive restorative signals, including the E→q_E→M→Thermal-K chain.

Check heat-null and heat-independent routing/flow, promoted source contractions, JVP/VJP, zero-information-measure physical M sources, empty/padded types, correct bias placement and measure application exactly once. Recompute maxima at endpoints. Ordinary FP32 endpoint subtraction remains a separate numerical scope.

## 13. Conditional small design-use test

This is deliberately not a generative inverse campaign.

Only after H has comparable field fidelity, genuine typed sparse execution, meaningful E-path use and a measured memory benefit, replay a small **existing saved candidate set** for at most four verified layouts. Use its original constraints and physical objectives. Report candidate rankings, feasibility, reference regret and agreement/disagreement with D-ME and D-M. All candidates and reference endpoints already exist; do not generate missing targets or optimize a new geometry on an unqualified hard-route gradient.

A saved candidate ranking test may reuse exposed development data; label that explicitly. It is limited decision evidence, not independent design generalization or a certified generator. If compatible saved candidates do not exist, deliver an inert interface test and list the missing physical evidence rather than inventing a design result.

No new solver call, inverse optimizer, learned generator, optimization objective selected on DEV, or broad design search is authorized by this plan. The next generation study is a separate user decision.

## 14. Concrete coding work packages

### W0 — source/physics/provenance audit, short and bounded

- Inspect live branch and user changes without resetting or changing branches unexpectedly.
- Resolve and hash the exact D8 e2500 parents from retained receipts; preserve Round7/8 and classic sources.
- Pin actual source/provider/normalizer/query/LR identities and current devices/mounts.
- Audit the active original loss and exposure, separating inactive historical Stage-A/port/H6 paths.
- Write one compact dependency table: what each output/read/route may depend on, which rules are physics, which are current dataset choices, and which are architectural restrictions.

### W1 — typed record and graph contracts

Suggested new package: `src/honf_forward_core/joint_native/` with `contracts.py`, `graph.py`, `routing.py`, `budget.py`. These are proposed names.

Implement typed IDs, all-native seeds, four route calibrations, mandatory member floors, protected unions, binary sampling, exact stochastic/relaxed work distinction and corrected cadence. Extend actual EM/EE native score capture with opt-in compatibility tests. Avoid quadratic wide latent tensors; incidence can remain a small explicit tensor.

### W2 — dense A2-DE and reusable physical readout

Suggested `readers.py`, `model.py`, `response.py` in the new package. Add original E-specific nonlinear values, within-type weighting, zero-initialized q-conditioning projection and reused M/local heads. Use one shared primitive for ordinary/affine/cached outputs and derivative exports. No fake E heating columns.

Introduce the minimal default-preserving hook before local decoding, or factor the existing feature decoder cleanly; do not duplicate the full model or patch only the flow branch. Test parent exact boundary, E activation over real optimizer boundaries and gradients through E into K.

### W3 — sparse typed executor and memory

Implement dense same-mask literal/reference and flat typed selected backends. Gather before wide values, preserve duplicates/denominators, provide pure checkpoint-compatible primitives, lazy trace exports and per-type execution counters. Confirm old weak-reference lifecycle fixes remain active. A ragged native list is sufficient; no grouped kernel is required to start learning.

### W4 — trained sparse control and managed runtime

Add `S9-ME`'s direct gates over the same physical reader. Add new family/recipe/factory/child declarations, loader, standalone inference export and a thin managed CLI around TrainingEngine. Preserve all original parameter names/moments and add named E/router groups. Do not alter old Round8 gate or schema validators.

Suggested runtime files: `src/honf_runtime/joint_native_{protocol,factory,runtime}.py`; CLI `tools/joint_native_train.py`. Register the public evaluator with explicit dataset role/monitor handling. Suggested commands: `prepare`, `fit`, `status`, `resume`, `evaluate`, `export`; only implement genuinely useful surfaces rather than a second orchestration framework.

### W5 — real matched learning

Run the stage sequence and all eight terminal histories in §8. Persist loss, hard/relaxed M/E work, E-path influence, dense coverage, actual updates/cases/queries, peak memory and resource state. Execute real reviews and finish the healthy 2,000-epoch sparse continuations. Tests and warm-up alone do not complete this package.

### W6 — physical/organizer evaluation and closeout

Produce native role boards, trained and same-weight controls, source/receiver invariance checks, layout-paired routing changes, original-input/read-path separation, saved physical-response replay and conditional saved-candidate ranking. Measure fair sparse memory before spending time on optional speed work.

Write one Round9 report, compact evidence index, reusable inference/training guide and inert future environmental-data specification. Commit/push code/config/tests/docs only; keep generated arrays/checkpoints ignored. Verify remote/local commit identity and preserve unrelated user changes.

## 15. Focused acceptance tests

The test count itself is not a research deliverable. Cover these mechanisms with compact fixtures and a small real-native subset.

| Family | Required checks |
|---|---|
| Parent and data | Exact original state migration and optimizer moments; common data/query hashes; current native loss terms/denominators; unchanged labels/frames/normalizers |
| Typed records | M/E ID collision avoidance; consistent permutation; variable counts/irregular coordinates; M-only/E-only/empty fixtures; masks, measures and mandatory floors |
| Value semantics | E values evaluated individually before reduction; E changes q_E and selected M/local messages; source-specific signed coefficients; no temperature bypass or fake E sources |
| Work algebra | Monte Carlo unit-noise work against analytical M/E marginals; exact mandatory/protection union; no duplicate rows; actual versus relaxed counts; direct sparse-control floors |
| Gradients | Nondead zero-A_E initialization; omitted E and M restoration; E→M→K derivative; selected fixed-support VJP; continuous primitive gradcheck; no masked-log zero-gate disconnection |
| Execution | Independent FP64 literal loop, dense/flat same-mask equality within envelope, actual value hooks, no selected wide dense rectangle or query-by-seed-pair tie tensor, lazy tracing and measured/unavailable counters |
| Sampling/cadence | Four-epoch balanced macro positions and exactly 25% primary case visits for 4/5-update datasets; response cadence separate; one scene draw across chunks; mid-phase resume exact |
| Lifetimes | Weak references and released graphs/trace lists; checkpoint recomputation does not redraw/increment; backward-consumed rejection; extended repeated real-objective memory check |
| Physics | Thermal heat-free routing/flow and affine K; native interpolation/curl/q proxy; recomputed peaks; Wind native U and aligned profile/residual conversion |
| Inference API | Receiver alone/permutation/chunk/mixed batch support identity; fresh-state counterfactual route API; stale owner/weights/geometry/normalizer rejection; target-poison no reads |
| Launch and release | No startup speed gate; protected GPU/storage checks precede allocation; standalone exported inference without ancestor/data files; exact old loaders remain readable |

For environment mass-splitting or duplicate-record tests, distinguish dense/fixed-support measure invariance from adaptive reselection. Do not demand an unproved global discretization invariant of a hard native-seed dispatcher; record any unsupported indistinguishable ties explicitly.

Use affected tests plus a limited legacy regression subset. Do not attempt to make every historical linter or unrelated experiment pass at the cost of the scientific runs. Report actual linter/test scope and pre-existing failures honestly.

## 16. Resource envelope and authorization boundaries

This is a **proposed new round envelope**, not a statement that Round8's closed reservation can be reused. Passing this plan to the coding agent as its execution goal requests the bounded work below, subject to actual device/source availability.

- Physical GPU2 only, UUID `GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`, unless the user explicitly changes authority. Recheck live processes/UUID. No killing, moving or appropriating unrelated jobs, including on GPU0/1.
- Proposed cap: **24 GPU-associated hours and 24 elapsed hours**, including at least 2 hours reserved for endpoint reporting/closeout. Main scientific budget: **81,000 updates**. At most **600 disposable native engineering updates**, including lifetime tests. No automatic extra scientific portfolio.
- Initial engineering should normally consume ≤4 hours; measure native A2-DE/H/S throughput and memory early and reserve all paired trajectories. Forecast from real complete-objective steps, not parameter count or a target-free forward. The E reader can be materially more expensive than M-only; old Round8 timings are not a sufficient forecast.
- If the full sequence cannot fit the envelope, stop at a matched declared milestone with intact checkpoints and an exact additional-time/update estimate. Do not shrink native data/queries, secretly omit S or call partial maturation complete. Ask for a new allowance before exceeding the cap.
- Use `/home/wanglz/miniconda3/envs/ModularDT/bin/python` only after verifying the installed stack. Preserve FP32, TF32 and deterministic settings; no framework upgrade or hidden mixed precision. Optional kernel work cannot consume the reserved learning/closeout budget.
- Keep at least max(8 GiB, 20% device memory) genuinely free, with a prospective preventive margin based on measured step peaks. Track monitor gaps; never claim unobserved memory safety. Clean resource pauses occur only at fully resumable boundaries.
- New payloads: verified `/data2/wanglz/ModularDT/diagnostics/<round9-owned-root>` and an ignored workspace symlink. Proposed limits: 40 GiB retained, 15 GiB temporary, 55 GiB peak new material, ≥30 GiB filesystem free headroom. Reuse source datasets and exact-key caches; no historical duplication/migration.
- Cleanup only verified round-owned disposable material with no open handles/dependencies. Preserve literal shared/terminal/best states, compact failures, core per-case arrays and enough training milestones for exact reconstruction. Do not spend disproportionate time hashing tens of thousands of unrelated historical files.
- No new reference solves: retain inherited solver accounting 326/326 with zero new attempts, subject to verifying the actual local ledger. No locked TEST target access. DEV is exposed validation. No full-TRAIN, new environment dataset or generative inverse run.

## 17. Report and figure contract

The first page must answer these questions in this order:

1. Were all planned scientific stages actually completed, with what inherited/new ages and exposure?
2. Did direct E access help the current data, and did E actually influence the physical output paths?
3. Did the joint hypergraph select both native types and preserve comparable fields against D-ME and D-M?
4. Did it add value over the **trained** direct sparse control and the same-weight final-count controls?
5. Did actual selected execution reduce fair memory, including the caveat from dense router steps?
6. What original-input/layout and physical-response evidence exists, and what remains unidentifiable?
7. What remains unsuitable for inverse geometry optimization or general changing-environment claims?

Use at most five main figure families:

- **Learning:** original physical terms, fixed native priority monitors, M/E hard versus relaxed work, dense exposure, E-conditioning amplitude/effect and memory history.
- **Fields:** shared-scale D-M/D-ME/H/S native predictions/errors, Thermal small-M surface/material/peak failures or improvements, Wind streamwise wakes and all-role tables.
- **Joint graph:** separate M→M/M→E/E→M/E→E incidence blocks; distinguish M/E seeds; two or three actual receiver read sets with E and M IDs; paired layout change. Distinguish learned, mandatory and protected records. Do not plot a pooled hyperedge as a physical value node.
- **Work/memory:** actual typed nonlinear rows, sparse/dense estimator/checkpoint work, fair allocated/reserved memory by scope and four-epoch cycle, complete latency without a mandatory speedup claim.
- **Interactions/decisions:** count-matched controls, E-to-M value effect, contextual-versus-final-read influence, saved reference responses, seams and conditional saved-candidate ranking.

Figures must use saved arrays tied to literal checkpoint, type catalogue, receiver/policy identity and native units. Keep masters plus necessary PNGs only; visually inspect every selected figure. No unreadable monolithic graph diagrams or hundreds of repeated diagnostic plots. Preserve numerical tables even for negative results.

Final classification fields should include:

```text
scientific_learning_completed
actual_direct_E_consumption
E_to_physical_response_coupling
direct_E_predictive_utility
joint_native_graph_adaptation
native_field_comparability
organizer_value_vs_trained_direct_control
actual_sparse_M_work
actual_sparse_E_work
selected_step_memory_gain
full_cadence_memory_scope
complete_latency_change
layout_response_evidence
independent_environment_physics_evidence
saved_design_ranking_scope
inverse_geometry_limitations
```

“E path active”, “hypergraph learned”, “memory reduced” and “physical interaction established” are not interchangeable results. Preserve negative and unavailable outcomes.

## 18. Explicit stop-list

Do not:

- Insert E only into a final flow head and call Thermal jointly decoded.
- Treat E records as heaters or enforce heat nonlinearity unsupported by the current dataset.
- Replace original M/E query-specific values with fixed H6 slots, pre-query pooled edges, moments, prototypes or a regular-cover intermediary.
- Add a frozen teacher, full partner predictor, cheap all-source completion, wake formula, learned native-ID embedding or error-dependent route.
- Force inference top-K after seeing errors or relax a historical numerical/physical gate to make an endpoint pass.
- Require inference/training speedup to admit a correct scientific fit; or claim memory savings from smaller batch/query/role scope.
- Reuse global-maximum grouped padding, count zero-masked dense values as sparse, or hide E work in unconsumed metadata.
- Change source losses, native units, current data semantics, normalizers or response objectives without a separately recorded and authorized scientific change.
- Infer causal physical edges from attention, source-ID labels, graph degrees or a same-weight deletion effect.
- Use original targets for unlabelled perturbed geometry, invent missing combined-response endpoints, or create new solver data under an “evaluation” label.
- Finish the goal with only infrastructure tests and tiny learning smokes while healthy matched runs remain feasible.

## 19. Source notes and external precedents

### Repository evidence

All repository links below are pinned to `6ecc1086454705dc791d9ea782dba0f68b459e55`. The local case/checkpoint references inside those files require the user's actual mounted data; their presence in Markdown is not evidence that the payload is committed.

- **[R8]** [Round8 complete report](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/docs/reports/HONF_Learned_Budgeted_Native_Hypergraph_Round8_Report.md), especially Predictor, Organizer, Execution, Native laws, Inverse and cost appendices.
- **[R8-index]** [Round8 evidence index](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/docs/reports/HONF_Learned_Budgeted_Native_Hypergraph_Round8_Evidence_Index.md).
- **[R8-guide]** [Implemented Round8 guide](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/docs/guides/_bk/20261029_Guide_Consolidation/02_Attention_and_Unified_Routing_Lineages/Learned_Budgeted_Native_Hypergraph.md).
- **[C1]** [Budgeted core](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/src/honf_forward_core/native_retrieval/budgeted_core.py), [graph](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/src/honf_forward_core/native_retrieval/budgeted_graph.py), [budget state](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/src/honf_forward_core/native_retrieval/budget_state.py).
- **[C2]** [Dense/flat/grouped reader](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/src/honf_forward_core/native_retrieval/budgeted_reader.py); global occupancy/source padding and group-level protection union are visible in `_grouped_native`.
- **[C3]** [Original reader and separate signed Thermal output](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/src/honf_forward_core/content_hypergraph/readers_v2.py), [typed contextualizer and capture](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/src/honf_forward_core/content_hypergraph/staged_attention_v2.py).
- **[C5]** [Native seed dispatch](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/src/honf_forward_core/native_retrieval/seed_dispatch.py), especially `_canonical_ranks` and the `tied_pairs` construction in `dispatch_native_seed`.
- **[C4]** [Round8 factory](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/src/honf_runtime/native_budgeted_factory.py), [runtime/loader](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/src/honf_runtime/native_budgeted_runtime.py), [protocol](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/src/honf_runtime/native_budgeted_protocol.py).
- **[P1]** [Thermal physical/data contract](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/Case_ThermalChannel/Dataset/PHYSICS_AND_DATA.md), [earlier source/dependency audit](https://github.com/cosmos2w/ModularDT/blob/6ecc1086454705dc791d9ea782dba0f68b459e55/HONF_Proj/docs/reports/_bk/20261006_154229Z_HONF_Dependency_Correct_Source_Audit.md). They distinguish fixed-step affine physics from saved converged-final numerical closure and limited excitation rank.

### External primary sources — precedents, not HONF evidence

- **[E1]** Hao et al., *GNOT: A General Neural Operator Transformer for Operator Learning*, ICML2023: https://proceedings.mlr.press/v202/hao23c.html . Heterogeneous-input operator learning is a relevant precedent; its architecture/results are not evidence for this proposed native joint directory.
- **[E2]** PyTorch2.6, activation checkpointing: https://docs.pytorch.org/docs/2.6/checkpoint.html . Used for explicit recomputation/lifetime semantics and the need to preserve the forward function/RNG during backward, not as a guarantee of a measured HONF memory saving.
- **[E3]** PyTorch2.6, numerical accuracy: https://docs.pytorch.org/docs/2.6/notes/numerical_accuracy.html . Different reduction/batching orders need not be bitwise identical; this does not excuse a changed physical field function.
- **[E4]** Martins and Astudillo, *From Softmax to Sparsemax*, ICML2016: https://proceedings.mlr.press/v48/martins16.html . A possible later sparse continuous-dispatch alternative, **not implemented or silently added in the primary Round9 experiment**. Sparse continuous dispatch would still need source-union, value and seam analysis.

**Main scientific question:** can individually expressive native M and E values be jointly organized by a learned input-adaptive hypergraph, retain useful physical responses, and reduce real memory relative to an equally matured dense ME reader? The required deliverable is the measured answer, including the limits imposed by present environmental data—not a promised speedup or an asserted causal graph.
