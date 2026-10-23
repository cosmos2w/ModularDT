# HONF next phase: active interaction organization and frozen-interface inverse reuse

**Working repository / branch:** `cosmos2w/ModularDT`, `agent/honf-core-next`  
**Reviewed source:** `bd56edf92a9998b59d15724b3b8cd2aa8c6ab6e5`  
**Planning date:** 28 September 2026  
**Execution mode:** one coordinated study, two parallel dataset lanes, two user-authorized GPUs.

This is a new research experiment, not an instruction to resume Run 2110 or overwrite any historical result. Inspect subsequent Git changes before implementation. Commit identities below document evidence, not a request to create another contract-freeze system.

---

## 0. Read this first: what this phase must answer

### The central goal

Learn a hypergraph-inspired organizer through physical-field reconstruction so that it encodes reusable module–environment interactions. Then test whether that **same learned organization**, without retraining it on inverse objectives, helps a conditional, stochastic modular-design task.

The three project objectives remain separate:

| Objective | This phase's question | What does not answer it |
|---|---|---|
| **Predictor** | Can organized fine computation reconstruct the relevant native fields and preserve useful responses? | Lower training loss, teacher agreement alone, or a large epoch count. |
| **Organizer** | Does a genuinely active, input-dependent interaction cover do useful work, with an appropriate case-dependent number of nonredundant packets? | A fixed full-access root, a forced K histogram, or a sparse mask on an inconsequential branch. |
| **Inverse system** | Does the frozen forward-learned interface improve a controlled conditional-generation/completion task? | Better forward error alone, diversity of invalid samples, or candidates checked only by the same surrogate that generated them. |

### Main change from the previous round

**Train useful sparse computations before asking an input-only selector to choose their grouping.**

The previous joint model was free to stay full access, and it did. This round explicitly measures a small information-capacity frontier. It learns nontrivial sparse computations at a declared capacity, then learns which receiver grouping each case needs. This prevents another nominally successful joint run whose selected hard organizer does nothing.

A work budget is an **experimental capacity setting**, not a claim that nature has that sparsity. K remains an outcome. A sparse model that cannot recover fidelity is an informative negative result; a full-access fallback is reported separately, not counted as sparse success.

### Deliverables

1. A shared, budget-conditioned packet learner and an independently trained direct-pair control.
2. Actual forward fitting and deterministic organization evaluation in **both** WindFarm and ThermalChannel.
3. A selected, fixed forward/organizer version per dataset, or a clearly identified best research-only version if fidelity remains inadequate.
4. A bounded frozen-interface conditional-generation/completion experiment, with matched no-organization controls and honest reference limits.
5. A report that begins with the predictor / organizer / inverse outcomes in plain English and includes readable measured visualizations. Section 18 makes this a standing reporting requirement.

Do not substitute a new pruning-penalty sweep, another algebra-only prototype, or a long forward-only continuation for these deliverables.

---

## 1. Evidence being carried forward

The following are source-derived observations, not new hypotheses.

**E1. Latest joint study:** `HONF_Proj/docs/reports/_bk/20260927_204904Z_HONF_Fidelity_Budgeted_Joint_Forward_Maturation_Study.md`.

- Both Run 2110 u1500 students improved the old WindFarm teacher in all five aggregate validation roles, but neither passed the transferred five-role guard uniformly. W-packet's selected hard plans remained full access on the audited endpoint rows; selected pair savings were zero. See the terminal and original-90-row review sections.
- On the saved endpoint cells, W-packet's selected-plan and same-student full-access predictions were identical. The difference from W-full was a difference in trained weights, not a demonstrated packet benefit.
- The native omitted-QE experiment demonstrated a missing restoration gradient in the old masked-log path, and an exact-hard-value / soft-organizer remedy. This is a learning-mechanism result, not a successful sparse endpoint.
- The Thermal expanded response fit improved training move responses but did not transfer their signs reliably to the Re90 panel; broad values also regressed.
- Compiled source-subset execution reduced some actual rows and memory, but did not establish a complete-call speed advantage.
- WindFarm targets are **stored OpenFOAM CFD velocities**. ThermalChannel's additional references are the **local analytic-wake/shared-grid thermal generator**. Earlier WindFarm `analytic-wake` strings were corrected in the latest source/report and must not be propagated.

**E2. Receiver-local study:** `HONF_Proj/docs/reports/_bk/20260927_101609Z_HONF_Receiver_Local_Interface_and_Decision_Recovery_Study.md`.

- An input-only organizer learned varying source supports and passed the Q1024 teacher criterion on 12/12 organizer-development directions, but reference error increased. The final K was constant at two.
- Only QE became sparse in the selected plans; other fine mechanisms and common branches still carried information.
- The physical-call reconciliation reported 300/320 standing attempts, leaving 20 at that time. Verify the current ledger before any new solve; do not infer a new allocation from this plan's GPU budget.

**E3. Historical controls:** mature Thermal Run1804 e4738, Run1502 e4794, Run1505, Run1507/1508, and WindFarm Run2103 e2475. Preserve them. Variable K in Run1505, universal closure in Run1507, and static source conditioning in Run1508 answer different questions.

### Deductions motivating the new experiment

For a parent and two children with receiver interpolation weight `a(q)`, access has the form

\[
w(q,s)=(1-g)m_p(s)+g[a(q)m_l(s)+(1-a(q))m_r(s)].
\]

If all three memberships are identical, then

\[
\partial w/\partial g=0.
\]

At a closed parent, the hard forward also does not exercise child-specific computations. Exact identity initialization is useful, but it does not itself teach a child to specialize. The earlier inactive-root observations are consistent with this difficulty; the equation is not a complete causal explanation of that training run.

Therefore, the new training curriculum deliberately exercises several **valid computational frontiers**, not only whichever untrained frontier the root gate initially chooses.

A second deduction is that a fidelity-constrained minimum-work objective always admits the incumbent full-access solution unless useful sparse alternatives become competitive. We will explicitly measure those alternatives rather than interpreting convergence to full access as progress on the organizer.

---

## 2. Scope and starting models

### WindFarm lane

- Main common initialization for grouped and direct arms: **Run2110 W-full exact u1500**.
- Frozen research references: the same W-full state, Run2110 W-packet u1500 under policy-free full access, and Run2103 e2475.
- W-full is the common initialization for a clean comparison, not a declaration that it beats W-packet in every role.
- Load native encoders, normalizers, fine readers, common head, coarse/local paths and source measures intact. No three-term head conversion.
- Use existing native layout-grouped splits. Retain the exclusions used by the latest joint study. Previously inspected validation is development evidence.

### ThermalChannel lane

- Main common initialization: **Run1804 selected e4738**, including its checkpoint-native local surrogate and predicted-port loop.
- Frozen comparator: **Run1502 selected e4794**. Use it for outcome context and, only if the primary initialization exposes a concrete issue, one bounded alternative pilot.
- Do not start from the response u200 weights that already failed transfer; preserve those as negative controls.
- Geometry, material-coordinate receivers, P0/P1/P2 contexts, and native output definitions must remain intact.

### Shared versus dataset-specific

Share the permission-budget logic, frontier representation, scorer interfaces, reporting, and reusable tests. Train separate physical and organizer weights per dataset. This phase does **not** claim zero-shot Thermal-to-Wind weight transfer.

ThermalChannel supplies multiphysics fields, interfaces, material temperatures, and bounded local-reference response checks. WindFarm supplies the larger 3-D stored-CFD field/composition test. Neither dataset must wait for the other to become perfect.

---

## 3. Research hypotheses and falsifiers

**H1 — Sparse sufficiency.** At a modest declared information budget, fine interactions can co-adapt to reconstruct the protected physical roles.

- Falsifier for the tested scope: both trained grouped and competent direct sparse models materially fail the same reference roles after the bounded remedies, while a matched full-access control remains sound.

**H2 — Shared organization.** Grouping receiver queries into shared source-access packets is useful beyond ungrouped pair selection.

- Evidence: a fidelity/work/learning-efficiency advantage, meaningful transferable packet behavior, or downstream reuse benefit against a competent matched direct control.
- Non-evidence: comparing against untrained geometry ranking, or comparing different trained weights without controlling access.

**H3 — Adaptive K.** Different input cases require different nonredundant frontiers under the same declared task-independent capacity setting.

- K may remain constant; report that result. Never add a K-variance bonus, arbitrary minimum K, or a desired histogram.
- Variation only across externally requested budgets is **budget adaptation**, not case adaptation. Report both separately.

**H4 — Reuse.** A frozen organizer learned before inverse training improves conditional generation/completion under matched downstream effort.

- Falsifier: matched no-organization or rewired controls perform as well or better, or any apparent gain depends on access to hidden target designs or on the organizer being retrained on the inverse task.

The study may succeed on one hypothesis and fail another. Its final conclusion must not compress these outcomes into a single pass/fail label.

---

## 4. Mathematical model: budget-conditioned shared interaction packets

Write design and operating context as `(d,c)`, and requested physical query as `q`:

\[
\widehat y(q;d,c,b)=F_\theta(q,d,c;H_\phi(d,c,b)).
\]

`b` is a small vector of experimental information budgets. The organizer never receives target fields, reference errors, inverse target observations, or an inverse objective during forward training or forward inference.

### 4.1 Typed, physically indexed packet

A packet contains

\[
e=(\tau,p,\mathcal R_e,\mathcal S_e,a_e,\text{source identities}),
\]

where `tau` is MM/ME/EM/QM/QE and `p` is the physical phase when relevant. The receiver cover uses an input-built tree. Source identities are physical module IDs or current-case environmental coordinates/adapter features; tensor positions are not cross-layout physical identities.

For a valid frontier `F`, let smooth receiver weights satisfy

\[
a_n(q)\geq0,\qquad \sum_{n\in F}a_n(q)=1.
\]

Use independently learned typed memberships `m^tau_ns`:

\[
w^\tau_{qs}=\sum_{n\in F}a_n(q)m^\tau_{ns}.
\]

The geometric candidate tree is an index, not a physical clustering result. The physically relevant learned object is the combination of receiver access, source permissions, phase, and actual field consequences.

### 4.2 Preserve fine interactions and collective response

Retain original fine source states, coordinates, source weights, pair-relative geometry and native nonlinear receiver updates. Evaluate each supported physical pair once after overlapping paths are deduplicated.

For environment attention, preserve one normalization over the unique source union:

\[
C_E(q)=
\frac{\sum_s \nu_s w_{qs}\exp(\ell_{qs})V_s}
{\sum_s \nu_s w_{qs}\exp(\ell_{qs})}.
\]

Empty-row behavior must follow a tested explicit convention. Do not silently restore a full environment as an unreported fallback. Keep additive message denominators and output biases exactly as in the native model.

Nonlinear updates after gathering multiple fine messages retain effective collective dependence. This does not prove an irreducible many-body law. An extra pooled group-value decoder is **not** part of this candidate.

### 4.3 Capacity is not K

For each targeted mechanism, define canonical valid-pair work on an input-only receiver catalogue:

\[
C_\tau(H)=\sum_{r,s}\omega_r\mathbf 1[w^\tau_{rs}>0]\,v_{rs}.
\]

`v_rs` excludes padding, invalid source measure, and MM self-pairs. Receiver weights `omega` are declared before targets are read. Module/environment preparation receiver sets are native; query-read catalogues use fixed physical anchors, independent of the requested output batch.

Impose an experimental cap

\[
C_\tau(H)\leq b_\tau C_\tau(H_{full}).
\]

This is a structural training constraint, not a physical-error certificate or an application latency guarantee. Also report unweighted unique pairs, raw paths, actual executor rows, and complete cost on the real query sets. Anchor-work compliance does not automatically imply compliance at every possible query distribution.

### 4.4 The initial budget panel

Use three declared settings rather than a penalty sweep:

- `b=1.00`: exact full-access reference and replay.
- `b=0.90`: primary modest-capacity experiment.
- `b=0.75`: a bounded stress point to measure the fidelity/capacity frontier.

Initially apply the cap to QE. Within the training-only pilot, choose **one additional consequential transport route**: MM or EM for WindFarm; MM or ME in Thermal P0/P1. Its choice uses native omission/restoration and loss-gradient evidence, not held inverse outcomes. Keep all other routes explicitly full access.

If a small-M mechanism cannot realize a precise fractional cap due to discrete pairs, report the actual attainable setting. Do not erase required validity conventions to manufacture 10% sparsity. Mechanisms with no eligible pairs are N/A.

The final primary comparison uses the same budget vector in grouped and direct arms. No claim of module-interaction organization is allowed if only QE was successfully changed.

### 4.5 Hard projection and soft optimization

Predict node-source logits using the existing input-only encodings plus relative receiver/source geometry and mechanism/phase labels. Add the budget as an input. Do not use a learned embedding of case ID or source-slot index.

For a frontier, choose the most permissive score threshold whose **compiled canonical unique-pair count** satisfies the cap. Bisection or a sorted threshold search is sufficient; the work is monotone in a shared source threshold. Retain valid sources according to logits, not a fixed geometric cutoff.

Do not assume a per-node top-k independently enforces the cover's unique-pair budget: overlap can increase the union. Count that union. Equal-score ties must have a declared permutation-respecting treatment; do not claim equivariance if an arbitrary tensor index breaks a tie. Identical quadrature atoms may be treated as a tied group without altering their individual physical weights.

The threshold is a detached structural decision during each hard forward. Use continuous probabilities around it for organizer gradients. This is an approximate topology derivative, not the derivative of a discrete projection.

The current hard-value/soft-organizer implementation remains the basis:

\[
y_h=F_\theta(d,c;H_h),\qquad
 y_s=F_{\operatorname{stopgrad}(\theta)}(d,c;H_s),
\]
\[
y_{train}=y_h+[y_s-\operatorname{stopgrad}(y_s)].
\]

The physical model and physical design-coordinate gradients come only from the hard branch; the soft shadow updates the organizer. Keep the native omitted-QE restoration test. Do not report a soft forward as the deployed physical prediction.

---

## 5. Learn useful frontiers before learning which frontier to choose

### 5.1 Bounded candidate family

Use the top three levels of the existing binary receiver tree: at most eight leaves, fifteen candidate nodes, and **26 possible complete cuts** for a fully populated depth-three binary tree. A cut contains exactly one selected ancestor for each leaf. Shallower/degenerate trees have fewer cuts.

This is a resource ceiling, not fixed effective K. Preserve smooth overlap access and the existing all-source split identity. Do not create a new global slot-merging tree or a high-capacity clustering solver.

### 5.2 Stage A: capacity-conditioned multi-frontier reconstruction

During the initial forward fitting stage, sample a valid frontier independently of the current split head. Exercise root, intermediate and leaf cuts so the model trains their actual hard computations.

Recommended initial curriculum:

1. First 50 updates: all-access behavior and physical adaptation; establish initialization parity.
2. Next 150 updates: introduce the 0.90 capacity setting across several sampled cuts.
3. Thereafter: 20% full replay, 60% primary 0.90, 20% stress 0.75, subject to the selected pilot recipe.

Sample cuts with reasonable coverage of small and larger frontiers, not simply uniformly over 26 cuts if this starves the root. Use the same physical cases, queries, budget schedule and update count for the independent direct model.

This scaffold is deliberately imposed during training. **It is not an adaptive-K result.** It addresses the untrained-child problem while preserving a clean later test of input-only selection.

### 5.3 Stage B: reconstruction-derived frontier evidence

After the candidate source scorers and physical student have learned, hold that snapshot fixed temporarily. On a training-only panel, evaluate the available cuts using the same stored reference queries per case.

For each cut, record:

- actual typed source support and canonical packet count;
- native reference errors by physical role;
- same-student full-access distortion;
- selected work and declared bypasses;
- whether its distinct packets actually change the action relative to a root source union.

Evaluate at most 26 cuts per case/budget. Start with 12 training cases per dataset and the primary budget; expand to at most 24 only if it changes the identification question. A second query set checks the selected cuts. No target field enters the runtime organizer.

Choose a target cut from the **measured fidelity/work Pareto set**, not from K alone. Among adequate cuts, prefer lower measured/canonical work and then fewer nonredundant packets. Preserve all inadequate and dominated observations as training diagnostics. If no cut is adequate, label that case `outside tested fidelity-capacity frontier`; do not label full access as a successful sparse target.

The adequacy comparison uses both reference error and the same student's full-access error. As an initial research allowance, use per-role RMSE no greater than `1.10 * same-student full-access RMSE + a measured numerical allowance`, and report the original retained incumbent error alongside it. This is a declared comparison threshold, not a physical safety certificate. Estimate numerical allowances from repeated/parity checks in the correct units, not from the desired outcome. Do not accept a deteriorated full-access student merely because its sparse ratio is small. The old global training-budget ratio is retained as a historical diagnostic, not silently equated with a per-layout engineering limit.

If no cut is adequate, the utility head can still learn the measured risks and rankings. Evaluate its least-risk **budget-feasible research plan**, explicitly flagged inadequate, as well as its deployment fallback. Do not return to `organizer target unavailable / zero updates` when measured candidate errors are available. The absence of an adequate cut is then a substantive fidelity-capacity result.

### 5.4 Stage C: amortize the choice and jointly refine

Train a small input-only frontier utility head over the candidate cuts. Reuse node/source embeddings; no dense forward or ground-truth probe may be called at inference to select a cut.

A practical implementation predicts per-role distortion for each cut, plus a relative utility/ranking score. Supervision is the measured Stage-B table. Treat these outputs as predictors, not certified error bounds.

At the selected budget, choose the cheapest cut predicted to meet the calibrated role tolerance. If none is predicted adequate, return an explicit `unsupported_at_budget` flag. A caller may then use full access as a separate fallback, but the fallback rate must be included in all sparse-result summaries.

Jointly refine physical and permission parameters under the **selected hard cuts**, with full replay retained. Permit one bounded train-only table refresh if the physical student has changed materially; do not build a permanent online oracle.

Evaluate hard K, physical roles and source sets throughout. A constant root subset, a constant two-packet cut, or all unsupported cases must remain visible outcomes.

### 5.5 Why this is not just another count penalty

The model cannot escape the primary sparse trial by silently returning every pair. Nor is it rewarded for collapsing all distinctions into one dense root. It must first learn which source information can be omitted at the declared capacity. K then reflects how sharing that information across receivers helps the reconstruction.

The experiment may show that direct selection is better or that only a mild compression is possible. Either outcome is more informative than another unchanged full-access endpoint.

---

## 6. Required controls and causal scope

### Formal forward controls

- **G: grouped packet student**, jointly trained as above.
- **P: independently trained direct-pair student**, same physical initialization, trainable scope, data, capacity settings and physical update budget. Its scorer operates on individual receivers and sources, not on G's logits. Extend `independent_direct_pair.py`; its current G2-label-only 200-update fit is not a competent task-trained control.
- **D: retained native full-access reference**, evaluated throughout. A small matched full-access continuation is allowed within the pilot reserve if G and P both regress for reasons unrelated to organization.

Do not expand this into a broad model zoo. The primary scientific comparison is G versus a competent P at matched fidelity and work. Report both the same canonical-budget comparison and a checkpoint-only direct comparison at G's actually measured per-mechanism pair counts on the evaluation queries. P selects sources using its own learned scores; borrowing G's count for that control must not borrow its memberships. Count P's independent scoring work. Because actual query distributions can differ from the canonical catalogue, equal budget labels alone do not prove equal realized work.

### Same-weight interventions

At selected G weights, compare:

1. learned hard cut;
2. the same case's root source union;
3. a geometry-only grouping with matched source/work scale;
4. a degree/size-matched rewiring of the learned packet membership;
5. full access.

Always recompute downstream numerical states. Fixed-weight interventions describe the current model's reliance on a route, not the outcome of retraining an alternative architecture.

### Bypasses and many-body meaning

Initially freeze coarse/local context builders so they cannot grow to compensate for pruning; report their existing influence. The field head and native fine interactions may adapt.

A packet is not a complete physical dependency graph while important nonlocal paths remain outside it. Test bypass ablations and sensitivity propagation on a bounded training panel. Do not delete established native branches solely to make the diagram look clean.

If the covered route is empirically inconsequential, broaden coverage to the consequential path in one bounded remedy or report the limitation. A few deleted QE pairs cannot stand in for module-interaction discovery.

For collective physical claims, use stored individual/joint perturbations only where their numerical discrepancy supports a comparison. A nonlinear maximum of independent module temperatures can create objective interaction without physical cross-module interaction; keep these distinct.

---

## 7. Physical training and data protocol

### Main training objective

The organizer is learned through field reconstruction, not inverse success labels:

\[
\mathcal L_{forward}
=\sum_r\lambda_r\,
\mathbb E_{q\sim\mu_r}\|\widehat y_r(q)-y_r(q)\|^2
+\lambda_{keep}\mathcal L_{full\ replay}.
\]

Use physical units or explicitly documented training-only scales. Distinguish volume integration from role-balanced objectives. Near-null changes are not divided by an arbitrarily tiny signal norm.

Do not reuse old G2 omissions as unquestionable current-student labels. Their role is historical initialization/control; the new labels come from current reconstruction evidence.

### WindFarm

- Fresh native query samples on every update, spanning the allowed training layouts and observed categorical directions.
- Use the existing role catalogue and native quadrature correctly; equal-layout and sampled-point-weighted errors are separate outputs.
- Primary training queries: total Q2048 initially, e.g. 1024 volume plus 256 in each of the four protected roles. Increase only if a measured fitting problem or memory headroom justifies it.
- Review with larger disjoint queries and predeclared whole-grid examples. Selected endpoint additionally replays the established 90-row development panel.
- Do not learn from previously excluded organizer-development layouts while continuing to call them held. Report teacher exposure separately from new student exposure.
- No invented yaw, power, AEP or continuous wind-direction labels.

### ThermalChannel

- Both G and P receive the same historical value replay and the same native predicted-port curriculum.
- Preserve field, interface and solid-temperature supervision and the original local-surrogate conventions. Verify the actual names/phase meanings in current code.
- Expand parameter scope to the fine interactions feeding P0/P1 as well as the existing nonlinear heads, following the demonstrated train-fit capability. Freeze the local surrogate initially.
- Use the available finite-response atlas as an equal-weight *additional forward task in both arms*, not as an inverse-specific training signal supplied only to G. Report value-only and response components separately.
- Use at most eight new reference attempts for a small train-only coverage remedy if the standing budget permits: select two training layouts and two contexts, then baseline/move pairs. Reuse exact existing outputs where available. Do not train on the previously held Re90 response families by relabeling them.
- Response endpoints use common-fluid masks or material coordinates as appropriate. Keep `q_normal` labelled as the generator's proxy.
- An empirically generator-supported fixed-geometry heat null is not imposed as a universal WindFarm or multiphysics law.

### Selection and evidence partitions

Use one declared selection rule per dataset based on forward/response fidelity and organization, before any inverse evaluation. Keep exact endpoints as well. Do not combine the accuracy of one checkpoint with the K or speed of another.

All existing repeatedly inspected cohorts are development. Reserve a small unused group partition only if one genuinely exists; otherwise explicitly report that external/generalization evidence is limited. Do not promise an untouched test by renaming files.

---

## 8. What counts as an active and useful organizer

Report these separately:

1. Candidate tree capacity.
2. Raw reached frontier nodes.
3. Source-bearing frontier nodes.
4. Canonical nonredundant packets per mechanism and phase.
5. Per-query positive packet support.
6. Exact unique physical source–receiver pairs.
7. Actual executor rows, padding and fallback use.
8. Coarse/local/uncovered information paths.

At the fixed primary budget, inspect variation within the same module-count stratum and across observed operating contexts. Do not attribute variation to physics when it is entirely due to a caller's different budget.

A useful organization should also withstand:

- physical module permutation and query order/chunk changes;
- environmental quadrature-atom splitting with conserved measures, where supported by the adapter;
- same-budget root-union and direct-pair comparisons;
- current-input versus population-mean organizer features;
- fixed-weight packet removal and source restoration with target-based error measurement;
- a limited design/operating sweep, with topology changes recorded rather than hidden.

The primary success target is **fidelity-preserving active organization with some evidence of useful conditionality**, not universal optimal K. A constant sparse K is partial progress and must be called partial progress.

---

## 9. Execution and differentiability

Use dense-masked hard computation for scientific fitting and exact reference comparisons. Use the compiled rectangular-subset path for measured sparse execution. Do not begin a custom CUDA/Triton project this round.

Compile fixed plan metadata once per current case and phase, not per output tile. Preserve existing stale-view/security checks at binding boundaries; do not remove them to improve timing.

Batch receiver blocks with shared source sets. Score source membership at packet nodes rather than every output query in G. Charge all scoring, preparation, common branches and regrouping to total inference cost. P's per-query scoring cost is also charged.

No cache may reuse continuous source states after design, heating, operating condition, or model weights change. A frozen combinatorial plan in a local inverse step does not freeze the physical state.

Differentiability checks must distinguish:

- hard-plan physical/design gradients at fixed connectivity;
- approximate organizer-training gradients;
- actual output changes at a learned plan switch.

A switch-free scan does not verify switch continuity. For local gradient optimization, use a documented frozen-connectivity inner region and reevaluate the graph outside that region. Conditional generation below is stochastic proposal generation and is not granted a global differentiability guarantee by these checks.

Measure a small matrix: Q64, Q1024, Q8192 and one large native read; two module-count strata; policy-free Dense, same-student full, hard grouped, and direct sparse. Three or five interleaved repetitions are enough for this bounded stage. Do not spend most of the study measuring variants already known to be much slower.

---

## 10. Freeze the learned interface and test actual reuse

### 10.1 Scope of the inverse experiment

This phase introduces a **small conditional stochastic design/completion pilot**. It is a first generative reuse test, not a full inverse-design product and not proof of global design optimality.

The organizer and forward model are selected and held fixed **before** inverse losses or inverse validation outcomes are used. The small inverse model may learn; the forward representation may not be fine-tuned to rescue a weak reuse result.

### 10.2 Thermal task: conditional heat-allocation generation

Given module geometry, operating conditions, a total heat budget and a sparse set of desired/observed temperature values, generate several nonnegative module heat allocations summing to the supplied total.

Use existing solved cases as supervised examples. The total is a legitimate supplied design requirement, not hidden information inferred from a reference during deployment. Module geometry is known; individual heats are hidden.

Represent heats through centered log-fractions:

\[
h_i=H_{tot}\,\operatorname{softmax}(z)_i,
\qquad \sum_i z_i=0.
\]

Obtain a fixed number of temperature observations from declared physical locations. Measurement coordinates may use known geometry in this task. Do not feed the unknown heat vector, an encoder computed with its clean hidden values, or the solved dense field as a concealed extra input.

Why begin here: the task uses existing multiphysics labels and avoids moving-grid ambiguity while still requiring conditional, potentially non-unique inverse generation. It does not establish layout optimization; retain a small stored position-response audit separately.

### 10.3 Wind task: conditional module-position completion

Given operating direction, domain, turbine count, visible turbine positions and sparse fixed-location native velocity observations, generate the positions of one or two hidden turbines. Increase the hidden fraction only after a usable small-task result.

Observed sensor locations must be chosen independently of hidden turbine positions. In particular, **do not condition on rotor-neighborhood query coordinates computed from the unknown clean layout**. Native coordinate snapping and known domain bounds must be documented.

Construct every candidate-dependent encoder, environment catalogue, receiver anchor and graph from the visible inputs plus the current candidate positions. Do not reuse a clean-layout plan or geometry-dependent cache from the supervised target. Respect the existing wind frame and categorical direction convention.

All hidden turbines of the same type are exchangeable. Use consistent set matching and randomly permute hidden-token assignments during training; do not exploit arbitrary turbine IDs.

The stored OpenFOAM field provides observations for the known clean design, not a CFD solution for each generated new layout. Geometric reconstruction, held-sensor consistency, and surrogate prediction consistency are different metrics.

### 10.4 Minimal stochastic inverse model

Prefer an existing compatible set-based inverse workflow after checking that it has the correct conditioning. Otherwise implement a small permutation-equivariant conditional denoiser, not a large new model family.

A standard noise objective is

\[
z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\mathcal L_{inv}=\mathbb E\|\epsilon_\psi(z_t,t,o,c,\mathcal I_\phi)-\epsilon\|^2.
\]

Use a standard diffusion schedule for training and at most 20 reverse sampling steps initially. Reuse existing code where available. For heat logits, center the noise in the zero-sum subspace. For position completion, keep visible coordinates fixed and report geometry repair/rejection separately. Projection or rejection changes the sampling procedure; do not claim calibrated posterior samples.

`o` contains supplied sparse observations, never an optimization oracle. `I_phi` is the frozen organizer interface evaluated from the **current noisy/candidate design**, not the hidden target. The denoiser may use a target-observation encoder, but no target signal is fed back into the forward organizer.

### 10.5 What is reused

Export source/receiver identities, typed memberships, receiver-access weights and frozen model-side embeddings through a small interface. In the inverse denoiser, use a shared set block to aggregate candidate module/environment features along the packet incidence. Aggregate observational features at their known sensor receivers and propagate them through the same physically indexed packet links.

The inverse model may perform latent aggregation; the forward physical reader still retains fine states. No expensive full physical field decode is required merely to obtain a graph at every noise step. Count actual organizer cost nonetheless.

If the learned graph is sparse only in QE while all paths relevant to module variables remain full, say so. The inverse reuse test may then have little useful module-level structure; do not invent a module–module adjacency from a pretty QE plot.

### 10.6 Matched inverse controls

Train two small denoisers with equal parameter count, same data, seed/noise stream, update budget and sampling count:

- **I-G:** frozen learned packet interface.
- **I-dense:** identical frozen embeddings but full interaction access in the inverse aggregation.

Additionally evaluate I-G with degree/packet-size-matched rewiring as a fixed-weight diagnostic. If an apparent graph gain emerges, a small separately trained rewired control is the first permissible follow-up within reserve. A from-scratch/dense-encoder inverse control is optional, not another mandatory large study.

The task was not used to train or select the forward organizer. This is the key reuse boundary.

### 10.7 Evaluation and limits

Use 8–12 development tasks per dataset, 8 samples per condition and identical random seeds. Report:

- observation and disjoint-observation consistency;
- valid-geometry / valid-heat fraction before and after repair;
- set reconstruction of hidden positions or heat-distribution error against the known design;
- diversity **among acceptable samples**, not raw spread of invalid samples;
- no-observation and changed-observation controls to show conditioning is used;
- performance with fewer observations or a different observed-channel subset not used for denoiser fitting;
- sample/organizer/full-forward call counts and complete generation time.

The original design is one feasible inverse solution, not necessarily unique. Distance to that design alone is not an inverse-quality criterion. Surrogate consistency alone is not reference validation. Independently retained dense predictors can reveal disagreement but are not new physical truth.

For Thermal, conditionally use at most 12 remaining reference attempts: one preselected candidate from each inverse arm on four tasks (8 calls), plus up to four declared numerical/refinement checks. Freeze model ranking before opening those reference outcomes. The benchmark's pressure limit remains a study constraint, not an engineering certificate. Geometry-fixed heat allocation has generator-specific flow behavior; label it correctly.

Physical-check candidates must be geometrically valid and handled as independently checked proposals, not autonomously accepted designs. A failed predictor can still be evaluated as a labelled negative control; do not advertise its generator as decision-ready.

If the current standing ledger leaves fewer calls, reduce the physical pilot **symmetrically** and report the smaller scope. New CFD solves for WindFarm are not authorized or assumed. A useful Wind result this round can be stored-case completion/generalization, not power or AEP gain.

Run the bounded inverse learning diagnostic even if the organizer remains a research-only candidate, provided inputs/outputs are valid. Mark failure of physical adequacy explicitly; do not promote or independently accept designs through an inadequate predictor. Lack of universal forward perfection must not indefinitely postpone testing reuse.

---

## 11. Implementation map

First inspect the current local code and reuse maintained paths. Suggested responsibilities, not mandatory filenames:

| Existing location | Required change |
|---|---|
| `src/honf_forward_core/interface_fields/adaptive_interaction_cover.py` | Enumerate bounded valid cuts; preserve access algebra; expose canonical packet count and budget support. |
| `.../input_cover_organizer.py` | Add budget conditioning, node-source scoring on exercised cuts, and a small frontier utility head. |
| `.../native_joint_shadow.py` | Reuse isolated organizer gradients; accept an explicit sampled/selected frontier and hard projected permissions. |
| `.../adaptive_cover_field.py` and `.../core.py` | Reuse typed fine reads and prepared execution; no new per-query Python metadata path. |
| `Case_WindFarm/src/windfarm/workflows/joint_forward.py` | Extend configuration for the new matched protocol, fresh queries and coverage. Do not duplicate its entire trainer. |
| `.../independent_direct_pair.py` | Turn the label-only bounded control into a task-trained, matched-capacity control; generalize spatial dimension through a small shared core if appropriate. |
| `Case_ThermalChannel/.../response_control/` | Reuse native absolute and response training, material receivers, value replay and per-module outputs. |
| Existing inverse workflows, or a small new shared module | Implement the frozen-interface conditioning and matched small denoisers. |
| Existing visualization/report utilities | Produce measured field, organization and inverse figures; avoid another experiment-specific framework. |

Expose a minimal `InteractionInterface` with source/receiver descriptors, typed permissions, frontier identity, numerical state version and evidence scope. A normal dataclass and existing plan binding are sufficient. Do not create a registry service, signature authority, or new artifact database.

Avoid long workflow functions tied to one old report directory, a particular old failed diagnostic, or a required historical optimizer count. Historical replay can keep those conditions; the new experiment should use explicit config arguments and ordinary tests.

Do not weaken existing security, checkpoint trust, artifact exclusion or pre-push behavior. Moving a mathematical check out of a repeated tile loop must preserve the check at the correct boundary.

---

## 12. Actual tests, not assertion-driven development

Focused tests should include:

1. Full-access initialization and typed-wrapper output/first-gradient parity.
2. Cut enumeration (26 for a full depth-three tree), complete receiver coverage, and identity when child masks copy their parent.
3. Canonical-budget projection with overlapping packets, MM diagonals, inactive modules, duplicates, and equal-score ties.
4. Exact hard training values and no soft-shadow gradient into physical weights/design inputs; nonzero restoration signal for a selected omitted source.
5. Fine pair deduplication, quadrature conservation, one environment softmax, and output-bias preservation.
6. Full native train steps for G and P on low/high module counts in both datasets.
7. Query order/chunk invariance of the input-built plan; no dependence on target fields or requested output batch for frontier selection.
8. Inverse no-leakage checks: poison hidden clean positions/heats and the clean graph cache; the candidate-conditioned path must remain unchanged when those forbidden objects are present but unused.
9. Frozen forward/organizer weights during inverse optimization; known design components stay fixed; masks and set matching work.
10. Figures load actual saved numerical arrays and agree with tabulated values.

Do not require bitwise equality where an intentionally different floating-point reduction order is used. Declare tolerances appropriate to dtype and scale. Test failures should identify a numerical/code issue; passing tests is not a scientific result.

---

## 13. Parallel execution on the two GPUs

The user has provided two GPUs. Do not assume their physical IDs are 0/1 or that historical `cuda:2` is the only permitted device.

- Resolve the two user-designated devices at startup using the existing environment and device inventory. Record the physical-to-logical mapping in the ordinary run metadata. Ask only if authorization/identity cannot be resolved from the environment.
- Assign **GPU-W** to WindFarm and **GPU-T** to ThermalChannel. Keep both dataset lanes progressing in parallel after shared code/tests are ready.
- Use independent processes, seeds, output directories and optimizer states. No distributed-data-parallel setup is needed for these bounded experiments.
- One model-training process per allocated device. Run matched arms sequentially or interleaved on that same device so they see the same hardware; do not compare an arm on a faster card with its control on a slower one.
- Avoid duplicate whole-dataset RAM caches. The native mmap sources are shared read-only. Bound the sum of role caches against host RAM, initially no more than 32 GiB total or one quarter of available RAM, whichever is smaller.
- Limit CPU workers to avoid starving the second lane or local reference jobs. Measure actual data-loading time before increasing worker counts.
- Do not edit shared runtime files beneath a running job. Finish a shared-code change, test it, then start/restart affected jobs through normal run machinery.
- Only the coordinator commits/pushes after merging the lanes' durable changes. Preserve unrelated GPU jobs and user files.

Parallel lanes are scientifically independent: a Thermal response failure is not a reason to stop Wind organization learning, and a Wind execution bottleneck is not a reason to skip Thermal inverse conditioning.

---

## 14. Computation envelope and reviews

### New-round budget

- **24 GPU-associated job-hours total**, summed over both devices, including failed GPU jobs, pilots, evaluation and generator fitting. Two simultaneous hours on two GPUs count as four GPU-associated hours.
- Initial allocation: 10 hours Wind, 10 hours Thermal, 4 hours shared reserve/report evaluation. Redistribute unused allowance without exceeding the total.
- **16,000 attempted optimizer calls total**, counted per optimizer step; a joint update of two optimizers counts twice.
- Wind primary G/P: at most **3,000 physical updates per arm**.
- Thermal primary G/P: at most **600 physical updates per arm** because its complete physical wrapper is more expensive.
- Frontier-supervision updates: at most **400 per dataset**.
- Small inverse denoisers: at most **800 updates per arm per dataset**, initially batch-scaled to the small task.
- At most **two remedy pilots per dataset**, 150 updates each, charged to the same total.
- Train-only native frontier/oracle evaluation: at most **2,048 complete candidate forwards per dataset**. Reuse exact saved computations when legitimate; never call reused data a new evaluation.
- At most **20 new local Thermal reference attempts**, and never more than the independently verified remaining standing allowance. Allocate up to 8 to training coverage and up to 12 to the selected inverse/reference checks.
- **Zero new WindFarm CFD solves** unless separately authorized with an available trusted execution path.
- No automatic 5,000-epoch continuation.

These are ceilings, not entitlements. Use the measured step cost to reserve time for both datasets' evaluations and report figures. Do not consume all time on the first training arm and omit its control.

### Reviews

Wind reviews: u100, u500, u1500, and endpoint up to u3000. Thermal reviews: u100, u300, and endpoint up to u600. Organizer and inverse reviews use their own actual update counters. Report native case/row coverage; do not call an optimizer update an epoch.

During early sparse fitting, a temporary accuracy loss is expected and is not by itself a reason to terminate before co-adaptation is tested. Stop or amend a recipe when measurements show a disconnected gradient, no ability to fit a small real batch, persistent divergence, or no improving fidelity/capacity trend over the bounded review interval.

Do not let a constant K alone stop the experiment: it is a negative adaptive-K outcome to retain and test against simpler controls. Do not let a nontrivial K alone justify continuation.

Reserve at least 20% of GPU time for reference/development checks and frozen-interface reuse. Reserve ordinary CPU/report time for visual inspection. No new monitoring daemon or external scheduling service is required.

---

## 15. Bounded remedies Codex is encouraged to try

Codex should solve ordinary problems rather than report the first failed invocation as a scientific limit. It may try up to two evidence-driven remedies per dataset within the pilot allocation.

| Observed problem | First bounded response | What not to do |
|---|---|---|
| Hard plan silently becomes full access | Inspect budget projection, exact supports and fallback flags; run a tiny real native case. | Increase an arbitrary sparsity penalty until something happens. |
| Child permissions never learn | Check that actual training cuts activate those children and their gradients reach source scorers. | Keep training a permanently closed root. |
| G and P both lose value fidelity | Verify normalizers/weights, reduce physical learning rate or broaden frozen-to-trainable scope once; check a matched full-access step. | Blame grouping before checking the common training change. |
| Only G fails at a matched budget | Check overlap work, source sharing and conditional cut scoring; try one deeper subtree only if a concrete spatial limitation appears. | Force a K minimum or erase reference roles. |
| Omitted sources do not reopen | Repeat the native restoration-gradient test; inspect threshold/shadow saturation. | Report a nonzero total gradient as sufficient evidence. |
| Inverse denoiser ignores observations | Test a small real conditional batch, scale observations using training statistics, and simplify conditioning once. | Use hidden target layouts or clean target graphs as inputs. |
| Generated layouts are invalid | Use explicit known geometry projection/rejection and count attempts; reduce diffusion noise/hidden count as a new labelled pilot. | Quietly discard failed samples or claim the repaired sampler is an exact posterior. |
| Sparse code is slow | Profile preparation versus reads; use batched rectangular subsets or dense-masked reference for training. | Begin another custom kernel project or exclude organizer cost from timing. |

Keep changes paired when required for interpretation. Do not try all listed remedies. When a scientifically hard issue remains, report the actual evidence, attempted remedies, resource use and what would distinguish the remaining hypotheses.

---

## 16. Claim levels and final decision logic

The final report should classify each dataset separately.

**Predictor success:** relevant role errors and response quantities remain competitive at a specified hard sparse capacity, with held/dev tails disclosed. Improving a teacher-normalized score is insufficient.

**Organizer success:** active input-dependent source selection, nonredundant packets that affect the predictor, and a demonstrated advantage or meaningful tradeoff over competent direct selection/root union. Case-dependent K is a separate required claim for full HONF success; fixed sparse K is partial progress.

**Inverse progress:** a controlled frozen-interface task uses the organizer beyond reconstruction. Better conditional completion with reference-supported checks is stronger than self-consistency; a complete new-layout physical-design or global-optimum claim is outside available Wind reference scope.

**Generalization:** distinguish query generalization, held layouts, operating contexts, observation/task transfer, and cross-dataset code/inductive-bias reuse. None automatically implies the others.

**Interpretability:** group overlays and native interventions can support an explanation of model computation. Physical causality or unique true hyperedges require evidence not supplied by appearance alone.

If G is good but P is equally good, retain the result as useful sparse prediction without proven grouping advantage. If the generator benefits only from the pretrained embeddings rather than adjacency, say that. If full access remains necessary at all tested tolerances, report the bounded negative result rather than inventing a successful graph.

---

## 17. Software process: simplify, do not add defensive infrastructure

Use existing Git, ordinary configs, native checkpoint/run tools, dataclasses, unit tests and explicit evidence labels. Do not introduce new cryptographic hashes, contract-freeze systems, baseline-snapshot frameworks, approval databases, event daemons, or hook gates.

Preserve existing trust/security controls, checkpoint behavior, and the repository's outgoing-artifact/pre-push rule. Never bypass or dilute them. Routine scientific heuristics belong in experiment summaries and configuration, not in new blocking infrastructure.

Blocking is appropriate for actual invalid data/device access, unsafe file overwrites, external solver authorization, destructive operations and publication/release boundaries. Scientific negative results should be measured and reported, not hidden behind an assertion that prevents the experiment from running.

Raw evidence, checkpoint files, generated figures and one-time renderers stay local and ignored under the existing rule. Commit durable implementation, reusable tests/configuration, and the report/captions. Do not upload a prohibited file and then delete it in another outgoing commit. Audit the outgoing range and push the non-default branch through the existing hook.

The present plan changes future experiment semantics; it does not rewrite previous code outcomes or historical reports.

---

## 18. Permanent reporting requirement: keep the project understandable

The user explicitly requires these additions in **every future Codex final report**. Append a concise reporting subsection to `AGENTS.md` or its existing project reporting guidance, preserving every current upload/security instruction. Do not add a new enforcement hook.

### 18.1 The first page: predictor, organizer, inverse

Begin the report with:

1. **One plain-language paragraph answering what changed this round.**
2. The following table, separately for both datasets where outcomes differ:

| Central goal | What we gained | What we missed | Key measured evidence | What remains next |
|---|---|---|---|---|
| Predictor | ... | ... | At most 2–3 key numbers | ... |
| Organizer | ... | ... | Actual hard K/support and a consequence/control | ... |
| Inverse system | ... | ... | Matched conditional/design result and reference level | ... |

3. **One explicit overall judgment:** not a single mixed pass flag, but which of the three claims is established, partial, negative or untested.
4. **A short visual reading guide.** Identify which figures are the main evidence and how to read them.

The detailed quantitative tables, per-case results, resource use, failures and mathematical checks remain in the full report. The plain-language summary complements them; it does not replace or sanitize them.

### 18.2 Required measured figures

Produce approximately 6–8 readable main figures, with dataset panels where useful. A negative outcome still needs honest figures.

**Figure A — Three-goal overview.** A simple predictor/organizer/inverse status diagram with links to evidence. This may be schematic and must be labelled schematic.

**Figure B — Physical fields.** Reference, same-student full, learned grouped, direct sparse and residuals on identical actual coordinates. Include one representative and one difficult example per dataset. Wind uses stored CFD velocity slices; Thermal uses the documented fluid/material fields. Shared scales, geometry overlays, units and clipping disclosures are mandatory.

**Figure C — Actual organization in physical space.** Show receiver regions, selected physical sources, typed mechanism, overlap, canonical K and source degree. Compare same-M different layouts or directions, plus a selected transition along a valid input change if available. All-access or constant-K outcomes must be plotted plainly.

**Figure D — Fidelity–capacity tradeoff.** Plot reference role error against exact selected pairs, actual executor work and/or total latency. Include G, competent P, full access and root union. Show the primary budget and stress point; do not mix metrics from different checkpoints in one purported operating point.

**Figure E — Response/intervention evidence.** Predicted versus reference finite changes with units and numerical-discrepancy information where known, and a spatial before/after or source-restoration panel. Unknown numerical resolution is marked unknown, not a zero bar.

**Figure F — Generative inverse samples and trails.** For Thermal, show actual generated heat allocations and conditioning values; do not draw fictitious position arrows. For Wind, show visible/hidden/generated turbines, several actual denoising/proposal stages and geometric constraints. These trajectories are sampling trails, not necessarily monotonic optimization progress.

**Figure G — Inverse quality and failures.** Matched conditional results, valid-sample diversity, predicted/reference outcomes for the physically checked Thermal samples, and at least one failure or disagreement. Wind surrogate-only checks are visibly separated from stored-reference measurements.

**Optional Figure H — Cost or graph intervention decomposition** only if it answers an unresolved question that the main figures cannot show.

### 18.3 Figure integrity

- Use real saved arrays or measured model runs. Never manufacture field maps from aggregated errors.
- Distinguish raw native samples, spatial-bin averages and interpolation. Do not claim a sampled plane is a complete-grid validation.
- Clip only the display, retain unclipped numerical errors, and state the clipped fraction/range.
- Use captions stating: what is shown; what supports the conclusion; what it does not establish.
- Identify train/development/test exposure, checkpoint, physical source and same-weight versus separately trained comparisons.
- Latent group colors and IDs are local unless an explicit matching was performed. Environment tensor order is not a spatial axis.
- Plot actual failed/invalid generated samples or their counts. Do not display only the best eight samples and call that the distribution.
- Choose representatives by predeclared geometry/strata plus an explicitly labelled worst-error diagnostic, not hidden cherry-picking.
- Visually inspect the final images at readable size. Fix cropped labels, tiny captions and mismatched color scales.

Provide a locally viewable overview document and relative links from the committed Markdown report to ignored local figures. External reviewers without the local images still need informative captions and the key numbers. Do not change artifact-upload rules to make images appear in Git.

---

## 19. Definition of done

The phase is complete when the budgeted experiment has produced, for **both** datasets:

- a tested shared packet/direct training route;
- actual physical training and data-coverage records;
- deterministic selected support, canonical K, physical role errors and controls;
- an honest answer about grouping versus direct selection;
- an explicitly selected frozen forward/organizer research version;
- a matched bounded inverse reuse pilot or a concrete implementation/data impossibility established by the permitted remedies;
- measured cost at representative use scopes;
- a first-page three-goal explanation and readable physical/organization/inverse visualizations;
- committed durable code/config/tests/report and a verified push through the existing artifact rules.

A successful outcome is not guaranteed. The required progress is a decisive comparison that can distinguish:

\[
\text{insufficient physical model}
\quad\text{vs}\quad
\text{insufficient sparse representation}
\quad\text{vs}\quad
\text{organization that does not transfer to inverse use}.
\]

Do not end another round with only a full-access field improvement and a nominal organizer whose selected hard computation is unchanged.

---

## 20. Methodological references and scope of the proposal

The combined design above is a new HONF experiment. The following sources motivate individual ingredients; none proves the proposed organizer, sparsity level, physical interpretation or inverse advantage.

**[R1] Kipf et al., Neural Relational Inference for Interacting Systems, ICML 2018.** Reconstruction through an explicitly inferred interaction graph is a precedent for structured representation learning. Their physical-dynamics examples do not identify the correct hypergraph in these steady-field datasets. Official paper page: `https://proceedings.mlr.press/v80/kipf18a.html`.

**[R2] Louizos et al., Learning Sparse Neural Networks through L0 Regularization, ICLR 2018.** Demonstrates trainable sparsity mechanisms. It does not guarantee conditional graph diversity or speed. The primary new capacity experiment is a hard-budget comparison, not another unconstrained expected-K objective. Paper: `https://arxiv.org/abs/1712.01312`.

**[R3] Ho et al., Denoising Diffusion Probabilistic Models, NeurIPS 2020.** Supplies the standard denoising generative framework. A small conditional modular completion model with geometry projection requires its own evaluation and is not automatically a calibrated physical posterior. Paper: `https://arxiv.org/abs/2006.11239`.

**[R4] Luo et al., Efficient PDE-Constrained Optimization under High-Dimensional Uncertainty Using Derivative-Informed Neural Operators.** Motivates distinguishing state and design-response approximation. Its results and speedups do not transfer to the current local generator or unverified Wind design derivatives. Paper: `https://arxiv.org/abs/2305.20053`.

Read only the implementation documentation needed by the actual environment. Do not upgrade PyTorch/CUDA or add a new numerical stack just to follow a paper. Check existing installed capabilities first.
