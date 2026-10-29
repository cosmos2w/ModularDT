# HONF Round13-5 supplement: consistent integration, measured latency and organization-conditioned member reading

**Planning date:** 27 October 2026. **Deliverable:** a bounded implementation and experimental campaign with two separately named formal candidates, A and B. This planning document launches no experiment. It builds on R13-4's central idea—relationships learned through useful interaction govern access to original records—and reuses the working four-block M/E-address implementation and completed R13-5 evidence. It does not roll the code back to M-only addressing. Preserve the existing [R13-5 plan](20261027_010044Z_HONF_Round13-5_Joint_ME_Organization_and_Member_Routing_Work_Plan.md), its user edits, [execution prompt](https://github.com/cosmos2w/ModularDT/blob/466a20590966c3f481a6a8743641aa165ecafb26/HONF_Proj/UpgradePlan/HONF_Round13-5_Execution_Prompt.md), [completed report](../../../docs/reports/_bk/20261029_R13_Consolidation/20261027_103711Z_HONF_Round13-5_Joint_ME_Organization_and_Member_Routing_Report.md), historical models and active jobs.

The existing `thermal_r13_5_me_joint95_train600_formal_v1` and its separately authorized 100→500 continuation are outside this supplement. Do not inspect its progress, latest alias or changing logs, assign an observer, alter its configuration, interrupt it or use its live state as a source. The old named-run continuation authorization and old campaign allowance do not transfer to new candidates. New A/B full-data lineages each end at epoch 100 unless the user explicitly requests an extension.

## 1. Scientific purpose and lessons to retain

HONF should learn structured information access and integration: information is read for a traceable reason, useful original detail remains available, and execution avoids unnecessary work. A group describes cooperating members, relations and permitted operations. Its definition is not a pooled vector that replaces its members. Query-dependent integration after organization-constrained member access is a legitimate primary path.

The scientific priority is useful organization together with physical-field reconstruction. Within the execution objective, complete latency is more important than allocator-reserved memory. Improve integration by making its equations consistent, and improve execution by measuring actual repeated work. Do not compensate for ambiguous organization with extra physical heads, correction branches, controllers or progressively more elaborate decoding.

| Existing lesson or measured result | Consequence for this supplement |
| --- | --- |
| A2-D established the value of contextual attention; U-H exposed differentiated, overlapping relationships worth organizing. | Retain actual interaction and reuse its live relations; do not restart from independent records or force geometric neighborhoods. |
| R12-3/R13 treated compressed group values as the obligatory information source and lost member detail. | Original M/E records remain available through organization. Shared summaries are optional operations, never the default exclusive value source. |
| Original R13-3 hard-null selection could disconnect input-dependent learning while a coordinate-driven head reduced loss. | Diagnose native learning paths early; a falling total loss cannot excuse widespread loss of scene/control dependence. |
| R13-4 moved useful member admission before expensive member scoring, but its independent e500 regressions also existed under full access. | Investigate the predictor and integration law as well as shortlist approximation; do not blame every physical error on sparsity. |
| R13-5 e100 improved all nine global VAL89 mean/p90/worst summaries, and working E addresses removed the M1 fixed-prior restriction. | Reuse this implementation and evidence. Do not repeat the old M-versus-ME campaign or remove E views. |
| Joint95 followed by separate type-normalized reads can amplify a tiny surviving E tail; the saved native stencil diagnosis located a real derivative-extraction error. | Carry joint contribution mass through final integration and examine full/restricted differences at actual native stencils. |
| About 80.180→7.224 member scores/query still left about 197.9 address scores/query; complete restricted calls were slower. | Start from end-to-end timing evidence, not nonzero counts. Address work, composition, support, indexing and backward replay all remain charged. |
| The current decoder uses only the member marginal after P=πR. | Candidate B should retain organization identity through a set-dependent read, without replacing original values or adding a new decoder network by default. |

Use the [R13-4 plan](20261026_171634Z_HONF_Round13-4_Relation_Guided_Member_Access_Work_Plan.md), [R13-4 report and e500 supplement](../../../docs/reports/_bk/20261029_R13_Consolidation/20261026_185102Z_HONF_Round13-4_Relation_Guided_Member_Access_Report.md), [R13-3 corrective report](../../../docs/reports/_bk/20261029_R13_Consolidation/20261026_044023Z_HONF_Round13-3_Corrective_Relative_Member_Routing_Report.md), [current R13-5 report](../../../docs/reports/_bk/20261029_R13_Consolidation/20261027_103711Z_HONF_Round13-5_Joint_ME_Organization_and_Member_Routing_Report.md), [development protocol](../../../docs/guides/Thermal_Model_Development_Protocol.md) and [standing instructions](../../../../AGENTS.md). Reuse saved measurements when checkpoint, equation, inputs and measurement conditions match. Earlier evidence is context, not a reason to rerun a portfolio.

## 2. Implementation scope and interfaces

Implement the supplement in the current non-default branch, preserving later user work. Keep the existing case adapter, runner, physical losses, record preparation, prescribed-control ownership and general joint physical-field head. Initially retain the two ordered MM→EM→EE→ME cycles, model width, precision and member inventory; changes to these are separate experiments, not hidden latency optimizations.

| Maintained source | Relevant work |
| --- | --- |
| [joint.py](https://github.com/cosmos2w/ModularDT/blob/7981c173e8b1d29ee2b3addb7b972d0f15003542/HONF_Proj/src/honf_forward_core/interaction_reused_member_routing/joint.py) | Four-block H, whole-row retrieval, M/E proposals, joint member prefix, integration and work counts |
| [guided.py](https://github.com/cosmos2w/ModularDT/blob/7981c173e8b1d29ee2b3addb7b972d0f15003542/HONF_Proj/src/honf_forward_core/interaction_reused_member_routing/guided.py) | Prior-weighted score calibration and explicit read-mode capture for checkpoint replay |
| [relative.py](https://github.com/cosmos2w/ModularDT/blob/7981c173e8b1d29ee2b3addb7b972d0f15003542/HONF_Proj/src/honf_forward_core/interaction_reused_member_routing/relative.py) | Feature normalization, selected-member scoring and smooth bounded read strength |
| [core.py](../../../src/honf_forward_core/interaction_reused_member_routing/core.py) | Applied relation policy, original values, pair scorer and tiled checkpoint execution |
| [staged_attention_v2.py](../../../src/honf_forward_core/content_hypergraph/staged_attention_v2.py) | Actual directed exchange and stage provenance |
| [joint_member_routing_task.py](../../../Case_ThermalChannel/src/channelthermal/training/joint_member_routing_task.py) | Shared native preparation, paired full/restricted execution and physical objectives |
| [joint_member_routing_evaluation.py](https://github.com/cosmos2w/ModularDT/blob/7981c173e8b1d29ee2b3addb7b972d0f15003542/HONF_Proj/src/honf_runtime/joint_member_routing_evaluation.py) | Fixed-panel native evaluation and fieldwise evidence |

Keep the public prepare/decode and case-adapter interfaces small. Encapsulate scene-local reusable tensors and query-local common planning behind the existing module; do not make every optimization a caller-visible switch. Use private interfaces only where behavior actually differs between A and B. No new training engine, generic admission service, plugin framework or speculative abstraction is needed. Final configurations identify one chosen equation per candidate; diagnostic alternatives are not an accumulating production option matrix.

### 2.1 Preserve actual relationships and member identity

Keep the live final-cycle applied gates and their original stage, receiver and donor identities:

\[
H_{\mathrm{exchange}}=
\begin{bmatrix}
G_{M\leftarrow M}&G_{M\leftarrow E}\\
G_{E\leftarrow M}&G_{E\leftarrow E}
\end{bmatrix},\qquad
H=H_{\mathrm{exchange}}+\operatorname{diag}(I_M,0_E).
\]

Retain the current explicit M self-access ledger and captured EE diagonal; do not invent an E identity edge or reinterpret reverse lookup as a reverse exchange. These scalar applied gates are stage-labelled relationships, not the complete head-specific message transport or an instantaneous physical influence matrix. Downstream retrieval normalization is a declared read operation; it did not itself run in the encoder. New task gradients must reach the actual applied exchange tensors through the prepared state.

Use `(role, original_id)` for all ties, overlaps and execution indices. Preserve coordinates, material/geometric measures and controls. Coalesce repeated contributions onto an original member before value gathering/reduction. Empty roles return zero context; no compulsory type winner, fixed-zero null, unorganized raw-value branch or nearest-neighbor fallback is introduced.

## 3. Measure latency before changing execution

Use a disposable model loaded from a fixed saved checkpoint, never the active run. Begin with established TRAIN0006/M1 and TRAIN0231/M12 and their existing native query recipe. Use the exact same device, FP32, model state, query tile, masks and inputs for each old/new execution comparison. Start with one warmup and five alternating measured calls per full/restricted mode; report median and spread. Add repetitions only when observed variability prevents resolving the proposed change. Do not expand this into a device, precision, tile-size and compiler sweep.

Measure ordinary complete execution first. Separately capture one bounded instrumented forward/backward trace per low/high-M scene, and one representative mixed-M training batch using the actual batch/microbatch and Q1024 training recipe. An 8192-query single-scene backward replay is not a substitute for that training batch. Profile selected frozen numerical states without modifying their saved parameters or optimizer history.

| Measurement layer | Required attribution |
| --- | --- |
| Ordinary inference | Host input/cache retrieval, transfer, scene/context preparation, decoding, native extraction and complete call |
| Decoder attribution | Query encoding, address scoring, relation composition/support, prefix sort, gather/scatter, selected-member scoring, value reduction and joint head |
| Ordinary training | Data/transfer, preparation, paired forward, physical loss construction, backward including checkpoint replay, and complete training-batch time |
| Resource evidence | Actual address/member rows, relation contractions, sort/index work, replay calls, allocated and reserved memory separately |

Use synchronized wall timing at coarse nonoverlapping regions and CUDA events or a short profiler trace for kernel attribution. Inclusive nested profiler spans overlap: do not sum them or turn them into percentages of an unrelated uninstrumented total. Do not synchronize after every small operation during ordinary timing. Identify host/device synchronization from counters and indexing as well as GPU kernels. Repeat the ordinary complete call after removing instrumentation and after each retained optimization.

Record hardware, shapes, dtype, batch/query policy, warmup, repetition variability and contention. Use available independent GPUs for development, preferably physical GPU0 and GPU1 when free; preserve existing jobs and declared device mappings. Do not run a competing arm on the same device while claiming isolated timing. A GPU ownership check for a new launch is not permission to monitor the existing formal continuation. Follow the host's existing blocked-device rule, without creating a reservation system.

### 3.1 Equivalent execution improvements, chosen by measured cost

Apply and validate one rewrite at a time. Retain the changes that reduce measured complete inference or training latency without changing the scientific equation. Fewer intermediate operations alone do not establish a gain.

**Scene-local normalized keys:** the current joint address scorer applies per-record LayerNorm after expanding keys across query-address pairs. Normalize original keys once per preparation, keep their live autograd path, and gather afterward. Reuse normalized query features per tile. Never persist graph-bearing keys across parameter changes, detach them or serialize them as a learned cache.

**First-affine factoring:** reuse the existing scorer weights and state identity while computing `Wq*q + Wk*k + Wg*geometry + bias`. Query and key affine terms can be reused; pair geometry, distance/asinh operations, bias placement and subsequent SiLU remain exact. Avoid large concatenated repeated features. Floating-point reassociation can affect near-threshold ties, so compare candidate IDs and prefix-boundary margins as well as numerical outputs. More small kernels may lose to a large GEMM; measure rather than mandate this rewrite.

**Structural reuse and deferred counts:** cache typed-ID ordering and genuinely scene-only base measures. Reuse reachable support only when address support is query-independent and numerically positive. Fixed-π interventions, address cuts, masks and finite-precision zeros retain the general path. Aggregate actual counters on-device and transfer them at a coarse existing reporting point instead of repeatedly converting individual tensor sums per tile. Do not delete execution accounting or silently broaden the allowed support.

**Shared paired planning:** compute common query states, normalized features, address scores, π, P and eligible support once for each paired tile, then execute full and restricted member reads separately. Gradients from both physical objectives must accumulate through the shared computation. Do not compute every full-reader member score and reuse it to advertise early restricted selection. Preserve RNG behavior, query unions and native extraction. The existing checkpoint closure explicitly captures the resolved read mode; keep full/restricted semantics immutable during backward replay. Prefer tile-local graph lifetime over mutable model-level caches. Measure the latency/activation-memory tradeoff.

For each retained rewrite, compare π, P, typed candidate IDs, contexts, native fields, physical losses and gradients to original values, scorer parameters and actual four-block relations. Cover low/high M, padding, masked queries, deterministic ties and empty-type reads. Follow with a short real segmented learning/recovery check within the common budget. Do not fit a separate long model for each cache. If a rewrite changes scientific selection or RNG semantics beyond ordinary floating-point tolerance, classify and investigate it as a model change rather than an equivalent optimization.

## 4. Candidate A: consistent joint contribution with efficient execution

Candidate A combines the selected equivalent implementation improvements with a coherent integration law. First hold the existing encoder, proposal, physical objective and head fixed so the diagnosis is interpretable. Use saved native replays to separate relation calibration, score calibration and contribution normalization before fitting; do not change all three silently and attribute the result to one mechanism.

### 4.1 Two relation interpretations, with no new controller

Initially retain the current retrieval measure: present M records carry equal discrete mass summing to one; positive E cell measures sum to one within E. Physical areas/volumes and native integration weights remain unchanged. These are explicit retrieval conventions, not proof that M/E relationship scales are physically commensurate. Show the distribution of raw block strengths, row masses, M self access and resulting member-role masses, including M1.

Compare at most two interpretations in fixed-state TRAIN diagnostics:

\[
r_a=\sum_iH_{ai}\mu_i,\qquad R_{ai}=\frac{H_{ai}\mu_i}{r_a},\quad r_a>0.
\]

| Interpretation | Address mixture used for member access | Question |
| --- | --- | --- |
| Normalized views, existing default | \(w_a=\pi_a\) | Should an entrance select a relative coverage shape independently of row strength? |
| Retained row strength, existing raw-composition alternative | \(w_a=\pi_a r_a/\sum_b\pi_b r_b\) | Does retaining total learned relation strength improve useful access or instead exaggerate self/type dominance? |

Both use the same R and form `P = w R`. Preserve raw H and stage provenance in either case. The second interpretation is equivalent to normalized composition of `π H μ`; use the existing diagnostic path where possible. Reuse earlier identical replays rather than repeating them. Do not independently unit-normalize each M/E block, force role totals, add learned block controllers or sweep arbitrary constants. If diagnostics locate a concrete equation defect, repair it before qualification. Otherwise choose one explicit primary convention from physical losses, gradients, role masses and useful member access; instantaneous replays are sensitivity evidence, not trained performance claims.

Select one calibration for both A and B. If both are implemented correctly and evidence is mixed, retain normalized views as the simpler existing convention and report the unresolved scientific choice. No automatic extra full-data calibration arm is authorized. Pure-E frequency is measured, not targeted; an impossible pure-E prefix under a fixed directory is diagnosed algebraically rather than awaited through more epochs.

### 4.2 Preserve joint mass through final member contribution

Use one joint95 member prefix before expensive member scoring. Let C(q) be that set, `e_i = exp(ell_i)` the positive query-conditioned member likelihood, and t(i) the input role. Define:

\[
Z_t(q)=\sum_{i\in C_t(q)}P_i(q)e_i(q),\qquad Z(q)=\sum_tZ_t(q),
\]

\[
b_i^{A}(q)=\mathbf1_{i\in C(q)}\frac{P_i(q)e_i(q)}{Z(q)},\qquad
\beta_i^{A}(q)=\rho_{t(i)}(q)b_i^{A}(q),\qquad
c_t(q)=\sum_{i\in C_t(q)}\beta_i^{A}(q)V_i.
\]

Retain the existing bounded typed strengths ρ initially to isolate the lost-joint-mass repair. Equivalently, the old typed conditional context is multiplied by `Z_t/Z` before its strength. Add no new amplitude network. For either candidate X, compute gate features from its own pre-strength distribution: `p_i^t = b_i^X / sum_{j in t} b_j^X`, its conditional expectation of the declared score ell, and its entropy normalized by the unchanged scene-eligible role count. Empty roles have zero defined distribution/expectation/entropy and zero context. Preserve the existing normalized query input and bounded strength mapping. For bounded scores, values and strengths, a vanishing role mass has a vanishing contribution instead of becoming a full-strength singleton. A later no-fit gate ablation may identify unnecessary historical gates, but a new trained gate-removal arm is not part of this first matrix.

Evaluate the documented zero/one-E native stencil case, nearby query changes, low/high-M inputs and controlled empty-role fixtures. Separate errors already present with full access from errors added by admission. No hard cutoff on learned strength and no fixed-score null are restored. Keep one joint physical head using the typed contexts; their roles do not assign output physics.

### 4.3 Simplify candidate-dependent score calibration only when diagnosed

The current prior-weighted mean/variance and bounded transformation are recomputed over retained members, so deleting a member changes the effective scores of the survivors. First quantify this on the fixed full/restricted replay. Compare the inherited law with one candidate-independent bounded transform at the existing scale, for example `ell_i = 4*tanh(s_i/4)` after the existing feature normalization. This is a proposed changed equation, not an exact implementation rewrite. Evaluate score ranges, original-member gradients and native stencil effects before choosing it. Do not calculate all full-member scores just to normalize the restricted path, introduce separate M/E standardization that erases meaningful offsets, or launch a normalizer sweep.

Choose the simplest coherent score law before A formal fitting and use it unchanged in B. If a short qualification exposes instability, inspect the score law and gradients rather than adding clipping schedules, new losses or a control network by default. Report the intentional changes together and keep fixed-state ablations for attribution.

### 4.4 Training objective and selection

Retain paired physical training on the same prepared scene: one full organized read and one joint95 read, with half weight for each native physical loss. Full means every reachable eligible original member under H, never an unorganized bypass. Keep current physical roles, normalization, weights and queries, with one coherent schedule. No Thermal-specific heat-affine head, flow-invariance check, analytical correction or additional smoothness/physics penalty is added. Query continuity is examined in the appropriate physical region and at the existing native stencils; do not enforce smoothness across real material interfaces.

Candidate A is selected from correct, actively learning equations with understood field behavior and actual complete-cost evidence. A short segmented loss ranking is not a mature accuracy result. An initially slower correct implementation prompts hotspot diagnosis and targeted optimization; speed is measured, not a new binary admission threshold.

## 5. Candidate B: retain organization identity without another decoder network

Candidate B inherits A's selected retrieval calibration, score law, physical objective, head, member95 policy and equivalent execution improvements. Its first scientific difference is where member competition is normalized. Do not add an address-conditioned triple MLP, a pooled value bottleneck, explicit triples or an additional hypergraph encoder to this initial comparison.

With context fixed, the existing marginal reader treats any two directories giving the same `P = w R` identically. B keeps a view's membership active during competition. For the same pre-score member mask C used by A, define:

\[
d_a(q)=\sum_{i\in C(q)}R_{ai},\qquad
Z_a(q)=\sum_{i\in C(q)}R_{ai}e_i(q),\qquad
D(q)=\sum_aw_a(q)d_a(q).
\]

The retained mass of a view is `w_a d_a / D`. Its conditional member distribution is `R_ai e_i / Z_a` on C. Combine these before the original-value reduction:

\[
b_i^{B}(q)=\mathbf1_{i\in C(q)}e_i(q)
\sum_a\frac{w_a(q)d_a(q)}{D(q)Z_a(q)}R_{ai},\qquad
\beta_i^{B}(q)=\rho_{t(i)}(q)b_i^{B}(q).
\]

Terms from empty views are exactly zero. If the entire eligible directory or query is empty, return zero contexts with no division by zero. In ordinary nonempty joint95 execution, D is the actual retained prior mass. The d factor makes a nearly empty view fade with its retained mass; merely renormalizing proposals over nonempty views would recreate a discontinuous tiny-tail problem. Weighting proposals by Z instead would cancel the organization normalizers and recover A, defeating this experiment.

Retain A's typed strength module and compute its features from B's own pre-strength b distribution as defined above; do not silently borrow full-path or A-path statistics. The common b weights sum to one before the existing bounded strengths. Overlapping views contribute through their normalized proposal mass and coalesce onto original member IDs; they do not cause duplicate value reads or multiply total contribution by membership count. A singleton view cannot identify an internal member preference, which is a property to report rather than a reason to force its size.

For constant member likelihoods, B reduces exactly to P renormalized on C, matching A before the same strengths. Identical relation rows also make A and B coincide; inspect real row overlap rather than demanding an artificial difference or adding a diversity loss. For nonuniform likelihoods and differentiated rows, within-view competition depends on the other members of that view. This is the minimal set-dependent read being tested. It is not an assertion that irreducible physical many-body interactions have been discovered, nor a requirement that a physically additive response become nonlinear.

### 5.1 Execution and provenance of B

Compute each selected original member's likelihood once, shared across its memberships. Use tiled contractions for `d = C R^T`, `Z = (C*e) R^T`, and the final row-weighted mixture with R. These are three additional contractions relative to an already computed member prior; d and Z may share a stacked call, but their work remains charged. Full execution can reuse d=1 for normalized nonempty rows. Keep only query×address and query×member intermediates, never materialize a full query×address×member tensor. Reuse scene preparation and paired common plans while retaining correct checkpoint replay and autograd.

Use numerically stable positive likelihood scaling and masked normalization; zero eligible entries stay zero. Subtract one common per-query score offset across selected M+E members, or use equivalent log-domain sums. Independent per-role exponent offsets change joint competition unless restored. Compare a small explicit reference with the contraction implementation, including fields and gradients. Record B's extra directory work, latency and memory rather than implying that an elegant formula is free. If directory contractions dominate, pursue exact reuse/fusion at this seam before considering a more complicated scoring architecture.

The same raw applied H still governed actual contextual exchanges. B's new nonlinear row normalization is a read operator over that live organization, not a newly executed encoder exchange. Preserve phase-labelled exchange/self ledgers and trace native gradients to them. Do not detach or replace H with a post-hoc clustering merely to produce cleaner figures.

### 5.2 A decisive structural control

Hold query, context, keys, original values, w and member scores fixed. Use the controlled fixture:

\[
w=(1/2,1/2),\quad
R^{(1)}=\begin{bmatrix}1&0&0\\0&1/2&1/2\end{bmatrix},\quad
R^{(2)}=\begin{bmatrix}1/2&1/4&1/4\\1/2&1/4&1/4\end{bmatrix}.
\]

Both give `P=(1/2,1/4,1/4)`. With all members retained and `e=(1,2,4)`, A gives `(1/4,1/4,1/2)` for either directory; B gives `(1/2,1/6,1/3)` for the first and `(1/4,1/4,1/2)` for the second, before strengths. Constant e removes the difference. This arithmetic control checks that organization survives beyond the marginal; it is not a learned physical result.

Then use bounded fixed native queries to compare preserved and reassigned memberships at fixed context with matched support/work where feasible, explicitly recording any imperfect marginal matching. Compare final contributions and physical outputs, not just a latent norm. Separately remove an actual applied exchange and reprepare context; label this different intervention accurately. Typed-ID permutation, empty-view limits, overlap coalescing and duplicated-row mass-splitting controls should behave as their equations require. Duplicating a row without splitting its proposal mass is not an invariance test.

Do not enumerate every subset or fit an extra pairwise baseline. Optional joint-control finite differences on existing reference panels may illustrate cooperation, but ordinary nonlinear heads can also produce nonadditivity. Attribute the organization effect using the fixed-context structural controls. No solver campaign or extra response-fitting objective is authorized.

## 6. Sequential experiment and formal-run budget

All training duration, reporting and agent handoffs use epochs only. Use the fixed input-stratified TRAIN150/DEV22 manifest for execution debugging, with its TRAIN-only normalization. The aggregate segmented allowance is at most 500 epochs across A, B, calibration alternatives, repairs and recovery checks; it is a ceiling, not a target. Normally start with six epochs per selected candidate, resume through twelve only when persistence/recovery needs checking, and use up to twenty-four only for a concrete remaining learning-path question. These windows do not force a warm-up, freeze, merge permission or loss schedule.

| Order | Experiment | Scope and decision |
| --- | --- | --- |
| 1 | Unchanged fixed-checkpoint native profile and same-state consistency/calibration replays | Two timing scenes, at most four fixed TRAIN mechanism cases; reuse existing numerical evidence where identical |
| 2 | Equivalent execution rewrites, then chosen A equation | Small reference/parity checks and short real segmented qualification; no long fit per optimization |
| 3 | Select A and start its independent full-data run | At most 100 TRAIN600 epochs; keep a resumable endpoint and perform one complete endpoint review |
| 4 | Implement B using A's selected conventions and run structural controls | Short segmented qualification with the same membership and initialization family; no new broad search |
| 5 | Select B and start its independent full-data run | At most 100 TRAIN600 epochs, matched to A; do not leave a healthy qualified B as only a paper proposal |
| 6 | Compare saved endpoints, fields, organization and ordinary complete timing | Reuse fixed panels and unchanged artifacts, deliver both trajectories and tested continuation commands |

No old M-versus-ME pair is repeated. No new full-data development portfolio precedes these two selected models. At most one extra calibration/score alternative may receive segmented fitting if fixed-state diagnosis leaves a concrete ambiguity; it consumes the same ceiling and changes one mechanism at a time. Do not run a Cartesian product of row calibration, score law, strength gates, cutoffs and interaction topologies. A stable but broad or mixed early candidate proceeds to the selected full-data experiment; segmented fitting is not a substitute for that evidence.

Propose these unused formal names, checking for collision without overwriting any history:

| Candidate | Proposed run name | Initial population and horizon |
| --- | --- | --- |
| A, consistent integration with measured execution improvements | `thermal_r13_5a_consistent_joint95_train600_formal_v1` | Independent seed-0 TRAIN600, through epoch 100 |
| B, A plus organization-conditioned competition | `thermal_r13_5b_organization_conditioned_joint95_train600_formal_v1` | Independent seed-0 TRAIN600, through epoch 100 |

Use the next ordinary version suffix if a name is occupied. Place payloads under a new clearly named supplemental round root on DATA2 and expose ordinary workspace run symlinks. Neither new run resumes or overwrites `thermal_r13_5_me_joint95_train600_formal_v1`.

A and B use matched independent starts with compatible tensors identically initialized, no segmented or transferred source weights, one common full-TRAIN-only normalization, matching input order/query draws, native losses, batch48/microbatch24, FP32 and Q1024 unless a measured resource issue justifies an explicitly matched adjustment before fitting. Reuse the declared constant learning rate 0.0003 for the initial comparison rather than adding a new schedule experiment. If B has no new trainable tensors, initial state can be identical while its read equation differs. B development follows A's design selection, not a transfer of A100 weights. This makes their equal-age comparison interpretable.

The total new full-data fitting allowance is 200 epochs: at most 100 per selected formal candidate, including any attempted fitting in those lineages. A new scratch restart or extra model does not acquire another allowance automatically. Preserve failing states, repair real defects and requalify within scope; if the remaining authorized fitting cannot complete a replacement, report the exact recovery proposal rather than silently expanding. An ordinary exact resume retains its lineage and cumulative epoch count. No new 100→500 or 1000 continuation is authorized here. Deliver working user-controlled launch/resume commands; issue extensions only on later explicit instruction.

Once A's equation and implementation are selected, its full run may proceed on one available GPU while B is developed/qualified on another. Do not modify A's live implementation/configuration under its process; use the repository's existing isolated source/worktree or process-local import practices, without creating an artificial baseline-freeze system. Keep isolated timing separate from concurrent development throughput. Actual device identities, initialization and epochs are recorded per run; no DDP or silent batch/sampling changes are needed.

## 7. Early diagnosis, persistence and inexpensive observation

Use existing in-process health callbacks and ordinary clean-stop/recovery behavior. During initial qualification and the first early full-data epochs, collect compact fixed TRAIN low/high-M observations each epoch; later use existing review cadence. Preserve RNG, mode and provider state during diagnostics. Existing native training and health signals should surface early failures without a human repeatedly reading logs.

Watch native physical losses separately for full and restricted reads; original-value and active relation task gradients; actual control/scene dependence; proposal and member-role mass; pre/post-strength contributions; candidate IDs; raw row masses; retained/full preference and native stencil changes. For B also record retained row mass d, row normalization Z, overlap-coalesced member contributions and matched-marginal sensitivity. A legitimate empty role at one query, zero gradient in an absent block, a pure-E/M read or broad active support is not a collapse.

Nonfinite computation, wrong IDs/provenance, failed state recovery, a wired-out promised path or widespread loss of previously active scene/control dependence calls for an immediate clean stop of the affected owned run and retention of its numerical state. Diagnose the equation, score/normalization scale, self access, eligibility, checkpoint mode and gradients before continuing. The parent actively investigates instead of accepting a coordinate-only predictor or launching full training from an already failed segmented model. Stop the affected experiment, not unrelated jobs.

An initially worse local metric, noisy loss, broad overlap or absence of speedup calls for analysis and a bounded corrective experiment, not automatic abandonment. Use the remaining agreed budget to repair concrete mistakes. At epoch 100, report trajectories and mechanism evidence as preliminary, then recommend the next user-controlled continuation; do not declare an architecture mature or failed solely from that endpoint. Do not require every field to improve, every type to be selected, a target group count or a forced spatial pattern.

After every new segmented or full-data launch, delegate read-only observation to an explicitly selected available Luna model with minimal history: exact owned run, epoch stop, logs/checkpoints, normal failure/health signals and notification condition. One startup check measures actual epoch timing and estimates completion. The next routine inspection is near 80–90% of estimated remaining duration or an already necessary milestone. Luna stays quiet unless completion, failure, meaningful delay or user action becomes actionable; the parent does necessary independent work or waits and does not duplicate polling. Luna may handle separately scoped lightweight subtasks. No daemon, external polling framework, reservation layer or new monitoring service is introduced. These new runs use normal sparse observation through their 100 endpoints; the older observer-free formal continuation remains untouched.

## 8. Evaluation and visual interpretation

For A100 and B100, evaluate the canonical VAL89 once per full/restricted reader using the maintained native extractor. Report all nine mean/p90/worst errors, worst IDs and per-M populations. Reuse saved R13-4/R13-5 independent e100 metrics as matched-age references when extraction is unchanged; use mature e500 references only as labelled context. No automatic reevaluation of every historical checkpoint is needed. VAL89 is exposed validation, not pristine TEST evidence.

Retain the fixed detailed panel TRAIN0003 and VAL0274/0293/0678, fixed health TRAIN0006/0051/0322/0231, and the already exposed VAL0286 stencil diagnosis only as an explicitly labelled mechanism audit. Do not select training membership or new showcase cases by errors. Examine local material/surface/peak fidelity and M3 behavior as well as global flow and thermal summaries. Measure prescribed-input response at fixed geometry/query through the owning input objects; do not impose heat-independent flow or claim sensitivity is solver-validated response accuracy.

Keep the postprocessing task bounded and use Python. Prefer the established side-by-side spatial style, with consistent geometry, original IDs, colors, coordinates and scales across candidates. Do not depict nonlocal membership by filling a giant convex hull across empty regions. Show disjoint same-color member regions or a small labelled incidence diagram. Separate the following stages visibly: applied exchange/self relations; selected organizational views; permitted/selected original members; final contribution and field effect.

| Selected visual family | Required content |
| --- | --- |
| Physical reconstruction | Aligned targets, A/B predictions and signed residuals on the fixed panel; compact original-module detail and full/restricted differences |
| Spatial organization | Two or three representative actual M/E views on the same scene, shared members, stage direction and separate self access |
| Query execution | Same fixed queries through view weights, member priors, actual candidate IDs, joint contributions and typed strengths |
| Organization-sensitive reading | Same-marginal arithmetic control plus a bounded native fixed-context membership reassignment; separate actual exchange-removal/reprepared results |
| Fidelity and complete work | Fieldwise fidelity versus executed members and measured complete latency; preparation, paired backward and allocated/reserved memory separately |
| Learning | Complete A/B physical histories, full/restricted terms and current loss curves, labelled by population and epoch |

Retain PDF masters and small PNG companions for direct Markdown embedding. Use English labels, one short figure index, meaningful alt text and captions with checkpoint, population, units, controls and quantitative meaning. Inspect decoded figures and the rendered report, verify relative PNG/PDF links, and preserve each prose paragraph/caption on a single source line. Generated figures, numerical exports and one-time renderers stay in ignored DATA2 paths.

## 9. Generality without a new dataset campaign

The shared core learns relations and structured reads; the case adapter supplies input attributes, units, train-only normalization, physical measures, targets and native extraction. M/E are input-object roles, not flow/temperature responsibilities. Retain a general joint physical-field predictor and controls on their owning objects before interaction. No Thermal-specific affine response, output split, invariant routing/flow rule, analytical correction or shortcut based on dataset generation enters the core.

Use focused core fixtures for typed permutation, variable M/E counts, changed eligible measures and E-only/M-only inventories. Keep physical geometry proportions and distinguish extensive controls from intensive attributes. An optional input-only E partition/refinement arithmetic check should preserve total measure and disclose that it is not a new physical simulation or a guarantee of field invariance. Shared architecture does not promise zero-shot weight transfer. No Wind fit, cross-dataset benchmark, solver run or inverse campaign is added to this supplement.

## 10. Resource discipline, safeguards and closeout

Before generated writes, verify DATA2 is mounted and the required diagnostics/Trained_Results symlinks resolve to their declared DATA2 targets. Record actual free space, a hard 8 GiB ceiling for all new payloads including temporary peaks, an estimated peak at most 6 GiB, and minimum headroom of 100 GiB on DATA2 and 20 GiB on home. Check actual checkpoint/export sizes before materialization and reduce duplicate payloads to stay within this budget. Home receives durable source/docs only, not experiment outputs. Existing source checkpoints and shared caches are read in place; do not copy full datasets or create arm-specific caches for the same recipe. A newly necessary temporary cache must have an owner, byte allowance inside the total and closeout expiry.

Retain latest state, one best-field alias, declared 100-epoch milestones and actual final/clean-stop state. Do not create 25-epoch snapshots, duplicate selectors or full graph exports at every review. Keep complete validation history, separate files only at milestones/final reviews, tqdm output and curve refresh every 10 epochs plus initial/final substantive qualification. Preserve native numerical precision and query draws.

Use ordinary Git history, existing dataset IDs/manifests, types, atomic writes and focused tests. Do not introduce cryptographic hashes, baseline snapshots, contract freezes, device-identity locks, admission services or research acceptance gates. An exceptional proposal requires a concrete failure and an explicit demonstration that ordinary primitives are structurally insufficient. Preserve existing security, TEST/data separation, job ownership, checkpoints and upload safeguards. New blocking gates belong only at irreversible changes, cross-system mutations, critical security perimeters or formal production releases; scientific research bands and runtime diagnosis are not release gates. Actual code execution, backward paths and measured native outputs take precedence over heuristic assertions.

At closeout, remove only verified round-owned disposable caches, failed temporary payloads and superseded exports after retaining required numerical/failure evidence. Follow the standing recursive allowlist, active-reader and protected-root checks; existing cleanup checksums are preserved without inventing a new training hash system. Never remove historical checkpoints, source data, active-run artifacts or the only shared reusable cache. Report removed files/bytes, retained artifacts/bytes and actual remaining capacity. Keep manifests, receipts and one-time diagnostics local and ignored.

Deliver maintained code/configs and meaningful focused tests; A/B full-data endpoint histories, resumable states and the compact English report; measured ordinary latency and backward/memory evidence; useful spatial organization and physical-field figures; and tested user-controlled launch/resume/evaluation commands. If an actual unresolved defect prevents one candidate from reaching full training, finish bounded diagnosis and provide the precise state, evidence, attempted repair and recovery proposal rather than a bare failure or a pretend second result. Inverse results are not requested.

Commit durable changes on the current non-default branch. Fetch/inspect the remote, audit every file in the entire outgoing commit range including earlier additions later deleted, enable the existing `.githooks` pre-push safeguard, push and verify local/remote tips match. Do not stage unrelated user changes or upload generated evidence/one-time code. Preserve the older formal continuation and its observer-free handoff throughout.

## 11. Concise execution handoff

Execute this R13-5 supplement in the current non-default ModularDT branch, preserving user edits, historical evidence and the existing observer-free formal_v1 continuation. Keep R13-4's organized original-member access and the working R13-5 four-block M/E views. First measure complete native/training latency on fixed small TRAIN panels; implement only justified equivalent reuse/factoring with forward/backward and recovery checks. Select Candidate A with joint-mass-consistent integration and at most two explicit relation-calibration interpretations, then Candidate B with organization-conditioned row competition over the same selected original members—no pooled replacement, new triple MLP, forced quotas or query×address×member tensor. Qualify briefly on fixed TRAIN150/DEV22, diagnose and repair genuine early path failures, and keep the aggregate segmented ceiling at 500 epochs. Run the two selected independent matched TRAIN600 formal lineages through 100 epochs each, 200 aggregate, using available independent GPUs and sparse ETA-based Luna observation; extensions require user instruction. Deliver fieldwise/local accuracy, actual relation/query/member traces, matched-marginal structural controls, complete latency/backward/memory evidence, inspected English figures, resumable states and tested commands. No new cross-dataset fitting, inverse/solver campaign, hashes/freezes/admission framework or automatic long continuation. Clean disposable outputs and commit/push durable changes through existing safeguards.
