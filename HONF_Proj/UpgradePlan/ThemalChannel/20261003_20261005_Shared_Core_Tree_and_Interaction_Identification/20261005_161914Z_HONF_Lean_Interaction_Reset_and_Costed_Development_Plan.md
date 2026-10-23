# HONF: end the Tree overhead and test a lean interaction representation

## 0. Decision, scope, and deliverables

**Scientific goal:** learn from multi-field reconstruction a reusable organization of module/environment interactions; later reuse that representation in modular inverse design. A receiver tree is one implementation hypothesis, not the definition of HONF.

**Decision:** preserve Tree-C as a research reference, but do not make further recursive-Tree engineering or an unconditional 5,000-epoch Tree continuation the main development path. Execute one bounded cost investigation, an algebraically specialized Global-C reader, and one matched, trained tensor-organizer experiment. Do not spend this round trying to win an arbitrary reconstruction leaderboard, forcing a K histogram, or training an inverse generator.

The new scientific question is:

> After giving ordinary global calibration its own inexpensive path, can a small, source-anchored, receiver-dependent interaction residual add useful reconstruction and response information without recursive planning or a second organizer-gradient program?

Required deliverables:

1. A measured explanation of the current formal epoch costs and separate remaining-time estimates for Run3501 and Run3502.
2. A reversible, explicitly authorized disposition of those two running jobs, with resumable states preserved. No silent termination or hot modification.
3. A tested, same-operator Global-C specialization that removes generic one-group pair-control expansion.
4. One tensor source-group residual candidate and its matched Global-C continuation, actually trained and reviewed on `fixed25_v1`.
5. A clear result for predictor, organizer, and inverse readiness, including A/B/C below and a small inspected visualization set.
6. An honest future-run recommendation with a measured cost forecast. No automatic full-data restart or replacement formal run.

This is an implementation-and-learning round. A profiler trace, parser pass, or a source-only diagnosis does not complete the scientific task. A genuine unresolved integration/numerical failure may limit completion, but try the bounded remedies below before reporting that outcome.

### Evidence reviewed

Scientific/source starting point: `agent/honf-core-next` at `ac0f1ca45ef01b32132c1ee8be6313d77faab301` and `docs/reports/HONF_Native_Context_Organizer_Confirmation_Report.md`.

This plan is based on the attached report and committed source, not a live replay of the user's tmux jobs. The reported early formal costs, 157 s/epoch for Tree and 33 s/epoch for Global, are user-supplied measurements. Verify them from the actual running jobs before using them as an operational forecast.

## 1. What the evidence does and does not establish

The last matched development experiment completed 500 epochs per arm, with 150 training cases, 22 exposed validation cases, FP32, Q1024, effective batch 48, and microbatch 8. Tree-C had 4,803,256 trainable parameters; Global-C had 4,670,953. That approximately 2.83% parameter difference does not explain the approximately 4.55-fold measured B8 optimizer-boundary difference.

Retain these findings:

- Tree-C improves six of eight core error means over Global-C, including fluid temperature by 16.26%, material temperature by 10.29%, and interface temperature by 14.22%. Pressure and vorticity worsen. Native Dense-D25 remains stronger on all eight core means at both retained reference ages.
- At the same Tree weights, route-uniform affine controls give four-case fluid/material/interface RMSE 1.121075/1.006809/1.069018, versus normal 1.121235/1.008320/1.070681. This retains the learned access/density paths. It does **not** prove that all Tree organization can be deleted without loss.
- Identity control removal is damaging. Useful global gain calibration is therefore not equivalent to useful nonuniform grouping.
- Both new models execute 43,972,896 fine MLP rows in 3,542 invocations on the reported all-22 full-grid evaluation. Their measured coarse/local work also matches. Logical MM/ME restrictions do not reduce the current fine executor.
- The high-M B8/Q1024 optimizer-boundary medians are 2.299978 s Tree and 0.505916 s Global. These are not complete effective-B48 updates or full-data epochs. The high-M Q8192 wrapper medians are 0.458013/0.269465 s.
- Small representation/interface-proxy numerical misses remain. The proxy `h*(Ts-Tenv)+delta_q` can amplify upstream floating-point differences. They do not explain the order-one reconstruction gap or all of the runtime.
- Positive response truth is still limited to an exposed training family. Known heat-null leakage remains. More baseline-field epochs do not manufacture held perturbation labels.

Preserve the distinction between:

**A. Added value:** a trained organizer must earn its complexity against an independently trained or matched continued nongroup control.

**B. Faithfulness:** exported control donors, physical value reads, global/planning paths, and phase ancestry must agree with execution.

**C. Transfer:** numerical derivatives and training-neighborhood response fit do not establish held physical-response generalization.

## 2. Running jobs: protect work, then make an explicit resource decision

### 2.1 Read first

Locate the actual Run3501/3502 directories, tmux panes, commands, PIDs, physical GPU assignments, environments, completed epochs, optimizer updates, checkpoints, and log paths. Do not infer a PID from a run label or terminate a broad Python/tmux pattern. Read the running resolved configurations as well as the committed templates.

The committed formal templates already request Q1024, microbatch 8/effective 48, receiver chunks 128, activation checkpointing, and the local-context shadow for Tree. Therefore, do not assume that a hidden Q8192 configuration or old whole-wrapper shadow caused the cost. Check for launch overrides and actual dispatch.

Separate measured epoch time into training, validation, scheduled response work, checkpoint/plot work, and unallocated elapsed residual. Report epochs before/after structural calibration and the epoch-101 objective/response transition separately. Identify external GPU/CPU/I/O contention without touching other users' processes.

Arithmetic using the reported initial values:

- Tree: `157 * 5000 / 3600 = 218.06 h`, or 9.09 days.
- Global: `33 * 5000 / 3600 = 45.83 h`, or 1.91 days.
- Concurrent completion is dominated by the slower job, not the sum of both durations. Do not say that both individually require 9–11 days.

Use remaining epochs, not the original total, for the updated ETA. A forecast is conditional on the current schedule, workload, and contention. Do not invent a confidence interval from two early epochs.

### 2.2 Proposed pause authorization

The accompanying execution prompt explicitly authorizes a **graceful checkpoint-and-pause of these two named formal runs only**. Confirm that the user has supplied that authorization when receiving this plan. Reading the plan alone is not an instruction to kill a process.

Use the trainer's existing supported resumable-stop mechanism after a completed optimizer boundary/epoch. Retain model, optimizer, scheduler, calibration, RNG/sampler state, completed-work counters, normalizers, and the original schedule. A specifically requested pause checkpoint is an exception to the ordinary 100-epoch retention cadence, not a new periodic checkpoint policy.

Do not use `kill -9`, `pkill python`, broad tmux shutdown, checkpoint overwrite, or a stale PID. Do not assume Ctrl-C saves a complete optimizer state: inspect the actual handler first. If safe saving is not supported, report the last durable state and unsaved work and obtain the user's choice before discarding it. Continue CPU/source work while resolving that boundary.

Preserve the original runs as **paused**, not failed or completed. Provide exact manual resume recipes without executing them. Do not edit imported source or configurations beneath still-running jobs; use an isolated ordinary Git worktree until they are safely paused. Preserve the repository hooks and artifact policy in the development worktree.

## 3. A bounded runtime diagnosis, not another audit campaign

### 3.1 What the source already explains

Relevant implementation:

- `src/honf_forward_core/interface_fields/typed_hypergraph_field.py`: local-shadow training calls the organizer twice per physical phase, once hard and once soft, although it calls the physical wrapper only once.
- `interface_fields/adaptive_receiver_hypergraph.py`: every prepare builds M/E/Q receiver indexes per case, computes all real node descriptors, applies five typed source scorers and both donor projections, and transfers split values to the host for frontier counting.
- `interface_fields/adaptive_interaction_cover.py`, `CaseLocalReceiverTree.build`: CPU transfers, Python sorting, recursive partition construction, and canonical block handling are in the live construction path.
- `interface_fields/routing_index/sparse_projection.py`: FP64 prefix/projection arithmetic is deliberately used for sensitive measure operations. Do not remove it globally to get a faster number.
- `Case_ThermalChannel/src/channelthermal/training/epoch.py`: ordinary training also invokes work telemetry, scalar diagnostics, calibration when due, response callbacks, and sampled gradient diagnostics.

For a 600-case full epoch, three phases, three receiver-index families, and hard/soft organizer preparation, the source implies roughly `600*3*3*2 = 10,800` index builds before validation and auxiliary work, assuming the ordinary three-phase call pattern. This is a **static count to verify**, not a measured profiler attribution. The local `index_cache` is local to one prepare call; its name does not make it a persistent case cache.

A 600-case epoch also contains 614,400 primary fluid queries; a 5,000-epoch schedule contains 3.072 billion primary fluid queries, before source pairs, ports, local outputs, backward, and callbacks. Small parameter count and a small number of layouts do not imply a small executed tensor program.

### 3.2 Measurements to execute

Allow at most **90 minutes of elapsed diagnosis plus engineering profiling**, not an expanding trace matrix. Use existing training cases and actual configured loss/backward. After job disposition, run:

1. One warmed low-M and high-M native microbatch per relevant mode; a small mixed-M batch as needed.
2. Three ordinary optimizer boundaries per measured mode with profiler off, preserving the effective-work interpretation.
3. One short profiler capture, at most three active microbatches, covering host and CUDA work. Separate execution time from profiler overhead.
4. At most one disposable full original-population epoch per existing formal recipe if the live logs do not already provide the necessary timing. This is explicitly authorized full-recipe engineering execution, with no scientific optimizer state retained or validation-based selection. Prefer reused real epoch logs plus subset optimizer measurements.

Measure host waiting/synchronization, index construction, learned planner/control heads, sparse projections, fine physics, Stage-A/coarse/local paths, backward/checkpoint replay, data movement, and diagnostics. Avoid adding nested times as disjoint totals. Count actual calls using a small temporary profiler/recorder, then remove that recorder for benchmark timing.

Compare at least ordinary hard inference and the configured training path. A large forward/backward difference does not by itself identify FP64 or CPU sorting as the entire cause. Record which overheads are measured and which remain inferred.

Do not respond with only 'organizer overhead'. Provide the three largest measured categories and the actual code entry points.

## 4. What to retain, freeze, and replace

### Retain

Native source-resolved nonlinear MM/ME/EM/QM/QE interactions; native coarse/local contexts; predicted-port/Stage-A coupling; physical identities and source measures; source-local initial feature conventions; prepared source projections; the already-tested projected-control algebra; honest donor/planning/phase exports; dataset/checkpoint security; and the fixed development split.

### Freeze as historical research, not delete

Tree-C/Tree-Lite weights and replay paths; its recursive geometry index; hard-value/local-soft-shadow training; existing intervention and invariance tools. Preserve all formal and development checkpoints and associated recipes.

### Replace in the next candidate's hot path

No recursive receiver tree, CPU sorting, `.cpu().tolist()` planning, per-case Python frontier walk, or 15 separate phase/route organizations. No hard physical masking followed by a second restorative organizer program. No dense receiver-source-control-vector expansion. No K target, entropy/diversity target, solver-selected merge, learned error-margin chooser, or repeated all-case intervention suite.

This is not a repudiation of hypergraphs. It removes one expensive implementation hypothesis whose marginal scientific advantage remains weak.

## 5. First implementation deliverable: a same-operator Global-C specialization

Before adding a new organizer, remove the generic one-group machinery from Global-C's numerical reader.

For a case-constant scalar affine control `a_tau`, let `g_tau = 1+tanh(a_tau)`. For non-attention routes:

`sum_s w_rs * mu_s * g_tau * message_rs = g_tau * sum_s w_rs * mu_s * message_rs`.

Apply the gain after the native sum but **before any subsequent affine output projection**. In particular, `query_module_output(g*x)` is generally not `g*query_module_output(x)` because of its bias. Respect MM self-exclusion, padding, empty rows, native denominators, and ME quadrature exactly.

For QE, the one case/head score is constant across sources and cancels in softmax. The headwise gain can be applied to the completed head context **before** the attention output projection. Keep the original unique-source softmax, geometric bias, keys/values, measures, and output-bias convention. Do not fuse gains across nonlinear module/environment updates or across P0/P1/P2.

Keep the Global-C control network and its live phase-current input summaries. Only remove pair-control/density expansion that is mathematically redundant for one full-access group. Diagnostic exports can reconstruct the old semantic view on request, outside ordinary timing.

Implement as an opt-in execution path under the existing native global model, with historical state-dict compatibility. Unit algebra checks are insufficient: compare native field/port/material outputs, representative physical parameter gradients, heat/query gradients, and actual optimizer steps at saved development weights. Do not claim a measured speedup until the profiler-off native timing is available.

Do not use this opportunity to alter source features, objectives, optimizer, precision, Stage-A, or physical architecture. Current formal Global-C may use the specialization only in a separately approved, checkpointed resume after these tests; no such resume is automatic here.

## 6. Main scientific candidate: tensor source-group residual HONF

Working label: **Tensor-H**. This is a proposal, not an already verified model.

### 6.1 Represent groups of sources, not a recursively partitioned query plane

Use one proposal associated with each present physical module and one environmental-background proposal. The environmental anchor is the positive-measure environmental centroid. Group descriptors use shared functions of the physical anchor, its local design state, and prescribed context. No slot-specific learned ID embedding or source-index tie breaker is allowed.

The maximum padded proposal count is `M_max+1`; the physical active proposal set is `M+1`. This is only candidate capacity, not learned K. Source groups may overlap; an anchor is a reference location, not a claim that its own module is the group's sole donor or physical owner.

Build group keys, case admission, and both M/E membership densities **once from initial encoded design/context per complete wrapper**. Carry them explicitly into P0/P1/P2. Phase-current source content is recomputed for controls at each phase, but membership structure is shared. This tests one reusable case-level organization instead of fifteen independently rebuilt organizations.

The once-per-wrapper plan is an explicit tensor value owned by that forward call. Never reuse learned plans after a parameter/input change or store the last plan as an unqualified global cache. Distinct inverse candidates get distinct fresh plans. Historical backends and Wind behavior remain unchanged.

### 6.2 Input-conditioned admission and actual K

Let `ell_e(d,c)` be a shared input-only score for each valid proposal. Use the existing masked sparsemax for final case admission:

\[
\pi = \operatorname{sparsemax}(\ell),\qquad
K(d,c)=\#\{e:\pi_e>0\}.
\]

There is at least one valid proposal and the probabilities sum to one. Do not penalize `sum(pi)` as an L1 sparsity objective: it is identically one. Do not set K from module count, force a distribution of K, permanently delete proposals, or use observed target errors in admission.

To avoid freezing unused new residual components before they receive useful task gradients, use one transparent, deterministic continuation shared by training and evaluation:

- Additional epochs 1–100: softmax admission and measure-softmax donor density.
- Additional epochs 101–200: linear blend to sparsemax admission and measure-sparsemax donor density.
- Additional epochs 201–500: the exact sparse projection only.

This is one evolving forward function, not a hard forward/soft backward pair. At soft/blended ages, report positive supports honestly; do not report thresholded probabilities as learned exact K. The new-branch age is additional age since attachment, not the inherited absolute model age.

Sparsemax is continuous and piecewise differentiable. It does not guarantee useful sparsity or gradient revival on every zero coefficient. Shared scorer parameters, broad initial exposure, and input variability can change future support; no guarantee is asserted. Inspect dormant proposals without automatically launching a temperature/penalty sweep.

### 6.3 Measure-aware donor memberships

For donor type `t in {M,E}`, normalize the case-owned physical source measure to `mu_s^t`. For each group use shared, signed relative-geometry and source-state features to score donors. Preserve spatial axis identity and the native frame.

Final donor density:

\[
b^t_{es}=[z^t_{es}-\tau^t_e]_+,
\qquad \sum_s\mu^t_s b^t_{es}=1.
\]

Reuse `source_measure_sparsemax`; return zero rows only for genuinely empty donor types. During the initial continuation, use the measure-normalized positive exponential density, not a source-count-normalized replacement. Keep the existing narrow high-precision projection arithmetic where required; the large physical feature tensors remain FP32.

The membership is a **density**. `mu_s*b_es` is the donor mass/probability. Unequal atom splitting must divide physical mass while retaining the same density, otherwise the source integral changes.

Scoring and pooling are tensor operations across `[B,G,S]`; there is no per-case CPU tree. Use batched matrix multiplication to pool projected source features; do not materialize `[B,G,S,H]` products merely to sum them. Initialize donor preferences from a modest directed geometry prior plus a shared trainable residual. Document its native length scale. This prior does not encode a measured causal interaction.

### 6.4 Collective phase content remains nonlinear, but physical values stay separate

At phase `p`, construct:

\[
 h_e^p = H\left(
 \sum_i\mu_i^M b^M_{ei}\phi_M(z_i^p),
 \sum_j\mu_j^E b^E_{ej}\phi_E(e_j^p),
 x_e,c,p\right).
\]

Use small control encoders, e.g. width 64 and control width 16. The source encoders in this formula serve control summaries only. The unchanged native fine MLPs still receive each original source state and coordinate separately. This is **not early pooling of physical value messages**.

The native QE value states can already contain EM transport. Raw-control donors, phase-current control content, native fine values, and earlier-phase ancestry must remain distinguishable in exports.

### 6.5 Receiver access without a discontinuous frontier

Let `s_re = 4*tanh(q(r,c)^T k_e / sqrt(d_k))`, with padded proposals masked. Use:

\[
 a_{re}=\frac{\pi_e\exp(s_{re})}{\sum_f\pi_f\exp(s_{rf})}.
\]

The bounded score keeps the denominator well scaled and avoids `log(pi)` at exact zeros. A group with zero admission contributes exactly zero; its contribution vanishes continuously as admission tends to zero. Use the same expression in training, validation, and input-gradient evaluation.

Access sees the receiver coordinates/role and current public case, not target fields or the rest of the requested query batch. MM/ME share the module-receiver access; QM/QE share query-receiver access. No arbitrary per-role independent graph is built. Recompute access for actual receiver positions, not by reading a precomputed field-grid label.

In this first candidate, every admitted group has positive access at every eligible receiver; spatial **weights**, not binary query subgraphs, differ. Exact Kq therefore generally equals case K. Report that limitation rather than claiming independently sparse per-query K. A later sparse query projection is not part of this round.

### 6.6 Give global calibration its own path; make the new part a contrast

Keep the Global-C case/phase route affine control `g_tau^p` as the baseline. Project `h_e^p` to scalar gains for MM/ME/EM/QM and headwise gain/score corrections for QE **before** receiver expansion.

For route `tau`, with its donor type `t`, define small-channel actions:

\[
 v^\tau_{rs}=\sum_e a_{re}\,b^t_{es}\,\gamma^\tau_e(h_e^p).
\]

Use densities, not donor probabilities, in this per-source action. Under splitting an identical source into two mass-partitioned copies, `v_rs` remains the same for both children; the native quadrature accounts for their divided mass.

Remove the case-global component with a declared, target-free reference receiver measure `nu`:

\[
 \bar v_\tau=\frac{\sum_{r\in\mathcal A_\tau,s}
     \nu_r\mu_s V_{rs}v^\tau_{rs}}
 {\sum_{r\in\mathcal A_\tau,s}\nu_r\mu_s V_{rs}},
 \qquad
 \delta a^\tau_{rs}=\tanh(v^\tau_{rs}-\bar v_\tau).
\]

`V` is native eligibility, including MM self-exclusion. Empty eligible routes return zero. The reference panel is a fixed input-only role-aware quadrature (native module receivers for MM/ME, environmental receivers for EM, and a stable environmental/port geometry panel for QM/QE); it must not be the current sampled training-query minibatch. Its locations/weights and global centering path are exported. Keep the panel small and shared between related routes. Do not claim post-tanh residuals have exactly zero mean: centering removes the **pre-tanh** constant mode.

The effective gain is `1+tanh(g_tau + delta_gain_rs)`. QE additionally uses `delta_score_rs` in its ordinary unique-source attention. Its base source-independent score remains absent because it cancels in softmax.

This deliberately avoids the potentially ill-conditioned pair-control quotient `sum(a*b*h)/sum(a*b)`. There is no division by a vanishing local membership. The residual varies continuously with admission/density; the original native local-neighbor/geometry operations still have their own differentiability limits.

Initialize the new final gamma projections at zero, but initialize group/content features nondegenerately. At attachment, Tensor-H must reproduce Global-C. On the first actual update gamma gets a task gradient; subsequently the group network can learn. Do not initialize every multiplicative factor to zero.

The centering path is global and must be named. A source excluded from one local h-content does not thereby have zero derivative through planning, centering, coarse/local context, or other groups.

### 6.7 What sparsity this candidate does and does not claim

**Claim to test:** an inexpensive, case-admitted, sparse-donor collective control representation can improve the native forward function and expose useful receiver-dependent interaction distinctions.

**Not a claim:** every excluded control donor is absent from the complete physical predictor, or fewer admitted groups reduce the existing fine physical source rows. This round retains all native fine values, global/coarse context, and local correction to avoid repeating the accuracy damage from premature information removal.

This is explicitly a **control-information hypergraph**, not yet the minimum physical-information graph. That narrower step is worthwhile only if its learned receiver-dependent part earns value and its execution cost remains modest. If it behaves as a decorative residual or a duplicate global calibrator, report failure and do not extend it through another series of forced sparsity repairs.

### 6.8 Implementation contract and expected cost

One shared implementation under `src/honf_forward_core/interface_fields/`, with an opt-in native-context architecture/profile. Prefer a small new tensor organizer and a narrow extension to the fine reader over copying the entire training/wrapper stack.

Prepared state contains prototype coordinates/validity, admission, M/E donor densities and measures, the initial group descriptors, and named global/planning dependencies. Phase state adds current collective controls and their small-channel projections. No persistent graph is keyed only by a case ID after inputs or weights change.

Computational shapes are approximately:

- group construction/projection: `[B,G,M+E]`, once per wrapper;
- live collective summaries: batched `[B,G,S] @ [B,S,64]`, per phase;
- receiver access: `[B,Q,G]` per chunk;
- final small-channel actions: `[B,Q,S,d_action]`, with `d_action=1` or QE head channels;
- original fine physical source computation: unchanged.

No `[B,Q,S,16]` control tensor, no `[B,Q,S,256]` organizer feature, and no all-pair scalar-to-Python loop. Padding by module count is allowed and measured; allocated groups and admitted groups remain separate. Do not infer sparse hardware execution merely from zeros in these tensors.

Use ordinary autograd through this actual continuous numerical function. No full or local soft-shadow gradient program, no straight-through hard frontier, and no active-set continuation exception should be introduced for the new organizer. Existing sparse projection's piecewise derivative is sufficient for this control-only experiment.

First implement and benchmark eager PyTorch. `torch.compile` is optional only after tensorization; it is not a substitute for removing Python/data-dependent planning. No new compiler/library upgrade, custom CUDA/Triton kernel, mixed-precision campaign, or distributed training project is needed.

## 7. One matched learning experiment, not a new portfolio

### 7.1 Starting state

Use the existing **development Global-C Run3402 exact e500**, which was trained only on the selected 150 cases. Do not use Run3502 full-data weights in a selected-data experiment. Do not use Tree500 weights for a supposedly identity-preserving attachment.

Arms:

- **G-fast:** same Global-C development e500, specialized exact reader, continued native training.
- **Tensor-H:** identical common development e500 weights/optimizer state, same specialized reader, new zero-output residual organizer.

Common physical tensors and their optimizer/scheduler states are restored from the same parent. New organizer parameters have new optimizer moments, with explicitly documented learning rate and the remaining absolute schedule. No old optimizer moments are assigned to new parameters. Stage-A remains frozen with its own normalization. The original 1,000-epoch development schedule is not reset.

This is a matched **warm-start adaptation** experiment, not two fresh 1,000-epoch architectures. Record 500 inherited epochs and up to 500 additional epochs. Checkpoints at absolute600/700/800/900/1000 are also additional100/200/300/400/500. Their two ages must be visible.

### 7.2 Exposure, objectives, and review

Follow `AGENTS.md` and the current development guide:

- unchanged `fixed25_v1`, 150 train/22 validation; no resampling, extra full-data training, or normalization leakage;
- identical seed, case order, primary queries, FP32, microbatch8/effective48, and common losses;
- normal checkpoint/latest/best-field/plots only every100 additional epochs;
- all22 statistical evaluation; four fixed detailed cases;
- same current native physical objective and same allowed TRAIN0348 response callback; null coefficient0;
- no new frontier/complexity penalty, teacher distillation, inverse loss, or development-fitted loss coefficient.

Both arms train for 100 additional development epochs and receive a real review. Complete **500 additional epochs per arm** when numerical behavior is healthy and the measured cost fits the round. Do not stop a healthy model simply because it has not beaten mature Dense at the first review. Conversely, a repeatedly disconnected/residual-zero candidate with no actionable remedy need not spend the remaining horizon.

The first100 review must include real task gradients in the new projection/content/admission blocks and changed parameters, exact finite predictions, native timing, and a train/development field trend. It is still the soft-membership stage; it cannot establish exact sparse K. At additional200, confirm the prescribed sparse transition executed. No rolling temperature search follows automatically.

Allow at most two evidence-driven engineering/numerical remedies and at most100 total disposable optimizer updates beyond the main matched runs. Preserve their artifacts and charges. A changed mathematical recipe is a separately identified child, not silently part of an unchanged curve.

### 7.3 Performance targets, not claimed outcomes

Target complete training/inference overhead of Tensor-H no more than roughly **1.5 times the specialized G-fast**, with substantially lower cost than the current Tree at matched settings. This is an engineering design target, not a promised result or a new blocking software gate.

Do not pursue a fivefold-overhead candidate to500 merely because its field loss decreases. After an executed100-epoch review, a large continuing cost overrun with no measured repair is grounds to end that candidate and retain G-fast as the useful deliverable. Explain the decision using measured cost and scientific value rather than a magic threshold alone.

Compare both error against actual epochs/case visits and error against measured elapsed training time. Same-epoch quality cannot hide grossly different cost; same-time quality cannot hide unequal data exposure.

## 8. Bounded tests and evidence

### 8.1 Engineering correctness

Before scientific continuation, execute actual native steps at the identity attachment. Keep the test panel small:

- same-weight Global specialization: outputs, representative parameter and heat/query gradients, low/high M;
- Tensor-H initial residual zero: common physical outputs/gradients match its Global parent;
- prototype/module permutation equivariance;
- unequal environmental atom splitting: source measures, donor density, content, action and output effects;
- arbitrary query order/chunking does not change the case graph or centering statistic;
- masked/padded/self-excluded/empty routes retain native behavior;
- ordinary autograd versus finite difference away from projection knots, and continuous output across a constructed admission change;
- actual excluded local control donors have zero fixed-membership h-content contribution, with global/planner/centering/ancestry paths shown separately.

Known q-proxy amplification remains a possible numerical limitation. Retain per-output-channel discrepancies; do not relax historical security or correctness tests or declare a new tolerance from the candidate's worst miss. Do not require every old exploratory recorder to be redesigned before the first real step.

### 8.2 Predictor and trained organizer value

At each100 checkpoint retain the common selector and role metrics. Run the full24-role all22 statistical panel at the first review and final selected/exact endpoint; lighter ordinary native validation remains on its maintained schedule. Reuse existing measurements when weights/protocol are unchanged.

Primary comparison: trained Tensor-H versus trained G-fast at matched additional age. Report all eight core roles, near-interface roles, initial/final ports, material peaks, and both pressure definitions without silently equating them. Include per-M counts and tails. One favorable thermal scalar is insufficient for promotion.

Four representative same-weight checks only:

1. Normal candidate.
2. Zero residual organizer, retaining its current global/coarse/local/fine base.
3. Replace receiver-dependent access by its declared reference-panel mean and recompute centering consistently.
4. Remove one input-selected active group's residual contribution without changing its physical fine source inventory.

If a control is numerically ineffective, report it as such, not a successful intervention. No large all22 rewire matrix. An ablation demonstrates current reliance; the trained G-fast comparison is the evidence for added learning value. The added-parameter difference remains disclosed.

Measure actual admission K, active donor sets, receiver access, residual action RMS and physical field effect, and current fine rows. Distinguish varying K at fixed M from variation caused solely by the number of candidate anchors. A nonzero hidden control is not physical causality.

### 8.3 Response and inverse readiness, not another inverse campaign

At final weights, use the existing small heat-null and TRAIN0348 response panels. Show thermal sensitivity as model sensitivity unless perturbed reference targets exist. Run only a small fixed-four numerical heat/input-derivative check. No inverse search, diffusion head, or generative sample portfolio is included.

A reliable inverse application also needs nonzero held physical response evidence. Do not promote because ordinary derivatives are finite. Retain the separate unexecuted physical-reference request; do not insert excluded atlas families into the selected training subset or launch new solves by default.

## 9. Resource plan and progress reporting

Maximum new experimental allocation: **12 aggregate GPU-associated hours and 8 elapsed hours**, including failed attempts and final evaluations. The preexisting formal jobs' historical time is reported separately; any overlap with this round must be visible, not double-counted as new exclusive development compute.

Suggested allocation:

- first90 minutes: live-cost accounting and a bounded exact-reader/profile repair;
- next implementation window: Tensor-H wiring and actual optimizer tests; G-fast can start its matched continuation when reproducible parent data are ready;
- main remaining GPU time: the paired additional100→500 development exposure;
- final45 minutes: saved-array figures, report, focused tests, commit/push.

This is an approximate work allocation, not a requirement to keep GPUs busy. After the first five real new epochs, forecast the paired100 and500 costs. Reduce repeated evaluation/export, not training-case coverage, when conserving time. The authorized recipe must fit the round or finish at the last shared100 boundary with an explicit partial outcome. Do not silently turn 500 required additional epochs into 500 optimizer updates.

External contention: leave foreign processes alone. Use the available authorized GPU, or safely co-reside only with sufficient memory and disclosed timing contamination. Do not spin for hours waiting. No third GPU, cloud purchase, precision change, or unauthorized full-data continuation.

## 10. Report and visualization contract

Open the final report with a short plain-English account of the scientific question, what changed, what improved, what failed, and the recommended next decision. Then give predictor/organizer/inverse and A/B/C results separately.

Required figure groups, at most five:

1. Measured current/formal cost breakdown and ETA, alongside new full-step/epoch timing and memory. Separate parameter count, fine rows, host time, kernel time, and profiler overhead.
2. Paired learning versus additional epochs and measured GPU-associated training time; mark inherited e500 and the membership continuation.
3. Four representative physical fields/residuals and interface/material summaries with native coordinates, common honest scales, units, and unclipped numerical metrics.
4. Actual source-anchored groups, admission, M/E donors, receiver access and residual effect maps. Do not draw a root/global group as sparse physical retrieval.
5. Existing response evidence and numerical derivative checks; explicitly state that no inverse generation/search or new reference solve occurred.

Do not require an unexecuted design trail merely to fill a figure slot. Use actual saved numerical outputs. Inspect each figure, embed a small raster companion in the Markdown report, retain PDF masters locally, and keep generated artifacts out of Git under the existing rule.

Include a concise retained/frozen/replaced code table, executed vs intended budgets, full-versus-development identities, all scientific changes, and unresolved hard failures. Routine minor fixes need not become a long narrated audit trail.

## 11. Exit decisions and future formal-run policy

Possible useful outcomes:

- **Fast base only:** Global specialization is accurate and faster, while the residual organizer adds no useful structure. Keep the base and retire this residual hypothesis; do not rename failure as dynamic-HONF success.
- **Useful lean organizer:** Tensor-H stays close to base cost, adds a meaningful physical/response advantage, and exports truthful input-conditioned groups. Recommend a later manual full-data comparative study, not immediate production/inverse readiness.
- **Engineering blocker:** real execution exposes an unresolved numerical/integration limitation after bounded remedies. Preserve native evidence and explain what remains, without starting another long formal run.

No fresh5k recipe may be described merely as 'ready' because its parser or inference smoke passes. Before a future manual launch provide measured ordinary training/validation epoch cost, scheduled-callback cost, expected initial/intermediate/full-horizon hours, memory, and a plain scientific reason to spend that budget. The user chooses the duration. 5,000 epochs is a possible ceiling, not the definition of 'fully trained'. This is a human resource decision at the formal-run boundary, not a new automated gate framework.

Do not automatically resume Run3501/3502, switch their objectives/architecture, transfer full-data weights into the development comparison, or overwrite their histories. Exact same-operator execution improvements may be proposed for a separately authorized resume; mathematical changes require a new identity.

## 12. Source map and external methodological context

Project evidence and implementation, inspected at `ac0f1ca`:

- `docs/reports/HONF_Native_Context_Organizer_Confirmation_Report.md`: matched means, uniform controls, actual work, runtime, numerical/response limitations and formal preparation scope.
- `docs/reports/HONF_Tree_Lite_Execution_and_Maturation_Report.md`: prior execution savings and hard/local-shadow semantics.
- `src/config_core/forward/thermal_native_context/{tree-c,global-c}_full5000.json`: actual committed formal settings.
- `src/honf_forward_core/interface_fields/{adaptive_receiver_hypergraph,adaptive_interaction_cover,typed_hypergraph_field,global_control_hypergraph}.py`: current planning, CPU tree construction, hard/soft preparation and one-group reader.
- `src/honf_forward_core/interface_fields/routing_index/sparse_projection.py`: existing masked/measure-aware sparsemax implementation.
- `Case_ThermalChannel/src/channelthermal/training/epoch.py`: actual loss/backward/telemetry and callback path.
- `AGENTS.md` and `docs/guides/Thermal_Model_Development_Protocol.md`: standing preservation/reporting/development rules.

Primary external references: Martins & Astudillo, ICML2016, “From Softmax to Sparsemax” (PMLR48:1614–1623), for a sparse continuous projection with efficient backpropagation; PyTorch official compiler documentation on data-dependent graph breaks and NVIDIA's CUDA Graph performance guidance on `.item()` synchronization and fragmented launches. These support numerical/execution primitives, not a claim that Tensor-H will discover physical groups. Use the installed PyTorch version's supported API; no upgrade is required.
