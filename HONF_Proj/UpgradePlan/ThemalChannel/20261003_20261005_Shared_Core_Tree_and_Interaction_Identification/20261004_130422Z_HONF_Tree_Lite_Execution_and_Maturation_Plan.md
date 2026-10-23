# HONF Tree-Lite: efficient faithful execution and a real maturation test

## 0. Goal, priority, and authorized scope

**Research goal:** learn a hypergraph-inspired organizer through multi-field reconstruction that organizes module–environment interactions and can be reused in inverse design. The organizer must eventually earn its place through useful physical predictions, faithful information paths, and transferred decision information—not simply a nonconstant K or attractive support map.

**This round's deliverable is a trained, substantially leaner Tree implementation and a credible 500-epoch comparison.** This is not another faithfulness-only audit, another 200-epoch stop, or a large inverse-generation campaign.

Reviewed source: `cosmos2w/ModularDT`, `agent/honf-core-next`, commit `de6f467b9466fd9f585255d49552fcaec1d857f2`. Read the actual branch tip and explain any subsequent changes before implementation. Do not rewrite historical results.

There are three priorities, in order:

1. Remove avoidable query-time and training overhead while preserving the current hard forward operator and measure-faithfulness properties.
2. Train the primary Tree candidate and its controls to **500 development epochs**, unless a concrete execution failure cannot be repaired or the declared resource limit is reached. A failure of the present weak heat-null auxiliary is **not**, by itself, a reason to terminate reconstruction learning.
3. At 500, decide whether one Tree and one relevant nongroup control should reach **1,000 development epochs** within the remaining explicit budget. Run a small frozen inverse-use evaluation, not a new diffusion-head campaign.

The development/formal distinction is mandatory. All scientific training here uses the existing fixed25_v1 150/22 subset. No full-data training, Wind training, new reference solve, or 5,000-epoch launch is authorized. A future full-data formal model is a separately labelled run, normally launched manually by the user.

### Plain-language explanation that must remain visible

Tree is a directory for receivers, not a replacement for physical sources. It recursively divides receiver locations into candidate regions. A learned split chooses whether a region can share one interaction rule or needs child rules. Each active rule specifies source weights and a small collective control vector. The ordinary fine source–receiver kernels still evaluate the physical source states.

At Tree-F e200, the tested native value access was fully open. What had learned was mainly **reweighting and collective modulation**, plus some excluded module donors in controls. That is not sparse physical value retrieval. Do not call it early pooling of all field values, but do not call it a sparse physical graph either.

## 1. What the existing evidence establishes

Use `HONF_Tree_Faithfulness_Fixed25_Development_Report.md` as the report basis. The following observations motivate this plan.

| Observation | Supported interpretation | Not established |
|---|---|---|
| Tree-F fluid T RMSE fell 5.47149 to 2.13424 from e100 to e200. | The current candidate was still learning substantially. | Convergence or an irreducible accuracy floor at e200. |
| At e200, Tree/Pair/Dense fluid T is 2.13424/3.06548/1.76051; material T is 2.28240/3.51092/1.35528. | Collective controls provide a thermal advantage over the trained Pair control; the stronger Dense architecture still wins. | Accuracy parity with Dense, or a comparison against Dense1000 at equal age. |
| Tree e200 Q8192 wrapper medians are 835.56/872.35 ms versus Pair 185.68/179.50 ms on M3/M10. Prepared P2 alone is 614.26/618.26 versus 135.97/133.14 ms. | There is large repeated decode overhead, not just one expensive tree construction. | A profiler-based percentage attribution to any one operator. |
| Incremental inference CUDA peaks are Tree 45.27/58.98 MiB versus Pair 81.58 MiB. | This measured inference allocation is smaller for Tree. | Whole-model resident memory, matched training peak memory, or a universal Tree memory advantage. |
| Subset packing removes only inactive source columns; every eligible value pair remains open. | Some validity padding can be skipped. | Learned physical-source sparsity or a sparse-executor speedup. |
| Geometry replacement changes fluid T by only 0.119%, while zeroing controls worsens it by 75.9%. | The trained controls matter; the particular learned grouping/source-weight arrangement has weak evidence of unique utility. | Causal physical groups or a sufficient argument that a simpler trained model cannot do as well. |
| Rebuild invariance and 64/64 pulled-back derivative checks pass. | Preserve these valuable properties. | Continuum response accuracy or beneficial inverse decisions. |
| Weighted null/native gradient ratios at the measured calibration are roughly 3.5e-6 and 4.8e-6. | The null term was an extremely weak force at that point. | A completed, effective test of enforcing heat–flow independence. |

The present Tree/Pair fine backbone is not the native Dense backbone. `ThreeTermInterfaceContext` removes the historical coarse bank/local-neighbor route. Tree-F/Pair-F also use source-local input semantics. Therefore the Dense gap is confounded by context capacity, input semantics, task weighting, and training age. The existing experiment did not merely add a tree to Dense.

Historical 1502/1505/1507/1508 results remain useful context, but their full-data membership, normalization, objectives, physical wrapper versions, and age differ. Never compare their numeric relative-L2 results directly with current physical RMSE as though they were one matched experiment.

## 2. Dataset, cadence, and controls

Follow `AGENTS.md`, `Thermal_Model_Development_Protocol.md`, and existing trusted loaders.

- Manifest: `/data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json`.
- Exactly 150 selected train cases and 22 selected validation cases; no redraw and no excluded-case injection.
- Initialization/query seed 0 and the existing selection seed; deterministic case/query ordering independent of architecture-specific RNG consumption.
- Train-only global normalization; preserve the attached frozen Stage-A model and its own normalizer.
- FP32 physical kernels, Q1024 primary training, effective batch48, the absolute 1,000-epoch schedule, existing physics losses and checkpoint rules.
- One development epoch: 150 unique train visits, 19 native microbatches, four ordinary optimizer boundaries, 153,600 primary fluid queries. Additional shadow/atlas/null work is reported separately.
- Statistical validation: all22 once per declared retained milestone. Detailed physical/graph exports: only 0277,0291,0294,0687 by default.
- Milestones/latest/best-field aliases and plots every100 epochs. Best is chosen only from retained monitoring checkpoints. Do not create a new collection of 25-epoch checkpoints or select unsaved epochs.
- Do not silently resume into a different normalization, membership, source-input schema, architecture, or optimizer schedule.

### Primary trained models

| Label | Initialization and training | Purpose |
|---|---|---|
| **Tree-Lite** | Exact Tree-F3204 e200 physical/organizer/optimizer state; explicit child lineage for the new training-gradient policy; continue to total e500. | Same hard model, lean execution and single-physical-pass organizer learning. |
| **Pair-F-long** | Exact Pair-F3202 e200 state; continue unchanged physical objective to total e500. | Competent independently trained, approximately parameter-matched nongroup control at the same final age. |
| **Fine-F** | Fresh common B-fine physical initializer and source_local_v3 inputs; full-access three-term backend without Tree or Pair control heads; train to e500. | Determine whether the shared fine/source-local backbone can learn the task without either modulation system. |

Retain Dense-D25 Run3101 at e200/e500/e1000 and mature Run1804 as read-only references. Dense500 is the main equal-age strong benchmark; Dense1000 is explicitly longer-budget until the candidate reaches1000. Do not retrain Dense.

The Fine-F control has fewer parameters; report this. It is a backbone-sufficiency control, not an exactly capacity-matched isolation. Pair-F-long is the stronger matched-capacity comparison. Do not invent another head solely to equalize a parameter count.

If an existing selected-data, source_local_v3, full-access Fine checkpoint genuinely matches this initializer, objective, normalization and schedule, reuse it. A historical full-data Fine checkpoint does not qualify.

This is one main upgraded Tree hypothesis with two controls—not permission to launch another architecture portfolio. No additional scientific arm without a concrete diagnosis and replacement of an existing slot.

## 3. First implementation step: measure the actual hot path

Use a bounded native profile, not a large diagnostic campaign. On one low-M and one high-M selected-training case, inspect:

1. no-grad Q8192 full wrapper and prepared P2 decode;
2. one real ordinary training microbatch at the native shape, including forward, backward and optimizer boundary;
3. hard-only and current hard-plus-full-shadow execution separately.

Include source encoding, canonical index construction, node descriptors, memberships/control formation, receiver access, source moments, control projection, fine kernels, attention, structural cost, auxiliary reporting, host/device synchronizations, and Stage-A/refinement. Attribute inclusive and self time correctly; do not add overlapping timers.

A small internal prepared projected-action record may be added for the numerical path. Keep the existing public graph export and diagnostic `TypedSourceAccess` semantics; reconstruct full control vectors only when explicitly requested, rather than changing callers to silently consume a new meaning.

Use CPU/CUDA profiler activities and a separate shape/memory trace. Turn the profiler off for timing comparisons. Record active tensor allocations and reserved memory separately from process/device residency; the earlier report did not measure comparable training memory. Keep profiler traces local and short.

A small profile of real execution is a means to implement improvements. It is not sufficient final completion. Budget roughly two hours for the initial performance implementation, with a bounded additional remedy if needed; start the inexpensive independent controls while it proceeds.

## 4. Exact execution repair: compile the action the reader actually consumes

### 4.1 Existing equations

For one typed mechanism, let:

- A_re: receiver-to-group access;
- B_es: group-to-fine-source membership density;
- h_e in R^d_c: collective control, d_c=16 currently;
- mu_s: physical source measure;
- rho_rs = sum_e A_re B_es;
- n_rs = sum_e A_re B_es h_e;
- c_rs = n_rs/rho_rs where rho_rs>0, zero otherwise;
- w_rs = rho_rs / [(sum_s mu_s rho_rs)/(sum_s mu_s)], with eligibility and empty-row conventions exactly as implemented.

The reader does not consume every coordinate of c_rs independently. It applies affine projections W_tau c_rs+b_tau to obtain one gain for MM/ME/EM/QM and headwise score/gain terms for QE, followed by the existing nonlinearities.

### 4.2 Project before expanding query–source controls

For the linear part, define p_e = W_tau h_e. Then:

    W_tau c_rs + b_tau
      = [sum_e A_re B_es p_e] / rho_rs + b_tau,  rho_rs > 0.

On rho_rs=0, preserve the original zero-control convention: the affine projection is b_tau before later support masking. Do not lose this bias behavior.

Preserve intervention semantics explicitly: `full_access_fixed_controls` changes access but retains the old pair controls (including the zero-control/bias behavior of newly admitted pairs); `control_identity` overrides the gain and score shifts exactly as the reference does. Do not recompute a different control merely because an intervention changes membership.

Compute the small p_e once per phase, form prepared source/group projected moments, and contract them with A for each receiver block. Do not construct [B,Q,S,16] only to reduce it to one or four channels immediately. QE can concatenate its score and gain projections into an eight-channel prepared representation, or compute them sequentially when memory favors it.

**Keep tanh after the projected mixture.** Moving tanh to h_e before mixing changes the model and is not this optimization. The same applies to nonlinear group controls: retain their source-resolved construction.

In exact real arithmetic this is the same operator. FP32 operation order and the FP64-to-FP32 cast in the old soft path can introduce bounded numerical differences. Prove native output and first-gradient agreement at the established tolerances rather than claiming bitwise identity everywhere.

The supporting `HONF_Projected_Control_Algebra_Check.py` tests the identity and all operand gradients in constructed float64 CPU tensors, including empty/invalid rows. It is not a native checkpoint, GPU, or physical result.

### 4.3 Preserve the hard plan but remove unnecessary reporting work

Current `TypedHypergraphField.read` evaluates `structural_cost` on every read, including inference, and repeats preparation/incidence calculations. `source_moments` also calculates overlap-path diagnostics even when no graph export is requested.

Implement explicit internal modes:

- numerical read only;
- training structural-loss read;
- diagnostic/export read.

Normal inference needs the first. Training needs the second only where the objective actually consumes it. Detailed exports use the third. Missing diagnostics must be absent or marked unmeasured, never reported as zero savings.

Compute phase-constant preparation and incidence terms once. Combine query-dependent numerator/denominator terms across chunks so the final structural objective and its gradient retain exactly the current weighting. Do not accidentally multiply a preparation penalty by the number of chunks or change case balancing.

Keep existing input validity, checkpoint trust, phase compatibility and mathematical empty-row handling. This optimization removes optional repeated analysis, not safety measures.

### 4.4 Cache only geometry that is genuinely unchanged

A phase-local prepared object can retain canonical block membership, tree integer indices, axes, node masks, gather indices, depth propagation and source catalogue metadata. Reuse immutable geometry structure across P0/P1/P2 and hard/soft evaluations only after verifying that their actual anchor coordinates/roles/measures are identical.

For training on fixed cases, discrete geometry may be reused across epochs through ordinary case-bound metadata. Every state-dependent feature, density, control and projected action is recomputed for the current weights and current physical phase. For inverse use, changing heat preserves some geometry but not the states; moving geometry requires reconstruction. A cache may not contain hidden inverse targets or stale autograd graphs.

Keep the canonical mass-weighted coordinates and their correct pullback alive where design coordinates require derivatives. Do not detach live continuous geometry merely to speed up differentiation. Do not introduce a new GPU-to-CPU hash on every numerical read.

### 4.5 Further bounded implementation remedies

After the above, allow at most two targeted choices supported by the profile:

- replace multi-operand query contractions with prepared bmm/matmul blocks;
- reuse common QM/QE receiver-tree access when actual geometry and gates match;
- avoid padding inactive hard nodes in evaluation while retaining required training exploration;
- batch node descriptor/pooling operations;
- adjust numerical query chunking while preserving the exact sample set, physical normalization and effective batch;
- use installed, already-supported checkpointing on explicit tensor functions.

No custom CUDA/Triton project, environment upgrade, new sparse compiler, or broad tuning sweep in this round. Dense masked fine execution is a legitimate choice when every source is open. Do not repeatedly pack fully open rectangles to demonstrate a sparse API.

## 5. Training upgrade: one physical pass, local soft organizer gradients

### 5.1 Why this is needed

The current `hard_value_soft_hypergraph_forward` performs **two complete physical wrapper forwards**. Only the organizer stays differentiable in the soft copy; physical parameters/inputs are detached, and activation checkpointing is disabled. This is a plausible source of training memory and time overhead, but its measured share still requires the native profile.

A frozen physical parameter can still require activations to propagate gradients from outputs back to organizer inputs. Therefore parameter freezing does not make that second pass cheap.

### 5.2 Proposed local-context bridge

Reuse the hard physical messages and hard current states instead of re-running the complete physical model. At each typed interaction reduction produce:

    z_h = Reduce_hard(messages_theta, hard_access, hard_projected_controls)
    z_s = Reduce_soft(stopgrad(messages_theta), soft_access,
                      soft_controls_with_detached_physical_projections)
    z   = z_h + (z_s - stopgrad(z_s)).

The forward value is z_h. The physical parameters receive the gradient of the hard computation. The organizer receives a surrogate gradient from the soft local reduction evaluated on the current hard physical states. Later physical layers and the Thermal coupling propagate that signal normally.

This is an **intentional change of organizer training gradient**, not an algebraic equivalence to the old complete-soft-wrapper gradient. Record it as `local_context_shadow_v1`; retain `whole_wrapper_shadow_v1` for historical replay and the bounded training comparison. The learned hard forward model and checkpoint tensor shapes need not change.

For MM/ME/EM/QM, form hard and soft weighted reductions using the same already evaluated physical message tensor. Use detached messages in z_s. Hard control projection parameters remain live in z_h; detach those physical parameters in z_s. Planner source inputs retain the existing detached-in-training convention.

### 5.3 QE must bridge the completed attention context, not its masked log prior

A bridge inserted before a hard zero-support mask cannot restore a missing source. For QE compute:

    z_h = AttentionContext(K_h, V_h, q_h, bias_h,
                           hard_density, hard_score_control, hard_gain_control)
    z_s = AttentionContext(stopgrad(K_h), stopgrad(V_h), stopgrad(q_h),
                           stopgrad(bias_h), soft_density,
                           soft_score_control, soft_gain_control)
    z   = z_h + (z_s - stopgrad(z_s)).

Normalize each over the correct unique source union and physical measures. The soft branch is positive on every eligible source under the established stable permission convention; invalid/self sources stay invalid. The hard prediction still uses the hard support. Apply shared downstream physical output projections once to the bridged context.

This performs additional small attention/reduction work, but does not execute a second environment encoder, fine geometry/message network, complete Stage-A path, or full P0/P1/P2 wrapper.

### 5.4 What remains differentiable, and what must be tested

At unchanged weights, require:

- exact hard forward selection and physical output parity within established tolerances;
- physical-parameter and input gradients matching the hard-only reference;
- useful finite organizer gradients in every applicable typed route and the split head;
- a positive restoration path for a currently omitted eligible QE source on a constructed control and a native example where available;
- only one ordinary physical wrapper and one execution of each hard physical fine kernel in the new training path, excluding explicit activation-checkpoint recomputation;
- no update to frozen Stage-A tensors/buffers and no extra physical optimizer steps;
- reproducible curriculum/RNG counters; soft preparation must not consume additional hard exploration choices;
- source permutation and measure-refinement behavior retained for hard inference and its tested input derivatives.

Do not demand equality of the old and new organizer gradients; they implement different surrogate derivatives. Report their norms/cosines and a short matched train-only learning probe instead.

A maximum twenty-update disposable pair from e200 on identical selected-training examples may compare the two gradient policies. This is an optimization/engineering probe, not a convergence or development-accuracy claim. Its temporary states are not candidates for best-field selection and do not add unscheduled scientific checkpoint milestones.

### 5.5 Numerical precision and checkpointing

Keep the proven FP64 sensitive permission/quotient/log chain initially. Reduce dimensions and repeated work before reducing precision. A blanket conversion of soft controls to FP32 reintroduces a previously observed overflow risk.

If checkpointing the new local reductions, use explicit input tensors and detached physical tensors bound inside the checkpointed function. Never rely on a mutable global permission flag or a temporary `functional_call` parameter swap being in the same state during backward recomputation. Use the installed supported non-reentrant API where needed. Prove the gradient scopes on real native losses.

If local-context shadow proves unstable after a bounded repair, fall back to the exact fast executor with the original full-wrapper shadow and complete the main Tree500 maturation if the measured cost fits. Do not pretend the training-gradient upgrade succeeded. Do not terminate the entire round merely because one performance technique fails.

## 6. Accuracy experiment and maturation policy

### 6.1 Keep the main training question stable

Continue Tree-Lite and Pair-F from total e200. Restore the existing normalizers, optimizer states, data cursor, absolute epoch schedule, current denominator policy and allowed response callback. The Tree child explicitly records its changed organizer-gradient implementation; its e201 physical values must reproduce the parent before any update.

Retain current scalar objective weights during this main comparison. Do not simultaneously tune null weight, structural cost, temperature, capacity, physical field weights and module encoders. That would prevent interpreting an improvement.

Fine-F uses the same source-local schema, global normalizer, frozen Stage-A model, primary samples and native physical objective; it follows the same e101 denominator/atlas/null schedule where applicable. It has no structural organizer term. Do not invent response labels for it. If the maintained helper cannot express these settings, add a small explicit profile rather than silently using the legacy global feature schema.

The tiny existing null term is a diagnostic limitation, not an accuracy stop rule. Keep it in this continuation for comparability, measure its effective gradient force at a small training-only panel, and do not claim it is adequately trained simply because its coefficient is nonzero. A new calibrated response-objective study is outside this main maturation comparison.

### 6.2 Reviews and continuation

- Existing e200 is the Tree/Pair entry point, not a new initialization.
- The mandatory 100-epoch screen for fresh Fine-F occurs at e100. Continue to500 if it executes and learns; missing mature-reference fidelity at100 alone is not grounds to stop.
- Tree/Pair reviews at300 and400 are progress/correctness reviews, not automatic fidelity-promotion barriers.
- All three primary arms should complete total e500 unless unrepaired numerical failure, persistent broad divergence, user interruption or the resource envelope prevents it.
- Evaluate exact e500 and the best retained field checkpoint separately. Do not combine a best-age field with an endpoint graph.

At e500, continue Tree-Lite plus one predeclared strongest relevant nongroup control to total e1000 when:

1. the run is stable and training/validation curves still show meaningful learning or delayed organizer specialization;
2. the most recent performance forecast fits the explicit remaining budget;
3. the question answered by the extra exposure is written in the milestone note.

Neither a nonsparse e500 value map nor the weak null objective automatically prevents an e1000 reconstruction test. However, extending solely to manufacture a varying K histogram is not justified. There is no automatic e5000 continuation.

Assess nonconvergence from multiple retained windows. A single role regression between two checkpoints is not persistent divergence. Distinguish fitting limitation, overfitting, source-support collapse and budget censoring.

### 6.3 What Fine-F resolves

At the same age and dataset:

- Fine-F near Dense, Tree much worse: investigate organizer/learning distortion, not missing Dense capacity.
- Fine-F and Pair both far from Dense: the simpler backbone/source-local representation is a major unresolved factor. Do not attribute that entire gap to the tree.
- Tree beats Fine and Pair on thermal values but loses flow: report a real cross-field inductive-bias tradeoff; inspect coupling and gradient conflict before enlarging K.
- All are still improving: retain a maturation conclusion with slopes, not an architecture verdict.

These are empirical interpretations, not exclusive causal proofs. Dense differs in capacity and feature/context paths. Restoring Dense's coarse/local branch is not part of this round; any later restoration must make its information path explicit and be compared as a new architecture.

## 7. Organizer quality: evaluate the learned computation, not just the tree picture

At e500 and, if reached, e1000, collect lightweight per-case/phase/typed-route statistics on all22:

- allocated nodes, active frontier and actual receiver participation;
- distinct action signatures based on densities and projected controls, not binary supports alone;
- native eligible value support, dual control donor support, repeated paths;
- source/control distribution concentration and projected-control RMS;
- whether same-M cases differ and whether a changed prescribed context changes a consequential action;
- actual fine rows, small projected-control rows and complete time separately.

On only four representatives run:

1. normal learned operation;
2. control identity at fixed access;
3. full access at retained controls;
4. existing degree/measure-matched geometry action replacement;
5. an effective rewire with changed native source/receiver assignments;
6. root union with its changed information/work explicitly reported.

Use the existing audited interventions rather than a new suite. Fixed-weight ablations show reliance, not superiority of the representation. The separately trained Pair and Fine controls remain necessary.

### Meaningful visual evidence

Prefer interpretable density/action maps to a forest of all-to-all lines. For one representative at each M show:

- receiver access A(q,e) on physical coordinates;
- actual module and environmental donor densities for one active group;
- effective scalar/headwise controls at those same source/receiver locations;
- native support percentage written explicitly;
- physical field and signed residual beside the control map;
- a small fixed-weight effect map from removing one group control.

Do not caption a full-support reweighting as sparse retrieval. K counts receiver rules, not physical modules, independent physical causes, or a unique decomposition of the PDE. Heatmap columns labelled by source slot are not spatial coordinates.

Retain value, control-content, planning and upstream P1/P2 ancestry as separate paths. Do not reintroduce total/maximum heat into every source/background feature merely to improve a score without disclosing that architectural change.

## 8. Response and inverse tasks remain separate readiness dimensions

### 8.1 Response panel

Normal values and known-null heat increments are reported separately at e300/e500, and final e1000 if executed. Use the existing all22 feasible fixed-total transfer panel and only the selected-training atlas anchor0348. Reuse unchanged reference arrays.

Do not divide by zero reference changes. Scale by fixed training channel scales and retain raw physical increments. Report the measured weighted/null-to-native gradient ratio at a few selected-training calibration points; do not extrapolate one ratio to every update.

The selected dataset has no held differing-heat physical-reference pairs. Nonzero held physical response accuracy therefore remains unavailable. No amount of additional training or surrogate perturbation replaces those labels. The prepared twelve-solve request may be retained for the user, but not executed.

### 8.2 Small completed inverse reuse at final selected weights

Do not train a generative denoiser in this round. A cheaper field-consistent inverse test is sufficient to check whether the improved forward interface is more usable.

Use the same four representatives and two public starts, but construct starts and proposals inside a **training-support-aware capped simplex**:

    q_min <= q_i <= q_max,   sum_i q_i = Q_total.

Bounds come from selected-training physical inputs, not hidden target heat. A task whose public total makes this set empty is reported unsupported; do not change its total or widen bounds after inspecting its hidden answer. Training-range inclusion is not a physical validity guarantee.

Compare Tree joint, Tree graph block and size-matched random updates, plus the strongest final nongroup model's joint update. At most10 attempted updates per trail. Persist every accepted state immediately; observed sensors alone select steps, held sensors are evaluation only. Retain rank diagnostics and failed trial charges.

Use existing strict topology checks. Where a fixed-active-set continuation becomes invalid, shrink within the declared trial budget or evaluate a newly rebuilt ordinary candidate under the recorded policy. Do not clip negative memberships, fabricate derivatives across a topology switch, or count repeated invalid trials as a physical optimizer failure. No extensive topology algorithm rewrite is authorized.

Only expand this to a generative training experiment after the forward candidate has demonstrated reliable conditioning and a consequential organizer. Thirty-two finished trials are enough; do not launch hundreds of samples automatically.

## 9. Efficiency measurements and acceptance of the implementation

Benchmark interleaved, without profiling, on one low-M and one high-M selected case at Q1024 and Q8192. Keep precision, physical tensor shapes and query count common. Record query chunk sizes as part of the result.

Compare:

- original Tree e200 numerical path;
- exactly lowered Tree e200 path, diagnostics off;
- original full-wrapper training shadow;
- single-physical-pass local shadow;
- Pair e200 and retained Dense in matched scopes.

Report preparation, reused P2 decode, complete wrapper, ordinary forward/backward/optimizer boundary, and input-VJP separately. Do not sum inclusive times. Count host synchronizations and fine/organizer work where captured. Do not call a PyTorch module invocation a GPU hardware kernel.

Performance goals, not claims or hard scientific gates:

- remove the [B,Q,S,16] materialization from the normal Tree reader;
- stop computing the structural regularizer on inference queries;
- eliminate the second complete physical wrapper in local-shadow training;
- demonstrate lower repeated-read latency and/or training memory at the same hard predictions;
- target a substantial reduction of the old 4.5–4.9x Pair latency overhead, without inventing a guaranteed ratio.

If a proposed optimization fails exact output/gradient tests, keep the reference path and repair locally. If it is exact but not faster, report it and use the fastest measured compatible route. No minimum speed threshold should prevent the actual accuracy experiment once its budget fits.

## 10. Execution budget and scheduling

Authorized devices: physical GPUs1/2 after verifying their actual IDs. Wind scientific training remains paused.

**Primary workload:** Tree200→500, Pair200→500, Fine0→500: 1,100 additional development candidate-epochs, 165,000 primary case visits, 4,400 ordinary optimizer boundaries and 168,960,000 primary fluid queries. Shadow/atlas/null/evaluation work is additional and separately charged.

**Conditional e1000 workload:** at most two models500→1000: another1,000 candidate-epochs. Do not add a second seed or an extra architecture during that extension.

**Round envelope:** at most18 aggregate GPU-associated job-hours and12 elapsed hours, including failed attempts and closeout. These are ceilings, not a target duration. Use the measured post-optimization first five epochs to forecast primary500 and possible1000 separately. If the extended stage does not fit, finish primary500 and leave an ordinary continuation recipe; do not silently shorten an epoch, reduce the selected population or call e5000 formal training.

Schedule the inexpensive Pair/Fine controls while implementing and checking Tree. Keep matched numerical comparisons on the same device where possible. If another job occupies a GPU, do not wait indefinitely or interfere with it. Use the other authorized device or safe microbatch accumulation preserving the effective batch; label contention-affected times.

Save progress notes at the completion of implementation, each100-epoch milestone and the e500 extension decision. Avoid a chain of short processes that repeatedly reload/normalize and discard unsaved work. Existing checkpoint cadence stays100; an explicitly requested emergency stop may save a recovery state under existing policy, not create a new best-selection candidate.

Evaluation should not consume the training budget. Reuse saved all22 results; run the detailed full-field/invariance/utility package at final selection, not in full after every minor code change. Reserve roughly the last hour for inspected figures, report and repository closeout. Do not end with only a profiler report if a healthy executable training path remains within the primary budget.

## 11. Bounded remedies and explicit non-goals

Codex may revise the implementation during measured testing. Document whether a change is:

1. an exact algebra/execution change;
2. a numerical implementation/derivative convention repair;
3. a changed organizer training gradient;
4. a changed predictive architecture or scientific objective.

The first three are the intended scope. A fourth-category change needs a separately named candidate and cannot be hidden inside the same learning curve. Do not launch it in addition to this portfolio by default.

Up to two bounded remedies are allowed for local-shadow stability/performance. Up to one training-only learning-rate/scope remedy is allowed if a newly added Fine control cannot perform a basic fitting test. No validation-driven sweep.

Do not:

- remove fine sources or pool their physical values before nonlinear interactions;
- equate fewer candidate nodes with measured computational savings;
- force K variation, reduce support by an arbitrary percentage, or accept a full fallback as sparse success;
- delete the measure-consistency and donor-locality fixes;
- remove trust/security/artifact rules under the label of simplification;
- change the dataset or Stage-A normalization;
- conflate same-weight compiler parity with fidelity to a physical reference;
- let an ineffective auxiliary loss veto ordinary reconstruction maturation;
- launch a new inverse-head portfolio or new physical solve.

## 12. Required report and artifacts

The final report must begin with a short direct explanation of the Tree idea and a three-row table:

| Goal | What improved | What is still missing | Decisive evidence | Next decision |
|---|---|---|---|---|
| Predictor | | | | |
| Organizer | | | | |
| Inverse system | | | | |

Then answer A/B/C explicitly:

- **A:** advantage versus trained Fine/Pair, not just sensitivity to deleting controls;
- **B:** hard operator equivalence, disclosed dependencies, and actual executed work;
- **C:** known-null behavior, nonzero response evidence limits, and frozen reuse.

Required figures, normally no more than six:

1. Learning curves through500/1000: native roles, age-matched references, true epochs/case visits and GPU-hours; mark the e201 gradient-policy change.
2. Representative and difficult physical fields, interfaces and material-temperature residuals, same scales/masks.
3. Actual receiver-group/donor/effective-control maps, with full or sparse support declared prominently.
4. Runtime/memory breakdown before and after the implementation, unprofiled benchmark results alongside a small explanatory profiler attribution.
5. Known-null and thermal-sensitivity curves, including the actual effective auxiliary force and missing nonzero truth.
6. Completed inverse heat trajectories versus charged calls, held/observed errors, capped-simplex feasibility, rejection and rank limits.

Plot only saved numerical arrays. Inspect every retained figure. Put PDF masters and minimal required raster companions in ignored local paths, embed relative links in the report, and verify the links. Keep units and reference provenance explicit: the Thermal benchmark is analytic/shared-grid and its q-normal quantity is a proxy.

Durable delivery:

- optional fast projected-control numerical path and local-context-shadow training path;
- focused invariance, physical-gradient, restore-source, cache and checkpoint tests;
- explicit development profiles and approved lineage transitions;
- selected-data500/1000 report and compact figure index;
- normal manual continuation commands for unfinished authorized development stages;
- a statement that full-data formal training has not been launched.

Commit and push reusable code, tests, configuration and report on the non-default branch. Audit the complete outgoing commit range and use the existing artifact hook. Check remote/local tips after pushing. Raw arrays, large traces, checkpoints and one-time renderers remain local. Do not add new cryptographic, snapshot, contract or gate frameworks.

## 13. Sources and evidence hierarchy

**Primary project report:** `HONF_Proj/docs/reports/_bk/20261004_091519Z_HONF_Tree_Faithfulness_Fixed25_Development_Report.md`, reviewed uploaded copy, especially Predictor, Physical executor work and latency, Response transfer, and Inverse sections.

**Reviewed source at de6f467:**

- `src/honf_forward_core/interface_fields/adaptive_receiver_hypergraph.py`
- `src/honf_forward_core/interface_fields/typed_hypergraph_state.py`
- `src/honf_forward_core/interface_fields/typed_hypergraph_field.py`
- `src/honf_forward_core/interface_fields/direct_pairwise_control.py`
- `src/honf_forward_core/interface_fields/core.py`
- `src/honf_forward_core/interface_fields/three_term_context.py`
- `src/honf_forward_core/training/hypergraph_shadow.py`
- `docs/guides/Thermal_Tree_Faithfulness_Development.md`
- `AGENTS.md` and the fixed-quarter development protocol.

**External implementation references:** official PyTorch profiler and checkpoint documentation. Use the installed environment's supported APIs; do not upgrade to the web documentation version. Profiling with shape/memory tracing can itself add overhead. Checkpointed backward must replay the same explicit function/parameter state, not a changed global mode.

- `https://docs.pytorch.org/docs/stable/profiler.html`
- `https://docs.pytorch.org/docs/stable/checkpoint.html`

The projected-control identity and local-context-shadow design in this plan are proposed mathematical/implementation work. The small CPU algebra check supports the former's real-arithmetic equivalence and constructed first gradients only. Native speed, memory, trained fidelity and inverse usefulness remain experiments to execute.
