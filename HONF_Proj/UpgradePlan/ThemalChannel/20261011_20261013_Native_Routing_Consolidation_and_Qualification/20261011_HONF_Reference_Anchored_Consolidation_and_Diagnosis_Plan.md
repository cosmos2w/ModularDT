# HONF: reference-anchored consolidation and decisive diagnosis

**Prepared:** 11 October 2026  
**Work mode:** bounded Codex Goal mode  
**Reviewed branch:** `agent/honf-core-next`  
**Reviewed commit:** `672614dd52772b767d0bddd22297266bd33575c2`  
**Status:** proposed work; none of the experiments or optimizations below has been executed by this review.

## 0. The decision this round must deliver

Do not start another architecture reset, another fresh P/P-G/P-H portfolio, or an automatic extension of T4203/W2501. Consolidate the useful model states that already exist, recover their computation faithfully, and identify the particular cause of the remaining learning difficulty before changing the forward representation again.

The research goal remains a self-contained, jointly trained Thermal model and a portable HONF interaction implementation whose organization has demonstrated usefulness. The goal is not a frozen D-sep plus R-direct system under a new wrapper. Nor is it a requirement that every physical output share every hidden feature or that an organizer simultaneously improve accuracy, latency, sparsity, physical interpretation, and inverse design.

This round has four concrete outcomes:

1. A verified, self-contained **Run2202-compatible direct Wind checkpoint/loader path**, retaining the mature learned function without requiring a fresh 5,000-epoch fit. This is a compatibility migration, not a newly trained predictor.
2. One **function-preserving execution improvement**, preferably factorized evaluation of the existing first affine layers, with actual output, gradient, update, memory, and complete-call comparisons. Retain the original backend wherever numerical qualification fails.
3. At most **one short causal learning comparison per dataset**: Wind tests the concrete old/new training-objective discrepancy; Thermal tests the dominant demonstrated source of adverse velocity updates. These are diagnostics, not a hyperparameter search or automatic production promotion.
4. A defensible **organizer decision**: identify the useful operation an organizer should perform and whether existing evidence supports investing in it. Do not train another organizer merely to make the exported graph more recognizable.

Completion does not require claiming a new universally superior model. It requires shipping useful compatibility/execution work and answering the stated questions with explicit scope. A failed short diagnostic must eliminate or weaken a hypothesis, not become the pretext for another unplanned family.

### Boundaries

Preserve the 953 previously protected observations/bindings, all older checkpoints and reports, current T4203/W2501 stopped states, and all user-owned files. Extend the existing inventory to newly created durable scientific artifacts without rewriting old receipts. Keep WindTEST locked and solver usage at 326/326. No new physical solve, synthetic replacement for a physical target, inverse campaign, new full-TRAIN epoch campaign, or automatic 5,000-epoch launch is authorized by this plan.

Read `AGENTS.md` and applicable repository instructions first. Check the actual working tree and branch. A documentation-only change beyond the reviewed commit can be recorded; a changed training/core implementation requires rechecking the affected assumptions. Do not reset, clean, stash, relocate, or overwrite unrelated user work.

## 1. What the current evidence establishes—and what it does not

Primary sources, relative to `HONF_Proj/`:

- `docs/reports/_bk/20261010_195650Z_HONF_Interaction_Preserving_Recovery_20261010.md`
- `docs/reports/_bk/20261010_214021Z_HONF_Interaction_Preserving_Recovery_20261010_Appendix.md`
- `docs/guides/Interaction_Preserving_Joint_Recovery.md`

Use the recent formal comparison and unified-refinement reports only to reconstruct named historical reference points. Do not rerun the entire historical research archive.

| Observation already reported | Supported conclusion | Conclusion not supported |
|---|---|---|
| The Run2202-to-P bridge maps 52 direct tensors and gives bit-exact physical outputs, standardized residuals, and center/query gradients on three target-free TRAIN scenes | The P implementation can recover the tested mature Run2202 computation; this is a real representability/compatibility result | Every fresh P recipe will reproduce Run2202's training trajectory, or every input/backend is already qualified |
| W2501/P stopped at e1000, with 18,000 updates, whereas Run2202 completed e5000 with 90,000 updates | Its present checkpoint is weaker than the mature references | P has been shown intrinsically incapable of the earlier accuracy, or its e5000 performance is known |
| W2501 Q4096 near/downstream monitor RMSE fell from 0.229295/0.161665 at e500 to 0.157642/0.106913 at e1000 | Substantial learning was still occurring before the stop | Another 4,000 epochs is guaranteed to solve the problem |
| The selected T4113/P-H improves full-native u/v/omega/T means over independently trained P, but remains above D-sep in v | There is a provisional matched-cohort collective learning benefit with unresolved absolute fidelity | All organization has failed, or the collective block uniquely explains the gain rather than added capacity/training differences |
| Three fixed TRAIN probes have errors close to the DEV errors in the P family | The earlier J-family overfitting diagnosis cannot simply be inherited | The full training population has no generalization gap; three probes are not representative |
| P-H costs more than P with every direct path still evaluated | The added dense collective branch has measured overhead | Any hypergraph organization must be slow, or a sparse executor was demonstrated |
| The native-curl revision improves omega and some flow metrics, with a flow-v/omega gradient cosine of about -0.521 on one selected TRAIN probe | The loss interaction merits a controlled check | One raw gradient cosine establishes the cause of all transverse-flow error |

### 1.1 Maintain a precise reference point, not a remembered aggregate

The recent efficient operating point is useful, but was not uniformly equal to Dense on every channel. The unified formal report gives Run2202 near/downstream vector means of approximately 0.092951/0.05414 m/s against Dense2103's 0.074455/0.039102; Thermal3904 fluid-T is approximately 0.306155 against Dense1804's 0.237375, with other Thermal roles favoring the cheaper model. Earlier HONF variants have their own distinct comparisons.

Do not tell the user that the earlier progress was imaginary. Identify the exact checkpoint, metric, physical panel, and computational scope that made it useful. Equally, do not define the recovery target as an artificial model assembled from the best value of every historical row.

### 1.2 Correct the computational scope before interpreting a regression

The latest appendix/main report measures these different workloads:

- Thermal segmented P/P-G/P-H: **0.080703/0.108318/0.133347 seconds summed across two detail-scene complete-call medians**. These are not individual-case latencies.
- Wind segmented P/P-G/P-H: **42.899/48.731/58.106 seconds summed across two complete 3-D volumes**, totaling 9,747,072 cells. These are not Q8192 role-panel times.
- Earlier Wind Run2202 millisecond timings concern much smaller declared receiver panels. Their numerical ratio to the full-volume seconds is not a speed regression.

The within-panel H/P overhead is real. Whether the direct implementation has become materially slower than Run2202 requires a matched-scope measurement.

### 1.3 Interpret the terminal failure correctly

Retain the report's terminal decision, “Recovery failed at strong-reference qualification,” unchanged. Explain its scope: the measured checkpoints did not qualify as replacements. This is different from rejecting the P function class, proving a mature plateau, or measuring full-data organizer harm. Full-TRAIN Thermal has only H, without a matched P; full-TRAIN Wind has only P, without an organizer.

The full follow-up used a 5,000-epoch schedule, warmup 20, hold through e1000, and stopped at e1000. The completed Run2202 recipe held through e2000. Learning age and schedule are consequential, not footnotes.

## 2. Findings from the implementation that should guide the work

The following source observations were inspected at the reviewed commit. Reconfirm only changed or required sections locally.

| File / symbol | Observed implementation | Consequence for the next experiment |
|---|---|---|
| `src/honf_forward_core/interface_fields/interaction_core.py`: `InteractionContextCore`, `NonlinearFieldReadout` | Two nonlinear MM/ME/EM rounds; source-conditioned values before reduction; source-specific receiver MLP before the nonlinear field head | Preserve these semantics. Do not diagnose a missing direct path that is now present |
| `.../interaction_preserving_joint.py`: `InteractionPreservingJointCore` | P inherits the direct readout and delegates its ordinary prediction to it | Use the existing bridge to recover mature weights, not another fresh-training exercise to prove compatibility |
| `.../interaction_refinement.py`: `RefinedNonlinearFieldReadout` | All-fine inference can bypass the refinement machinery; training can still compute it | Inference equivalence does not establish training equivalence |
| `Case_WindFarm/src/windfarm/training/unified_task.py`: `loss_terms` | The old path includes `base_loss`, with weight `0.1 / message_scale^2`; main/full physical terms are also assembled | Audit the actual Run2202 training identity and auxiliary gradient reach |
| `Case_WindFarm/src/windfarm/training/joint_task.py`: `_loss_terms` | The new P/J provider returns the five component-scaled physical role terms, without the old base approximation term | This is a concrete changed training problem, not just an organizer change |
| `.../interaction_refinement.py`: `SourceRefinement.auxiliary`, `source_features`, `read` | The approximation uses `base - fine.detach()`; the base is computed from live context states | Detaching the fine target does not by itself stop the auxiliary from updating shared context through the base; measure the actual path |
| `.../interaction_preserving_joint.py`: `_read_source_features`, `_read_affine_source_features` | Thermal already has separate flow and affine source-read MLPs; their context and collective receiver projection remain shared | Do not propose separating these already separate read MLPs as a new repair |
| `Case_ThermalChannel/src/channelthermal/joint_regional.py` | Predicted environmental-flow features are explicitly rejected in this joint configuration | Shared latent features do not enforce predicted-flow-to-temperature transport |
| `Case_ThermalChannel/src/channelthermal/flow_curl.py` | Physical u/v are differenced on a native cell-centered stencil; omega has no independent projection; off-grid requests are rejected | Preserve the actual discrete capability and measure its cost/gradient effects; do not advertise unrestricted continuous-query omega |
| `Case_ThermalChannel/src/channelthermal/training/joint_task.py` | Equal flow/temperature family weights, separate q/response/operator terms; stored TRAIN velocity appears only in the supervision-side thermal operator | Do not claim predicted-flow thermal consistency, or infer auxiliary dominance from coefficients alone |

The old Wind base approximation might have helped context learning, been neutral, or been harmful. The code establishes the difference, not which of those explanations is correct. Missing historical calibration must remain missing unless recovered from actual artifacts. A newly measured scale is a new calibration, never a retroactively recovered receipt.

## 3. Research principles for this reset

### 3.1 Separate four types of change

**Compatibility:** different packaging that recovers a known trained function.  
**Execution:** cheaper evaluation of the same mathematical function.  
**Training:** different gradients, objectives, samples, or optimizer trajectory for a given function class.  
**Representation:** different information pathways or representable functions.

Every proposed edit and experiment must name which type it changes. Do not describe a representation/training change as an execution optimization. Do not require fresh random initialization to validate a compatibility change.

### 3.2 Preserve competence, not just an initial identity

Zero collective projections recover a randomly initialized P function. They do not preserve the accuracy of a mature predictor after co-adaptation. The Run2202 trained-state bridge is therefore more immediately useful than another demonstration that P-G and P-H coincide at random initialization.

For new architecture work after this round, preserve a competent reference at attachment and retain a matched continuing control. Fresh-training scientific comparisons may still be needed later, but should not be the default way to recover a function already stored in a checkpoint.

### 3.3 Keep joint Thermal learning; stop treating maximal sharing as the definition

No frozen D-sep inference dependency may enter the new joint candidate. Read-only D-sep comparisons and diagnostic flow-only updates on copies of the current joint model are permitted, with clear labels. They are not deliverable joint-model replacements.

One model can have output-specific projections and a physically directed training dependency. Conversely, concatenating two independently trained legacy predictors does not establish the requested unified model. Any eventual sharing change must be justified by the diagnostic evidence below, not made in advance.

### 3.4 An organizer needs a declared job

An additional learned context block, an execution-sharing index, and a physical causal interaction graph are different objects. P-H primarily tests the first. It retains the full direct work and adds collective work; sparse execution is not a built-in consequence.

The next organizer investment must target either:

- better generalization at a comparable capacity/cost, with an independently trained competent non-organized control; or
- reusable interaction computation at a specified physical-error allowance, with measured complete cost.

Choose one primary benefit before training an organizer. Physical source causality and inverse-design success are not prerequisites for calling a computational representation useful, but must not be inferred from it either.

## 4. Stage A — restore the reference and isolate changed training conditions

### A1. Create one compact comparison manifest

Reuse the current manifests and receipt readers. For each selected reference record: actual checkpoint hash/age, data membership, initialization lineage, transform/profile, E, Q and role mix, effective/microbatch sizes, source frame, output law, loss terms, optimizer schedule/moments, sampling namespace, backend, and evaluation panel.

Prioritize:

- Wind Run2202 e5000, W2501 e1000, W2401/P e1000/e2500, and Dense2103's retained mature checkpoint.
- Thermal3904/3901 as the efficient historical composed reference, Dense1804 as the strong nominal reference, 3906 as a completed self-contained joint reference, and T4111/T4113 plus T4203 as current joint candidates.

These references serve different purposes; do not declare a single overall winner by averaging their incompatible channels. Include other old HONF checkpoints only when they resolve the user's identified accuracy–cost milestone. Do not launch a broad historical tournament.

Recover any **already saved** Run2202 e1000/e2000 physical or monitoring records. Compare with W2501 at the same age and common physical metric where possible. If a necessary checkpoint is absent, report the comparison unavailable. Do not interpolate a missing model state or convert a log into a new native evaluation receipt.

### A2. Complete the existing Wind bridge into a useful migration

Start from the reported 52-tensor mapping; do not invent a second mapper. Strictly map the mature Run2202 direct weights and retain its exact normalizer, height profile, feature definitions, physical source identities, and E8 inventory.

On a small native low/mid/high-M TRAIN panel, verify ordinary predictions, standardized residuals, source-center/query gradients, chunk behavior, and self-contained reload. Then evaluate the migrated and original checkpoint on the same frozen fullVALID90/Q8192 panel, reusing identical native IDs, targets, masks, and reducers. The migrated checkpoint must not access the old learned checkpoint after loading.

Keep the original checkpoint untouched. The new artifact is explicitly named a **Run2202-trained compatibility migration**, with inherited full-TRAIN exposure and e5000 age; it is not a fresh subset-trained P result, a new physical accuracy gain, or proof that the current P trainer reproduces the old fit.

A mismatch is localized before any training: input/transform, direct state mapping, preprocessing, backend arithmetic, then output conversion. Do not fix it by fitting weights or hiding a residual correction to the old predictor.

### A3. Audit a real optimizer boundary, not just the inference graph

Using cloned states and fixed native TRAIN batches, enumerate the exact old and new loss terms, coefficients, denominators, active parameter groups, per-group gradient norms, clipping, and AdamW state. Distinguish:

1. physical-loss-only gradients under identical tensors and samples;
2. old base-approximation gradients and which shared tensors they reach;
3. effects of including auxiliary parameters in global clipping;
4. sampler namespace, query identities, and transforms;
5. initial state and learning-rate history.

The all-fine main/full replay may reduce to the same physical objective when both predictions coincide. Check that algebra and the actual values; do not count an identical replay as extra independent supervision. The base approximation remains a separate term.

Use no more than 32 disposable update steps per compared path. These are numerical/optimizer probes, not scientific fitting or a mature trajectory comparison. Restore all states between counterfactual updates. Missing old message calibration forbids a claim of exact historical optimizer reproduction.

**Stage A decision:** is the current gap already attributable to a software/contract mismatch, only to unequal maturity/changed training conditions, or still unexplained? Multiple possibilities may remain; report which observation would distinguish them. Do not call an unisolated difference the root cause.

## 5. Stage B — make the same computation cheaper

### B1. Measure the current cost once, at matched scopes

Use an actually available authorized GPU, not a historical device index assumption. Preserve unrelated processes. Time original Run2202, migrated P, current segmented P/P-H, and the relevant Thermal joint path on the same scope within each comparison.

Required scopes are Q1024 and Q8192, low/high-M, ordinary complete physical output conversion and CPU return. Thermal must include its declared interface/material extraction; identify its fluid/temperature/native-curl query unions separately. Price a prepared read separately. A single larger Wind plane may be added if needed; do not repeat two full 3-D volume exports just to compare millisecond paths.

Use two warmups and at least seven interleaved synchronized measurements, alternating order. Report median, range, device/dtype, allocated/reserved memory, and host/transfer boundaries. Do not add overlapping preparation/read/complete timers. Take only one short profiler capture of the dominant scope; do not build a profiling campaign.

### B2. Primary optimization: factor the existing affine projections

The receiver MLP currently begins with an affine map of concatenated repeated features. Write its preactivation as

\[
 a_{qi}=W_s z_i+W_g g+W_q\gamma(q)
       +W_d\gamma((q-x_i)/L)
       +W_\ell\gamma((q-x_i)/\ell_i)+W_f f_q+b.
\]

Its original message is

\[
 m_i(q)=W_2\,\mathrm{SiLU}(a_{qi})+b_2.
\]

Evaluate the source projection once per source, global projection once per scene, and query-only projection once per receiver. Only the relative-geometry projections remain pair-dependent. Sum these terms **before the original SiLU**, then apply the original second affine layer and the unchanged physical reduction.

Apply the same idea to the first affine layer of the MM/ME/EM message MLPs when profiling supports it. The destination and donor projections can be reused while their relative geometry remains pair-specific.

This is real-algebra factorization, not a new low-rank model. It does not move a nonlinearity through a sum, merge physical sources, alter normalized measures, or replace nonlinear messages with pooled donor states.

Implementation requirements:

- Reuse slices of the existing weight tensor, not independent learned copies. Preserve checkpoint keys and parameter count.
- Add the bias exactly once. Keep MM self-exclusion, masks, physical measures, input features, normalization, and final reductions unchanged.
- Preserve autograd to both original parameters and physical coordinates. Do not detach cached projections.
- Cache learned projections only within a valid current preparation/forward lifecycle. Parameter or configuration changes invalidate them through existing guards.
- Do not reuse an old training autograd graph across optimizer steps.
- Avoid constructing a full concatenated Q×M×input-width tensor merely to feed the factored path. Query chunks and native masks must retain their existing meanings.
- Default historical checkpoint execution remains unchanged. Introduce a narrowly scoped, explicit execution option, not a new architecture family.
- Do not enable compilation, precision reduction, altered activation functions, new samplers, or larger batches in the same comparison.

If the measured dominant cost is not these projections, make one alternative **semantics-preserving** edit supported by the profiler, such as reuse of identical request-local geometry or removal of unrequested diagnostic tensors. Do not remove correctness guards globally to win a timer. Document the substitution before implementing it.

### B3. Numerical qualification and useful deployment

Test forward outputs, physical-unit outputs, losses, parameter gradients, input JVP/VJP where the interface supports them, and one AdamW update against the original backend on identical batches. Test padding, input permutations, chunks, stale-state rejection, reload, zero heat, and affine temperature increments using the existing test helpers.

Reassociation need not be bit-exact in FP32. Reuse the relevant pre-existing numerical criteria and record failures without widening them after inspection. A same-weight FP64 check can establish the algebraic relation but cannot erase an FP32 deployment failure.

The reported B48-versus-six-B8 omega/interface failures are inherited numerical qualifications, not evidence that every new same-B8 optimization fails. Keep these tests separate. Conversely, passing loss/gradient tolerances does not mean every derived output passed.

The practical target is a noticeable complete-call or peak-memory saving—approximately 20% is a useful initial engineering target, not a promised outcome. No minimum speedup may be obtained by changing the prediction, output scope, or precision. Retain the original backend as authoritative for unqualified paths. Ship a qualified optimization even if a different research fidelity gate remains open.

## 6. Stage C — two small learning questions, not another model portfolio

### C0. Establish a representative fixed-weight fit diagnosis

The old three-scene TRAIN probes are insufficient. Select from metadata a bounded fixed TRAIN audit with about 24 Thermal cases spanning module count and operating context, and 12 Wind layouts with their available three directions. Keep these within each model's declared fitting exposure and do not treat a full-TRAIN-trained checkpoint as subset-trained.

Use fixed receiver IDs and the same output metrics/reducers as DEV, with common strata or explicitly standardized comparisons. Reuse all DEV22/DEV24 and saved fullVALID measurements rather than repeatedly rebuilding full fields. Audit both e1000 and the selected segmented endpoint when already available.

Determine whether errors are mainly present on TRAIN as well as DEV, whether late validation still improves, and which channels/regions dominate. Read physical field amplitude and residual shape; do not rely on a total normalized score or a near-zero relative-error denominator.

### C1. Wind: test training equivalence before inventing a new interaction law

The primary question is whether the old base-message approximation supplied consequential training gradients to the shared context that the new field-only P recipe lacks.

First, recover the actual Run2202 auxiliary setting and initial/moment provenance where available. Inspect the live `base - fine.detach()` path, not only its label. The code suggests possible shared-context gradients through the base, but their scale and effect must be measured.

Authorize one pair only if this term has non-negligible measured shared-context influence and the comparison can be made honestly:

- **W-control:** the selected direct physical objective.
- **W-aux:** identical direct computation and physical objective, plus the declared historical-style training-only base approximation.

Both must use the same starting direct tensors, native samples, transforms, batch denominators, optimizer state convention, LR schedule, and auxiliary initialization. Both may compute the auxiliary for measurement; only its gradient contribution differs. Reuse the existing low-width `SourceRefinement` implementation. No router/sparsity objective, new contextual network, or changed field head is added. The auxiliary is discarded for P deployment, so the inference function class and cost remain unchanged.

Prefer a compatible retained segmented checkpoint, such as W2401/P e1000, under its exact subset exposure, or an already retained matching Full checkpoint with its own explicit lineage. If a full-TRAIN-trained checkpoint is used, label the entire exercise full-exposure warm-start diagnosis even when updates sample only a subset. Never present it as a fresh subset-generalization comparison.

If historical calibration cannot be recovered, there are two honest choices: stop at the measured gradient discrepancy, or perform a **newly calibrated** bounded diagnostic using a single TRAIN-only rule stated before evaluation. Do not claim that the latter reproduces Run2202. Any preparatory fitting of the small auxiliary must fit inside the disposable/calibration update allowance and be identical before the pair branches.

Use up to **400 optimizer updates per arm**, reviewing at 100/200/400, with identical held-fixed TRAIN and exposed DEV metrics. Preserve existing optimizer moments when compatible; otherwise reset both identically and label the reset. Do not silently apply an e2500 terminal learning rate to a recovery experiment, or change only the candidate's LR.

A positive result supports a training-only refinement candidate, not the claim that the omitted auxiliary explains the full historical gap. A negative result closes this particular hypothesis for the tested state and scope. If Stage A instead finds a concrete implementation bug, spend this pair on corrected-versus-original execution/training under the same conditions; do not add another pair.

### C2. Thermal: identify the harmful update before changing sharing again

Use the existing joint P model as the diagnostic host, preferably the retained T4111/P e1000 state with its known live schedule. P avoids confounding the question with an additional organizer. Also inspect the selected P/H endpoint without training to check whether the observed interaction persists.

On several fixed native TRAIN macrobatches, record gradients by actual parameter path for u, v, p, native-curl omega, fluid/surface/material T, q, response, and the thermal operator. Use one common zero-filled parameter basis; distinguish task-specific heads, shared encoders/context, and collective projections.

Go beyond raw gradient cosines. Clone the checkpoint and AdamW state, calculate the actual proposed update, and evaluate the one-step change on a disjoint fixed TRAIN probe. For interpretation, report

\[
 d_v=\nabla_\theta L_v^\top\Delta\theta,
\]

where positive values predict a local increase in v loss. AdamW, momentum and clipping make the mapping from gradients to updates nonlinear: do not add per-loss Adam updates as an exact attribution. Use independently cloned counterfactual updates to check the dominant hypothesis.

Select **one**, using TRAIN evidence only:

- If adverse velocity updates are primarily associated with temperature-family gradients on shared context, test a diagnostic that blocks those gradients into the shared flow/context path while still fitting the temperature read/head. The control retains ordinary joint gradients.
- If adverse updates are primarily associated with the derived-omega objective, test a diagnostic with the omega supervision contribution removed while retaining the exact native-curl output computation. All other joint losses and channels remain unchanged.
- If neither pattern is reproducible, do not invent a gradient surgery. Use a flow-only diagnostic on the same current joint core to distinguish inadequate flow fitting from competition, or stop with the hypothesis unresolved if the remaining budget cannot support a clean comparison. This replaces, not supplements, the pair.

The selected pair has at most **400 updates per arm**, the same checkpoint, sampling and optimizer convention, and reviews at 100/200/400. No new heads, larger widths, changed environment resolution, extra losses, PCGrad/FAMO/regularizer sweep, or loss-coefficient search is allowed.

All modified-gradient/removed-loss variants are **diagnostic controls**, not automatically acceptable joint replacements. A flow-only control cannot be promoted as the requested joint model. Removing a target term and sacrificing that channel is evidence of a tradeoff, not an accuracy solution.

Judge v together with u/p/omega and all temperature roles/tails. Roughly 10% v improvement against the continuing control, without material harm to other priority fields, is a useful signal for a later targeted change; it is not a retroactive qualification threshold for the old runs. Report all outcomes even when the signal is weaker or inconsistent.

### C3. Stop and decide

Both datasets together have a maximum of **1,600 scientific optimizer updates**: two pairs × two arms × 400. No pair is mandatory when its preconditions fail. At most **128 additional disposable/calibration updates** may be used across the entire campaign, including optimizer parity, profiling, and any auxiliary preparation. Failed attempts count.

Do not add a third recipe when a diagnostic loses. Do not extend a short diagnostic to 2,500 epochs. Keep T4203/W2501's existing e1000 run artifacts/caps unchanged. An evidence-backed suggestion to mature an existing full-data lineage is a user decision after this report, not an implicit launch.

## 7. Stage D — make organizer value a precise research question

### D1. Reassess what the current P-H result buys

Use already saved P/P-G/P-H metrics, parameters, interventions and costs. Preserve the positive Thermal comparison: H improves several fields versus P and keeps its stated priority guards. Also preserve that H fluid-T does not beat P-G's full-native mean, the shared direct path is still dense, and H adds approximately 65% cost in the reported two-scene Thermal measurement.

For Wind, near-wake deterioration and additional cost outweigh a small downstream mean gain for the current stated preference. This is a negative result for this residual construction, not for every possible use of hypergraphs.

Do not train another matched-capacity control in this round just to turn the existing result into a publication claim. Specify that comparison as missing where it matters. Same-weight interventions demonstrate dependence; they do not replace independently trained comparisons.

### D2. One bounded opportunity probe, only when cost evidence justifies it

The preferred future direction is **sharing/reusing receiver-local source-conditioned computation**, rather than adding another dense scene-to-edge-to-scene network while keeping every original read. Before authorizing a learned implementation, check whether the relevant current interaction values actually admit useful reuse.

This probe is optional, target-free in its formation, uses frozen weights, and has no optimizer or new solver calls. Limit it to four metadata-selected TRAIN scenes per dataset, with small disjoint formation/check receiver panels inside native support. Preserve source identity and type.

On the actually dominant receiver-message path from Stage B, measure whether nearby/role-compatible receiver groups have sufficiently similar **source-resolved nonlinear messages or affine coefficient functions**, not just similar attention weights. Use a fixed geometry grouping and one deterministic feature-based grouping on the same bounded panels, with equal group capacity and the same explicitly stated message approximation/reconstruction rule. Fix that rule before comparing reference errors. Do not fit an entire second predictor.

A possible implementation is a low-dimensional local message basis fitted on the formation panel, with coefficients obtained only from legal query/geometry features; evaluate it on the disjoint check panel and pass its approximate messages through the unchanged downstream physical decoder. A full-message SVD alone is only an offline compressibility diagnostic. It is not a deployable method, a speed result, or permission to obtain missing coefficients by evaluating all of the supposedly avoided fine messages.

Reject this opportunity unless all three are demonstrated:

1. the approximation retains the necessary component/role distinctions on the disjoint check receivers;
2. physical-reference error, not only model distortion, remains within a declared allowance relative to the frozen host;
3. the deployable work estimate, including formation, access, coefficient computation, fallback, and remaining all-source/context reads, has a credible break-even point for a named workload.

Use approximately 5% per-role mean and 10% tail degradation as a **prospective opportunity screen**, not a physical guarantee or a revised historical gate. Apply stricter existing application requirements where defined. Do not select the allowance after inspecting results. If the frozen host is physically weak in a role, passing its relative allowance is not application qualification.

Do not construct recursive trees, force small K, sweep rank/group counts, or write a new sparse GPU kernel. If formation itself requires the full fine bank, identify the limited repeated-use workload that could amortize it; otherwise there is no saving. Negative or merely low-rank evidence should stop the learned-organizer proposal rather than trigger more fresh forward fits.

This probe is not another pruning/omission campaign: it tests sharing of receiver-dependent computation after source-conditioned nonlinear formation. Existing R-group, packet and coalescence failures remain relevant limitations. Do not rename an old failed cover as a new successful organizer.

### D3. Interpretation and excitation limits

For a fixed Thermal geometry, the qualified response block is

\[
 T(q,h)=\sum_i K_i(q)h_i.
\]

A single nominal heating vector constrains a contraction of K, not every column independently. Opposite signed transfers supply one independent direction, not two. Extra receiver points do not increase the number of independently excited heating directions at that geometry. Shared parameterization and the supervision-side operator can add constraints, but do not convert every learned column into an independently validated physical Green's function.

Therefore, physical causal hyperedge identification is not an automatic consequence of reconstruction training. Computational usefulness can still be studied without causal claims. Keep every original source column/ID and distinguish input-to-output sensitivity, learned message routing, and physical finite-response validation.

The frozen solver ledger is not permission to fabricate response labels. Existing saved response pools can be replayed only as a small final check after a material prediction change; do not run them repeatedly to select a short diagnostic.

## 8. Data, numerical and execution safeguards

### Existing invariants

Maintain input-only organization, separation of source IDs from numerical features, heat-independent Thermal flow, affine heat application, nonlinear Wind output, native coordinates and vector transforms, correct solid masks, and exact definitions of physical roles. Do not apply a second Wind rotation. Do not impose zero divergence on the analytic Thermal benchmark: its stored reference is not divergence-free.

Native-curl omega is a discrete adapter capability. Report the additional receiver union, padded execution and derivative domain. Do not infer arbitrary-query or resolution-transfer support from native stencil tests. Do not remove the solid-node rule or replace the stored q-proxy with continuum AD flux.

The inherited strict batch-shape output mismatches remain visible. Their small absolute size does not explain a several-fold physical error by itself. Test prediction/metric sensitivity before attributing learning failure to numerical order.

### Dataset exposure

Thermal fixed25 is TRAIN150/DEV22; Wind fixed24 is TRAIN72 direction rows from 24 layouts and DEV24 from eight layouts, not literally 25% of original TRAIN. Preserve the missing-M18 disclosure. The canonical89 and fullVALID90 panels have been exposed for model selection. Wind direction rows are clustered by layout and physical role masks overlap. Do not label point counts, directions, or repeated perturbations as independent samples.

Checkpoint migration inherits its original training exposure. New calibration uses TRAIN only. Formation/check query separation in the optional opportunity probe does not create unseen-geometry validation.

### Storage and process handling

Use the established `/data` artifact/cache location for large new output. Check space before launch and reserve space for a checkpoint plus atomic-save temporary file. Never fill the root filesystem with full-volume diagnostic arrays. Store compact metrics and selected planes by default.

Respect `HONF_WIND_CATALOGUE_CACHE_DIR` and the proven file-descriptor requirement. Validate the environment's current hard/soft limits before launching; the previous working setting was soft 16384 under hard 1048576, not an assumption about the present machine. A cache loader must distinguish an observable resource failure from an ordinary cache miss. If fixing `EMFILE` handling or mmap lifetime is necessary, keep it a narrow regression-tested engineering change and do not alter cache identity or scientific sampling.

Do not silently rebuild gigabytes of cache after catching every `OSError`. Reuse existing verified entries; do not move or delete inherited caches. Keep process ownership, clean-stop status, and failed-attempt charges explicit without a new scheduler or monitoring service.

## 9. Budget and bounded autonomy

Proposed campaign ceiling: **8 aggregate GPU-associated hours and 10 elapsed hours**, with the last **two elapsed hours** reserved for testing/report/Git closeout. Count setup, stalls, failed attempts, profiling, evaluation and disposable optimization. A ceiling is not a target to spend.

The scientific update limits in Stage C are independent hard caps. All new short fits remain confined to their explicitly sealed diagnostic sampling protocol. Full-data checkpoints may be read, migrated, benchmarked, or used as labelled warm-start controls, but no new full-TRAIN epoch campaign may be launched.

Suggested allocation priorities are Stage A first, one Stage B implementation/measurement cycle second, then the two Stage C questions. Stage D is optional and must fit inside the remaining allowance. Do not sacrifice a decisive comparison or delivery reserve for more figure exports.

Codex may autonomously fix a directly demonstrated bug, improve the factored executor, or correct a diagnostic harness before delivery. It may not change the scientific question, open an extra arm, widen a numerical tolerance, alter data membership, or add a new architecture after seeing a disappointing result. A source/math change creates a named new child/protocol; it never silently resumes an old scientific identity.

## 10. Deliverables and decision gates

### Required durable code and artifacts

- Reusable strict Run2202 direct-state migration/loading support, extending rather than duplicating the existing bridge.
- An opt-in function-preserving executor/helper and focused tests, or a documented measured rejection with the original execution retained.
- A reusable comparison/diagnostic entry point using existing providers, native samplers, LossTerm denominators, checkpoint loaders and TrainingEngine infrastructure. Do not create a parallel training framework.
- Explicit short diagnostic configurations and exact exposure/optimizer conventions for whichever pairs actually ran.
- A compact main report, small machine-readable comparison tables, and one evidence index pointing to existing/native receipts. Generated checkpoints, arrays, figures and one-time renderers stay ignored.

Suggested new documentation names are `docs/reports/_bk/20261011_224701Z_HONF_Reference_Anchored_Consolidation_20261011.md` and `docs/guides/HONF_Reference_Anchored_Development.md`. These are proposed output names, not existing files. Do not overwrite the previous recovery reports.

### Four decisions, not one opaque pass/fail

| Decision | Required evidence |
|---|---|
| Mature Wind function retained? | Strict learned-state migration, same-panel physical comparison, self-contained reload, stated numerical coverage |
| Same-function execution improved? | Qualified original/factored parity plus complete cost at identical workload; no prediction/precision/scope change |
| What limits present joint learning? | Matched-age evidence, representative fixed-weight fit audit, and the bounded controlled gradient/objective comparison where justified |
| What organizer experiment is now justified? | A specified useful operation and evidence of either learning value or exploitable computational reuse—not heatmap appearance alone |

The first two can succeed without claiming that Thermal's strong-reference accuracy gap is solved. A provisional organizer finding can be recorded without pretending it is deployment-ready. Conversely, a restored mature function is not evidence that a different fresh trainer is equivalent.

### Report format

Keep the main report to roughly 1,500–2,500 words plus compact tables. Use at most four figure families:

1. reference/age/physical-fidelity comparison with the relevant learning interval;
2. original versus factored complete cost at common Q, and one native residual parity example;
3. the two diagnostic comparisons, including unchanged-priority-channel tradeoffs;
4. one actual computational organization/reuse example only if Stage D produced meaningful evidence.

For native field plots, annotate quantitative relative L2 and RMSE from unclipped valid arrays. Retain common scales within a component, native masks, source IDs, and explicit query scope. Do not export every checkpoint and every field simply to make the report look complete.

The opening conclusion must say what can now be used, what remains weaker, and which single next research investment is justified. Put hashes, rejected harness attempts, and exhaustive numerical details in the existing evidence index rather than hundreds of lines of repeated prose. Do not omit failures; organize them so they do not obscure the scientific result.

### Final checks

Run focused affected tests and a bounded legacy compatibility replay. Preserve inherited failed-test classifications; do not claim a fully green broad suite that was not run. Confirm all owned candidate processes stopped and all scientific states saved. Recheck the protected inventory, user-owned files, TEST lock and solver ledger.

Audit all outgoing commit objects and execute the existing pre-push artifact checks. Push only durable allowed source/tests/configuration/documentation. Verify remote/local branch tips. Report completed optimizer updates, actual epochs where meaningful, measured resource use, backend qualification, and unresolved questions.

## 11. The intended outcome

The practical reset is to stop repeatedly exchanging an existing useful predictor for an unproved organization hypothesis. Recover the trained computation where it already exists, eliminate demonstrably redundant execution without changing its meaning, and test a small number of concrete learning differences.

Only after that should HONF add a new learned organization mechanism. Its value must come from a named reusable computation or a controlled learning advantage—not from the fact that its diagram contains hyperedges.
