# HONF Round11 — Thermal-first backbone development and formal-run handoff

**Purpose:** develop an actual reusable hypergraph-organized neural field, make its execution practical, and give the user a working route to full-TRAIN experiments without another multi-day autonomous portfolio.

**Reviewed source:** `cosmos2w/ModularDT`, `agent/honf-core-next`, remote tip `b87773f21c1da6433702a16df9d3a9c6c21b8e5c` on 21 October 2026. This identifies the review, not a requirement to reset later work. Read the current workspace and preserve subsequent user changes.

**Evidence:** the supplied `HONF_Round10_Continuous_Unified_Organization_Report.md` and the source paths in §14. The report describes U training as deferred. The user's newer instruction says U-D/U-H/U-S have been launched independently; that newer instruction governs. Their live progress has not been verified in this plan.

**This is an implementation and bounded research task, not a historical audit-closeout task.** Do not complete missing Round10 paperwork, reproduce all 104 boards, rebuild its operation ledger, or make old campaign acceptance services a prerequisite to useful work.

## 0. The job, in order

1. Read existing Thermal U run status once. If suitable completed checkpoints exist, evaluate them first. If they are running, leave them running and work independently. Do not restart, migrate, duplicate or extend the user's jobs.
2. Deliver a small research entry point that can train, resume and evaluate on a user-selected permitted GPU, without importing Round10's campaign-reservation machinery into new experiments.
3. Optimize the shared continuous/unified execution path without changing its mathematical model. The priority is eliminating dense reduction scratch, redundant computation and frequent device synchronization—not inventing another router.
4. Use actual short forward/backward execution and a small native numerical comparison to diagnose executor differences. Existing parity failures remain failures; do not rewrite their tolerances or evidence.
5. Evaluate the best available **hypergraph** candidate on Thermal fields, actual retrieval and saved physical responses/design candidates. Keep dense/direct models as controls, not the primary research direction.
6. Only if final-access densification remains a material issue after these measurements, run the single architectural experiment in §7. At most **two new research trajectories**, normally 150 added subset epochs each and never more than 200 each in this goal. No new Wind or HET-MF fits.
7. Deliver tested full-TRAIN commands/configurations, a concise recommendation and resumable checkpoints. **Stop before autonomous formal training. The user launches formal runs.** After the user supplies the run or results, the same tools support analysis and an evidence-based continuation decision.

The default development allowance is **up to six active GPU process-hours**, with at most four spent in new research fitting and at most fifteen minutes in optional clean-device cost comparisons. These are workload limits, not predictions of completion time. Time spent waiting for a GPU, subscription downtime and CPU documentation do not consume a scientific training horizon. Record worker elapsed time simply; do not build another reservation ledger. If contention prevents the fitting work, finish independent code/tests/launch instructions and clearly label measurements not performed. Never spend the goal idly polling.

No new full-data run, physical-reference generation, locked TEST evaluation, broad hyperparameter sweep, long U campaign or production release is authorized here.

## 1. Scientific objective and what Round10 actually supports

The project remains:

> Learn from multi-field reconstruction → discover reusable organization of module/environment interactions → reuse that organization for modular design.

The intended reusable backbone contains original typed M/E records, learned scene-dependent incidence, receiver-dependent retrieval, individual native-value evaluation, and an E-conditioned module response. Applied unified preparation should make incidence govern contextual value exchange as well as decoding. Thermal is the development case; thermal-specific field extraction and heat algebra belong in its adapter.

### 1.1 Source-derived findings, not new measurements

- All twelve non-U endpoints reached e5000. More subset training is not an unresolved prerequisite for this round.
- Thermal C10-S improves the reported DEV temperature means; C10-H has surface/peak misses. Neither result proves how the corresponding architecture will perform on all 600 original TRAIN cases.
- H/S actual work is not matched. Their accuracy ordering cannot be called a grouping-only comparison.
- C10-H's case0006 trace has one M and 192 E records; all nine queried union vertices execute all present records. The full Thermal DEV board nevertheless reports only 32.059% E retention. **Do not generalize one dense trace to the entire dataset.**
- Continuous selected execution is about 2.0–2.3 times slower than the all-open controls in the reported replays; the all-open control changes the function. These timings do not isolate executor overhead at an identical mask.
- Four same-mask comparisons fail their fixed tests. Thermal also has field discrepancies; Wind field tensors pass while Adam-state tensors fail. Small absolute differences are not a retrospective pass and are not, by themselves, proof of a wrong scientific model.
- Saved geometry/heat responses and six design candidates already exist. They can support a small decision-use diagnostic without new simulation or a speed/memory prerequisite.

### 1.2 Decisions made by this plan

**Do not run another maturation campaign.** Reuse completed models and the user-launched U work.

**Do not abandon the hypergraph merely because direct S predicts well.** Use S to distinguish the value of grouping from the value of E conditioning, but devote new implementation effort primarily to H and its shared executor.

**Do not make sparsity the sole objective.** An applied hypergraph can be scientifically useful without a wall-time win; conversely, a small gate coefficient is not a saved nonlinear evaluation. Report reconstruction, organization, physical response, logical support, executed work and runtime separately.

**Do not make subset superiority a gate to full-TRAIN exploration.** A stable, plausible H model with disclosed misses can be a reasonable formal-run candidate. The user makes that resource decision.

## 2. Lightweight operation, without weakening real security

### 2.1 Ordinary research bookkeeping is sufficient

For a new experiment, keep:

- a human-readable configuration and actual case IDs/split name;
- existing Git commit information, package versions and initialization checkpoint path;
- one ordinary training history, `latest.pt`, `best.pt`, and the final checkpoint;
- optimizer/scheduler, epoch/update counters and RNG state for resumption;
- a small result table and selected figures.

Use Git history, existing typed inputs, ordinary exceptions/tests and atomic checkpoint replacement. Do **not** introduce new SHA256 trees, baseline snapshots, source freezes, admission receipts, cryptographic seals, operation reservations, watcher leases, final-scope reconciliation services or nested approval files. Do not recursively rehash historical datasets/checkpoints as a startup routine.

A normal checkpoint may retain existing metadata already produced by shared utilities. This is not a request to strip old metadata or rewrite old files. Do not add another validation layer around it.

A proposed new blocking mechanism must first name the concrete failure it prevents and demonstrate why ordinary software primitives cannot prevent it. No such new mechanism is requested by this plan. Numerical tests are software diagnosis and regression tests, not signed campaign admission certificates.

### 2.2 Preserve actual protections

Keep credentials, repository/service permissions, operating-system ownership, trusted checkpoint loading, data-access permissions, locked TEST separation, reference/target isolation and protection of other users' processes. Do not patch historical validators to always return true, forge old receipts, remove authentication, disable file permissions or delete provenance to make old launchers accept a new run.

Historical campaign restrictions remain on their historical entry points. New research uses an ordinary separately named runner with user-configurable scope; it does not impersonate a resumed closed Round10 operation. Reuse existing model loaders as appropriate, including their existing safety behavior. If importing historical ancestry is expensive, use the already-created standalone model export for new weight initialization; do not call an inference export an optimizer-resume checkpoint.

Blocking checks belong at destructive writes, cross-system mutations, security boundaries and formal production releases. A missing benchmark, busy GPU, unmatched optional field table or unhelpful forecast is a reported condition, not a reason to block unrelated development. Actual invalid dimensions, nonfinite training and an out-of-memory exception still require handling the affected computation honestly.

### 2.3 GPU behavior

Implement explicit `--device` and a small optional `--devices` preference list. The existing user-approved examples are GPU1/GPU2; additional devices are permitted when the user/site authorizes them, not merely because they look idle. Do not embed immutable GPU UUIDs or an exclusive-device condition into model/checkpoint identity.

- Resolve visible device ordinals correctly under `CUDA_VISIBLE_DEVICES`; display the selected physical/visible mapping once.
- Respect any real scheduler allocation. A free-memory query is an advisory snapshot, not a reservation and not permission to take another user's allocation.
- Training may share a permitted device when actual memory allows. Nonzero utilization alone must not veto a launch.
- If memory becomes insufficient, save/reuse the last complete checkpoint, reduce microbatch while keeping effective batch/loss normalization unchanged, or resume in a new process on another permitted device. Log the change once. Do not attempt live migration of a running CUDA process.
- Catch a real CUDA OOM, retry at most twice with a smaller microbatch or a permitted alternate device. Restore the last complete optimizer update before retrying; do not silently skip cases or continue from a partially applied update.
- Do not kill foreign jobs, change their device visibility, impose unexplained large free-memory floors, or monopolize memory as a reservation.
- If no useful training device is available after one brief recheck, continue CPU implementation/reporting and provide the resumable command. No indefinite watcher or wait loop.

Use a standard output-directory lock only to prevent two writers to the **same run**. Different runs need no campaign-global execution lock.

### 2.4 Checkpoints and interruption

Write a temporary checkpoint in the destination directory and atomically rename it. Save latest at least every ten epochs or approximately ten minutes, and on graceful interrupt. Keep best/latest/final and a few informative checkpoints; do not save every microstep. `resume` accepts a different permitted GPU and relocates optimizer tensors normally. Interruption is not a reason to repeat completed phases.

Progress should show epoch, successful updates, native loss components, validation summary when available and observed epoch time. A lost chat/watchdog connection is not evidence that a training process is dead. On return, inspect the process and output once; adopt an existing live job rather than duplicate it.

## 3. Handle the already-running unified experiments

Read actual status from their normal logs/checkpoints/process information. Do not infer status from the report's historical zero-U statement. Do not require complete triplet availability before evaluating an individual finished model.

| Observed state | Action |
|---|---|
| Thermal U-D/U-H/U-S all completed | Evaluate a common available epoch first; use existing results where adequate. Compare fields, original-record reads and cost observations. |
| Some completed, others running | Evaluate completed checkpoints separately. A common earlier saved epoch can provide a labelled partial comparison; never present mismatched ages as a matched experiment. |
| All running | Leave them untouched; optimize/test shared code against completed C10/H9 checkpoints and small fixtures. Recheck once at the end if convenient. |
| Location/state unavailable | Report unknown and continue the independent work. Do not launch replacements. |

Prefer a completed U-H as the developmental scientific model because its incidence is applied during preparation. If unavailable, use C10-H to develop the shared executor and retain its **dense contextualization** limitation. Do not silently call C10-H unified. If U-H quality is poor, compare it with U-D before blaming grouping: local-input removal and applied preparation changed the problem seen by the network.

Avoid editing modules/configurations imported by the user's live U workers. Use a separate normal Git worktree/branch for development when needed; this is source isolation, not a new cryptographic snapshot system. Integrate without resetting their branch or outputs.

## 4. Reusable interface, not another model framework

Improve existing `continuous_joint` and its Thermal runtime/provider integration. Do not write a second TrainingEngine or a plugin platform.

The interfaces should express:

```python
prepared = model.prepare(scene)                 # typed original records and applied organization
result = model.decode(prepared, queries, controls=controls,
                      execution=execution_options, diagnostics=False)
```

The exact existing signature can remain compatible; this is the desired separation, not a mandatory wholesale rename.

**Scientific model:** typed records, local-input capability, contextual equations/incidence, receiver routing, E-conditioning, native M values, output law.

**Execution options:** dense same-function or packed reduction, query tile size, optional activation checkpointing, device and telemetry. Switching these does not replace trained gates with ones or change fields/losses.

**Case adapter:** units, normalizers, geometry, native interpolation/curl/interface extraction, supervision, and physical controls. Retain Thermal's signed source-column contraction and heat-independent flow. Do not force that affine response into future Wind/HET-MF adapters.

Move the `thermal`/`wind` local-record dispatch out of the generic backbone's main execution logic into a small adapter-owned transform or registered capability. Preserve old constructors/loaders through a compatibility shim. New core code should consume the transformed scene and its feature specification, not branch on a dataset name. Do not build future Wind/HET-MF implementations in this round.

Preserve original source IDs, masks and measures. Scalar source coefficients `B × Q × M` are reasonable compact outputs; they are not the wide `B × Q × E × width` scratch targeted below. Model updates invalidate prepared states; no stale prepared-state reuse across training updates. Keep these basic semantic protections at public boundaries, not repeatedly inside pure numerical tile functions.

For unified U, keep both contextual cycles in their current MM → EM → EE → ME order. Preserve removal of learned scene-summary broadcasts and layout-aggregate neural value features. Do not reintroduce them to make a short run look better. Retain own-record residuals and legitimate prescribed global operating information. Dense score dependencies remain acknowledged control paths; an omitted final read is not a claim of physical conditional independence.

## 5. First implementation priority: same-function execution optimization

### 5.1 Remove dense reduction scratch

`continuous_joint/reader.py::_selected_source_order_reduce` evaluates selected messages, scatters them into a zero tensor shaped `[B, Qtile, N, H, Dh]`, then sums over `N`. This recreates a wide dense allocation despite sparse value evaluation.

Add a `segment` executor using packed contributions and a reduction by destination `(batch, query)`:

\[
C_{bqh}=\sum_{(b,q,n)\in\mathcal R}\alpha_{bqnh}\,m_{bqnh}.
\]

Use existing PyTorch `index_add` or a supported segment-sum primitive. The allocated output should be `[B, Qtile, H, Dh]`, with packed intermediate storage proportional to selected rows. Preserve zero rows, ordering of returned queries, physical source columns, zero-information-measure heaters and gradients through the coefficients. Keep the historical source-order reducer as an explicit diagnostic backend, not the default performance requirement.

Do not introduce a custom CUDA/Triton dependency unless profiling after these ordinary changes demonstrates that it is necessary. Kernel authoring is not the first job.

### 5.2 Reuse the existing reduced-projection algebra

There is a particularly concrete optimization already supported by the repository. `content_hypergraph/source_read_execution.py` implements `reduce_headwise_linear`, `compose_scalar_projection` and `reduce_indexed_linear`. The continuous path currently calls `_flat_messages`, which evaluates `message_output` separately on every admitted pair before reduction.

Let the original per-record nonlinear hidden value be `h_i`, the output projection be `W h_i + b`, and head-specific information weights be `alpha_ih`. Then

\[
\sum_i\alpha_{ih}(W_h h_i+b_h)
= W_h\left(\sum_i\alpha_{ih}h_i\right)
+b_h\sum_i\alpha_{ih}.
\]

Reduce the original query-dependent hidden values by head first, then apply the corresponding output-weight blocks. This is an algebraic reorder, **not** replacement of original records by a scene-pooled hyperedge. It preserves the nonlinear query/record computation and the actual per-head coefficient mass. Do not assume that mass is one: the guarded normalizer deliberately allows it to differ.

For Thermal's signed scalar response, compose the two linear projections while retaining their gradients:

\[
k_i=(w_TW)h_i+(w_Tb+b_T),
\]

then apply the physical access/source law as before. This avoids a full width-by-width projected message solely to obtain a scalar source coefficient. Do not multiply the signed scalar by information-attention weights.

Reuse the existing dense helpers; implement only the missing packed headwise equivalent. Avoid materializing selected_rows × heads × width solely for convenience: accumulate head blocks or use bounded head loops/appropriate contractions. Reconstruct projected per-record messages only for small explicit diagnostic requests.

This changes floating-point association and needs the scoped forward/gradient tests; it does not warrant a new model fit or a claim of bitwise equality. Profile the complete native operation to determine the actual saving. Update counters truthfully: nonlinear hidden/value rows, full message projection rows, head-reduction projections and scalar projections are now different quantities. Do not keep the old `message_projection_rows = value_rows` alias when it is no longer true.

### 5.3 Support a fast dense **same-function** executor

When access is nearly dense, compacting and gathering can cost more than a regular dense operation. Provide a `dense_weighted` backend that evaluates the exact trained incidence/access/normalizer, including the guarded reduction and Thermal scalar coefficients. It is **not** the all-open model.

Backend selection may depend on measured workload shape/density without changing the model's mathematical function. First let the user/config choose `dense_weighted` or `segment`. A small optional per-shape heuristic can be added after actual measurements; do not build an autotuning service or inspect target errors to choose execution. Charge every executed dense value row even when its contribution is zero. A faster weighted dense backend is a useful implementation result, not a sparse-execution result.

For applied U contextual exchange, compare packed weighted donor reduction with a regular batched contraction of the **same** coefficients. Preserve message-mass scaling of the projection bias. Do not substitute ordinary all-open SDPA or renormalize away the guarded mass. With only around 200 records, a dense weighted contraction may be more practical than irregular gathers; measure rather than assume.

### 5.4 Remove synchronization and diagnostic work from hot paths

The current reader converts GPU reductions to Python integers, and tile functions construct diagnostic tensors/dictionaries even during checkpoint recomputation. The shared configuration key also converts a device buffer to `float`. These are profiling targets, not claims that each causes the entire slowdown.

- Keep row totals as device scalars during a macrostep; transfer one compact summary when logging.
- Do not build support/access/message trace dictionaries unless requested.
- Do not compute physical-row telemetry again during checkpoint backward. Instrumentation should be pure or outside recomputed regions.
- Construct source-static projections once per prepared scene and query projections once per tile, not once per selected row.
- Reuse pre-E E scores in the E reader. **Do not reuse pre-E M scores for post-E M attention**, whose query has changed.
- Avoid `int(tensor)`, `.item()`, `.cpu()`, repeated Python boolean reductions and cache-key scalar transfers inside tile loops when not needed by the actual computation.
- `nonzero`-based compaction is a genuine synchronization cost on CUDA. Reduce repeated compaction; do not replace it with dense wide computation and still claim sparse execution.
- Avoid an `N × N` wide expansion for contextual messages; source-local V projections plus weighted reduction are sufficient for the existing contextual law.

Preserve normal public input validation and safe loading. Do not use `unsafe=True` or disable protections as a blanket performance tactic.

### 5.5 Reconsider checkpoint granularity using actual steps

The current `_checkpoint_work` recomputes routing separately from the physical tile. That separation was intended to avoid retaining wide values during work-only differentiation. Do not simply delete it or attach detached work to the loss.

Run a short profiler on one representative native forward/backward. If the TrainingEngine permits a single correct combined macro objective, return compact differentiable work numerators from the physical tile and avoid a second identical routing forward. If microbatch/dual accumulation needs separate derivatives, keep the pure score-only replay, but eliminate redundant metadata, projections and validation. Verify the native loss, penalty and gradients on a small actual macrobatch.

Expose `checkpoint=off|values|tile` as execution choices only where supported. Test two sensible tile sizes (256 and 512 initially), not a large grid. On a roomy GPU, redundant recomputation may be slower than retaining activations. On a constrained GPU, checkpointing is useful. No universal fastest option is assumed.

Optional compilation is limited to the stable pure value/reduction primitive after an eager implementation works. Report compilation warm-up separately. Do not spend the goal fixing full-model graph capture or recompiling variable-sized scenes.

### 5.6 Do not change science while optimizing execution

Keep gates, guarded normalization, E-conditioning order, local branch, source-coefficient algebra, field losses, physical units and query counts unchanged for this phase. Fixed coefficients are useful for local diagnostics, but training must still propagate gradients through live routing and contextualization.

A same-weight dense/segment discrepancy must be investigated at the first divergent quantity: IDs/gates → scores/weights → per-record messages → reduction → coefficients/native extraction → losses/gradients. Do not start with four epochs of evolving Adam states.

## 6. A small, useful numerical test set

Round10's four negative parity results remain untouched. New tests assess the new implementation; they do not amend history.

Required tests are conventional tests, with an ordinary test report—not admission artifacts:

1. A small independent FP64 explicit sum for E and M information reduction and signed Thermal contraction; compare forward values and gradients away from activation/support boundaries.
2. Dense-weighted versus segment with identical trained gates, then with live routing, on M1 and M12 native Thermal fixtures. Check primitive outputs, individual physical fields, loss and parameter/coordinate gradients separately.
3. Empty support, exact-zero contribution, zero-measure information records, duplicate receiver occurrences, variable M, source permutation and query chunking.
4. Thermal zero heat, linear heating increments, source-column order and heat-independent flow; original native curl and q-proxy extraction stay unchanged.
5. Fresh model save/load and actual resume for a few updates on a new disposable run; no all-history reconstruction.
6. For U changes, one two-cycle applied-incidence fixture and one frozen-coefficient disconnected-value test. Keep score-path dependence explicit.

Use a genuinely independent FP64 reference rather than the same helper under two names. For FP32, report per-field differences normalized by existing TRAIN scales and per-parameter gradient norms, including cancellation/near-zero cases. As initial research regression expectations use normalized RMS around `1e-5` for primitive fields, `1e-4` for derived curl/q-proxy and relative gradient-L2 around `1e-4`; record maximum errors too. These are starting engineering tolerances, not new physical guarantees or a campaign gate. Do not silently loosen a failed test. Diagnose a failure and use the known-correct backend for affected work while the alternative remains experimental.

Adam-state near-zero elementwise tests are diagnostics, not the definition of executor correctness. The important questions include gradient and actual parameter-update error, physical field error, and whether the implementation follows the intended equations. Floating-point association can cause differences, but that fact alone is not an explanation for a particular failure. Preserve old tests as historical evidence and add the separate scoped tests above.

A demonstrably wrong backend must not be promoted as correct. This does not require halting CPU work, evaluation through the reference backend, user handoff or all other scientific work.

## 7. The only optional model change: final-access sparsification

### 7.1 Diagnose composition, not heatmap appearance

The continuous hypergraph currently composes nonnegative seed weights and incidence:

\[
a_i(q)=\sum_s\pi_s(q)b_{si}.
\]

Ignoring floating-point underflow, its positive support is exactly

\[
\{i:a_i(q)>0\}=\bigcup_{s:\pi_s(q)>0}\{i:b_{si}>0\}.
\]

Therefore one active broad seed can make final access dense. Sparse seed weights and a sparse-looking incidence plot are insufficient. Self identity, active native M seeds and broad ME rows can matter, particularly for the singleton example. Tiny positive coefficients remain full nonlinear reads.

On at most six input-selected TRAIN cases covering M1/small/medium/high M, and the normal DEV board when available, report:

- incidence degree per seed and its distribution;
- positive seed count per query;
- final positive M/E access and executed M/E rows;
- optional mass-based effective counts, clearly distinct from execution;
- final support overlap and union inflation, including the role of broad seeds.

Use actual gates for counts. Do not threshold a visualization and call it executable sparsity. The nonnegative formula above is a structural explanation; the panel establishes where it matters in this model. A dense M1 case alone is not grounds to redesign the backbone when the aggregate reader is sparse.

### 7.2 When to run a model trial

Default to the unchanged mathematical model with the improved executor. Consider the following one experiment only if broad final E access remains common and costly, or the composed-support proxy substantially understates actual positive reads. Decide once from the small panel and profile. Do not launch several alternative routing schemes.

Keep M access and contextual incidence unchanged. Add one optional, trained-in-the-loop **E final-access map** after composition:

\[
\widetilde a_E=\chi_{\theta}(a_E),\qquad
\chi_\theta(a)=
\begin{cases}
0,&a\le\theta,\\
(a-\theta)^2/a,&a>\theta.
\end{cases}
\]

For nonnegative `a`, define `chi_0(a)=a` exactly. For `theta>0`, the value and derivative with respect to `a` approach zero at the support boundary. Evaluate the inactive branch with a safe denominator, not `0/0` followed by masking. Keep `theta` in `[0,1)` and serialize its value in the ordinary model config.

This is a new scientific function, not an executor tolerance. It attenuates E context and may hurt accuracy. Use the unchanged guarded normalizer and the transformed E gate in both the actual reader and work objective. Do not clip positive weights after training, invent a top-K fallback, or detach the gate.

Initialization is identity. From a small TRAIN-only input panel, choose a conservative target threshold once: start with the 20th percentile of currently positive final E accesses, capped at `0.05`. This attempts to remove only the weakest tail; the cap can mean few or no rows disappear. Keep zero as the control. Ramp from zero over the first 40 local trial epochs, then hold fixed while ordinary model parameters adapt. Do not repeatedly tune the threshold against DEV. Record its empirical effect rather than promise a retention rate.

The exact-zero map has no local derivative below threshold; omitted records are not guaranteed to recover. The identity start, gradual ramp and shared upstream parameters are pragmatic, limited mitigations—not a claim of an unbiased restoration estimator. If fields degrade, reject this candidate rather than adding dense restorative passes, another predictor or more policies in this goal.

This experiment changes only final E access. It does not fix every contextual information path or establish a hypergraph advantage over S. It tests whether reusable groups can retain their mechanism while avoiding unnecessary union-tail reads.

### 7.3 Trial design

Use one available H parent, preferably completed U-H, otherwise C10-H. Fork an unchanged-math H control and the single E-map candidate with the same weights, native objectives, subset, normalizer, sampler, effective batch and optimizer treatment. Preserve compatible optimizer moments in both or reset them in both; never mix treatments without labeling it.

This is the maximum new research pair, not an additional pair on top of a large baseline portfolio. Existing S/dense checkpoints provide contextual references. Do not claim grouping superiority from their unmatched trajectories.

If the panel does not justify this change, **skip it** and deliver the execution improvement plus formal-run package. Success does not require inventing a new architecture.

## 8. Short experiments that hand off to full TRAIN

### 8.1 Subset screens have a limited job

Reuse the current 150-case Thermal development TRAIN and its exposed DEV22. Do not search for an easier subset or read locked TEST. A short subset run answers implementation/adaptation questions, not final generalization.

For the optional pair:

- default 150 **additional local epochs** per arm, maximum 200;
- evaluate every 25 epochs on DEV22 and a fixed 24-case stratified TRAIN panel;
- do not stop for ordinary noise before epoch 75; after epoch 100, stop the pair at a common review point if three review windows show no useful change or validation deteriorates while training improves;
- a clearly broken/nonfinite computation is handled immediately; ordinary error increases remain reported results;
- keep native query/auxiliary/response losses intact; no one-epoch smoke is called a scientific result;
- no further automatic reruns, seeds or 5000-epoch endpoints.

Use a small advisory ranking statistic—the geometric mean of fluid/surface/material/peak RMSE normalized by the **same parent's** values—to observe trend. Always show per-field and stratum results; do not average away a severe local failure. Plateau guidance is an early decision aid, not a new authorization gate.

The existing 150/48 batching produces four updates/epoch, so two 150-epoch arms are about 1,200 new main updates, not 54,000 or 144,000. Recompute this from the actual provider; do not hard-code it into a generic runtime.

Separate `experiment_epoch`, optimizer step and parent training age. New short schedules must not spend their entire horizon waiting for the old absolute e3000/e4000 boundaries. Preserve historical schedules in historical models. The new runner should take an explicit local schedule start; default to the parent's terminal work target for a resumed mature model, not a fabricated restarted thousand-epoch warm-up. For a fresh router, explicitly use a short near-open ramp and label the screen immature rather than extrapolate its endpoint.

### 8.2 Full-TRAIN launch is prepared now, not after another perfect subset result

The current manifest declares **600 TRAIN / 90 source-test cases**. Full TRAIN means the existing 600 original TRAIN IDs, not 690, not the separate local-module dataset, and not an expansion of DEV into training. DEV22 is exposed and remains distinct from any locked formal TEST.

Deliver:

- `thermal_honf_full.json`: recommended H backbone, all 600 TRAIN cases;
- `thermal_dense_full.json`: compatible dense control;
- `thermal_direct_full.json`: optional S control, ready but not automatically launched;
- `FORMAL_RUNS.md`: exact tested commands, initialization choice, normalizer handling and the first review point.

Prefer the U family for these presets when it is usable, because it actually applies the organizer in preparation. If U remains incomplete or clearly degraded, provide the stable C/H option as an explicitly decoder-organized alternative and retain the U preset as experimental. Do not silently restore aggregate/global shortcuts or label C unified.

For warm starts, keep the parent's TRAIN-only transforms so loading does not silently change physical output units. Those transforms may have been fitted on a TRAIN subset; that is permissible if disclosed and used consistently. Do not fit full-TRAIN normalization and copy unchanged old scaled weights as though the forward function were preserved. Offer fresh training with common full-TRAIN statistics as a separate explicit mode, not an automatic parallel experiment.

When comparing formal H/D/S runs, use compatible parents/local-input capability, matching initialization policy, the same TRAIN population and the same query/loss schedule. If the available parents have unequal exposure or different ancestry, state that and label the results practical warm-start comparisons rather than a clean grouping ablation.

### 8.3 User-facing workflow

The following are **target CLI interfaces to implement**, not claims that these commands already exist:

```bash
# User initiates full-TRAIN work; default first segment is modest, not e5000.
python HONF_Proj/tools/honf_research.py train \
  --config HONF_Proj/configs/research/thermal_honf_full.json \
  --device cuda:1 --epochs 200 --output /data2/.../thermal_honf_full_run01

# Resume on a different permitted device without rewriting model identity.
python HONF_Proj/tools/honf_research.py resume \
  --run /data2/.../thermal_honf_full_run01 --device cuda:2 --additional-epochs 100

# Analyze available results; never launch an automatic new training portfolio.
python HONF_Proj/tools/honf_research.py assess \
  --run /data2/.../thermal_honf_full_run01 --dataset thermal
```

Replace placeholders with resolved real paths in the delivered launcher document. The generic core must not contain physical device IDs. Validate parser and actual execution, not only `--help`.

A default formal first segment is **200 full-TRAIN epochs**, with reviews at 50/100/150/200; at effective batch 48 this is about 2,600 updates, conditional on the actual provider. It is a user-run segment, not automatic authorization in this goal. Continuation can be another 100–200 epochs when native fields are still improving. Give observed time-to-next-review estimates from the user's hardware, not a fixed claim about convergence.

At every review provide a clear recommendation: continue the same run; stop at best; fix a named defect; or compare against a matched dense control. If training error falls and DEV rises, avoid automatically adding epochs. If full-TRAIN errors improve despite a weak subset screen, treat that as evidence for continuing—not as a contract violation.

The Codex goal ends with a launchable package and useful results. It does not remain suspended waiting for the user to finish formal training. On the next user invocation, inspect the existing run and continue analysis without recreating setup or demanding a new identity audit.

## 9. Minimal evaluation that answers the scientific questions

### 9.1 Fields

Evaluate the chosen H checkpoint, available paired U-D/U-S checkpoints when finished, and relevant saved controls. Reuse adequate old numerical tables; do not repeat six or twelve historical model boards just to fill a new report.

Use DEV22 plus a fixed 24-case original-TRAIN panel spanning available module counts, including M1/M2. Report u/v/p/omega, fluid/surface/material T, q-proxy and module peaks, means and p90, with populated M-stratum summaries. Full TRAIN600 evaluation is optional after the user launches full training; do not make it a precondition to the present handoff.

Show absolute native units and changes relative to the selected controls. The pressure and temperature units are the dataset's native units; do not invent Pa/K calibration. The flux proxy remains a proxy, not a conservation proof.

### 9.2 Organization

Produce one useful M/E incidence/receiver-read figure for a low-M and a high-M case, plus a small table of actual support/work. For U show both **applied** preparation incidence and final retrieval. For C show the retrieval directory and disclose dense preparation.

One E-context/value removal and one group-coefficient intervention on a few fixed cases are enough. If a group rewire changes final work, say so. A work-matched replacement should preserve final per-type read counts and state which weights/normalizers change; it is still a same-model reliance test, not a trained H-versus-S proof.

Do not require that H/S work align before any field result can be reported. Work mismatch limits attribution. An actual matched-work formal comparison can be planned later rather than forcing a new controller/sweep now.

### 9.3 Continuity and physical response

Replay a few existing seams, and search a small fresh query panel for current support changes instead of only testing old boundaries that may have disappeared. Use several represented step sizes; discard collapsed coordinates, keep actual gaps, and compare the same query pairs across models. Distinguish continuous-core queries from native-grid interpolation/curl.

Use the four existing Thermal reference families for a compact heat/geometry response table. No new physical solve or independently forced environment truth is implied. Do not rebuild the entire old response inventory.

### 9.4 Inverse-use test now

Complete the six-candidate saved ranking for the selected H and up to two relevant available controls (dense and direct). Read actual objective/constraint definitions from the existing candidate set; do not invent new objectives to favor H. Report objective error/ranking, feasibility confusion, the chosen candidate and its stored-reference constraint margin. Reference regret is defined only for feasible selections.

Do not require a memory saving to run this diagnostic. The previous D9-M infeasibility remains in history. A small pressure exceedance may be close to unresolved reference numerical uncertainty, but it must not be relabeled feasible without a pre-existing justified tolerance. Six correlated candidates from one layout are a diagnostic, not general inverse success.

Retain a simple fixed-layout heat-allocation interface using the affine response `T = K h`, with bounds and total-heat constraints supplied by the user/task. A small surrogate optimization may exercise the API; label proposals **unverified** unless compatible physical truth exists. Flow is heat-independent in this benchmark, so changing heat cannot repair a pressure-infeasible layout. Do not launch new solver calls, broad layout generation or train a generative model in this goal. Unavailable reference data does not prevent finishing the forward package.

## 10. Cost checks are optional, independent and disposable

Use them when an appropriate GPU is sufficiently idle. They must not delay training, block formal-run preparation, or mutate a scientific checkpoint.

A useful quick check is one mixed low/high-M native batch at the actual training query scope, three warm-up steps and about ten measured forward/backward steps per backend. Use a disposable model/optimizer in a separate process; delete temporary payloads on exit. Synchronize the timing boundary or use CUDA events. Run selected and dense-weighted on the same device with the same data, precision, effective batch and math.

Report wall time, peak allocated/reserved memory and actual executed M/E rows. Capture one short profiler trace only if it answers an identified bottleneck; remove it after extracting the result. Retain, at most, a small optional CSV/console summary with hardware, shapes and timing scope. No new immutable evidence tree, model hashes, benchmark-admission files or eight four-epoch cost cycles.

A computation-cost probe leaves **no persistent changes to model, optimizer, RNG, run history or prepared caches of the real run**. It may have a small result summary if the finding is quoted. Omit performance claims when a clean comparison was unavailable. Contended training remains valid training, but its epoch times are not an uncontended architecture benchmark.

The three important comparisons are different:

1. Old versus improved executor at the same trained gates/function.
2. Segment versus dense-weighted execution at the same trained gates/function.
3. Independently trained H versus dense/direct scientific models.

Never replace comparison 1 or 2 with all-open gates and call it numerical parity or pure executor acceleration.

## 11. Concrete implementation tasks and scope boundaries

### P0 — Make the research path usable

- Inspect current U state once; preserve active work.
- Add the small runner and a research task builder using existing providers/TrainingEngine.
- Support initialization from a safe existing model export and ordinary new-run resume.
- Make dataset membership, local epoch horizon, execution choices and device parameters explicit.
- Leave historical Round9/10 launchers, closed ledgers and validators unchanged.
- Add ordinary SIGINT/graceful checkpoint behavior and a per-output writer lock where absent.

**Output:** a real native mini training/resume execution and working evaluation command. No dummy dataset result substitutes for this.

### P1 — Improve shared numerical execution

- Implement segment reduction, reuse the existing reduced-projection algebra, and provide a dense same-function backend.
- Make tracing/counters opt-in or batch-aggregated; eliminate repeated Python/device scalar synchronization where practical.
- Reuse local projections/scores correctly; preserve the E→M dependency.
- Profile one actual step; choose a reasonable tile/checkpoint configuration.
- Add the focused tests in §6 and retain the known-correct backend when an alternative is still experimental.

**Output:** code, measured local numerical differences, and optional brief cost result. A speedup is not assumed.

### P2 — Decide once on the final-access trial

- Measure incidence/seed/final-access structure on the small panel.
- If justified, implement the single E access map in a new explicitly named model option and train the one short H/control pair.
- Otherwise skip new scientific fits; do not manufacture a trial to occupy the budget.
- Never launch replacement U/D/S campaigns or additional Wind work.

**Output:** a recommendation based on actual fields/work, including a negative result if applicable.

### P3 — Evaluate and deliver the full-TRAIN handoff

- Integrate any Thermal U checkpoints that have become available; do not wait for them.
- Run the minimal field, organization, saved-response and candidate-ranking checks.
- Produce actual launchable full-TRAIN H/D/S configurations, with H the recommendation and controls clearly explained.
- End with exact run commands, what has/has not run, and the next review point. Do not launch full training automatically.

The stages are not independently signed gates. When one measurement is unavailable, carry that limitation into the handoff and finish the other useful work.

## 12. Files to add/change

Prefer targeted changes over a full refactor:

| Location | Intended work |
|---|---|
| `src/honf_forward_core/continuous_joint/reader.py` | Add segment reduction and explicit same-function execution choices; retain old reducer for compatibility/diagnosis. |
| `src/honf_forward_core/content_hypergraph/source_read_execution.py` | Reuse existing linear-reduction/scalar-composition helpers; add a packed headwise helper only if needed. |
| `src/honf_forward_core/continuous_joint/model.py` | Compact optional diagnostics, avoid repeated score/projection work and separate execution settings from scientific state. |
| `src/honf_forward_core/continuous_joint/preparation.py` | Efficient weighted contextual contraction; preserve applied gates and bias/mass law. |
| `src/honf_forward_core/continuous_joint/routing.py` | Only the optional final E access map and useful aggregate support diagnostics; no new router portfolio. |
| `src/honf_forward_core/continuous_joint/unified.py` | Preserve applied stage behavior; move dataset transform dispatch behind a case capability without breaking old constructors. |
| `src/honf_runtime/research_training.py` (or equivalent small module) | Build ordinary research tasks, local schedules, checkpoints and assessment using the existing engine. |
| `tools/honf_research.py` | `train`, `resume`, `evaluate`, `assess`, optional `cost`; no campaign service. |
| `configs/research/thermal_*.json` | Short H/control only if needed, full-TRAIN H/D/S user-launchable presets. |
| Existing core/runtime tests | Add small targeted numerical, resume and native-adapter tests; reuse existing relevant tests. |
| `docs/reports/HONF_Round11_Thermal_Development_Report.md` | Approximately 2–4 pages of findings plus small tables/figures; no audit inventory dump. |
| `docs/guides/HONF_Thermal_Formal_Runs.md` | Commands, choices, initial full-data horizon and how to resume on another device. |

All generated checkpoints, predictions, plots, temporary probes and processed outputs belong under the existing `/data2` storage arrangement, e.g. `/data2/wanglz/ModularDT/diagnostics/thermal_honf_round11_<run_id>/`, browsable from an ignored project symlink. Use existing dataset locations; do not copy datasets or change HET-MF links.

Do not overwrite old runs or user-created outputs. Cleanup is limited to new, known disposable temporary outputs owned by this task. There is no recursive historical cleanup obligation. Source/config/docs go to Git; generated data do not.

## 13. What a completed goal must deliver

1. A usable Thermal-first training/evaluation/resume workflow with configurable permitted devices and no new campaign admission system.
2. Shared HONF executor improvements tested on real native forward/backward work; honest numerical/cost findings.
3. A clear assessment of whichever Thermal U outputs were actually available, without interfering with the user's jobs.
4. At most one short paired scientific experiment, or a documented reason it was unnecessary/not feasible.
5. A compact report on fields, actual applied organization/retrieval, response and saved-candidate utility. Missing evidence is named, not replaced by proxy claims.
6. Tested full-TRAIN commands/configurations and a specific recommendation for the user to launch. Subset misses are disclosed; they do not automatically veto the proposed full-data experiment.
7. A normal commit of the relevant changes and, when requested through the goal prompt, a normal non-force push preserving unrelated changes. No security/repository-permission bypass.

A successful outcome can be: **“The H backbone is scientifically plausible; execution is improved or its remaining bottleneck is identified; the short screen is informative; here is the full-TRAIN experiment to run next.”** It need not be an invented claim of H superiority, a sparse speedup, physical causality or validated inverse generation.

## 14. Source notes and interpretation boundaries

### Directly reviewed repository files

At the review commit identified at the top:

- `HONF_Proj/src/honf_forward_core/continuous_joint/{model,reader,routing,primitives,budget,preparation,unified}.py`.
- `HONF_Proj/src/honf_forward_core/content_hypergraph/{readers_v2,source_read_execution}.py`.
- `HONF_Proj/src/honf_runtime/continuous_joint_factory.py`.
- `HONF_Proj/tools/continuous_joint_train.py`.
- `HONF_Proj/Case_ThermalChannel/src/channelthermal/local_records.py`.
- `HONF_Proj/Case_ThermalChannel/Dataset/dataset_manifest.json`.
- `HONF_Proj/docs/guides/HONF_Round10_Reusable_Core_Guide.md`.

The full user-supplied Round10 report is the basis for the empirical findings. The review did not execute private checkpoints, measure current GPU occupancy or inspect uncommitted U progress. The new user instruction supersedes the report's historical U deferral; it does not supply measurements of U results.

### Main code observations

- The packed reducer creates dense wide source-order scratch; this is actual code behavior, not a measured attribution of the whole memory peak.
- Continuous routing uses `bmm(seeds, incidence)`, so positive-path support unions can become dense. The formula proves the mechanism, not its prevalence across all cases.
- Existing source-read helpers support reducing nonlinear hidden values before linear output projection and composing the Thermal scalar head. The continuous path does not currently use that algebra in its packed message calculation.
- Full query-tile checkpointing already exists. The next task is to simplify its internal work/lifetimes, not propose that same change as though it were absent.
- A separate routing/work replay exists and has an intended derivative/lifetime purpose; remove duplication only when equivalent objective gradients are preserved.
- U already applies typed contextual incidence and removes specified scene-summary/aggregate input paths. Preserve and test it rather than rebuilding it from scratch.
- Current source manifest/admission/resource requirements are campaign-specific implementation choices. They should not be copied into a new ordinary research workflow; actual security, historical behavior and trusted loading stay intact.

### External technical context, checked separately

Official documentation supports floating-point non-associativity and differences across equivalent batched implementations; this does **not** establish the cause of Round10's failed tests. Segment reduction and CUDA timing/synchronization guidance inform possible implementation choices. Test the locally installed version and do not upgrade the active environment solely for this plan.

- PyTorch, Numerical accuracy: `https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html`
- PyTorch, `torch.segment_reduce`: `https://docs.pytorch.org/docs/stable/generated/torch.segment_reduce.html`
- PyTorch, CUDA semantics/timing: `https://docs.pytorch.org/docs/stable/notes/cuda.html`
- NVIDIA, Writing Sync-Free Code: `https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/sync-free-code.html`

The short-run limits, local schedule, optional E final-access map, backend choices and formal handoff are **new recommendations**, not findings of Round10 or promises of improved performance.
