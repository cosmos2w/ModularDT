# HONF Round12 — compact, reusable hypergraph organization on ThermalChannel

**Prepared:** 23 October 2026\
**Reviewed repository:** `cosmos2w/ModularDT`, `agent/honf-core-next`, `43fcdd2fc9d8cc470d1b46552c74f18ef9398e65`\
**Deliverable:** a Codex goal-mode implementation and bounded learning plan, not a claim of implemented or validated compact groups.

## 0. Decision and scope

**Select the U-H route, and implement only its Level R: directly constructed compact retrieval groups over the existing applied unified encoder.** Retain U-S epoch 1000 as a strong frozen predictive/decision reference. Do not implement or train the U-S E-grid prior in this round. Do not automatically proceed to the factorized Level U encoder.

The scientific question is:

> Can an input-generated, reusable bank of organizational groups be smaller than the physical-record inventory, while preserving useful original-record responses and layout-dependent organization?

This is not a race to beat every classic-model field or to obtain a sparse speedup before studying the representation. The order of priorities is: build the intended organizer; train it on real native objectives; test organization, field quality and response/decision use; report practical cost when it is easy to measure. Fewer groups alone are not a scientific success, and a slower but informative model is not automatically a failed experiment.

### Authorized goal when this plan is supplied to Codex with the accompanying prompt

Implement the new capability, make one common 20-epoch full-TRAIN adaptation, fork one uncompressed control and one sibling-merging candidate, and run both to local epoch 100. Evaluate them and supply ordinary continuation commands. This is **2340 new scientific optimizer updates**, not a new 5000-epoch portfolio. Existing long-running U-D/H/S jobs are unrelated and must remain untouched.

The current conversation prepares this plan; it does not itself launch jobs. The older proposals' “design only” wording describes their original status. This plan is the execution scope for the next goal, while the user's latest workflow instructions override their proposed new hash/freeze/admission procedures.

**Out of scope:** Wind/HET-MF training, new datasets, factorized contextual encoding, cross-type ancestor merging, recursive multi-level compression, a second routing amendment, radius/regularity sweeps, new physical solver calls, locked-test access, broad geometry inverse search, and automatic long formal training.

## 1. What the sources support, and what this plan adds

### 1.1 Current evidence — not a new measurement

The attached Round11 report's **full-data epoch-1000** section is the current predictor evidence. Its older U10-e5000 section is a different development lineage and must not supply the new models' response or cost results.

- U-S has the lowest mean on seven of nine quantities within the U triplet. U-H has the best fluid-temperature and vorticity means. U-H fluid-T RMSE is 0.394110 versus U-D 0.510812; classic1804 remains better on transverse velocity and some temperature tails.
- U-H learns applied contextual exchange and group-composed retrieval. On the six-case organizer panel, every M-anchored group's E membership is positive across all 192 E records. Even M-seed mass 0.000366 can therefore reopen every E value.
- On canonical VAL89, H selects 72.65%/39.79% of eligible M/E rows and S selects 48.71%/44.46%. Their configured dense final executors nevertheless evaluate **100%** of final value rows.
- No epoch-1000 response or saved-candidate replay has yet validated these new weights. Historical weak geometry responses/rankings remain warnings, not measurements of the new checkpoints.
- VAL89 is exposed validation, with M3/M5/M7/M10 strata; M1/M2/M12 structural examples are TRAIN evidence. Classic models have different normalization/objective/training histories.

**Basis:** Round11 report, current comparison and organizer sections, plus the two revised proposals. The source inventory at the end names all inspected files.

### 1.2 Why choose H rather than the easier S amendment?

The S proposal is a reasonable narrow experiment, but its new term is an E-grid locality preference. It does not introduce explicit reusable groups or a new module-translation dependence at fixed query/domain. Its strong current predictor does not require this amendment to remain a useful reference.

H's proposal directly tests the representation central to HONF: **record identity, group identity and physical values need not share one axis**. Its first-stage risk and scope are manageable only if we keep the existing encoder and train one controlled compact-retrieval comparison. Developing both proposals would recreate the portfolio problem.

### 1.3 Explicit decisions and refinements made here

These are planning choices, not findings from the report:

| Issue | Round12 decision |
|---|---|
| Main architecture | U-H Level R only; original applied MM→EM→EE→ME encoder remains |
| New learned object | Compact group descriptors, typed group-to-record memberships, pre-E query-to-group scorer, shared sibling-compatibility rule |
| Training source | Literal full-TRAIN U-H epoch 1000, with inherited normalizer |
| Comparison | Common 20 epochs; `R0` keeps singleton groups, `R1` learns sibling tying and activates one hard cut |
| Duration | Both terminal branches stop at local 100; no automatic extension |
| Compaction review | A bounded actual before/after experiment, not an admission service or a requirement for immediate field equality |
| Efficiency | Use existing dense-weighted execution for scientific fitting; optional same-function segment measurements separately |
| New infrastructure | Extend the maintained research runner; no second engine, hash tree, source freezer, or GPU reservation service |
| Inverse evidence | Replay current-parent and final-child responses/ranking; demonstrate cached K reuse without claiming new verified designs |

## 2. The architecture and the limits it must make visible

### 2.1 Preserve the original records and physical laws

Let `N=M+E` index original typed records. Preserve their IDs, coordinates, masks, characteristic lengths, measures, contextual states and individual query-conditioned values. E records do not become heaters. Every original M source retains its column in the response matrix, including a physically present source with zero information measure.

The existing Thermal dependency remains:

\[
q_E=q_0+A_Ec_E(q),\qquad
T(q;X,h)=\sum_{i=1}^{M} K_i(q;X)h_i.
\]

`K_i` includes the gated signed individual source coefficient and the unchanged smooth local response. Information-attention weights or quadrature are not applied again to the physical source sum. Heat amplitudes remain outside routing/contextualization for this benchmark. All native flow, surface/material extraction, curl, q-proxy and peak evaluation stay with the Thermal adapter.

### 2.2 Introduce an organizational axis, not a pooled physical value

The new scene preparation returns live groups of count `K=K_M+K_E`, with:

\[
C(X)\in[0,1]^{K\times N},\qquad
\pi(q,X)=\operatorname{entmax}_{1.5}(s(q,X)),\qquad
 a(q,X)=\pi(q,X)C(X).
\]

The final gates are:

\[
g_i^M=\ell_i+(1-\ell_i)a_i^M,\qquad g_j^E=a_j^E.
\]

`ell` is the existing local M envelope. Each positive-access original record still has its own nonlinear value. Evaluate a record once per query even if multiple groups reach it. Preserve the existing guarded information reduction and E-first decoding order.

A group descriptor is used **only for control scoring**. Never feed it into the physical field head as a replacement E/M value. Never replace M source columns with group temperatures or merge heat amplitudes. Original module coordinates remain the physical value coordinates; a group centroid is only a routing coordinate.

### 2.3 Do not promise that compaction cures dense E access

For nonnegative `pi,C`:

\[
\{i:a_i(q)>0\}=\bigcup_{k:\pi_k(q)>0}\{i:C_{ki}>0\}.
\]

Convexly averaging two positive rows also takes their support union. Therefore grouping fewer seeds does not by itself close the broad M→E access route. No extra threshold, E-grid prior, top-K or final-access clipping is added to disguise this. Report group count, final support and actual execution as different quantities.

### 2.4 Level R is deliberately not a factorized encoder

The current two applied contextual cycles remain unchanged in equation and input capability; their weights continue to learn. They still score native pairs of order `N²` and may retain their native stage tensors for backward.

The **new retrieval path** constructs `K×N` memberships directly and queries `K` groups. It must not call `applied_native_directory` merely to pool an `N×N` final directory, or compute all old query-to-native-seed probabilities to sum them afterward.

These are different assertions:

- Required: no additional retired native retrieval directory or old query-seed bank in ordinary compact retrieval.
- Not claimed: absence of native contextual `N×N` work in Level R.

The final compact membership is derived from the learned record states, not literally identical to the encoder's applied incidence. Label this loss of literal encoder/retrieval coefficient reuse. Level U would address that later; do not quietly describe Level R as having already achieved it.

## 3. Concrete Level-R model specification

### 3.1 Same-type singleton sibling candidates

Follow the selected H proposal's first-level policy. Form M and E ancestor banks separately. Canonically sort distinct normalized coordinate tuples lexicographically in the adapter frame, pair adjacent singleton entries, and carry an odd singleton. Exact-coordinate tie bundles stay unmerged; do not choose one record by numeric ID.

Only the leaf-sibling pairs are eligible in this version. Higher tree levels need not be materialized because they are not used. Group ancestry is disjoint, but donor membership can overlap across all groups and reach either type.

For `P` accepted pairs:

\[
K=N-P,\qquad K_X\geq\lceil N_X/2\rceil.
\]

Tie bundles can make the bound stricter. At `M=12,E=192`, the attainable best case is `K=102`, not 32 or 6. This is a restricted candidate family, not unrestricted physical clustering.

Compute the hierarchy/candidate metadata once per prepared scene, not per receiver tile. A small scene loop is acceptable initially. Batch-pad to the largest **live K in that batch**, not permanently to N; report both live and allocated counts.

### 3.2 Two separate representations: topology features and live group controls

**Topology features:** a small bank-specific shared compatibility MLP consumes only legitimate local input features, declared operating conditions, and symmetric sibling geometry. Use child feature sum, absolute feature difference, absolute normalized separation, extents and characteristic lengths. Do not use target fields, true errors, heat amplitudes, case IDs or source-ID embeddings.

Use one hidden layer of width 32 with SiLU per destination bank; a scalar final layer produces `b_v`. Initialize its final layer to zero weights and bias `-2`, giving conservative soft compatibility. This is a declared heuristic, not a learned physical threshold. It introduces no trainable per-case state.

**Live control descriptor:** for group ancestry `A_k`, form the arithmetic mean of current prepared native states, coordinate centroid, per-axis extent and destination-bank type. Use lineage multiplicity, not heat or donor quadrature, for these means. One width-preserving linear latent map plus a zero-initialized geometry/type correction is sufficient; initialize the latent map to identity. The descriptor is recalculated from current states in every `prepare`.

Use a positive group characteristic length, for example:

\[
\ell_k=\sqrt{|A_k|^{-1}\sum_{s\in A_k}\ell_s^2
                 +|A_k|^{-1}\sum_{s\in A_k}\|x_s-\bar x_k\|^2},
\]

with the existing physical coordinate convention. Include normalized extents as control features; never overwrite any original record length. Singleton ancestry has the original length and zero extent.

**Important limitation:** Thermal local E inputs are a fixed grid with module-distance/occupancy aggregates removed. At fixed domain/material/operating inputs, this input-only E cut has no explicit module-translation dependence. Its *live memberships and query access* can still change through contextualized states. Report these separately. A fixed E cut across layouts is not automatically a bug, but is not evidence of layout-adaptive E topology either. A context-conditioned coarsener is not added here.

### 3.3 Direct compact group-to-record construction

After the unchanged encoder, construct the four final retrieval blocks with group destination axes:

\[
C=\begin{bmatrix}C^{MM}&C^{ME}\\C^{EM}&C^{EE}\end{bmatrix}.
\]

For each block use a normalized group-query/original-record-key dot product, the maintained relative-geometry helper, inherited bounded temperature, typed score calibration, and the existing positive quadratic gate. Copy compatible projections/calibrators from the parent's final contextual block into a **separate compact-retrieval module**; do not alias them to the encoder's parameter objects. Record copied/new/dropped parameter names once in plain text.

All four Level-R blocks use **final prepared record states** in this prototype. This is a concrete implementation choice: they are new retrieval scores, not old stage-time raw scores. The original MM/EM/EE/ME encoder retains its actual sequential state versions. Do not relabel the new blocks as applied contextual edges.

Eligible compact destinations are live groups. Eligible M donors for final source access are physically present M records; E donors require the existing positive-information-measure convention. The information reader still excludes zero-measure records from its normalization without deleting physical M source columns.

There is no general compact-group self record. Do not restore a unit incidence for every ancestral ID. Own-record contextual residuals and the original local M envelope remain unchanged. Removing the native retrieval diagonal is an intentional representation change already present in the uncompressed compact control.

### 3.4 Direct query-to-K scoring

Compute scores from `q0`, live group latent controls, group centroids/lengths/extents and destination-bank identity. Reuse compatible query/key/geometry projections from the parent's M/E reader as initialization, and use the inherited type-specific calibration and one entmax over all valid live groups. Newly introduced extent contributions start at zero. No global scene-value summary is introduced.

The zero/new-feature initialization reduces avoidable initial disruption; it **does not** make the new function identical to native U-H. The descriptor, score construction and retrieval-identity rules change even when K=N.

No ordinary query should compute N old seed logits and then pool their probabilities into K. The reader's independent query-to-original-E information scores and post-E query-to-M information scores remain legitimate `O(QN)` work. They are not the retired seed scorer and must be counted separately.

### 3.5 What soft tying changes

For disjoint sibling pair `(u,v)`, compute a live pair descriptor `p_uv` from the combined ancestry. During local epochs 21–40, with phase factor `alpha`:

\[
t_v=\sigma(b_v),\quad w_v=\alpha t_v,\quad
 p'_u=(1-w_v)p_u+w_vp_{uv},\quad
 p'_v=(1-w_v)p_v+w_vp_{uv}.
\]

Apply the blend consistently to the latent **and routing-geometry** components. Tying latent features alone while leaving different score centroids would not tie the complete group functions. Original physical donor coordinates and values are never blended.

Both live leaf slots remain allocated during this phase, and their actual blended memberships are built by the new compact builder. Use those already computed rows for the tying loss; no extra pre-blend/native-directory capture is needed.

At the end of local epoch 40, activate the same rule in every scene: merge eligible siblings when `b_v>0`; otherwise keep both leaves. A zero logit does not merge. Freeze the shared topology-policy weights after this event. Continue updating live descriptor/scorer/membership and original model parameters. Regenerate cuts for new geometries from the frozen rule, not a TRAIN-case lookup table.

This changes score multiplicity and function; entmax on one group is not generally the sum of the old two entmax weights. Therefore the boundary gets an actual before/after native measurement, followed by the planned recovery phase. It is not advertised as an exact merge.

## 4. Learning objective and one structural transition

Retain the parent's native field/auxiliary/response objective and M/E/context work-controller equations. Add only the proposal's two weak, transient compression terms during soft tying:

\[
\mathcal L=\mathcal L_{native}+\mathcal L_{existing\ work}
 +\alpha\lambda_{tie}\mathcal L_{tie}
 +\alpha\lambda_{cap}\mathcal L_{cap}.
\]

Define `L_tie` as the mean over eligible sibling pairs of `t_v` times the equal-donor-type mean absolute difference between their **blended** membership rows. Within each nonempty type, average over its eligible donor records, not the combined E-dominated catalogue. Reduce per scene before the equal-case macro average.

\[
\mathcal L_{cap}=\operatorname{mean}_{cases}
 \left[\frac{N-\sum_{v\in eligible\ pairs}\sigma(b_v)}{N}\right].
\]

This is a differentiable capacity surrogate, not allocated K or executed work. The v1 forward does not sample Bernoulli merges. There is no fixed K quota, balance loss, entropy loss, physical residual amendment or new pruning dual.

Choose the two coefficients once at the common epoch-20 state using at most two TRAIN minibatches and ordinary native backward evaluation in a disposable copy at `alpha=1`. As a concrete weak default, scale each term's gradient norm on its participating new parameters to 2% of the native objective's norm on those same parameters. Record norms and resulting coefficients in a small JSON/config entry. If a comparison norm is zero/nonfinite, disable the unsupported term and report it rather than divide by a tiny number, tune on VAL, or block the whole goal. These are practical defaults, not optimum settings.

After hard activation, drop the soft-tying/capacity losses; K is determined by the frozen policy, and recovery uses the native plus existing work objectives. Do not keep a constant capacity term in the loss and call it ongoing learned compression.

### One review, not another admission bureaucracy

At epoch 40 save the ordinary candidate checkpoint, run before/after hard-cut predictions on the six representative scenes, and record K, final support, field/response-coefficient changes and nonfinite/semantic errors. Use a separate scene preparation after activation.

A finite accuracy drop is **not** a reason to suppress the planned recovery experiment or demand another user decision. Proceed through epochs 41–100. Genuine exceptions—invalid native correspondence, malformed tensors or nonfinite computation—stop only the defective candidate and are debugged normally; the independent control/evaluations remain useful.

If no pairs merge, retain that result. Do not force merges or search thresholds. There is no repeated cut/revert/retune cycle against VAL89. If the final compact model is unhelpful, recommend the control or current parent and preserve the negative comparison.

## 5. Training protocol: one full-data pair, no subset marathon

### 5.1 Parent and data

Use:

`HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_u_h_full5000_v1/checkpoints/epoch_1000_model.pt`

Resolve that path through the existing workstation symlink. The reference for U-S is the sibling `thermal_u_s_full5000_v1/checkpoints/epoch_1000_model.pt`. Do not substitute old U10-e5000, the superseded 18-epoch run, or a moving `latest` alias. If a requested parent is missing, provide that concrete failure and complete independent coding/tests; do not silently change the experiment.

Keep original TRAIN600 and canonical exposed VAL89. Inherit the parent's actual saved A2-D TRAIN600 normalizer; do not refit it. Keep Q1024, the native auxiliary receivers, saved response addendum, equal-case losses, source sampling namespace and current work settings. No quarter-data scientific screen is inserted.

### 5.2 Phases and counts

| Local adaptation age | Computation | Learning purpose |
|---|---|---|
| 1–20 | One common K=N Level-R model | Adapt the new descriptors/scorer/identity convention to real fields |
| 21–40 | Fork `R0` singleton control and `R1` soft-tying candidate | Learn proposed redundancy; identical case/query exposure |
| Boundary after 40 | One candidate hard-cut activation, ordinary checkpoint and six-case probe | Make the bank physically compact |
| 41–100 | Continue both branches | Recover/evaluate at the same native exposure |
| After 100 | Evaluation and handoff | No automatic extension or Level U/S campaign |

Use seed0, FP32, TF32 disabled, effective batch48, nominal microbatch24. Each epoch has twelve 48-case macros and one 24-case macro: **13 updates and 25 microbatches**. Normalize the last macro by 24, not 48. Reduce packing when needed without duplicating cases or changing effective batch/query identities.

Each endpoint has 100 local epochs of scientific exposure: 60,000 case visits, 1300 updates and 61,440,000 primary queries, excluding auxiliaries. The shared prefix is executed once. Total new physical training is:

- `20 + 80 + 80 = 180` epoch-equivalents;
- `180 × 13 = 2340` optimizer updates;
- `180 × 600 = 108000` case visits;
- `108000 × 1024 = 110592000` primary query occurrences.

The native e1000 H and S parents are frozen inference references, not additional training arms. R0 is the matched compression control; the parent alone cannot separate additional learning or the new scorer from merging.

### 5.3 Optimizer and four distinct clocks

Start the common adaptation with fresh AdamW moments, using the parent's actual `formal_joint` settings: LR3e-4, WD1e-5, betas(0.9,0.999), epsilon1e-8, clipping1.0. The report discloses a contradictory provider-summary WD; use the saved optimizer contract, not that summary. Copy compatible neural weights and parent M/E/context dual state. Name the experiment an adaptation, not an exact resume of parent training.

Fork both children from the exact common epoch-20 weights, optimizer, controller, sampler and RNG state. Do not reset them again at the fork. The compatibility module can be instantiated in both arms; only R1 uses it for tying. Remove/freeze its optimizer participation at epoch 40 without disturbing the remaining shared parameter moments. Use ordinary named-parameter state transfer; no per-scene trainable group slots or averaged Adam moments exist.

Serialize these clocks explicitly:

| Clock | Local epoch `s` | Purpose |
|---|---|---|
| Parent scientific age | 1000, constant metadata | Training ancestry |
| Child local epoch | `s`, 1…100 | History, stopping and milestones |
| Sampler epoch | `1000+s` | Same case/query and response schedule for R0/R1 |
| Existing work-controller coordinate | Loaded parent coordinate + `s` (normally `4000+s`) | Preserve existing mature sparse schedule |
| Soft-tying phase/cursor | Local epoch/update within 1…100 | New 20/40 transition, no inherited waiting phase |

Confirm the loaded parent coordinate instead of assuming all checkpoint epoch fields mean the same thing. Within a macro, every microbatch/tile uses the same phase. The existing once-per-epoch response family follows the parent's **sampler** epoch and final-macro placement.

### 5.4 Practical bounds and interruptions

Aim to finish the bounded comparison within eight GPU-worker hours of active scientific work; this is a planning ceiling, not a claimed throughput or an exclusive reservation. Use measured actual epochs to report progress. At a resource limit, save ordinary resumable state and deliver the attained result/commands; do not manufacture completion or wait indefinitely. User-launched long formal continuations remain a separate decision.

Save common20, candidate40-before/after structural state, final100, latest and the existing best monitor state. Refresh ordinary curves/monitoring every10 epochs. Do not create an extra checkpoint grid. Review fields at the end; transient finite misses do not trigger automatic hyperparameter searches.

## 6. Concrete repository changes

### 6.1 Current code findings that determine the patch

At the reviewed revision:

- `ContinuousRouter.forward` composes native seeds with `torch.bmm(seeds, incidence)`.
- `UnifiedPreparationMixin._finalize_native_preparation` unconditionally assembles `applied_native_directory` for hypergraph mode. A compact model cannot call that default and then claim it did not build B.
- `ContinuousJointCore._continuous_route_tile` obtains pre-E M/E scores and routes them through the native seed interface. Compact query scoring needs its own K-axis branch; original E information scores and post-E M scores remain.
- `read_continuous` already offers `dense_weighted`, `segment`, and historical `source_order`. Reduced projection and empty-support gradient handling already exist. Do not reimplement Round11's reader optimizations.
- The research runner has fresh classic-metadata initialization and a warm-start branch restricted to the old completed U-triplet workflow. A new full-data research e1000 adaptation is not just a config edit.
- `ResearchTrainingEngine` presently couples parent-age sampler/model-clock mappings in the warm-start mode. The new adaptation needs separate sampler and controller offsets.

### 6.2 Minimal new model surface

Add a small reusable module, for example `honf_forward_core/compact_groups/`, with only the numerical responsibilities actually needed:

| Component | Responsibility |
|---|---|
| `groups.py` | Same-type singleton pairing; symmetric compatibility; live ancestry/centroid/extent reductions |
| `retrieval.py` | Direct K×N typed membership and Q×K scoring; no original-seed fallback |
| `model.py` | `CompactRetrievalCore` using unchanged unified preparation and original readers |

Ordinary dataclasses should distinguish native record metadata from group metadata: `group_mask`, bank type, ancestry indices/offsets, live descriptor, live membership, and phase. No giant generic group-management framework or dense assignment matrix is needed.

Make a narrow overridable final-directory hook in the unified preparation owner. The legacy default continues to build its applied native directory. The compact override builds C directly from final record states without executing that legacy final-directory branch. Native contextual scoring remains unchanged and legitimately quadratic.

Override the routing part of `_continuous_route_tile` so it computes query-to-K scores. Retain the shared local-envelope, E read, E-conditioned M read, scalar response and physical extraction. Adapt `capture_continuous_trace` or add `describe_organization` to expose group IDs/lineage separately from record IDs; do not mislabel group columns as native seeds. Traces stay opt-in.

Final compact rows are not the native stage snapshots. Use clear names such as `compact_retrieval_membership`; keep existing `applied_context_incidence` metadata for the unchanged encoder. This prevents incorrect causal/complexity pictures.

In the new capability, retain the contextual incidence policy but exclude unused native-seed dispatch/direct-selector parameters from the active optimizer. Record that ordinary parameter migration; do not modify historical model classes or leave unexercised trainable branches solely to satisfy an old parameter inventory.

### 6.3 Extend the ordinary runner rather than cloning it

Use `HONF_Proj/tools/honf_research.py`, `honf_runtime/research_training.py` and the maintained `TrainingEngine`/Thermal provider. Add an ordinary initialization mode such as `research_checkpoint_adaptation` with `inherit_parent_train_stats`. The exact config names are implementation choices, not already-supported flags.

Load a new e1000 parent through the existing safe run-checkpoint reader and self-contained adapter-state reconstruction (`_read_run_checkpoint` / `_model_from_run_payload` or factored public equivalents). Do not send it through `classic_identity_metadata_only`, which intentionally does not load learned weights, or pretend it is an old `continuous_joint_train_v1` export. Preserve existing safe-loading and run-ownership checks.

The initialization path must copy the parent's real TRAIN600/VAL89 membership, normalizer, objective/response recipe, sampling namespace, model config and controller state, and then perform the documented compact-model parameter migration. Construct optimizer groups over the actual new/shared trainable parameter names. Ordinary model config versions and tests are sufficient; add no new source-closure manifest.

Provide a small explicit fork operation for common20 into two fresh output directories, carrying optimizer/controller/RNG/sampler states. Preserve the common prefix's history as inherited exposure while counting its physical compute once. Do not copy the whole source run tree or datasets. A same-run `resume` remains a resume; a fork is a different named child.

Add independent `sampler_parent_epoch` and `controller_parent_coordinate` mappings to the research engine for this capability. Do not overload `source_parent_age` to fix one clock at the expense of another.

Native and structural losses must participate in the actual engine's task/work derivative path. A scalar recorded in logs is not trained. Use the existing additive-loss hook if available; otherwise make one narrow scene-level structural-loss hook, aggregated once per scene/microbatch with the real macro denominator. It must not be charged once per query tile or once per auxiliary role. Replayed work-only VJPs must not apply it twice.

Integrate compact configs into ordinary self-contained adapter save/load, evaluation and resume. No new checkpoint format should require old parent files during inference. Copy safe loader behavior rather than setting unsafe deserialization flags.

### 6.4 Shared backbone boundaries

The compact numerical modules accept typed record tensors and declared frame/scales, not a dataset name. Thermal-specific normalization, legal geometry, T=Kh and response-target extraction remain in its adapter. Keep local-input capability resolution behind the existing adapter boundary. Do not add Wind/HET-MF wiring or hypothetical future APIs in this round.

## 7. Execution settings and GPU practicality

Start with the full-data parent's effective settings: final reader `dense_weighted`, query tile1024, value checkpointing, and the parent's contextual backend. Both R0/R1 use the same settings unless actual OOM requires an execution-only adjustment.

Dense-weighted is permitted because organization is the primary experiment and Round11 did not show a segment speed advantage. It must report all executed rectangular values. A compact directory can reduce organization dimensions while final values remain dense; that is an honest partial result.

Optional cost checks compare `dense_weighted` and `segment` on the **same trained function**, not by setting gates to one. Preserve guarded denominators, signed source coefficients and E conditioning. Known source-order/segment numerical limitations remain disclosed, not retroactively passed.

Use any currently user-permitted GPU; no permanently bound device identity. Existing active formal/HET-MF/other-user jobs are not to be stopped or modified. GPU occupancy alone is not a reason to wait: shared use is allowed when feasible. Use the maintained OOM rollback/repacking and resume on another permitted device when necessary. If none is usable, finish CPU work and deliver exact resume commands rather than indefinite polling.

No new GPU reservation service, free-memory admission daemon or campaign-global writer lock. The existing **same-output-directory** writer lock remains necessary to avoid two writers overwriting one run. It is not a lock on an entire GPU.

## 8. Minimum tests, then actual learning

Keep a compact affected-test set, organized around new code rather than reopening historical campaigns:

1. **Group algebra and identities:** singleton/tie/odd counts, permutation-consistent pairing, original record/source axes intact, direct compact axes, empty donor type and zero-measure M source handling.
2. **Numerical function:** a small FP64 literal membership/access/value reference with parameter and geometry derivatives under fixed topology. Include the identical-row/summed-mass identity and a test showing that naive merged entmax logits do not generally preserve it.
3. **Actual model execution:** a real low/high-M Thermal forward/backward with native auxiliary/response extraction and gradients reaching new descriptor/scorer/coarsener parameters during tying. Preserve zero heat, heat superposition and heat-independent flow/routing.
4. **State transitions:** ordinary save/load, common20 fork, hard-cut/frozen-policy restore, separate clocks, and one interruption/resume. No retired N-seed directory is constructed by the compact retrieval hook; contextual N² allocations are explicitly allowed.

These are normal software tests. No new receipt service, hash-based admission stage, comprehensive source audit or bitwise-Adam certificate is required. A failed new primitive gets a specific bug report/fix; do not use an unrelated optional parity/timing failure to block other actual model work. Preserve existing access controls, safe deserialization, physical data separation and protected run artifacts.

## 9. Evaluation: a small set that answers the scientific question

### 9.1 Models and panels

Main trained comparison: **R0-local100 versus R1-local100**. Reuse available parent H/S epoch-1000 field boards when compatible; obtain their missing response/ranking evidence rather than rerunning classic1404/1804. Do not promote old U10-e5000 response results into current-parent evidence.

Score all canonical VAL89 native receivers at terminal evaluation. Report the nine native quantities separately: u/v/p/omega, fluid/surface/material T, q-proxy and recomputed module peaks. Retain equal-case means, p90, worst IDs and M strata. Use the existing fixed TRAIN24 for low/high-M diagnostics; M1/M2/M12 remain TRAIN evidence.

Use the six representative cases already in the new organizer panel: VAL0274/0658/0293/0692 and TRAIN0006/0254. Include valid surface/material queries in addition to the eight old grid indices when diagnosing rare roles. The saved index5850 in case0293 is not a valid fluid-error receiver; keep any such core probe separately labeled.

### 9.2 Organization, not just small K

Record by scene and receiver role:

- N, K_M/K_E, accepted eligible sibling count, and padded K allocation;
- actual C coefficients, exact-positive members, query-group mass, final M/E support and actual value rows;
- compact descriptor/membership/scorer work, native encoder work and retained reader score work separately;
- per-query union inflation and full-E-query frequency, including small M cases;
- group reuse across fluid/surface/material roles, including rare groups;
- live changes of C and query access under a legal module shift, versus changes in the cut itself.

Do not use effective seed count as a target K, threshold small positive entries out of the execution count, or equate duplicate-looking heatmap rows with redundant physical influence.

At the epoch-40 boundary, measure the **combined** cut, not a sum of independently tested pair changes. At the terminal endpoint, make one bounded group intervention: disable the most-used group selected by input/query mass, recompute the remaining group normalization and native prediction, then restore. Report changed final M/E work, K and fields. This is trained-model reliance, not physical causal discovery or a work-matched H-versus-S result.

A successful organization outcome means a genuinely smaller directly constructed reusable bank with useful fields/response retained; graph appearance or compression without utility is insufficient. S remains a stronger or weaker reference on its own lineage, not a matched compression control.

### 9.3 Current-weight response and decision evidence

Evaluate native H/S e1000, R0-local100 and R1-local100 on the existing four reference families 0001/0318/0333/0348 and the original six-candidate pool. Run missing current-parent measurements once and reuse them thereafter.

Keep exposed rank-1 heat-response supervision separate from saved geometry responses and near-zero mixed-heat nulls. Use absolute errors when the physical difference is near numerical zero. Do not invent missing geometry-plus-heat or independent-E factorial endpoints.

For ranking, preserve the original smooth module-peak objective (tau2.0), constraints, candidate IDs and pressure limit **0.1364936370**. Report objective errors, rank correlation, feasibility confusion and the reference outcome of the model's selected candidate. A reference-infeasible selection has undefined feasible regret; a layout with no predicted feasible candidates yields “no feasible recommendation,” not a forced winner.

Demonstrate existing fixed-layout K reuse by comparing cached K contraction against fresh native predictions for a few admissible heat vectors. A single small `allocate_minimax_heat` example may be run using the already defined bounds/total heat from a compatible saved design request. Without such bounds, test contraction only. Cached K reuse is available to all relevant models and is not a unique compact-hypergraph achievement. Pressure-infeasible geometry cannot be repaired by heat allocation. No new proposed heat vector is called physically verified without reference truth.

### 9.4 Geometry derivatives and new topology seams

Use two legal module translations on the fixed low/high-M scenes. Compare complete live-model autodiff with represented finite differences under a fixed cut, rebuilding original states and current group descriptors. Separately inspect whether ordering or `b_v>0` changes the cut under those shifts. If a topology change is found, show shrinking-gap outputs on both sides; do not freeze assignments and call that a live derivative.

The inherited row-max calibration is continuous but can kink at ties. The hard sibling cut and coordinate sorting can introduce genuine topology seams. A smooth positive gate does not make the whole compact model globally C1. If no crossing is found in the bounded sample, say so; that is not a global continuity proof.

These diagnostics, saved-reference ranking and fixed-layout heat reuse are the inverse-use scope. No new geometry optimization campaign or physical solver authorization is implied.

### 9.5 Practical cost, only opportunistically

When an idle permitted GPU is available, spend at most about 15 minutes on disposable cost checks: a low/high-M native call, one representative full native training macro, and optional segment/dense-weighted comparisons at identical coefficients. Use small warm-ups/repeats and state the timing boundary and hardware. Include native extraction and work derivatives when claiming training-macro cost. Model loading is excluded from steady-call numbers and stated separately.

Do not save changed optimizer/model states or alter scientific histories. Retain only a compact timing/count/memory table if quoting results; remove disposable profiler dumps. If hardware is contended, record cost as unavailable or observational and finish the scientific report. No efficiency win is a prerequisite for reporting the model result or preparing a user-controlled formal continuation.

## 10. Interpretation and stop/handoff decisions

Use separate conclusions for predictor, organizer, response/decision and cost. As practical reporting bands—not blocking admission rules—show whether R1 stays within +10% mean and +15% p90 of R0 on each priority temperature/peak quantity and whether any M stratum has a large regression. Keep all flow/q-proxy quantities and absolute errors visible. These bands are choices for this plan, not physical validity limits or statistical significance.

| Result | Conclusion and next handoff |
|---|---|
| Smaller K with comparable fields and no important response deterioration | Compact retrieval is promising; prepare a user-launched continuation recipe |
| Smaller K but still broad/full E access | Organizational compression only; do not claim value sparsity |
| Smaller K but slower cost | Useful representation may still exist; no speed claim or automatic executor rewrite |
| R0 already degrades strongly relative to parent | New retrieval representation/scorer is the limiting transition, not demonstrated harm from merging |
| R1 degrades relative to matched R0 | This merge policy is not useful at this horizon; retain R0 and the negative result |
| K remains N | Compression hypothesis did not materialize; no forced count reduction |
| New topology seam harms live geometry response | Fixed-topology/heat reuse may remain usable; withhold broad geometry-inverse claims |
| Ranking remains weak despite good field means | Do not claim design reliability; report the actual failures |

No result automatically launches Level U, the U-S prior, a new coarsener or a long training portfolio. Provide exact ordinary launch/resume commands for the recommended state. The user decides whether to continue formal training. If the declared short experiment is interrupted by resources, hand off resumable checkpoints and the attained evidence rather than seeking more elaborate “closure.”

## 11. Operational simplicity and preservation

Use Git history, a small model-capability version, normal config files, checkpoint state, focused tests, atomic writes and the existing run-local writer lock. Do not introduce new cryptographic hashes, source snapshots, checksum inventories, frozen manifests, launch-admission receipts, global reservations or output-reconciliation services. Existing checkpoint/data authentication that the maintained loader already performs stays intact; do not disable it or add another parallel layer.

The latest user instructions supersede proposal text requesting new hash freezes, extensive preflight proof or dataset-wide cleanup attestations. New enforcement mechanisms would need a concrete failure that ordinary software primitives cannot address. None is proposed here.

Protect active runs using a separate ordinary Git worktree/branch when code imports could be affected. Do not patch sources under live workers. At delivery, commit relevant code/docs normally and push through existing access controls/hooks without force-pushing or bypassing security.

All new generated results belong under a dedicated `/data2/wanglz/ModularDT/diagnostics/thermal_compact_round12_<run>/` root, browsable through the existing ignored project diagnostics symlink. Keep source/config/docs in Git, payloads out. Reuse existing packed data and response caches; no copies of the old result trees. Use the maintained storage writer's existing checks; do not write to a fake unmounted `/data2` path. Plan roughly 10 GiB for the entire bounded output, mostly checkpoints and selected arrays, using ordinary configurable storage limits rather than a new global quota service.

Retain common20, before/after40, final100, latest/best, histories and small evaluation tables. Limit report figures to approximately six readable pages: fields/tails; compact group lineage/C; query access including dense exceptions; cut/shift behavior; physical responses/ranking; optional cost. No exhaustive 100-board catalogue is requested.

Delete only newly owned disposable test/profiler files; never erase historical negatives, datasets, old checkpoints, other-user files or active-run outputs. No broad cleanup campaign is part of completion.

## 12. Required goal deliverables

1. One implemented Level-R compact-retrieval capability with original physical record/source axes preserved, plus ordinary save/load/fork/resume support.
2. One common20 and one matched R0/R1 local100 comparison on TRAIN600/VAL89, or a clearly stated attained resumable stage if actual resources prevent completion.
3. A concise report with per-field/tail/stratum results, direct K×N construction evidence, live/allocated K and final work, actual current-checkpoint response/ranking, and topology/derivative limits.
4. Fixed-layout K reuse demonstrated; no unsupported claim of physical group causality or generated-design success.
5. Optional disposable cost measurements, never a requirement for completing the other deliverables.
6. Tested launch/resume/evaluate instructions for the user to extend the recommended model without a second campaign framework.

The report should answer a simple question: **Did learning and compacting reusable groups preserve useful physical information, and what precisely remains to be improved?**

## 13. Source basis and reading map

### Supplied sources

- `HONF_Round11_Thermal_Development_Report.md`: current full-data e1000 comparison; historical U10 evidence must stay separately labeled.
- `HONF_UH_Soft_Merged_Hyperedge_Upgrade_Design.md`: Sections 5–7, 8.1, 10.1, 11.2–11.5, 12, 19.2 provide the Level-R axis separation, sibling policy, transition and matched100-epoch starting design. Level U sections are deferred.
- `HONF_US_Geometry_Conditioned_Implicit_Hypergraph_Upgrade_Design.md`: Sections 1, 6 and 20 establish the narrower fixed-E-grid prior and its distinct purpose. It is not implemented in Round12.

### Current repository files inspected at 43fcdd2

| Path under `HONF_Proj/` | Used for |
|---|---|
| `src/honf_forward_core/continuous_joint/routing.py` | Native seed composition and direct selector |
| `src/honf_forward_core/continuous_joint/incidence.py` | Typed gates and retrieval identity |
| `src/honf_forward_core/continuous_joint/unified.py` | Applied preparation and final native-directory construction |
| `src/honf_forward_core/continuous_joint/model.py` | Preparation/route/read ownership, execution options and work replay |
| `src/honf_forward_core/continuous_joint/reader.py` | Existing dense-weighted/segment/source-order executors |
| `src/honf_forward_core/continuous_joint/budget.py` | Existing raw clock offset, typed actual/proxy work |
| `src/honf_forward_core/native_retrieval/route_capture.py` | Actual contextual stage snapshot contents and correspondence |
| `src/honf_runtime/research_training.py` | Initialization limitation, provider, clocks, device/packing and resume behavior |
| `tools/honf_research.py` | Maintained public entry point |
| `configs/research/thermal_u_h_full5000_v1.json` | Actual source profile and execution defaults |
| `docs/guides/HONF_Thermal_Unified_Full5000_Protocol.md` | Current protocol, normalization and loss details; old S launch-status prose is superseded by the newer report |

This is source/report analysis, not a replay of private e1000 weights or an observation of live workstation processes.

### External precedents checked, not validation of this new model

- Peters, Niculae and Martins (2019), *Sparse Sequence-to-Sequence Models*, ACL, DOI `10.18653/v1/P19-1146`. Supports the existing exact-sparse entmax mapping, not compact-group equivalence or HONF physical significance. https://aclanthology.org/P19-1146/
- Ying et al. (2018), *Hierarchical Graph Representation Learning with Differentiable Pooling*, NeurIPS 31. A precedent for learned soft assignments; this plan does **not** transplant pooled node values into the physical decoder. https://proceedings.neurips.cc/paper_files/paper/2018/hash/e77dbaf6759253c7c6d0efc5690369c7-Abstract.html

All phase boundaries, weak penalty choices, implementation hooks and reporting bands in this plan are prospective research decisions. None is represented as an already measured improvement.
