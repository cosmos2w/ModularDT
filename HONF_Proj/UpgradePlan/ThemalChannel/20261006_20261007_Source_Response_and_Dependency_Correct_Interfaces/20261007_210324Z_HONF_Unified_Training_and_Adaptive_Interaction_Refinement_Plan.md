# HONF next phase: unified training and adaptive, source-preserving interaction refinement

**Mode:** Goal mode; bounded model development, not formal full-data training. **Reviewed source:** `agent/honf-core-next`, commit `2af49e7564ba127a5a427c40545a5bfdaccb3d7a`. **Primary evidence:** `HONF_Proj/docs/reports/HONF_Shared_Interaction_Core_and_Receiver_Packets_Report.md`. **Status:** this document proposes work; its new architecture, targets and expected benefits are not completed results. Repository paths below are relative to `HONF_Proj/` unless stated otherwise.

## 0. Decision in plain language

Keep the efficient, physical-source-resolved family. Keep the shared context implementation and the distinction between Thermal's affine response law and Wind's nonlinear field law. Do not continue the optional predicted-flow Thermal branch, polish the same post-fit deletion policy again, revive recursive Trees, or start another broad architecture portfolio.

The next scientific question is different from the last one: **can reconstruction training teach a small organizer where accurate individual interactions are needed, while a cheap source-resolved approximation carries the remaining information?** A source that does not receive detailed processing must not automatically disappear from the prediction.

Implement one common training workflow and one new model family, with two matched execution/training arms per dataset: **Full-detail** and **Adaptive-detail**. Both share the same fine model, inexpensive approximation, dataset inputs, initialization, warmup, reconstruction objectives and data stream. Adaptive-detail adds learned selection of fine corrections. The full-detail path must remain recoverable by opening every refinement gate. This is an explicit escape from an irreversible group-pooling bottleneck, not a guarantee of successful optimization or useful sparsity.

The result we seek is a **promising research direction**, not a mature universal hypergraph or a win over every Dense1804 quantity. A successful result combines a credible field predictor, faithful adaptive information processing, an acceptable measured accuracy–cost frontier, and a real common training implementation across Thermal and Wind. A meaningful memory, repeated-read or derivative-cost advantage can justify further research even when cold latency does not improve, provided costs and limitations are reported honestly.

### Required deliverables

1. One maintained training engine used by both real datasets, including native sampling, optimizer accumulation, scheduling, checkpoint/resume, validation, counters and reporting. Dataset-owned physics remains in adapters/objective providers.
2. An efficient full-detail reference and a matched, task-trained adaptive-detail candidate on each dataset. Give healthy fits meaningful maturity rather than treating another short Wind pilot as a mature architectural comparison.
3. An organizer whose decisions are executed during field learning and inference, with physical source IDs and explicit coarse/fine/environment paths; not only a frozen-teacher visualization.
4. A tensor-first packet/compiled-request path, differentiable fixed-route consumption, and preservation of stale-state, normalization, source and receiver ownership protections.
5. A short decision report, intuitive physical figures, detailed numerical appendix, and manual-only formal recipes for any candidate that earns continued investment.

## 1. What the preceding round established—and what it did not

The report below is the source of the measured numbers in this section. The following interpretations must remain separate from new hypotheses.

| Previous question | Measured answer | Consequence for this phase |
|---|---|---|
| Does predicted environmental flow improve Thermal? | Selected fluid-T RMSE changes from 0.733013 to 0.732888: 0.0170%. The optional edge is live, but counted responses do not improve and inference is roughly 8% slower at selected weights. | Keep geometry-only preparation as the default; do not spend another fit on this injection. This negative result concerns this specific frozen provider, injection and continuation, not all physical-context conditioning. |
| Does the packet proposer select useful donors? | Learned proposals meet the measured distortion request in 336/336 eligible patch checks at each budget and match exact equal-K selection. Only 10.71%/16.37% of eligible patches are partial. | Keep the learned-sensitivity and ownership infrastructure. Change the approximation problem rather than demand fewer genuinely important sources. |
| Is there real sparse execution? | On TRAIN0097 fluid Q7898, active pairs fall from 31,592 to 30,089: only 4.76% are omitted. Padded rows fall much more, but complete guarded calls are 312.394 versus 304.849 ms. | Separate padding cleanup from scientific interaction savings; do not call either count a latency win. |
| Is the core genuinely shared? | Thermal and Wind execute the same `InteractionContextCore` functions with affine and nonlinear readouts. Compatibility extraction adds about 0.99% median latency and preserves the measured native behavior. | Build on the current implementation; do not re-extract another shared facade. |
| Is the Wind predictor competitive? | At 2000 updates, volume/near-turbine component RMS is 0.109503/0.451953 m/s versus retained 0.010517/0.043943. Only 27.778 effective passes over 72 TRAIN direction rows were completed, using H32/message32 and eight environment donors. | This proves learning feasibility, not mature accuracy. Test a reasonably sized shared baseline with sustained training before judging the new family. |
| Is inverse use established? | Precise Thermal increments, local Wind JVPs and four software consumers exist. No recovered design or independent finite Wind response was validated. | Continue interface/reuse checks, not a new inverse-generation campaign. |

The Thermal continuation itself has a tradeoff independent of the flow-context hypothesis. Relative to development R-direct2500, selected R-geom1200 improves nominal fluid error 5.37% but worsens counted fluid/surface/material response errors 10.06%/19.33%/21.71%. Terminal2500 nominal fluid error improves further while response errors regress further. Do not replace the parent response reference merely because a field-only selector improves.

The many negative single-heater coefficients also remain a physical warning. In the fixed TRAIN audit, their norm share is approximately 3.65–6.13%; these are not all negligible roundoff values. The report's conditional steady M-matrix argument supports nonnegative single-heater responses under its qualified assumptions. A negative *balanced-transfer* response is a different, potentially valid quantity. Do not repair interpretation by clipping an old checkpoint or by calling signed computational terms causal Green's functions.

The original formal3901/3902 family retains its separate completed full-TRAIN result and strong measured cost advantage. The present development children do not supersede that result. Full-data and quarter-data errors must never be placed in one unqualified ranking.

## 2. Source review: concrete starting points

| Existing file | Inspected finding | Required action |
|---|---|---|
| `src/honf_forward_core/interface_fields/interaction_core.py` | Shared two-round source/environment contextualization; source-wise reads; physical-measure reduction; affine/nonlinear dependency contracts. Nonlinear readout reduces source messages before a small field head. | Retain the shared base. Add the approximation/refinement mechanism beside the readout, without reintroducing the old field core. Keep physical-source identity through the fine computation. |
| `src/honf_forward_core/interface_fields/source_response_operator.py` | Source-resolved near/far coefficients; separately applied heating; precise accumulation; inherited shared context. | Preserve existing class/checkpoint behavior. Add an opt-in refinement wrapper/readout with all-fine equivalence, stable physical units and precise increments. |
| `src/honf_forward_core/interface_fields/receiver_packets.py` | Host-side immutable IDs and fingerprints are repeatedly converted/validated. `union_receiver_packets` still loops over requested receivers and writes mask entries individually. | Compile IDs, row maps, source masks and local fallback at a request boundary; execute tensor work afterward. Keep the current path as an audited reference. |
| Same packet file, `apply_affine_packet_increment` | Decorated with `@torch.no_grad()`. It is a value-only packet consumer, not an existing differentiable packet inverse path. | Keep a clearly named inference-only convenience API; provide and test a separate differentiable fixed-route tensor consumer. No blanket `no_grad` around a derivative-capable public call. |
| Same packet consumer | `has_packet_fallback = any(packet_union.fallback)` disables subset use for the entire request if any row falls back. | Partition qualified and unsupported rows/roles, process the latter fully, and join once. Do not downgrade all fluid rows because some solid rows are unsupported. Preserve fail-closed semantics per row. |
| `Case_WindFarm/src/windfarm/shared_interaction.py` | Real shared nonlinear model and physical denormalization; `_prepared_ownership_signature` binds scene/core metadata but does not visibly include `velocity_transform`. | First add a normalizer-mutation regression. Bind the physical transform to prepared/model identity and reject stale physical-output/linearization use. This is a static-review concern to reproduce, not a claim that recorded results are invalid. |
| `tools/wind_shared_interaction_pilot.py` | Separate update-based H32 pilot, hard cap2000, one direction-case visit/update; native catalogue and role preparation are expensive outside the neural read. | Preserve historical pilot. Migrate new scientific training to the common runner; reuse static native catalogues and sample fresh epoch-indexed queries rather than prebuilding another entire update-history array. |
| `tools/thermal_source_response_flow_context_fit.py` and `tools/thermal_source_response_fit.py` | Separate Thermal fitting workflow with valuable native losses, optimizer lineage, fixed data binding and response objectives. | Reuse the math and loaders as providers to the common engine. Do not silently change their denominators, scales or scientific histories. |
| `Case_ThermalChannel/src/channelthermal/source_response.py` | Thermal-native grid/role/proxy extraction and frozen D-sep composition. | Keep native extraction and heat independence dataset-owned. Leave optional flow-context disabled for this comparison. |
| `docs/guides/_bk/20261029_Guide_Consolidation/01_Legacy_Field_and_Response_Interfaces/Shared_Interaction_Core_and_Receiver_Packets.md`, development protocol, repository `AGENTS.md` | Current contracts and scientific/format/artifact rules. | Update documentation to distinguish unified execution, unified training, separate weights and unproved cross-dataset transfer. |

The report's cold Wind application takes about 1.6–1.8 s at M8 and 10.7 s at M30, whereas prepared reads including their guard take 2.26–10.43 ms. CPU geometry/role catalogues dominate that first-use measurement. This is an application bottleneck, not evidence that neural inference itself needs ten seconds. Likewise, the packet consumer's hundreds of milliseconds cannot be directly compared with the ordinary approximately26 ms Thermal call without matching receiver/role/precision/setup scopes.

## 3. General model contract

### 3.1 Common objects; dataset-owned meanings

Use the existing scene and dependency types. Let X contain physical modules M, environmental context E, prescribed conditions c, measures, identities and dependency declarations. Requested receivers Q have physical coordinates, roles, weights and identities. Observations/targets remain outside inference inputs.

The shared encoder produces source states Z_i, environmental states Z^E_j and a global state z_g. Whole-layout contextualization remains explicit. A coefficient attached to source i may depend on other modules and environment through those states; source-resolved output does not imply independent geometric effects.

Keep separate definitions for configuration variables, separately applicable controls, output laws, control units and local linearizations. Thermal can declare affine heating and heat-independent flow. Wind cannot inherit this law by renaming turbines as heaters. Different dataset heads, losses, normalization and parameter values are allowed. One common trainer does not require mixed-dataset batches or shared trained weights.

### 3.2 The new approximation problem

Current deletion approximates an interaction F(q,i) by zero when source i is omitted. Instead learn an inexpensive approximation B(q,i), then refine selected pairs:

\[
\widetilde F(q,i)=B(q,i)+g(q,i;X,\eta)\,[F(q,i)-B(q,i)].
\]

Here F is the full source-wise fine read, B is a low-cost source-wise approximation, g is a learned gate, and eta is a declared accuracy/cost setting if used. Neither B nor g may read reference values at inference.

Three limits have unambiguous meanings: g=1 exactly recovers the fine model; g=0 keeps the coarse approximation rather than deleting the source; intermediate soft g is a training/smooth-read mode whose actual computation must be counted. The core retains a full-detail escape path. This avoids an architectural accuracy ceiling caused by forcing all source information through a small latent-group bank, although it does not guarantee that a trained router will be useful.

The organizer answers: **for this receiver and current scene, which source interactions need the expensive correction, and which are adequately represented by the inexpensive path?** This is adaptive fidelity allocation, not a claim that unrefined sources are physically irrelevant.

### 3.3 Thermal specialization

At fixed scene, let F(q,i)=K^fine_i(q;X), B(q,i)=K^base_i(q;X). Then

\[
\widehat T(q;X,h)=b(q;X)+\sum_i\left[K^{base}_i(q;X)+g_{qi}(X,\eta)\left(K^{fine}_i(q;X)-K^{base}_i(q;X)\right)\right]h_i.
\]

Retain b=0 for the existing qualified zero-boundary benchmark. An all-fine request must dispatch directly to the retained fine read, not subtract and re-add B in FP32 and assume bitwise identity. For mixed routes and precise increments, widen the underlying factors before kernel assembly/contraction, preserve the existing precision contract, and report operation-order differences separately. All fine gates, base coefficients and fine coefficients must be independent of current heating. In particular, do not feed nominal heating, heating-dependent action radii, observed temperatures or heat-conditioned global summaries into the ordinary kernel/router. Otherwise the advertised exact affine law is lost.

Use a fixed TRAIN-qualified forcing scale/envelope when training an affine refinement-risk score. An action-specific inverse packet may carry a separate bound, but its route must be frozen within an exact affine endpoint/increment comparison. If a route changes with the action, label that change and do not call the resulting global map affine.

Native fluid/surface/material temperatures continue to come from the same scalar field and original extraction. Evaluate maxima and effective-h proxies at endpoints; they are not affine functionals. Keep the precise contraction of delta-h and preserve the known heat-null flow dependency through actual inputs.

For a fixed route and admissible forcing increment, the approximation error relative to the fine learned model is

\[
\Delta T_{fine}-\Delta T_{adaptive}=\sum_{i:g_{qi}=0}(K^{fine}_i-K^{base}_i)\,\Delta h_i.
\]

An analogous triangle bound uses residual sensitivities `RMS_P(Kfine_i-Kbase_i)` rather than `RMS_P(Kfine_i)`. This can be substantially less conservative *if B learns the broad interaction well*. The reduction is a hypothesis to measure, not a guaranteed result. The bound still concerns the learned fine model, not physical truth.

### 3.4 Wind and other nonlinear specializations

Use F(q,i) and B(q,i) as vector source messages. Apply refinement before physical-source reduction:

\[
\widehat u(q;X)=u_{background}(q;c)+\Psi_\theta\!\left(q,z_g,\sum_i\mu_i\,[B(q,i)+g_{qi}(F(q,i)-B(q,i))]\right).
\]

Retain the adapter's actual normalization and the executed measure convention in both arms. Do not change mean-versus-sum reduction in only one arm. Source messages and their norms are latent computational quantities, not physical velocity contributions or control derivatives. Export actual physical JVP/VJP quantities separately.

For this new Wind comparison, use H64/message64 as a single predeclared baseline size, rather than repeating the H32 feasibility pilot. Retain its eight environmental records initially so that the main comparison does not also sweep environment resolution. Both arms use the same native support features and source-length-relative geometry. The source read already sees local and domain scales; do not falsely describe the existing implementation as lacking them.

A TRAIN-fitted height-profile output baseline may be used in **both** new Wind arms to separate broad background reconstruction from interaction detail. Fix this choice in the common profile before training, reuse the already implemented TRAIN-only fit, and record it as a learned empirical background—not prescribed physical inflow. It must be added through the physical output transform with exact normalized/raw tests. Its use is common to Full-detail and Adaptive-detail; improvements over the older pilot cannot be attributed solely to routing.

Nonlinear finite response has no inherited triangle certificate through Psi. Compare actual full/adaptive output differences, native targets and local AD derivatives. Do not treat a message-space approximation bound as a physical velocity-error bound.

### 3.5 Cheap path and parameter economy

Prefer a small factorized per-source read rather than another complete model: project each source state once and each query once to an 8–16 dimensional feature, combine them by elementwise product/dot product, and add a small relative-geometry term. For vector messages, project that small combined feature to the fine-message width. This is an implementation suggestion, not a latent hyperedge count.

The inexpensive representation is not the only information path: full individual fine reads remain available and protected near interactions remain fine. No source is replaced by an averaged module token before its available nonlinear fine computation. Build cheap features without first allocating the full hidden-width Q×M fine tensor.

Retain the current shared context rounds in this phase. Do not add a new dense Q×E stack, recursive partition solver, global latent group bank, source-merging loop or extra full predictor. The current environment path is context ancestry, not a separately pruned environmental read.

### 3.6 What counts as the hypergraph-inspired organization

The packet is a directed receiver region/list, selected detailed physical donors, and explicit context/environment ancestry, with a recorded coarse path for every remaining donor. Packets can overlap; execution counts each pair once. Physical modules are stable identities; packet IDs are computational labels, not material objects or ground-truth physical regions.

Primary learned gates should be receiver-specific. Do not make a fixed4×4 grid the resolution of the learned physics. It can remain a reporting aggregation. A tensor executor can group equal masks into execution blocks after selection, but such grouping must not alter the mathematical gate or average distinct fine values.

Export coarse and fine path strengths separately. An unrefined donor still affects the coarse prediction and potentially the contextual states. Draw it as a faint coarse link, not as an absent physical dependency. State honestly that this is source-preserving adaptive interaction refinement, not yet an independently identified many-body causal hypergraph.

## 4. Workstream A — application efficiency and contract repair

### A1. Compile once at the trust boundary

Convert immutable source/receiver IDs to integer maps when constructing a validated request. Store coordinates, native row IDs, source slots, role partitions, packet-to-receiver incidence and masks as tensors. Validate serialization hashes, checkpoint identity and external metadata at load/compile boundaries. During repeated reads, use bounded ownership/version/configuration/normalization checks on those compiled objects.

Do not suppress stale-state errors, make guards optional by default, or trust user-owned mutable lists without copying/binding them. Every current ownership test must remain meaningful. Add tests for parameter replacement/reload, optimizer steps, changed normalization, source order, receiver coordinates/features, unit metadata, packet mask mutation and scene changes. Inverse geometry work must rebuild a live computational graph and cannot reuse detached cached geometry.

Replace row-by-row CUDA scalar writes and per-read Python reconstruction of entire coordinate catalogues with batched integer/tensor operations. Retain human-readable IDs as export metadata. A large speed gain from this work is an engineering gain, not scientific sparsity.

### A2. Row-local fallback

Resolve unsupported roles and unqualified rows individually. For example, solid receivers can use full fine access while qualified fluid receivers use the adaptive path. Join outputs in the exact original order. One unsupported row must not automatically force unrelated qualified rows to full computation unless a real shared dependency requires it; document such a dependency if found.

### A3. Differentiable consumers and normalizer ownership

Preserve the existing value-only packet consumer for reproducibility. Add a separate fixed-route consumer whose kernel/forcing/native extraction operations retain autograd. Host metadata compilation may be nondifferentiable; physical tensors and their legal derivatives must remain live. Hard membership derivatives are piecewise fixed-route derivatives, not gradients through a discrete topology switch.

Reproduce the Wind normalizer-binding concern before fixing it. A prepared request followed by a mutation/replacement of `velocity_transform.mean`, `safe_std`, `u_ref_mps` or its identity must not silently return a differently scaled physical field. Reject stale use, rebuild explicitly, and test endpoint and local-linearization paths. This test must not modify preserved scientific checkpoints.

### A4. Native Wind data preparation

Build static geometry/role catalogues once per fixed source row or actual geometry identity and reuse them during training. Open memory maps once. Keep target arrays on the supervision side. Store native index/measure maps, not a second copy of the complete CFD fields. Draw fresh queries from deterministic `(seed, epoch, layout, direction, role)` streams; do not allocate an enormous precomputed training history.

Separate first-ever dataset/catalogue initialization, a new-scene prediction, and repeated prepared reads in timings. A genuine geometry update must rebuild affected features and role validity; it cannot reuse the static catalogue blindly. Do not time the hot cache and label it first-use inference.

### A5. Small, decisive preflight

Use fixed TRAIN low/high-M cases and the existing representative panels. Confirm compatibility of unchanged fine predictions and derivatives before performance claims. Measure end-to-end calls, setup, prepared reads, training steps, memory and role coverage under identical scopes. Limit performance-tuning to two implementation attempts, with a combined engineering GPU cap of one hour. The task must proceed to learning rather than becoming another guard/profiler campaign.

## 5. Workstream B — one actual training workflow

### B1. Shared runner, not another framework

Reuse `honf_runtime` checkpoint/run-store/compatibility utilities. Create one maintained runner or extend the smallest suitable existing runner. Old scripts may remain as thin entry points/adapters; new Thermal and Wind optimizer loops must call the same engine. Do not create another registry, approval database, provenance framework or independent checkpoint format without a demonstrated necessity.

The shared engine owns: seeding, deterministic epoch visitation, collation/microbatching, gradient accumulation and last-batch denominators, optimizer/schedule steps, common stage transitions, train/eval modes, exact resume, checkpoint selection, every100-epoch monitoring, counters and process accounting. Dataset providers own: scene construction, target sampling, native output extraction, loss terms/scales, applicable response/null checks and physical metrics.

Suggested provider contract, to adapt to existing repository types rather than copy literally:

```python
class DatasetTask:
    def epoch_cases(self, epoch: int, seed: int): ...
    def collate_and_sample(self, case_ids, epoch: int, seed: int): ...
    def make_scene(self, batch): ...
    def predict_native(self, model, scene, batch, execution_mode: str): ...
    def losses(self, predictions, batch, auxiliary_state): ...
    def validation_batches(self): ...
    def reduce_native_metrics(self, records): ...
    def identity_payload(self): ...
```

Targets must not be accessible through `make_scene` or inference preparation. Do not let an adapter secretly execute a different optimizer loop or hardcoded architecture. Trace one real step and one exact resume for each dataset.

### B2. Data and exposure

Thermal remains `fixed25_v1`: 150 primary TRAIN/22 exposed DEV, semantic SHA256 `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`. Retain the four existing TRAIN response addenda and their distinct membership/exposure. Do not initialize a development child from formal3902 and call it quarter-data development.

Wind initially remains `wind_shared_fixed24_v1`: 24 TRAIN layouts/72 direction rows, eight DEV layouts/24 direction rows, SHA256 `b223e174d7262682a7e1ac1977cb38f721993e14c0340b2dc0f5fb5f95bfa715`. Preserve all three existing directions per selected layout and disclose the missing TRAIN M18 category. Do not silently resample to make a result look better. A later broader Wind cohort requires a new identity and is not this paired comparison.

A Thermal epoch visits all150 selected cases. A Wind epoch visits all72 selected direction rows once; the layout is the grouping unit for splits and uncertainty summaries. Use compatible but not necessarily identical microbatch/effective-batch sizes: start with Thermal micro8/effective48 and Wind micro4/effective24, reducing only microbatch under memory pressure while preserving the effective update boundary. This gives four Thermal and three Wind ordinary updates per complete epoch under these sizes. Log the actual final partial boundary and never divide by an assumed full batch.

Cross-dataset equality of epochs or updates is not a fairness claim. The matched arms **within each dataset** must have the same scenes, queries, initialization, schedules, objectives, normalizer, sampled roles and exposure. Report case visits, optimizer updates, role queries and actual training time explicitly.

### B3. Checkpoints and validation

Save latest, best-field and declared milestones every100 complete development epochs; keep only those aliases/milestones and any explicit clean stop. Checkpoint optimizer, schedule, RNG and sampler cursor so a resumed continuation is exact under the declared stochastic streams. No25-epoch snapshot flood or duplicate selected aliases.

Compute statistics on all22 Thermal cases and all24 Wind direction rows. Keep detailed maps to the fixed four Thermal representatives and two Wind low/high-M planes, with additional maps only when a predeclared question requires them. Do not evaluate all600/90 or full Wind populations during routine development.

Preserve the field-only selector for comparability, but add a separately named **response-guarded selection** for Thermal: among saved checkpoints satisfying the declared response-development guard, choose the best common field score. The guard uses only the existing four response-development families, not counted final-audit outcomes. Default exploratory guard is no more than10% mean response-RMSE deterioration versus the starting development parent in fluid/surface/material T, with near-zero denominators handled using fixed TRAIN response scales. This is a proposed selection policy, not a physical certificate. If no child qualifies, retain the parent for response use rather than invent a passing alias.

Always report literal1000 and literal2500 when reached, separately from both selectors. Wind has no analogous independent finite-response labels; do not fabricate a matching physical guard. Use its native role fidelity and numerical derivative checks with their distinct meanings.

### B4. Warmup and matched arms

Thermal's fine path begins from the preserved development R-direct2500 parent and the same D-sep2500 flow partner; keep that flow frozen and flow-context disabled. Fine-path loading must reproduce the existing model under all-fine execution. The new cheap path is initialized separately and learns its approximation; inherited histories remain explicit.

Wind uses the single fresh H64/message64 specification above and the fixed TRAIN-only normalization/background choice. Its H32 pilot remains an unequal-history reference, not an initialization silently promoted to H64.

Per dataset, run a common all-fine warmup through500 new epochs, training the fine predictor with native losses and the cheap path with its approximation loss. Review at100. Then branch from exactly the same warmup state into Full-detail and Adaptive-detail and continue to1000 new epochs. Healthy arms should reach this stage even if a100-epoch checkpoint is underfit. Numerical failures, data leakage, unusable application cost or exhausted budget—not failure to beat every mature reference—are reasons to stop early.

The requested maturation ceiling is2500 new epochs per arm, including the shared warmup. Continue both matched arms together when learning is healthy and the resource estimate fits. Wind should normally receive this fuller exposure because the previous pilot covered only27.8 passes. Do not repeatedly restart the same schedule. All shared warmup work is counted once, with an amortized and per-arm-equivalent view if useful.

Use AdamW with the inherited/common weight decay and clipping rules. As a predeclared default, Thermal fine parameters preserve their parent moments, warm from3e-6 to5e-5 during new epochs1–20, hold through1000, then cosine-decay to3e-6 by2500. New cheap/router parameters use their own fresh moments and a3e-4 peak learning rate, with the same declared endpoint horizon. Fresh Wind fine/base parameters use3e-4 through1000 and cosine decay to3e-6 by2500. Initialize the router at its stated activation stage; record its separate age. Keep the Full-detail/Adaptive-detail schedules identical for corresponding parameters, and do not restart the physical-model learning-rate clock at the500 branch. These are proposed defaults to seal after correctness/cost preflight, not retrospective optimizations on DEV.

One bounded, pre-fit four-layout Wind overfit check of at most300 disposable updates can diagnose units, masking or an inability to fit wake structure. It is not a new architecture search or a selection result. Fix a proven correctness defect; do not select hyperparameters by that small set's validation outcomes. If the full-detail model remains unable to learn native wakes after adequate fitting, report that the base is not yet a suitable organizer host.

## 6. Workstream C — learn the refinement organizer during reconstruction

### C1. Exactly two main arms per dataset

**Full-detail:** uses F for every valid source/receiver pair. The cheap path receives the same approximation training as in the other arm but does not replace F in ordinary outputs. This is the capacity-preserving reference.

**Adaptive-detail:** uses the same fine network, cheap network and contextualization, plus a small source/receiver scorer. Fine computation is kept for near pairs and learned selected pairs; other pairs use B. This arm's router and predictor co-adapt through the reconstruction task. It is not merely a frozen-teacher mask fitted after the predictor has stopped learning.

Both arms carry the same useful fine capacity. Do not reduce the full reference's width, increase the adaptive branch's physics budget invisibly, or initialize one from a more mature state. Additional router parameters and replay costs are reported. The old post-fit deletion proposer stays as an archival control, not a third long-running main arm.

### C2. Train all relevant paths without dead gates

Let D=F−B. A cheap-path loss fits B to a detached fine target on the actual native TRAIN queries, using fixed output/message scales. For affine Thermal, include both kernel-residual diagnostics and native endpoint/increment consequences. For nonlinear Wind, message-space fit is auxiliary; native output loss is authoritative because latent scaling can change.

A practical hard-forward/surrogate-router-gradient construction is

\[
V_{hard}=B+g_{hard}D,\qquad
V_{train}=V_{hard}+(p-\operatorname{stopgrad}(p))\operatorname{stopgrad}(D),
\]

where p is a soft gate. It gives exact hard-forward values while allowing omitted interactions to send a restoring signal to the router. Physical fine gradients still follow the actually selected route, except during explicitly included full-detail replay. The full fine terms evaluated to obtain this training signal are real training work; do not advertise sparse training speed from hard-forward masks.

This surrogate must be confined to router training. Deterministic input-gradient evaluation uses the actual fixed hard route or a separately labeled soft model. Do not report a straight-through geometry gradient as the derivative of the discrete deployed map.

Use a common mixture of adaptive/main and all-fine replay objectives, for example 0.75 native main +0.25 native all-fine, reusing the already computed fine messages. For Full-detail both terms are the same fine prediction; their sum retains unit weight. This keeps omitted fine interactions trainable and permits recovery. Count repeated field-head/native extraction work. Do not instantiate a second complete frozen teacher wrapper per batch.

### C3. Router objective and schedule

The router consumes cheap source/context states, relative physical geometry, receiver role and an optional declared fidelity setting. It must not evaluate F to form an inference proposal. Use one small scorer (default hidden32) and one fixed cost regularization coefficient calibrated on a bounded TRAIN-only gradient panel, rather than another K/risk/entropy portfolio.

The loss contains native reconstruction, inherited dataset response terms, cheap approximation loss and a normalized expected fine-work cost. Freeze their scales before the paired branch begins. Start cost pressure at a modest TRAIN-calibrated gradient share, such as5% of the router reconstruction-gradient norm, with a documented cap. If that gradient is numerically unavailable at all-open initialization, first fit the router to detached residual-importance targets for the declared warm period and calibrate once afterward; do not divide by a zero norm or tune on DEV.

Use a short open/soft warm period after the shared500 checkpoint, then a smooth transition to hard-forward task learning. A default schedule is epochs501–600 open-forward/router learning,601–800 soft-to-hard transition,801 onward hard-forward with live router surrogate and full replay. Record actual mode at each saved checkpoint. Do not impose a global K, a histogram target, mandatory source-count variation or a minimum partial-patch percentage.

Evaluate a small predeclared set of gate thresholds/fidelity settings to form a frontier, including exact full detail. Choose any proposed operating point from TRAIN calibration and seal it before DEV review. Changing the threshold trades measured approximation for work; it does not produce an inherited rigorous certificate for nonlinear fields.

### C4. Protected locality and grouping

Keep all original positive-support near interactions detailed initially. Use per-receiver protection rather than marking a whole large patch protected because one point is near a source. Report its computational cost and any limit this places on possible savings.

The cheap source-wise layer covers every active source; environmental/global preparation remains full. Source masks and equal-mask execution blocks are tensor objects, with batching and chunking designed around actual shapes. Do not build a recursive tree or write one CUDA scalar per physical pair from Python.

### C5. Necessary control comparisons, without a large portfolio

Use each adaptive checkpoint's all-fine mode, its normal learned route, its all-base mode, and equal-count nearest/upstream routing where the dataset supplies a meaningful flow direction. Add one degree-matched source-permutation or routing shuffle per fixed representative. Keep the actual number of fine rows/near protections equal for these controls. A permutation of labels together with their physical objects is an invariance check, not an intervention.

Compare the separately trained Full-detail and Adaptive-detail arms at matched epochs. Same-weight interventions test use of a learned route; the independent Full-detail comparison tests what the learning recipe buys over a capable full model. Neither alone proves physical causality or unique global optimality.

No same-weight success can replace field fidelity against native stored targets. Teacher closeness, physical error, actual fine work and end-to-end cost must remain separate columns.

## 7. Efficiency measurements that answer the user's question

Measure the same public scopes for both datasets and both arms. Warm allocator/kernels consistently and rotate model order. Default five repeated measurements, not dozens of repeats per matrix cell; increase only if the ordering is unresolved and material.

| Scope | Include | Keep separate |
|---|---|---|
| First application | Dataset/catalogue setup, scene construction, learned context, selection, fine/base reads, native extraction and CPU output | One-off checkpoint load and framework startup, reported alongside |
| New scene with dataset open | Rebuild scene-dependent geometry/context/selection and complete outputs | Static data catalogue opening |
| Prepared repeated read | Required guards, receiver processing, base/detail reads, extraction and output | Previous paid preparation; show amortized totals for1/10/100 reads |
| Input derivative | Complete forward + physical heat/geometry/query VJP or JVP, actual route semantics | Parameter-training backward and unsupported discrete-switch derivatives |
| Training | All objectives, all-fine replay, coarse fitting, router surrogate and optimizer step | Validation, saves and data preparation; give complete process totals too |

Use low/high actual M and Q1024/Q8192, plus the existing native Wind planes for visual fidelity. Separate full padding, all-active padding cleanup and learned detail selection. Count context pairs, cheap rows, fine rows, near rows, gate rows, valid physical pairs and output-role coverage. Layer-row counts are not FLOPs and cannot be summed across different widths as equivalent work.

The cost model to test is `T_context + T_base + T_router + T_selected_fine + T_native_output + T_guards`, not simply retained-K/full-K. Report peak allocated, peak reserved and process residency separately, together with host preparation time. Never preserve slow validation machinery in a hot loop merely because it was previously written; preserve its guarantees through a cheaper implementation.

A practical exploration target is at least20% fewer **active expensive detail rows** at a usable accuracy point, with no more than25% complete-call overhead versus the same full-detail model. A real latency or memory improvement is preferred. These are proposed decision bands, not universal scientific laws, and failing them should not be described as a numerical-safety failure. Full-detail must itself remain reasonably priced relative to the efficient existing family; do not hide a slow new base behind a favorable adaptive/full ratio.

Do not promise the previous6–9× classic speed factors will persist on Wind or on a richer readout. Measure same-scope end-to-end ratios and retain the older actual-system result with its original history and hardware scope.

## 8. Physical fidelity and organizing evidence

### 8.1 Predictor board

Thermal: all22 equal-case native mean/p90 for fluid T, surface T, material T and module peak; near/downstream diagnostics; q proxy separately; frozen flow roles and heat-null checks. State clearly when a flow quantity is unchanged rather than newly learned. Use the development parent and same-cohort controls; formal3902/Dense1804 remain labeled contextual references.

Wind: all24 direction-row metrics in m/s for volume, hub slab, downstream, near turbine and background, plus separate u/v/w. Show uncertainty/dispersion grouped by layout, not by millions of points. Compare with the previous pilot, TRAIN height profile and retained Wind reference under their different histories, and with the matched new Full-detail arm.

No requirement to beat Dense everywhere. First establish that the full-detail model learns the important field structure. On Wind, a substantial reduction of the pilot's near-turbine/wake deficit is required for a useful host; a merely falling scalar loss or beating the height-only profile again is not enough. A roughly twofold pilot-error reduction is a useful directional target, not a mature-accuracy certificate. If full and adaptive remain an order of magnitude behind the retained reference, say that a credible joint accuracy–efficiency direction has not yet been established.

### 8.2 Responses and sign limits

Use only existing allowed response records. Keep TRAIN-fit, response-development and counted-audit families distinct. Select before final counted replay; previously exposed outcomes remain exposed even if opened late this round. No new solves, synthetic physical labels or fabricated missing0277 baseline.

Measure native finite heating-response errors, mean direction/amplitude, material-peak changes, response-development transfer and exact heat-null flow. Use saved geometry perturbations as a challenge panel, not an implied independent geometry validation. Report negative single-heater coefficient norms and nominal negative contribution mass without clipping. This phase does not prove independent donor-column identification.

Do not add a simultaneous positivity-head search. If sign defects remain a limiting issue, describe a separate dataset-qualified monotone-head experiment as a future choice. Generic message signs in Wind have no analogous positivity requirement. The current conditional Thermal steady-generator argument must not be exported as universal multiphysics physics.

### 8.3 Organizer board

Show the same physical receiver regions under Full-detail, learned refinement and matched geometry controls. Plot every source ID, which sources retain coarse influence, which receive detailed correction, and the resulting physical output difference. Show how gates change between different receivers in one scene and across the existing Wind directions or saved geometry states after rebuilding context.

Report raw approximation difficulty `F−B`, requested accuracy, gate, actual fine work, physical field error and full-model distortion separately. Demonstrate that fine removal/restoration changes the executed computation and is locally useful. Show environmental contextual ancestry explicitly; a packet does not prove that unpictured environmental donors are irrelevant.

Use current coarse/fine output reconstructions to verify the exported graph algebra. For Thermal, source coefficients have temperature/heating units. For Wind, source messages are latent; use actual field changes or local derivatives when making a physical-response plot. Do not compare normalized membership weights with physical coefficients on the same numerical scale.

### 8.4 Direction-worthiness, not perfection

Classify each dataset independently as: **continue this direction**, **useful full model but organizer not yet useful**, or **base predictor not yet adequate**. The overall shared direction earns further investment when both datasets actually train through the same workflow, the fine path has credible field learning, at least one nontrivial refinement frontier is useful, and the interpretation matches executed information paths. The strongest next evidence would be consistent receiver/scene adaptation and a measurable equal-work advantage over geometry controls, without sacrificing critical field/response behavior.

A full-detail outcome everywhere is an admissible measured negative result, not a reason to force K downward. A weak frozen teacher cannot certify a good organizer merely because its predictions are preserved. Conversely, failure to beat every historical role must not erase a genuine adaptive accuracy–work improvement.

## 9. Inverse-facing scope

Do not train a denoiser, launch a generative inverse campaign or claim new physically validated designs in this phase. Provide a small differentiable consumer demonstration using the fixed representative inputs, existing stored heating alternatives and a few operating-point perturbations explicitly labeled model-only.

At fixed Thermal scene and fixed route, compare precise full/adaptive endpoints, directly contracted increments, and physical heat VJPs. Verify source IDs, unit conversion and role extraction. For geometry moves, rebuild scene, gates and exact baseline; compare local AD/FD only away from a gate transition. Report transitions, do not hide them with detached geometry.

For Wind, expose local physical JVP/VJP at a declared scene and reject exact affine finite-increment calls. Do not treat a successful local derivative test as validation of a finite layout move. Normalization and receiver-support feature derivatives must be included through the adapter.

Show one reusable-interface example with six observed and three held native receivers, but do not optimize to those observations or manufacture a design-success statistic. Saved-pool ranking is an optional cheap replay and must retain the fact that existing controls already make the same choices.

## 10. Scope, resources and stop rules

The proposed ceiling is **16 aggregate GPU-associated process-hours and10 elapsed hours**, across the two devices currently authorized by the user, with the final60 minutes reserved for closeout. This is a plan budget, not a requirement to consume it. Verify device UUIDs and current authorization; do not infer permission to use GPU0 from a previous report. Never terminate or interfere with unrelated jobs. If one authorized device is occupied, use the other, progress on CPU preprocessing/tests, or run a documented contended fit when memory permits. Do not wait indefinitely or invent exclusive-GPU timing.

Suggested allocation: up to1 GPU-hour for compatibility/contracts/application preflight, up to5 for the Thermal paired learning, up to7 for Wind paired learning, and3 for organizer/native evaluations, failures and contingency. CPU preprocessing has an initial60-minute target; diagnose repeated catalogue work instead of rerunning it. These are flexible suballocations within the fixed total, not additional budgets.

At startup and each100-epoch review, forecast completion using measured full scheduled objectives plus validation/save cadence. Measure both devices separately under contention. Preserve a clean resumable checkpoint before the ceiling binds. No silent shortening, resetting of epoch counters, unreported failures, or promotion of a startup into a scientific run.

Scientific scope is two main arms per dataset after a shared warmup, one fixed architecture size per dataset, one small router and one cost calibration. One additional bounded correctness remedy is allowed if an actual bug is demonstrated. No additional backbone portfolio, seed hunt, threshold search on DEV, Tree revival, full-data expansion or solver attempt. Numerical controls and same-weight route interventions are not extra training arms.

Prefer reaching2500 new epochs when healthy learning and forecast permit, especially in Wind. Do not extend past2500, start5k or expand the data population automatically. Unused budget is not a reason for unrelated diagnostics. The authoritative physical solver ledger remains **326/326**.

## 11. Development versus formal training

All new development work uses the fixed subsets and TRAIN-only transforms above. Existing full-data3901/3902, classics1401/1502/1804, all parents and histories remain preserved. No new formal optimizer or run directory is created as part of this goal.

For any worthwhile candidate, prepare a manual full-data5000 recipe using the same common runner, an explicitly new data binding, fresh TRAIN-only normalization and declared component histories. Thermal can retain a staged flow/thermal schedule; this is a capability-managed workflow, not a requirement for one monolithic optimizer. Wind has its own output law and full-data profile. The user decides whether to launch either.

Provide explicit prepare, dry-run, start, status, clean stop and exact resume commands that actually exist after implementation. Parser/identity/resume checks must pass. Include a **manual** short full-data startup benchmark command for a measured time estimate; do not execute that training benchmark in this development scope. Never extrapolate the old H32 pilot time as a guaranteed full-H64 5000 runtime.

Formal promotion and direction-worthiness are different decisions. It is acceptable to recommend a promising source-refinement direction while deferring its expensive full-data comparison until the matched base predictor and actual costs are clear.

## 12. Required final report and visualization design

Begin with no more than600 words answering: what changed, what became useful, what failed, and what to retain. Give a direct Predictor/Organizer/Inverse statement and A/B/C assessment, distinguishing model fidelity, physical fidelity, software contracts and measured execution. Follow with one compact scorecard. Detailed metrics belong in the appendix, not in a wall of qualifying prose before the main answer.

Use these six intuitive figure families. At most six panels per page; split a dense page instead of shrinking text. Include PDF masters and small raster companions only for the chosen Markdown embeds. Generate figures from saved arrays, not redrawn imagined fields.

1. **What field do we predict?** Native reference, Full-detail, Adaptive-detail and signed residuals on the same Thermal scene and two Wind scenes. One shared physical scale and one shared residual scale within each comparison. Show source outlines, units, checkpoint ages and unclipped RMSE.
2. **Where is detailed interaction needed?** A fixed physical receiver, every source marked, coarse links faint and detailed links bold; fine/base difference map and the actual prediction change from removing/restoring one selected correction. Include environment ancestry without implying sparse environmental support.
3. **Does the organization adapt?** Three input-selected receivers in one scene and one existing changed-scene/direction pair. Preserve physical source correspondence where the dataset actually establishes it. Display selected-detail count and work, not a latent K label.
4. **What happens when a source or layout changes?** Existing stored response, full/adaptive response and residual, including an unfavorable example. Display heating allocations or changed geometry beside the physical field. No sampler trajectory is invented.
5. **What is the price?** Per-role physical error versus complete milliseconds, memory and expensive active rows. Separate first-use setup, new-scene and repeated-read totals; show all-active padding control separately from learned refinement.
6. **Did the common workflow really train both datasets?** Native learning curves against epochs, optimizer updates, scene/query exposure and measured time. Annotate shared warmup, route-training stages and selected versus terminal checkpoints. Include one small executed pipeline diagram.

Immediately below every figure, write one quantitative sentence and one plain-language conclusion. Report representative and unfavorable cases without giving a hand-picked case the status of a population estimate. Use equal aspect for physical geometry; show display clipping and preserve unclipped metrics. Label Thermal references as analytic-flow/shared-grid benchmark and Wind references as stored OpenFOAM data. No SI units or independent-CFD claims are invented for Thermal.

Report all22/all24 native statistics in machine-readable files and a concise appendix, including per-layout/tail errors and response exposure. Keep source IDs, coefficients, masks, cheap/fine decompositions, data normalization, exact checkpoint ages and measured timing scopes recoverable. Avoid unnecessary duplicate maps or thousands of row images.

Prose paragraphs and figure captions must each occupy one continuous Markdown source line, as required by repository AGENTS.md. Check rendered figures, legends, aspect ratios, all links and report width. Keep generated data, checkpoint files, figures and one-off renderers local and ignored; durable code/tests/profiles/guides/report are the commit scope.

## 13. Acceptance checklist and final decision

### Software and numerical requirements

- Same real training engine and shared interaction functions execute both datasets; dataset adapters do not hide a second training loop.
- All-fine compatibility is measured at retained Thermal weights; new Wind has a meaningful supervised learning curve and native output checks.
- No target values enter scene/preparation/router; no Thermal heating enters affine kernel gates; exact heat-null flow remains.
- Differentiable fixed-route consumers, stale geometry/weights/receivers/normalizers, source/role/unit joins and precise increments are tested. Historical FP32 misses remain separately documented.
- Row-local fallback, tensorized packet compilation, padded/all-active/detail counts and complete-call scopes are explicit.

### Scientific directional requirements

- The new full-detail reference learns important field structure at an acceptable cost; merely matching the previous weak Wind pilot is insufficient.
- Adaptive-detail is judged against matched Full-detail and equal-fine-work controls, not only against a frozen teacher or an unmatched classic.
- There is a nonvacuous measured accuracy–detail-work frontier, or a clearly reported reason it failed. A promising point need not beat every historical role, but serious response or wake errors cannot be hidden by a scalar aggregate.
- Exports agree with actual coarse/fine/global paths. Source identity and computational relevance are distinguished from physical causal attribution.
- The report makes one concrete retain/reject/next-investment decision per dataset and for the common organizer direction. It does not open a new portfolio automatically.

### Delivery

Before the final push, inspect the remote branch and audit the entire outgoing commit range with the repository artifact gate. Preserve local backups/history if unpublished commits need cleaning. Do not overwrite another contributor's changes or rewrite preserved scientific checkpoints. Commit/push durable changes to the non-default working branch and verify tips. Release only owned processes/resources. Final message: elapsed/GPU allocation, exact completed ages, one sentence each for Predictor/Organizer/Inverse, and paths to the report and manual recipes.

## 14. Source basis and limited external context

The inspected report and sources at2af49e7 are the evidence for Section1 and the code review in Section2. Runtime and scientific measurements in this plan are reported results, not independently rerun measurements by the plan author.

- [Measured shared-core/packet report](https://github.com/cosmos2w/ModularDT/blob/2af49e7/HONF_Proj/docs/reports/HONF_Shared_Interaction_Core_and_Receiver_Packets_Report.md).
- [Shared core implementation](https://github.com/cosmos2w/ModularDT/blob/2af49e7/HONF_Proj/src/honf_forward_core/interface_fields/interaction_core.py).
- [Receiver packet implementation](https://github.com/cosmos2w/ModularDT/blob/2af49e7/HONF_Proj/src/honf_forward_core/interface_fields/receiver_packets.py).
- [Wind adapter and physical transform](https://github.com/cosmos2w/ModularDT/blob/2af49e7/HONF_Proj/Case_WindFarm/src/windfarm/shared_interaction.py).
- [Thermal continuation workflow](https://github.com/cosmos2w/ModularDT/blob/2af49e7/HONF_Proj/tools/thermal_source_response_flow_context_fit.py) and [Wind pilot workflow](https://github.com/cosmos2w/ModularDT/blob/2af49e7/HONF_Proj/tools/wind_shared_interaction_pilot.py).

External methodological context is limited to the general usefulness of separating geometry handling, multiscale computation and physical output reads. GINO (NeurIPS2023) combines geometry-aware graph operators with a regular latent-grid operator; RIGNO (NeurIPS2025) uses a multiscale regional graph between input/output point clouds. These are precedents for multilevel organization, not evidence that the proposed HONF refinement mechanism will work or a novelty claim. This plan deliberately retains identified physical sources and an exact all-fine recovery path rather than copying either model.

- Li et al., *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, NeurIPS2023: https://papers.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html
- Mousavi et al., *RIGNO: A Graph-based Framework For Robust And Accurate Operator Learning For PDEs On Arbitrary Domains*, NeurIPS2025: https://proceedings.nips.cc/paper_files/paper/2025/hash/dcb91f43033bb1d367d1848806dee98d-Abstract-Conference.html

**Final instruction:** deliver actual matched learning and an executed organizing mechanism. Do not substitute a larger quantity of diagnostics for progress on the common predictor–organizer family, and do not turn an honest negative mechanism result into a claim that the whole HONF hypothesis is impossible.
