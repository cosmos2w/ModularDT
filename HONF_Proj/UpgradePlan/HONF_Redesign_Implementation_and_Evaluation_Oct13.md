# HONF Redesign: Implementation and First Evaluation

**Companion to:** `HONF_Redesign_Oct13.md` (the mathematical specification, referred to below as **Design**).

**Mode:** Codex Goal mode; implementation, bounded learning on both datasets, and conditional fresh full-TRAIN 1,000-epoch research runs. This is an executable work plan, not a report of completed work.

**Reviewed repository:** `cosmos2w/ModularDT`, branch `agent/honf-core-next`, remote snapshot `7971a7ed57ff889b226f36cd4b5eaed07f96b9cd`. This snapshot follows the Design review at `f10aa863`; the new commit archives closed plans/reports. Resolve current paths from the archive index or `git ls-files`, rather than restoring obsolete report paths. Inspect the local checkout and any later changes before implementing. No local GPU state, installed package version, private checkpoint availability, or runtime performance was independently verified while writing this plan.

**Relationship between the documents:** Design owns the model equations and physical capabilities. This document selects a first implementation, maps reuse boundaries, defines training/evaluation scope, and grants bounded revision room when adopted by the user. It does not replace Design with a different architecture. Proposed file names, family names, defaults, and thresholds below are new implementation decisions, not existing repository APIs.

---

## 0. Goal, scope, and what counts as completion

Build a new, self-contained HONF family that uses standard typed attention to contextualize native module/environment tokens, prepares content-bearing hyperedges once per scene, and decodes fields through that bank plus a bounded local-detail path. Demonstrate real learning on ThermalChannel and WindFarm through the **same core implementation and TrainingEngine**, with separate dataset weights and explicitly different physical output laws.

The first version must answer three practical questions: Does the model learn recognizable, reasonably accurate fields? Does its executed computation have a credible cost/scaling advantage? Does the exported organizer describe information actually used by the decoder? It does **not** have to beat every classic metric, establish independently identified physical hyperedges, or deliver sparse wall-clock acceleration before it can be a useful first result.

Deliver implemented code, focused correctness tests, four primary segmented fits, understandable native-field/organization/cost evidence, and a clear next decision. Allow up to two additional candidate revisions, not an expanding architecture search. At most one selected organized model per dataset may proceed to a fresh full-TRAIN 1,000-epoch research run under Section 13. No 5,000-epoch launch, inverse generator, new physical solve, or modification of a protected historical run is authorized by this plan.

**Do not finish with only an implementation preflight or an audit report if healthy first-version models can still be trained within the declared envelope. Conversely, do not keep adding mechanisms until an arbitrary collection of old promotion gates passes.**

### 0.1 Operating permissions and bounded resources

Use physical **GPU 0 and GPU 1 only**, normally one dataset/job per GPU. Verify their current UUID mapping before launching; logical `cuda:0` inside a one-device process is not physical GPU 0 by definition. Do not touch unrelated jobs or use GPU 2. No DDP or model parallelism is needed for this first comparison: parallel independent fits are the default.

Use the existing ModularDT environment. Check installed Torch/CUDA versions and supported attention/optimizer backends; do not upgrade the shared environment merely to obtain an advertised kernel. Existing process ownership, checkpoint, storage, and upload rules remain binding.

Adopt a planning ceiling of **16 aggregate GPU-associated process-hours and 14 elapsed hours from the first GPU experiment**, including failed/profiling runs and both conditional full-data follow-ups. Coding/CPU preparation time is reported separately. These are ceilings, not spending targets or an ETA. Reserve approximately the last two elapsed hours for endpoint evaluation and closeout; allow a full-data launch only when its measured forecast, evaluation, and reserve fit. Count overlapping process lifetimes separately for allocation, but do not call them GPU utilization. Log actual epochs, updates, visits, query work, and train-loop time as well.

A foreign process occupying an authorized GPU is not permission to terminate it or wait indefinitely. Use the other authorized GPU, safe co-residency when measured headroom permits, or CPU development work. Contended training remains valid learning evidence; comparative timings are marked contended or postponed. Do not silently alter effective batch, supervision, precision, or schedule to fit an interruption. At a resource boundary, save an epoch/optimizer-safe state and report the actual stop.

## 1. The first implementation: one family and one useful control

Use these provisional names consistently in configs, reports, and checkpoints:

| Name | Contextualization | Field decoder | Purpose |
|---|---|---|---|
| **A-D** | New typed measure-aware attention | Source-resolved direct readout | Competent no-compression control; reveals whether attention/input/learning is already weak |
| **A-H6** | Identical attention design | Six content-bearing edges, four Thermal response channels per edge, local detail | Main HONF redesign |

A-D and A-H6 train independently from fresh weights. Match initial tensors for identically named/shaped shared encoders/contextualizer where possible; initialize private modules from separate reproducible namespaces so construction order cannot alter shared initialization or query streams. They need not have identical parameter counts. Report real counts and widths; do not distort either architecture just to manufacture a parameter match.

There are **four primary scientific runs**, two per dataset. Historical P/R1, Wind2202, Thermal1404/1804, and Wind2102/2103 are read-only context. Do not retrain every historical family. The new A-D/A-H6 comparison isolates the decoder design under a common new contextualizer and recipe; comparison with historical P is not a clean causal estimate of the attention substitution if its training or loss differs.

The default A-H6 implementation follows Design Sections 5–10 and 15:

| Parameter | Initial choice |
|---|---|
| Token/edge width | 128 |
| Attention heads / context depth | 4 heads / 2 simultaneous typed blocks |
| Typed routes | MM, ME, EM; no EE stack or edge–edge Transformer in v1 |
| Token FFN | Shared-by-type standard pre-LN residual FFN, expansion 4; SiLU or GELU, fixed for both arms |
| Original-token re-injection | Not active initially; ordinary residual paths remain |
| Hyperedges | H=6, soft overlapping typed incidence, distinct reproducible reference-domain seeds |
| Edge reader | Shared nonlinear reader, internal width 64; edge/query affine projections cached separately |
| Thermal response channels | r=4 per edge; global capacity Hr=24, not six scalar response modes |
| Local reader | One shared width-64 reader, smooth compact support; actual selected local pairs only |
| Edge access | Full softmax across all six edges during primary training and inference |
| Precision baseline | FP32 weights and a qualified FP32 reference path; optional optimized settings only under Section 7 |
| Dropout | 0 initially, including SDPA's explicit `dropout_p` |

Retain Thermal's 24×8 geometry-only environment catalogue. Use a **4×4×4 geometry-only Wind catalogue, E64**, supported by the existing Wind provider, for both new arms. This is a deliberate new-family input choice to give the attention contextualizer an interior spatial representation, not evidence that E8 caused the old error. Preserve the original seven Wind geometry features, measures and coordinate transforms. Do not add target flow. Historical E8/E512 models retain their native catalogues in comparisons. E is not an extra hyperparameter sweep.

Use full smooth edge access in the research candidate. The optional sparse diagnostic later must not be turned into a sparsity curriculum that takes over this first version.

## 2. Repository reuse: keep the infrastructure, replace the computation

The inspected code provides a viable integration boundary, but several existing classes combine reusable physics with historical architecture restrictions. Reuse functions/contracts by composition or a narrow extraction; do not instantiate a legacy model to access a helper.

All paths below are relative to `HONF_Proj/`, except root `AGENTS.md`.

| Existing location | Reuse | Do not carry into the new family |
|---|---|---|
| `src/honf_runtime/unified_training.py` | `TrainingEngine`, `TaskProvider`, `TaskBatch`, `LossTerm`, `SamplingKey`, `EngineConfig`, `ScheduleSpec`, `SelectionPolicy`, named optimizer state, run lock, epoch-boundary resume | Historical refinement stages or experiment-specific branch amendments; use `training_mode="joint"` and the new family identity |
| `src/honf_runtime/run_layout.py`, `checkpoints.py`, `reproducibility.py`, `compat.py`, configuration/path utilities | Existing output layouts, safe loading, source/version information, tensor-device handling | A new competing checkpoint, job, or provenance framework |
| `src/honf_forward_core/interface_fields/interaction_core.py` | Scene/dependency vocabulary, explicit identities, masks/measures, ownership principles; small independent utilities where appropriate | `InteractionContextCore`'s pair-value MM/ME/EM implementation and accumulated routing-generation branches |
| `.../interaction_preserving_joint.py` | Reference formulas for the direct nonlinear/affine readouts and current interface behavior | Inheritance from the full P/G/H/R dispatch tree; dormant collective networks or default legacy fallbacks |
| `Case_ThermalChannel/src/channelthermal/data/datasets.py`, `data/development_split.py` | HDF5 access, fixed membership, TRAIN-only transforms, native targets and geometry | Candidate input preparation through a classic checkpoint's normalized-data assumptions |
| `Case_ThermalChannel/src/channelthermal/source_response.py` | Heat-independent `context_tensors` logic, source IDs, native interpolation, interface/material coordinates, conductivity/proxy formulas | `SourceResponseThermalModel` and its frozen D-sep composition; ordinary application via automatic dense K export |
| `Case_ThermalChannel/src/channelthermal/flow_curl.py` | `build_native_curl_read_stencil`, `physical_uvp_native_curl`, maintained solid/boundary conventions | Replacing native curl with a fresh unrelated omega head, or imposing an unsupported incompressible target |
| `Case_ThermalChannel/src/channelthermal/training/joint_task.py`, `training/unified_task.py` | Target-separated data preparation, losses/reducers, original TRAIN response records, role extraction and sampler | Pretending a new mode is P to get past family checks; forcing all-source operator-kernel exports into ordinary training |
| `Case_ThermalChannel/src/channelthermal/source_response_residual.py` | Keep intact as future optional, generator-qualified training machinery | Executing a generator, stored velocity field, or operator residual inside inference |
| `Case_WindFarm/src/windfarm/data.py`, `io.py`, `geometry.py`, `splits.py`, `normalization.py` | Read-only native arrays, source/receiver frame, split grouping, TRAIN height profile and residual transforms | A second rotation, target-derived inference features, or a new formula inferred from an old plot |
| `Case_WindFarm/src/windfarm/workflows/joint_forward.py`, `workflows/native_role_cache.py` | Native role sampler, shared geometry catalogue cache, bounded cache ownership | Per-arm copies of expensive role catalogues or caching learned states across optimizer updates |
| Both `training/joint_task.py` providers | Model-neutral sampling/loss machinery via narrow common helpers | Copying entire providers, including all old family guards, into new files |
| `tools/joint_regional_train.py` | Read its CLI/identity conventions and schedule integration | Copying its large experiment-specific launcher or widening sealed legacy mode schemas |
| Thermal `evaluation/loading.py` and checkpoint-native tests | Correct classic raw/local versus normalized/global input assembly for read-only references | Feeding classic models the new candidate's transforms or using quarantined old arrays |

**Concrete reuse warning:** `ThermalSourceResponse.apply_native(..., accumulation_dtype=...)` currently calls `export_native_kernels`, which materializes all source columns. The new factorized response must implement precise application as factor contraction followed by native extraction; keep full K export explicit and optional. Similarly, the current `prepare_native` performs per-case stencil union construction but also invokes its specific core receiver preparation. Extract the former without dragging in the latter. [S3–S8]

### 2.1 Proposed new code organization

Create a small opt-in package; equivalent names are acceptable if the responsibilities stay separate:

```text
src/honf_forward_core/content_hypergraph/
    contracts.py          # prepared-state/data containers; capability validation
    attention.py          # typed SDPA blocks and reference attention
    organization.py       # typed incidence, edge content, source coefficients
    decoding.py           # full-edge/local reads; factor application; direct control
    model.py              # one core, A-D/A-H6 mode, prepare/read/application APIs

Case_ThermalChannel/src/channelthermal/
    content_hypergraph.py
    native_observations.py                 # narrow extraction, only if needed
    training/content_hypergraph_task.py

Case_WindFarm/src/windfarm/
    content_hypergraph.py
    training/content_hypergraph_task.py

tools/content_hypergraph_train.py
src/config_core/forward/content_hypergraph/  # explicit development/full1000 recipes
```

Do not interpret the tree as a demand for a fixed file count. Consolidate trivial files; split substantial code by responsibility. The shared package must import neither `channelthermal` nor `windfarm`. Dataset adapters may import the shared package and their own native helpers. Add family registration through the maintained loader/factory, with strict new-family checkpoint validation. Do not monkeypatch existing mode lists or weaken old resume checks.

## 3. Implement the dataflow, not just the model facade

### 3.1 Public objects and lifecycle

Implement the four objects in Design Section 13 using the existing ownership infrastructure wherever it fits:

```python
scene = adapter.make_scene(prescribed_inputs)       # no labels; no Thermal heat
organization = core.prepare(scene)                 # live once per training forward
query = adapter.prepare_queries(scene, request)    # native coordinate/stencil plan
outputs = adapter.predict_prepared(
    organization, query, controls=physical_controls
)
```

Provide equivalent explicit methods for `apply_increment`, heat JVP/VJP, `export_source_kernels`, and opt-in trace export. These names are proposed API semantics; integrate with existing native-facing methods where that avoids needless call-site changes.

`PreparedOrganization` owns contextualized M/E tokens, six edge states, typed incidence and actual pooling weights, spatial anchors/scales, projected edge keys/content, and Thermal C when applicable. It is independent of receiver batches and Thermal heating. `PreparedQuery` owns receiver identity/frame, native gather/scatter maps, selected local pairs, and optionally reusable spatial factors. No model parameter appears on a second separately frozen predictor.

During ordinary training, prefer a fused decode-and-apply path that returns physical fields and small metadata, not a permanently retained Q×H×width diagnostic tensor. Query factors may be retained deliberately for repeated inference controls; they are not required for every learning batch.

Keep mandatory validation at the public preparation/application boundary. Within an owned request, tensor-only chunk reads reuse that validation. Reject geometry, model-weight, source-ID, normalizer, frame, receiver, or policy changes. Do not hash the entire parameter collection, inspect CUDA tensors with `.item()`, or copy geometry to CPU separately for every inner query tile. Do not remove stale-state checks to achieve a benchmark.

### 3.2 Tensor axes to bind explicitly

| Object | Shape or meaning |
|---|---|
| Native tokens | M: `[B,M,d]`; E: `[B,E,d]`, with independent raw input widths |
| Typed context attention | `[B,heads,N_dst,N_src]` scores; donor values `[B,heads,N_src,d_head]` |
| Incidence | `B_M[B,M,H]`, `B_E[B,E,H]`; normalized over H per valid native token |
| Pooling weights | `W_M[B,H,M]`, `W_E[B,H,E]`; normalized over the corresponding donor measure |
| Edge bank | `[B,H,d]`; anchors `[B,H,D]`; positive spatial scales |
| Thermal coefficient bank | `C[B,H,r,M]`; physical source columns, zero on padded/absent modules |
| Query routing | `alpha[B,Q,H]`; no dependence on other requested receivers or actual heat |
| Thermal spatial modes | `Phi[B,Q,H,r]`, only retained when requested |
| Local reads | Ragged/packed `(batch,receiver,source)` IDs; evaluate exactly those admitted local pairs |
| Native output | Small channel tensors plus adapter-owned role extraction; full K `[B,Q,M]` only on explicit request |

Use clear names for hyperedge count, attention heads, response rank, message width, and physical module count. Never overload `H` to mean all five.

### 3.3 Attention and edge construction

Implement Design Equations 7–20 directly. Project donor K/V once per type/route per block. Use all pre-update M/E states for simultaneous MM/ME/EM updates. Retain individual native states through both blocks. Mask inactive donors and MM self-pairs; handle M=1/empty environment types without NaNs or invented physical donors.

Use log donor measure in attention logits exactly once, with type-specific positive units/reference. For edge pooling, expose incidence B and actual W separately; do not multiply quadrature twice. Preserve raw edge mass as a nondimensional feature. Do not inject environment-record count as physics. A node's physical ID is correspondence metadata, not a learnable ordinal embedding.

Start with distinct deterministic scene-frame edge seed positions, a weak smooth geometry bias, and trainable content scores as described in Design. Do not use an upstream-only hard mask. Edge construction is query-independent. No unrestricted learned whole-scene vector feeds a large parallel final coordinate MLP in A-H6; prescribed operating conditions are allowed. Any extra learned background token must count within the six-edge budget.

### 3.4 The decoder and heat application

Use the shared edge-conditioned reader from Design Equations 25–30. Define its split affine projections directly in the new network; there is no requirement to recover an old checkpoint's arithmetic ordering. Share its geometry-conditioned features across fresh flow and temperature output projections when their native query nodes coincide.

For Thermal, implement:

\[
a_{gr}=\sum_i C_{gri}\,h_i,\qquad
T_q=\sum_{g,r}\Phi_{qgr}a_{gr}
+\sum_{i\in\mathcal N(q)}\chi_{qi}k^{\mathrm{loc}}_{qi}h_i.
\]

No nonlinear normalization, activation, bias, or heat-dependent routing is allowed after the separately applicable heat enters this contraction. Use the zero physical offset required by the present Thermal capability. Source measures used for attention are not automatically a physical heat-volume multiplier. Normalization must preserve `T(h=0)=0` in physical output units, even if the loss is centered/scaled.

For ordinary application and heat increments, contract `C @ h` or `C @ delta_h` first. For JVP/VJP, use the factors and local pair lists; do not silently export a dense K or execute one forward per source. Precise accumulation casts the retained factors/controls and native reductions to FP64 where requested; it does not claim FP64 neural inference when the factors were computed in FP32.

For A-D, use a clean direct-source readout with the same new attention states and physical laws. Match the local branch architecture, neighborhood rule, input descriptors and common initial tensors between A-D and A-H6; the intended difference is the global decoder, not unequal near-body protection. Reuse tested readout formulas or extract small modules; do not create a dormant old `InteractionContextCore` just to borrow its methods. A-D is the explicit control that retains dense final source work, not a hidden fallback in A-H6.

### 3.5 Local detail without a second dense global model

Choose a fixed smooth compact-support window before outcomes: initially distance from module center less than **3 radii in Thermal**, and less than **1.5 rotor diameters in Wind**. These are model-design neighborhood sizes, not physical influence cutoffs. Use Design's endpoint-smooth support weight. All nonlocal influence remains available through the edge bank.

Build local pair IDs before the local neural reader. Keep every eligible pair, including overlaps; never silently truncate a near-interface neighbor to satisfy a small K. Use a degree bucket or a packed gather rather than a Python loop over receivers. Record active/padded local counts and the chosen implementation. At M≤30, a cheap tiled distance matrix is acceptable; a wide value tensor for every source is not.

The local branch has separate flow-feature and Thermal-coefficient outputs from a shared small reader. Keep physical region features continuous or explicitly piecewise as dictated by the adapter. Do not create independent surface/material/fluid temperature heads or an all-source bypass to rescue a poor edge model.

## 4. Dataset adapters: retain the hard-won physical lessons

### 4.1 ThermalChannel

Reuse the actual target-free feature inventory and source/control units from `source_response.py`. Its existing context builder coerces some arrays to `.float()`; the new tensor path must support explicit test precision without unintentionally severing differentiable coordinates. Treat a pure helper extraction as a compatibility-sensitive change and leave the old public behavior unchanged.

Build one per-case **native receiver union** covering primary flow queries, required curl neighbors, surface/outside interpolation, and material interpolation. Decode only the needed flow/temperature channels at each portion of the union; share prepared context and overlapping query work. Do not run preparation once for flow and again for each thermal role. Retain all interpolation neighbors when their weights have live derivatives, including zero-weight neighbors required at a cell boundary. Deduplication must not merge equal coordinates that have different derivative ownership; preserve each observation's interpolation weights and pullback.

Derive omega from physically transformed u/v using the maintained native stencil, including its boundary and solid-cell conventions. Keep surface/outside/material coordinates in their native module-attached frame. The normal-flux proxy remains the original difference/conductivity formula; do not substitute continuum AD flux. Compute module maxima from reconstructed endpoint material temperatures. Report historical initial ports as not applicable, rather than fabricating a P0/P1 process.

Moving a module requires a new geometric state. Stored nominal masks are valid only for their bound nominal geometry; regenerate target-free masks/stencils under the current adapter rule after movement. No case-ID cache may return a nominal mask for changed coordinates. Keep the piecewise-smooth nature of masking and interpolation visible in derivative tests.

### 4.2 WindFarm

Retain the native read-only field source, literal turbine IDs, rotor-diameter coordinates, prescribed direction handling, source height/radius, support bounds, TRAIN background profile, and native velocity residual transform. Resolve the frame from the dataset itself; do not rotate twice. Environment records must contain prescribed geometry only. Changing E8 to the proposed E64 catalogue does not change the native target/sampling support.

Use the maintained five-role sampler and TRAIN-only component-balanced scale rules. Q4096 means **4096 total queries per direction row**, not 4096 per role. Three directions of one layout remain one correlated layout group. Preserve 3-D source/receiver geometry even when plotting a 2-D hub plane. Wind output is nonlinear; expose ordinary input VJPs/local derivatives, not exact affine finite-control application.

Geometry preprocessing can dominate cold costs. Reuse the existing bounded native role catalogue keyed by source geometry, direction, role recipe and source-data identity. Separate fixed role masks from current network content. Do not create an E64 catalogue by reading CFD target velocity at those locations.

### 4.3 One workflow does not mean one physical law

Both adapters instantiate the same core class and implement the same TaskProvider protocol. They train separate weights. Thermal builds one fresh joint model with no frozen D-sep or Stage-A partner; Wind has its own nonlinear head and profile transform. Common preparation, attention, edge formation, decoder, optimizer engine, and checkpoint lifecycle must actually execute in both datasets; identical method names around two unrelated networks are insufficient.

## 5. One simple, explicit learning recipe

### 5.1 Data and sampling

Keep the existing fixed development memberships: **Thermal fixed25_v1 TRAIN150/DEV22**, and **Wind wind_shared_fixed24_v1 TRAIN72 direction rows from 24 layouts / DEV24 rows from eight layouts**. Preserve all available strata in those manifests. Do not resample validation to repair an inconvenient case or silently add the missing Wind TRAIN M18 category. These are repeatedly exposed development sets, not independent tests. [S1, S6, S7]

Fit candidate normalization, Wind profile, and loss scales from the permitted TRAIN inventory only. Both new arms in a dataset use the same serialized transforms. The existing four original-TRAIN response families are separately disclosed auxiliary exposure, not additional independent test cases. Do not load held-response outcomes for scale fitting. Keep Wind TEST and unexposed Thermal targets locked.

Use `SamplingKey.CASE_EPOCH_VERSION` / `case_epoch_v1`, with dataset, seed, epoch, physical case, role and stream identifiers. Query draws and case order must not depend on arm, microbatch index, graph construction RNG, or model width. Preserve native query IDs, masks and loss denominators when changing packing. Hash streams compactly; do not store every training coordinate array.

### 5.2 Losses: retain physical balance without retaining every historical objective

The following is an explicit **new-family recipe**. It intentionally simplifies the old P objectives; therefore old saved P scores remain contextual references rather than matched loss controls.

For Thermal, use equal-case normalized losses with half the core weight on flow and half on temperatures:

\[
\mathcal L_T=\tfrac18\sum_{c\in\{u,v,p,\omega\}}\mathcal L_c
+\tfrac16\sum_{a\in\{T_f,T_s,T_m\}}\mathcal L_a
+0.05\mathcal L_{q\text{-proxy}}.
\]

Each term uses its own TRAIN-owned physical scale and exact native valid support. Surface/material aggregation first respects the per-case/per-module convention, rather than allowing high-M cases to dominate by element count. Do not add units directly. Report each raw physical error separately from the normalized optimization objective.

Retain a **small supervised heat-response anchor**, using the existing original-TRAIN families only: once per epoch, on the final optimizer macrobatch, add coefficient **0.1** times the equally weighted normalized fluid/surface/material increment loss. Cycle families deterministically; use saved TRAIN response RMS scales or recompute them solely from those TRAIN records. This coefficient and cadence are v1 design choices, not a recovered optimum. Match them between A-D and A-H6. Do not repeat the callback once per microbatch or multiply it by the primary case count. Log its standalone magnitude and gradient norm on a small training probe, not at every step.

Set the old **all-source discrete operator-residual loss off in v1**. Its implementation remains preserved, but adapting a dense K-column residual can erase the new cost advantage before the ordinary representation is tested. The response anchor preserves some slope supervision without making first delivery contingent on port/solver machinery. Enabling an operator residual later is a separate recipe revision with its own work accounting, not a silent reuse default.

For Wind, keep the established five-role, component-balanced residual objective and TRAIN-fitted floors. Use the same formula and scales in training, monitoring and checkpoint selection. Return component RMSE and vector RMSE separately: vector RMSE is `sqrt(mean(sum(component_error**2)))`, not divided by three. No new PDE loss, teacher field, sparsity penalty, entropy target, or multi-task balancing algorithm enters the first fit.

### 5.3 Optimizer, queries, and genuine maturation

| Setting | Thermal | Wind |
|---|---|---|
| Seed | 0 | 42 |
| Primary Q/case or direction row | 1024 fluid plus native auxiliaries | 4096 total over five roles |
| Effective batch | 48 cases | 24 direction rows |
| Initial microbatch probe | 8 → 16 → 24/48 if safe | 8 → 12 → 24 if safe |
| Context | E192, d128 | E64, d128 |
| Optimizer | AdamW, betas (0.9,0.999), eps 1e-8 | Same |
| Weight decay | 1e-4, bias/norm exclusion explicit | 1e-5, same exclusion rule |
| Gradient clip | 1.0 after the full accumulated update | Same |
| Development LR | 20-epoch warmup 3e-5→3e-4; hold to 1000; cosine to 3e-6 at 2500 | Same |
| Save/latest/best/plots | Every 100 epochs plus a declared non-cadence stop | Same |

Bias/norm exclusion is part of the new resolved recipe; do not relabel it as unchanged legacy decay. All active new model tensors train from the first update. Dataset-specific output projections are not frozen to isolate the organizer.

A development epoch visits every selected item. The last partial effective batch is retained with correct denominators. Review at 100, 500 and 1000; healthy primary pairs may continue to **2500** under this plan without another user request. Do not declare the architecture inadequate because it does not match an e5000 full-data classic at e100. Also do not extend to 2500 merely to fill time when an implementation error or unmistakable collapse is present.

Normal monitoring and selectors run only at the saved 100-epoch cadence. Per-epoch training scalar logs are cheap and retained. Full native development evaluation is needed at e1000 and at the final compared endpoint; e100/e500 can use the fixed light monitor plus the small representative panel. Reuse evaluations when selected and endpoint checkpoints coincide.

## 6. First-version correctness: a compact test suite, not another research campaign

Test the new model against its declared equations, not against random-weight equality with P or a requirement to reproduce classic predictions. Protect historical APIs with targeted regression tests only where shared code is changed.

Complete these **eight test families**, with CPU synthetic tests where possible and a small low/high-M native GPU confirmation. Parametrization is appropriate; hundreds of physical trials are not required.

| Test family | Essential checks |
|---|---|
| **1. Attention and measure laws** | Explicit reference softmax versus SDPA; correct mask polarity; MM M=1; empty E; source padding/permutation; environment-record split into identical positive-measure copies |
| **2. New-model graph connectivity** | Finite nonzero task gradients and actual optimizer changes in encoder/context, incidence/content, edge reader, local reader where eligible, and applicable heads; every trainable parameter belongs to an optimizer group |
| **3. Query consistency** | Same receiver alone/in shuffled subsets/in several chunk sizes gives the same field to declared precision; no query-set normalization or query self-attention |
| **4. Thermal laws and axes** | Zero physical T at zero h; exact computational heat independence of flow; affine heat JVP/increment identities; factor contraction versus explicitly formed K on a small fixture; correct C source permutation and Hr axis; no forced rank-six reshape |
| **5. Native extraction** | Same grid field yields the maintained surface/outside/material/q/curl outputs; physical conversion occurs before curl; peak differences use endpoint maxima; active module/source IDs and empty/padded roles are correct |
| **6. Geometry and reuse** | Live source/receiver-coordinate VJPs on one smooth interior fixture; two finite-difference steps; material-attached receiver motion; local-boundary window continuity; mutated geometry/IDs/weights/normalizer/policy rejected; rebuild works |
| **7. Packing, serialization and resume** | Equivalent effective-batch loss denominators and gradients for two microbatch packings; self-contained save/load; one short interrupted/resumed TRAIN sequence reproduces the same data/schedule/optimizer state |
| **8. Operation ownership** | One contextualization/edge preparation per scene per training forward; no Q×M-wide far-value path in A-H6; no dense K export in ordinary/precise application; local neural rows equal admitted pairs; diagnostics off by default |

Choose the new-family numeric assertions **before** measuring candidate differences. Suggested starting reference tolerances are FP64 `atol=1e-9, rtol=1e-7`, and FP32 primary fields `atol=3e-6 × TRAIN_scale, rtol=3e-5`; evaluate gradient comparisons in their own scaled spaces, initially `atol=1e-5, rtol=1e-4`. These are new software precision budgets, not physical-error budgets or replacements for existing tests. Report absolute and relative norms and worst components. Derived curl/q tolerances must account explicitly for the native difference operator and its spacing/conductivity amplification; do not copy a temperature tolerance onto a flux proxy or tune a tolerance to a failing result.

If optimized execution fails a precision comparison, retain the FP32 reference execution while recording the optimized miss. If the reference equations, heat dependency, units, source identities, stale-state rejection, or main gradients fail, repair before scientific fitting. Small backend roundoff is not a reason to rewrite the architecture or spend a day forcing bitwise equality across all GPU batch shapes. A genuine FP64 algebra/derivative failure is not dismissed as harmless roundoff.

A short TRAIN-only learnability probe uses two cases/layouts per dataset with fixed observations for at most 100–200 updates, discarded before scientific initialization. It checks that both flow and Thermal temperature losses decrease and that edge content receives gradients. It is not a new validation run or a hyperparameter competition. Use at most 800 such disposable optimizer updates across the whole campaign, including packing checks.

## 7. Efficiency engineering to do before long fitting

### 7.1 Standard kernels, with their semantics intact

Use `torch.nn.functional.scaled_dot_product_attention` for typed context attention. Assemble geometry bias and log measure as a correctly broadcast additive mask; masked donors remain truly ineligible. SDPA boolean `True` means allowed, unlike some key-padding-mask APIs. Set `is_causal=False`. Set `dropout_p=0.0` explicitly. Do not remove geometry or measure bias to obtain a preferred fused backend. Keep a small explicit attention reference for diagnostics and unsupported numerical cases. Backend eligibility depends on installed software, dtype, masks and shapes; record what actually ran rather than calling every SDPA invocation FlashAttention. [D1]

First optimize regular batched operations, cached projections, receiver-union reuse, and removal of host synchronization. Then allow **one bounded compile experiment** on tensor-only contextualizer/decoder functions and **one bounded selective BF16 experiment** per dataset. They are performance probes at retained disposable weights, not additional model families. Do not compile file IO, Python receipt construction, model ownership guards or dynamic HDF5 loading. Record graph breaks and compilation startup; discard compilation as a default when repeated recompilation consumes its expected benefit. [D2]

Selective BF16 may cover qualified attention/FFN blocks. Keep physical geometry/stencil construction, normalized loss reductions, Thermal response contraction and difference-based q/curl in FP32 or the declared precise dtype. Casting a BF16 result back to FP32 does not recover lost information: compare actual per-field outputs and gradients. Use FP32 masters; if gradient scaling is enabled, unscale only after accumulation and before clipping. Keep one common production precision policy within each A-D/A-H6 comparison where feasible, and report any unavoidable difference. [D3]

Do not make AMP, compilation, a custom CUDA kernel, or linear attention a prerequisite for finishing v1. No Torch upgrade, Triton kernel project, quantization, or distributed-training rewrite is part of this round.

### 7.2 Avoid recomputation and hidden activation growth

Cache **raw geometry** and native role catalogues when their identity is unchanged. Rebuild **learned context** after every parameter update. Within one training forward, keep shared preparation live and reuse it across query/role chunks. Do not detach learned tokens merely to reduce memory. A prepared graph from the previous optimizer step is invalid even for the same case.

A Python loop that appends all chunk activations and concatenates them can still retain memory proportional to the complete unchunked backward graph. Measure training peak memory, not just a no-grad decode. Stream inference outputs. For larger training Q, use standard **non-reentrant activation checkpointing on decoder tiles**, passing shared live prepared tensors into each tile. This may recompute tile readers during backward; it must not recompute scene attention or edge construction per tile. Leave cheap shared preparation outside checkpointed tiles. Measure the extra work and validate gradient behavior. [D4]

Do not retain full attention maps, `[B,Q,H,width]` messages, dense source K, CPU trace copies, or repeated role outputs in ordinary training state. A raw geometry cache may preserve indices; coordinate-dependent weights/features must be recomputed live for geometry derivatives. Use `zero_grad(set_to_none=True)` and ordinary supported optimizer kernels when available. Avoid `empty_cache()` in the training loop and repeated whole-model scans inside the inner reader.

### 7.3 Use more GPU memory for useful work

Hold effective batch, queries and loss denominators fixed while testing up to three microbatch packings. Choose the largest **measured useful throughput**, not the largest allocation. Maintain headroom of at least 20% of physical capacity or 4 GiB, whichever is larger, with additional allowance for foreign residency. Do not reserve that headroom speculatively or infer free memory from model parameters alone.

Begin at Q1024 Thermal and Q4096 Wind. Extra Q or width is a representation/supervision change, not a packing optimization. It may be used only as the bounded scientific revision in Section 10, with changed exposure explicitly recorded. Do not silently give one arm more labels because it uses less VRAM.

### 7.4 A small but honest cost matrix

Use low/high-M input-selected TRAIN fixtures for each dataset. Benchmark A-D, A-H6 and one useful current P/mature reference where its compatible native load is available. Repeat new-family and reference measurements on the **same device**, not Thermal GPU0 versus Wind GPU1 as if they were a paired ranking.

Measure four scopes: prepared tensor decode, fresh-scene complete GPU-native forward, complete application including native extraction and CPU returns, and a real effective-batch training update including backward/optimizer and scheduled auxiliary work. Cold filesystem/catalogue construction is a separate startup number; warmed case preparation still belongs to fresh-scene inference. Do not sum overlapping medians.

Use two warmups and five alternating timed repeats for the main native inference matrix. For learning throughput use at least five complete warm epochs, plus the actual campaign log. Save parameter/optimizer bytes, peak allocated and reserved memory, baseline residency, and complete optimizer-boundary peaks. One short representative profiler trace is enough to explain a surprising result.

For query-density scaling, keep **B fixed** and measure Q=1024/4096/8192 at the high-M fixture. Include a Q32768 no-grad core stress point and, where safe, a real B8 backward stress point. Wind can use actual native coordinates; Thermal's larger-Q point is a target-free coordinate-density/scalability probe, not a new physical-grid accuracy test. At least one B8/Q8192 full training-forward/backward test is required for each new family/dataset unless it OOMs; report OOM honestly rather than shrinking B and claiming success. Lower-B fallbacks are separate evidence.

Publish requested Q, unique neural receiver union, executed padded/active edge and local rows, and bytes of output/factor buffers. Native Thermal Q can saturate the original 8192 grid even when many material requests are made; explain that distinction. Fixed-H decoding should not create Q² attention or hidden wide Q×M/Q×E far-message tensors. Real memory may grow linearly in Q; the aim is bounded, explainable scaling, not an impossible constant-memory training claim.

## 8. The staged Goal-mode sequence

### Stage 0 — Inspect, isolate, and freeze the new scope

Read Design and this plan, root `AGENTS.md`, the current fixed-data protocol, the final R1/source-read report and its Thermal classic reconciliation addendum. Resolve archived paths with the maintained index. Inspect only the relevant live core, adapters, providers, and engine first; this round is not a mandatory reread of every historical report.

Record the actual local/remote source state, active process ownership, authorized device mapping, current disk headroom and shared-cache roots. Use a separate worktree/source snapshot when other tasks are running so new edits cannot hot-change their imported helpers. Preserve all historical checkpoints/normalizers and the source locks. Declare a round-owned disk budget before creating caches or checkpoint directories.

Produce one compact resolved design/recipe JSON and a short implementation map. Bind the Design file's actual hash, the new core specification, dataset manifests, source/role units, selected losses, optimizer, schedule, precision and intended experiment names. A single maintained manifest is sufficient; do not invent another approval/receipt framework.

### Stage 1 — Implement the new core and adapters

Implement A-D and A-H6 through the new shared package, then integrate the native TaskProviders with TrainingEngine. Finish the explicit factorized Thermal application and the nonlinear Wind conversion before adding optional sparse access. Verify no frozen learned partner and no dormant historical model are constructed.

Retain all affected historical API behavior and serialization. If a shared helper extraction is necessary, prove its old caller still computes the same function under the old numerical contract. New-family mathematical differences are intentional and do not require old-function parity.

### Stage 2 — Correctness, learnability, and packing

Run Section 6's compact suite and TRAIN learnability probes. Measure Section 7's packing and one representative complete cost. Fix an all-masked attention error, normalization leak, wrong native support, repeated preparation, or dead gradient here—not after a 2500-epoch run.

Freeze the initial scientific settings and fresh initial states after disposable probes. Record realistic per-epoch and full-run forecasts from measured work; do not multiply a tiny no-auxiliary smoke time by the number of epochs.

### Stage 3 — Four primary development fits

Run A-D and A-H6 for each dataset to e100. Review learning health, physical monitor values, connectivity and resource forecasts. Continue healthy fits to 500, then 1000. Run the all-DEV native comparison at 1000 and decide whether maturation or the single permitted targeted revision is more informative.

The intended mature endpoint is 2500 for healthy main comparisons. Complete the same 2500 schedule rather than replacing it at each review. Models may have different selected best ages, but the primary architecture comparison uses a common completed literal age; the selected-state board is separate. Preserve every saved 100 milestone and latest/best aliases under repository retention rules.

Do not use a poor response audit as an automatic veto on learning to reconstruct fields. Do not ignore response failures either: they remain a clearly separate limitation. A-H6 may legitimately have mixed fields and still be a worthwhile first representation.

### Stage 4 — One compact organization and response study

Evaluate the final selected A-H6 and A-D on the fixed development panel. Execute the limited interventions and subset diagnostic in Section 11. Replay the existing six signed Thermal transfers at most once per selected model, plus a small software-law/geometry derivative check. No optimization trajectory or new physical response labels are created.

### Stage 5 — Conditional fresh full-TRAIN1000

Apply Section 13 independently to each dataset. If qualified and budgeted, launch one selected organized recipe per dataset with new full-TRAIN transforms, fresh parameters and moments, and an explicit stop after 1000. Do not carry segmented weights or silently inherit their optimizer history. If a dataset does not qualify or cannot finish within the envelope, prepare its exact recipe and state the reason; continue the other dataset rather than blocking both.

### Stage 6 — Closeout with a readable answer

Deliver the report, compact evidence index, tested development and conditional/manual commands, standalone checkpoints and inference example, cleanup receipt and verified remote push. Clearly state what implementation and learning are complete, which optional formal runs actually finished, and the one most informative next refinement. Do not describe an unlaunched recipe or startup process as a completed formal result.

## 9. Reviews and moderate first-version decision rules

Separate **mandatory correctness** from **research promise** and **mature replacement**.

Mandatory correctness comprises legal inputs, correct source/control axes and units, actual joint training, output capability laws, native extraction, finite gradients, self-contained loading, query consistency, and stale-state rejection. An error in these is fixed before more scientific work.

Research promise is assessed against equally trained A-D, inexpensive input-only baselines, learning curves and measured cost. No field must beat Dense1804/2103 at e1000. No edge-degree histogram or uniform occupancy is required. No measured sparse speedup is required when full H=6 already provides a compact useful representation.

Use these prespecified **coarse research-follow-up guardrails**, not a dozens-row promotion score:

* At a common development age of at least 1000, A-H6 should be within **25% of A-D on the shared dataset field-selection score**, with Thermal fluid T and Wind near-turbine vector error also within 25%. A clear lower-cost tradeoff can be recommended separately, but it does not automatically bypass these launch guardrails.
* No well-resolved primary component/role error should exceed **2× A-D** without being repaired or explicitly preventing that dataset's automatic full1000 launch. Retain small-component absolute errors and TRAIN numerical floors so a near-zero denominator cannot manufacture a failure or hide one.
* A-H6 must learn nontrivial spatial fields on both TRAIN and DEV, use its edge bank in an actual prediction, and avoid a grossly pathological runtime. Complete high-M Q8192 inference should be within **2× A-D** for automatic follow-up; a slower case warrants diagnosis, not immediate abandonment of the mathematical idea. State stronger advantages versus mature references separately.

These deliberately broad bounds permit an informative first full-data experiment; they do not claim noninferiority, statistical significance, or production readiness. If neither new model is competent against simple TRAIN-fitted/known-input reference predictions, diagnose the common attention/input recipe instead of increasing edge count.

At e100, require healthy loss/gradient/physical behavior, not these final relative bounds. At e500/1000, compare TRAIN and DEV on fixed checkpoint weights to distinguish underfitting from generalization gaps. Do not compare a running training average with a fixed validation checkpoint and call the difference a rigorous generalization estimator.

## 10. Bounded self-revision: improve the model, do not start another portfolio

Codex may make **at most one substantive candidate revision per dataset**, two total, producing at most six segmented scientific model lineages including the original four. Bug fixes that restore the declared equations are documented separately; they do not authorize unlimited restarts. Preserve invalidated/failed work and charge it.

Use one diagnosed change, not a bundle:

| Observed limitation | Suitable single revision |
|---|---|
| A-D and A-H6 both underfit source/environment exchange | One common-context depth or width revision, recorded as a broader new-family change; do not attribute its effect solely to edge compression |
| A-H6 loses broad field detail while A-D is competent | H6→H12 **or** edge-reader width64→128, chosen from TRAIN residual/representation evidence |
| Thermal source responses show a demonstrable modal-rank limitation | r4→r8; do not confuse increasing r with discovering more physical groups |
| Accurate broad fields but poor near-interface/body structure | Improve local descriptor/reader capacity or its one support radius; do not add a complete second global predictor |
| Structured fields learn but generalization deteriorates | One simple regularization/geometry-encoding correction with a stated hypothesis; do not launch a loss-weight or augmentation sweep |

Maintain the core principles: standard attention, late content-bearing organization, no whole-scene decoding bypass, one self-contained Thermal model, legal affine/nonlinear capabilities, and bounded local detail. Do not revive Tree, shadow passes, old port refinement, or a frozen D-sep fallback.

Before the change, write a short reason and predicted effect. Use a new recipe/model identity. Shape-changing architectural revisions start fresh for the matched primary board; a warm child is permitted only as explicitly labeled exploratory evidence and must not masquerade as the same-age fresh arm. Do not transplant differently shaped tensors just to claim retained maturity. Any revision to shared contextualization changes the interpretation of the A-D comparison; keep the unchanged primary pair as the clean decoder comparison.

Stop revising once a coherent promising family is established. Small residual gaps and mixed minor metrics belong in the next round, not in another emergency mechanism.

## 11. Evaluate organizer meaning with a few interpretable operations

### 11.1 Fixed scenes and exported objects

Use the established four Thermal representatives 0277/0291/0294/0687 for detailed field exports. Use two fixed Wind DEV layouts spanning low/high M, including the maintained native row69/row426 scenes when membership is verified. Do not choose cases from error rankings. Detailed organizer interventions need only **two scenes per dataset**, at three input-selected receivers: near-body, downstream overlap and background. Numerical field statistics still use all DEV22/DEV24 cases.

Export the actual six-edge bank: physical source IDs and positions; environment positions/measures; B_M/B_E and W_M/W_E; edge anchors and typed masses; receiver alpha; signed decoded edge contributions; local pair IDs/contributions; and Thermal source-indexed C and small requested K slices. Separate the attention heads from the hyperedges. Do not replace measured assignments with a prettier geometric graph.

Show whether different receivers use different edge mixtures and whether edges aggregate more than one module/environment record. Compute inexpensive usage/effective-rank summaries, but set no required entropy or degree histogram. The current checkpoints may validly use all six edges. Thermal mode signs and edge indices are latent coordinates, not identified physical labels.

### 11.2 Four same-weight readout conditions, not another fitting portfolio

For each intervention scene compare normal A-H6 with:

1. **Uniform receiver access:** replace alpha by the uniform valid-edge weights, while keeping prepared edge content and local reads unchanged.
2. **One-edge removal:** remove the input-selected largest-access edge at the fixed probe without renormalizing the other edge weights; keep context and local reads fixed.
3. **Local-detail removal:** zero the explicit local residual, retaining the full edge bank.

Normal plus these three conditions is four forwards per scene. Rebuild/restore the unmodified prediction afterward. For Thermal one-edge removal, verify the output change equals the removed factor contribution, including native linear extraction; for Wind verify the recomputed nonlinear output change, not an incorrectly assumed linear sum. Compare both output changes and errors against the stored field. An intervention that changes a prediction is evidence of use; it is not automatically beneficial use.

A full simultaneous permutation of edges, coefficients and receiver access is an invariance check, not an organizing intervention. No expensive all-edge/all-source deletion oracle is required. No physical causal interpretation is made from incidence or nonzero gradients.

### 11.3 Optional sparse execution: one diagnostic only

After full-edge accuracy is measured, implement one opt-in **top-three input-access edge read** on the same two scenes per dataset, plus the full-H path. Keep full softmax alpha and zero omitted contributions **without renormalization**, so the approximation is explicitly term deletion. Compute the cheap router first, gather selected edge projections, then run the nonlinear reader only on those rows. No target error or dense edge-message norm selects support. Keep all required local detail.

Compare selected execution against a dense-masked implementation of the **same truncated function**, and compare both against the full learned model and physical targets. Record actual edge rows and complete timing. This asks whether the code supports a real reduced-work operation and whether any approximation looks promising. It does not qualify a learned sparse policy, and a poor top-three result does not fail the full-edge model.

Use full smooth access for official fields, training and derivative interfaces. Do not launch sparse fine-tuning, a proposer, an adaptive-K controller, or a new dense/sparse training pair in this phase. At H=6 the dense full-edge kernel may be the best executor; that is an acceptable result.

## 12. Physical validation that remains easy to understand

### 12.1 Headline field metrics

Use one common physical evaluation support within each board. Report mean and p90/p95 of per-case/layout errors, with worst-case ID in the appendix. Include units in every table and do not turn different dimensional fields into a single physical norm.

**Thermal:** fluid T, u, v, p and native omega; surface/material T; normal-flux proxy; and per-module material peak error. The headline uses full native fluid-grid support at selected endpoints. Surface/material supports must be identical across compared models and explicitly distinguish monitoring samples from full stored samples. Preserve M3/M5/M7/M10 summaries. Use all full material points for final native peak comparisons when available; streaming their native interpolation does not require retaining repeated neural activations. No initial-port metric exists in the new family.

**Wind:** component and vector RMSE for each of the five roles; emphasize near/downstream Uy/Uz as well as Ux. Aggregate means at the layout level, retaining direction-row tails separately. Use the maintained Q8192 TOTAL fixed native panel. A small fractional full-vector L2 is not enough to hide inaccurate transverse components.

Report relative L2 next to absolute RMSE for field figures, with the valid support and reference norm specified. A nearly zero reference component has no stable relative-error interpretation; its absolute error remains primary. Native Wind dense hub-plane maps and wake cuts should use stored cells, not an XY projection of a thick random hub slab presented as a plane.

### 12.2 Minimal response and inverse-interface evidence

Replay only existing compatible saved Thermal baseline/plus/minus records for 0291/0294/0687, no more than the existing six signed transfers per selected model. Use precise direct increments and compare with native endpoint differences separately. Report one signed mean fluid change, fluid/surface/material response RMSE, heat-null flow, and endpoint material-peak changes. The opposite signs do not double the independent excitation rank. Missing 0277 physical baseline remains missing.

Add one smooth geometry VJP/finite-difference fixture per dataset and the saved-state invalidation checks. When geometry changes, distinguish fixed Eulerian receivers from moving material-attached receivers and compare on a common valid support. No new Wind physical displacement truth is implied. This is software/interface assessment and limited saved-response evaluation, not an inverse-design campaign or a criterion requiring already valid generated designs.

Do not optimize response thresholds against these repeatedly exposed audit cases. If field learning improves while saved responses worsen, state both and propose a next response study; do not start it automatically.

### 12.3 Six figure groups maximum in the main report

Keep the report readable, with separate figures rather than a 40-panel collage:

| Figure group | What a reader should immediately learn |
|---|---|
| **1. Learning** | Fixed-weight TRAIN/DEV physical errors at saved ages; exact completed epochs, visits and update counts; age versus cumulative measured training time |
| **2. Thermal fields** | Reference, A-D, A-H6 and a qualified reference on identical full native grids; T and transverse-flow residuals; one clear interface/material curve |
| **3. Wind fields** | Ux/Uy/Uz native hub maps and downstream wake cuts on low/high-M layouts; avoid showing only Ux |
| **4. Actual hypergraph** | Physical modules/environment → six content-bearing edges → three receivers; actual weights and one intervention effect, with local paths visible |
| **5. Cost and scaling** | Complete inference and optimizer-boundary time, allocated/reserved training memory, fixed-B Q scaling, actual local/edge work; optional selected versus dense-masked read |
| **6. Responses and decision** | A compact saved heat-response/gradient summary and the final first-version decision; no invented design trail |

Use shared field color scales across models, readable source IDs, and unclipped numerical metrics. Any visual clipping is disclosed. Retain PDF masters and small embedded raster companions, following AGENTS.md; native arrays needed to reproduce the chosen results remain locally available. Put full per-role/per-case tables in a machine-readable file and appendix rather than repeating them in every caption. Inspect every selected figure and link.

## 13. Conditional full-TRAIN 1,000-epoch research runs

The user's permission is used here as **conditional authorization when this plan and its prompt are adopted**. It authorizes at most two new full-data jobs: one organized candidate per dataset, after the first-version checks and available-budget forecast. It does not authorize restarting old runs or extending to 5000 now.

### 13.1 Launch identity and scope

Use Thermal original TRAIN600 and the already exposed canonical89 validation catalogue excluding duplicate0273; use Wind original TRAIN420 and fullVALID90, with WindTEST locked. Build new full-TRAIN-only normalization/profile/loss-scale state. Initialize fresh model weights and fresh optimizer moments. Select the architecture/configuration from development, not its weights. Preserve the declared TRAIN response addendum separately.

Freeze a prospective **5000-epoch LR schedule** with warmup20, hold through2000, and decay to3e-6 at5000, but enforce an **actual execution stop at epoch1000**. This produces a resumable full-data research checkpoint under a possible later user-authorized 5000 trajectory; it is not a converged 1000-horizon fit. Report that distinction. A subsequent continuation beyond1000 requires new user authorization, not automatic Goal-mode persistence.

Keep the selected architecture, physical output laws, loss weights, query counts, environment resolution, effective batch, and backend settings unchanged from the winning development recipe except for explicitly necessary full-TRAIN transforms and the declared full-run schedule. Do not drop a chosen locality rule or silently change an auxiliary coefficient at formal launch. Print and review a resolved development-versus-formal configuration diff before creating the run.

Do not reuse sealed old P-only full1000 protocol names for the new model. Add a narrowly named new-family follow-up recipe/schema with an explicit `approved_stop_after=1000`, preserving strict legacy behavior. The recipe should fail if asked to exceed that execution stop without a new authorization record. CPU dry-run must be inert: no optimizer update and no formal run allocation.

### 13.2 Compare with classics without pretending the experiments are identical

Prepare three clearly separated boards:

**Board A — matched development:** A-D versus A-H6 at the same completed segmented age, same populations, query streams, scales and losses. This is the main first-version model comparison.

**Board B — full-data early age:** the new full-TRAIN e1000 checkpoints versus actual saved historical e1000 checkpoints **when they exist and can be loaded natively**. Match output receivers and native metric definitions. Record differing initializations, objectives, pretraining/frozen components, effective updates and schedules. Same epoch and full dataset do not alone make this a causal architecture ablation. Do not infer a missing historical e1000 checkpoint by interpolating curves or substituting a later best model.

**Board C — mature-system distance:** the new e1000 versus Thermal1404/1804 literal mature endpoints and Wind2102/2103 their actual saved mature checkpoints. Wind2102/2103 were reported as selected e2440/e2475 from 2500-epoch histories; do **not** label them e5000 because the Thermal classics reached5000. Include mature Wind2202 if useful as an efficient contemporary reference. This board measures practical distance, not equal-age superiority. [S1, S9]

Every reference uses its own native input transforms, local-surrogate path if present, environment catalogue and output conversion. In particular, preserve Wind classic E512. Do not give classics the new E64 input or apply a new TRAIN normalizer to their checkpoints.

### 13.3 Thermal classic reconciliation is a bounded independent task

The most recent R0 addendum identifies a real raw/global-normalization error and still quarantines its incomplete corrected GPU board. Old field arrays from the wrong input path are not eligible comparison targets. It also records exact CPU public/helper prediction agreement on two cases, while a repeated public GPU interface call itself failed its strict tolerance. These facts do not establish a completed corrected population board. [S10]

Allow a bounded **new CPU public-path reference board** on the required comparison panel if needed: first reproduce the actual HDF5 reader's normalized global heat/interface inputs **and raw-derived local inputs**, then verify public/helper and repeated-public predictions on low/high-M probes. Only after this qualification, complete the new board using the original public pipeline and record its own source, input and checkpoint bindings. Keep the older quarantine and GPU failure unchanged; a new qualified CPU field board is not a retrospective GPU-parity pass.

Do not spend more than two CPU wall-hours or the remaining evaluation allowance on this recovery. If native reference qualification cannot be completed, mark that reference unavailable on the current panel and compare with verified A-D/P/Wind references. Lack of a corrected classic score must not block the new architecture's implementation and segmented assessment. Never fill an unavailable baseline with an unsupported number from a past chat summary.

### 13.4 Independent follow-up decisions

Each dataset can qualify separately. Do not force an unsuccessful Wind candidate into a formal run because Thermal is promising, or vice versa. Do not launch A-D alone and call that evidence for an organized HONF. If A-H6 is still poor but the new attention-based A-D is strong, the valuable result is a localized decoder problem and a tested implementation—not a justification for disguising the control as the proposed model.

## 14. Storage, historical preservation, and source changes

Follow root AGENTS.md without creating a larger administration project. Before materialization, declare round-owned caches/output roots, disk-byte ceiling, temporary peak and filesystem reserve. Reuse verified shared geometry/catalogue caches. Do not duplicate full HDF5/mmap datasets per arm or checkpoint. Keep sufficient reserve for final e1000/e2500 saves and report exports.

Maintain the prescribed 100-epoch checkpoint cadence, latest/best aliases, compact histories, resolved recipes and selected numerical evidence. Remove only owned disposable benchmarks, duplicate caches and superseded figure payloads after verifying the retained evidence. Never delete protected historical checkpoints, the only source/reference array copy, active-run files or needed report dependencies. Use the existing allowlist/ownership cleanup procedure; record bytes removed and remaining capacity.

Leave old family code and checkpoints replayable. Prefer additive new-family files and narrowly tested helper extractions. Do not retire or rename historical scientific families to make the new code look cleaner. Do not relax the solver ledger, data locks or old numerical tests. Current solver allowance remains **326/326 with zero new attempts** unless the user separately changes it.

Commit and push durable code, tests, configuration, guides and the final report to the current agreed non-default branch. Inspect the entire outgoing history under the existing pre-push artifact gate. Generated figures, datasets, weight files and one-time diagnostics stay local/ignored. Verify the remote tip matches the local commit. No force push or shared-history rewrite except a separately permitted unpublished-artifact cleanup under the existing repository rule.

## 15. Required handoff

Deliver these items, reusing equivalent maintained structures rather than duplicating them:

* **One new reusable family:** core plus two physical adapters, self-contained loader, factorized Thermal application, training integration and focused tests.
* **One reproducible experiment registry:** primary/revised runs, exact completed/selected ages, dataset/scale/query identity, precision, actual compute, and concise change reasons.
* **One readable report and one evidence index:** the six figure groups, all development statistics, conditional full1000 results if executed, numerical limitations, and explicit code/learning/physical evidence boundaries.
* **Tested commands:** dry-run, development start/stop/resume, complete native evaluation, trace export, precise heat increment, conditional full1000 launch, and manual later continuation recipe that does not run without authorization.

The report's first page should contain a small board with the actual values for Thermal fluid T and flow, Wind near/downstream vector and transverse errors, complete inference time, real training peak memory, and whether edge content influences the output. Then answer in ordinary language:

**Predictor:** Are the fields now recognizable and competitive with the matched direct attention control? Which remaining error is largest and spatially where?

**Organizer:** Is the bank a real intermediate representation, what does a receiver gather through it, and is the main gain representation compression, selective work, or neither?

**Cost:** What does one complete prediction and one real update cost at fixed B/Q? What actually grows with Q and M? Is optional sparse execution faster than the same dense-masked function?

**Generality and inverse interface:** Does the same code train both datasets, which assumptions remain in adapters, and which response/derivative/reuse operations work? Do not replace this answer with a claim that inverse designs are validated.

Conclude with one of: **promising first HONF family; continue focused refinement**, **attention foundation works but edge decoding needs revision**, or **common implementation/learning defect remains localized**. Do not summarize every result as merely “mixed” or use “formal promotion failed” as the first-version verdict. Strong-reference replacement remains a later question.

## 16. Acceptance checklist for Goal closeout

The goal is complete when the new model has actually been implemented and trained in both datasets; the shared-core/data-adapter boundary is real; native physical laws and essential software checks work; the four main experiments or explicitly justified bounded replacements have measured endpoints; accuracy, cost/scaling and real edge use are clearly reported; optional full1000 jobs have either finished or are explicitly deferred; and durable changes are pushed with historical/data protections intact.

It is **not** necessary for every field to beat the classic, every receiver to use fewer than six edges, every optimized backend to pass, every legacy unrelated test to be repaired, or the inverse application to be ready. It **is** necessary not to hide illegal physical dependencies, incorrect normalization, disappearing source identities, stale geometry, unmeasured memory scaling, failed jobs, or dense computation presented as sparse.

## 17. Basis and implementation references

**Design basis.** The supplied `HONF_Redesign_Oct13.md`, read in full. The reviewed copy has SHA256 `6887c5436f6006fa8b6451cca5abe753e08a2d27ca3fed08cbbd336caeb63a44`. Resolve and bind the user's actual copy at startup; a mismatch calls for a content comparison, not blind rejection. Sections 5–10 define the math; Sections 11–16 define its reuse, numerical, interpretation and cost boundaries. This plan supplies the concrete recipes and modest first-version evaluation policy left open there.

**Repository sources inspected for this plan.** Source snapshot `7971a7ed57ff889b226f36cd4b5eaed07f96b9cd`; targeted code review, not a claim that the entire repository or local data were executed.

| Key | Source | Basis used |
|---|---|---|
| S1 | Root `AGENTS.md` | Fixed-subset policy, 100-epoch cadence, reporting, cleanup and push requirements |
| S2 | `HONF_Proj/src/honf_runtime/unified_training.py` | TaskProvider protocol, deterministic SamplingKey, joint mode, exact denominators and state handling |
| S3 | `HONF_Proj/src/honf_forward_core/interface_fields/interaction_core.py` | Existing dense pair-value preparation; scene/measure/source ownership |
| S4 | `HONF_Proj/Case_ThermalChannel/src/channelthermal/source_response.py` | Native union/extraction, interpolation derivative caveat, dense export in precise application, frozen composition to exclude |
| S5 | `HONF_Proj/Case_ThermalChannel/src/channelthermal/flow_curl.py` | Generator/native curl laws and physical conversion |
| S6 | `HONF_Proj/Case_ThermalChannel/src/channelthermal/training/joint_task.py` | Data/normalizer/mask ownership, role losses, model-bound provider restrictions and optional auxiliaries |
| S7 | `HONF_Proj/Case_WindFarm/src/windfarm/training/joint_task.py` | Five-role sampler, E64 support, full split validation, TRAIN-only component/profile scaling and native-cache dependencies |
| S8 | `HONF_Proj/tools/joint_regional_train.py` | Existing family-specific launcher, sealed profiles and reusable physical model/objective descriptions |
| S9 | Supplied multiview, unified-comparison and recovery reports | Exposed dataset populations, historical actual ages and context/architecture differences; report-derived, not rerun |
| S10 | `HONF_R0_Thermal_Classic_Native_Input_Reconciliation_20261013.md`, read at `f10aa863` before the archive move | Raw/global input defect; qualified two-case CPU predictions, incomplete corrected population board and retained GPU failure |

**Official implementation documentation checked for the proposed optimization choices.** Documentation availability does not establish the installed environment's capabilities. Consult its version-matched documentation before using optional features.

* D1 — PyTorch, `torch.nn.functional.scaled_dot_product_attention`: `https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html`. Mask semantics, dropout and backend behavior; no advertised speedup is imported into HONF.
* D2 — PyTorch, Dynamo core concepts: `https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html`. Graph breaks and compiled-region scope.
* D3 — PyTorch, Automatic Mixed Precision examples: `https://docs.pytorch.org/docs/2.14/notes/amp_examples.html`. Precision regions, gradient accumulation, unscale-before-clipping and optimizer-step boundaries.
* D4 — PyTorch, `torch.utils.checkpoint`: `https://docs.pytorch.org/docs/stable/checkpoint`. Non-reentrant activation checkpointing and recomputation/memory tradeoffs. No new experimental checkpoint API is required.

**One-sentence execution mandate:** Implement and mature a clean attention-contextualized, content-bearing HONF on both native datasets, keep physical interfaces and training infrastructure, measure its actual field/organization/cost behavior, and use only a small number of evidence-driven revisions before deciding whether a fresh full-TRAIN1000 continuation study is worthwhile.
