# HONF attention round 2: geometry-carrying exchange and content-led decoding

**Mode:** Codex Goal mode; implement, train, inspect, revise within scope, and conditionally run fresh full-TRAIN fits to an enforced epoch-1000 stop.

**Reviewed repository:** `cosmos2w/ModularDT`, `agent/honf-core-next`, commit `852f995403a91d8feb0e5df94f3022c7b1f3467b`, verified through the live GitHub connector on 14 October 2026. Check the actual checkout and any subsequent changes before implementing. No training, checkpoint-array replay, GPU benchmark, or local-machine verification was performed while writing this plan.

**Primary evidence:** `HONF_Proj/docs/reports/HONF_Content_Hypergraph_Oct13_Report.md` and `HONF_Proj/docs/reports/HONF_Content_Hypergraph_Oct13_Evidence_Index.md`, plus the source files identified in Section 2. The report was read in full; the evidence index was retrieved from the same pinned repository snapshot. Numerical claims below are reported measurements, not newly reproduced results.

**Relationship to the previous documents:** retain `HONF_Redesign_Oct13.md` as the architectural motivation and `HONF_Redesign_Implementation_and_Evaluation_Oct13.md` as the first-round record. This document explicitly amends the first version's simultaneous context schedule, geometry representation, node-to-edge competition, and reader laws. It is not an assertion that those equations were already implemented. New names and configuration fields below are proposed APIs, to be implemented and tested rather than passed to old launchers.

## 0. The decision and the bounded goal

Continue the attention/content-hypergraph direction. Do not return to recursive Trees, dense pair-value contextualization, a frozen D-sep partner, Stage-A, or the P0/P1/P2 thermal loop. The useful foundation is one self-contained, jointly trained model per dataset, reusable native interfaces, inexpensive scene preparation, and a decoder whose expensive work can depend on a small content bank rather than every original donor.

The second round must address **information sufficiency before sparsity**. The current direct attention model is also inaccurate, so replacing only the edge bank cannot be assumed sufficient. Conversely, a more elaborate encoder alone does not fix the way Thermal incidence suppresses response channels. Build one coherent replacement of these two information paths, with a direct-reading control that makes failure localizable.

The intended pipeline is:

```text
Prescribed attributes + explicit spatial identities + operating conditions
    -> geometry-carrying typed attention, staged M/E exchange and reinjection
    -> contextualized physical modules and environmental records
    -> six spatially registered contents that actively gather those records
    -> multi-head receiver reads of the contents + compact local detail
    -> dataset-owned nonlinear fields or affine control application
    -> unchanged native observations and units
```

**Primary outcome:** two improved organized research candidates, one for Thermal and one for Wind, or an honest, localized reason one candidate remains unsuitable. Give healthy comparisons 2500 segmented epochs. After the measured improvement review, launch at most one fresh full-TRAIN organized candidate per dataset to 1000 epochs. A dataset can qualify independently. No 5000-epoch continuation, new solver call, inverse generator, or historical replacement is authorized.

### 0.1 Resource and ownership contract

Use physical GPU **0 and 1 only**, with UUIDs rechecked at arrival. The previous mapping was GPU0 `GPU-233fcd85-5c6a-6212-3f44-655252afec70` and GPU1 `GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e`; do not assume it remains unchanged. Expose one authorized UUID per process and use logical `cuda:0`. Normally assign Thermal and Wind separate lanes. Do not terminate unrelated processes, hot-edit their imported code, or use GPU2. No DDP is required.

Use ModularDT's existing environment; the reviewed report used PyTorch 2.6.0+cu124. Record the actual environment. Do not upgrade a shared installation to obtain a fashionable attention kernel.

Use a **planning ceiling of 16 aggregate GPU-associated process-hours and 14 elapsed hours from the first GPU experiment**, including diagnostics, failures, revisions, the two possible full-data runs, and their evaluation. Reserve the last two elapsed hours for evidence and delivery. These are ceilings, not an ETA or an obligation to consume the allowance. Report CPU/coding preparation separately. Forecast with measured complete epochs, cold catalogue preparation, monitoring and evaluation; do not extrapolate warm GPU steps alone. The previous Wind full run's approximately 90.6 GiB peak host RSS makes host memory a real scheduling constraint even when VRAM is almost empty.

At most **800 disposable optimizer updates in total** may be spent on all tiny-fit, packing, and numerical qualification work. Aim substantially below that ceiling. They never initialize scientific runs. At a budget boundary use the maintained epoch-safe stop/resume facility; save and report the actual state rather than deleting failed runs or implying the requested horizon completed.

## 1. What the first round established—and what it did not

The literal-2500 comparison is useful because it shares the new contextualizer, dataset membership, sampling, objective and schedule within each dataset. It does not establish that this contextualizer is a competitive replacement for the older physical predictor. [R1, R2]

| Measured endpoint | Thermal A-D / A-H6 | Wind A-D / A-H6 |
|---|---|---|
| Priority physical RMSE | Fluid T: 2.47444 / 3.30279 native T | Near-turbine vector: 0.64306 / 0.64282 m/s |
| Other important error | v: 0.01037 / 0.01061 native units | Downstream vector: 0.45548 / 0.47714 m/s |
| Native exposed-DEV field score | 0.06240 / 0.12543 | 0.19483 / 0.20558 |
| Complete high-M native B1 call | 17.20 / 19.59 ms | 7.72 / 8.70 ms |
| Fixed B8/Q8192 forward/backward peak allocation | 1.506 / 1.004 decimal GB | 2.474 / 0.809 decimal GB |

The memory reduction is real at fixed batch and requested query density. The complete-call latency is slightly worse for H6, and neither field accuracy nor sparse wall-clock acceleration is established. All six edges execute. Low GPU memory is available capacity, not a reason to fill memory without a learning or throughput hypothesis.

Thermal's width-128 reader revision improved the original H6 native field score to 0.11080, but remained 1.775 times A-D. At literal2500 its fixed TRAIN aggregate was close to A-D while DEV thermal errors remained high. More width alone did not resolve the limitation; neither this result nor a loss curve identifies a unique cause.

Wind's fresh full-TRAIN H6 e1000 near-turbine error was 0.51648 m/s versus 0.11663 for classic2102 e1000 and 0.08843 for dense2103 e1000. These are same-panel, same-literal-age references, **not matched training budgets**: the classics used 27000 updates, batch16, Q8192 and E512, while H6 used 18000 updates, batch24, Q4096 and E64. This substantial practical gap cannot be dismissed solely as a segmented-data artifact, but it also does not isolate architecture from objective/exposure differences.

The new report **completed a separately qualified Thermal classic CPU reference board**. Reuse that valid path and its exact inputs/transforms. Do not repeat the previous claim that all Thermal classic results are unavailable. The older wrong-input arrays and failed GPU parity attempt remain quarantined; the new CPU board does not retroactively validate them. Wind mature2102/2103 ages are actual e2440/e2475 from 2500 histories, not invented e5000 ages.

## 2. Code-grounded diagnosis before changing the architecture

All paths in this section are relative to `HONF_Proj/`. Read these actual functions, not just the architecture drawing.

| Reviewed file / symbol | Actual finding | Consequence and required response |
|---|---|---|
| `src/honf_forward_core/content_hypergraph/model.py`, `prepare`, `_operating_state` | Initial token values use encoded attributes plus a prescribed operating embedding. Physical coordinates are separate arguments. No learned layout-global summary is built by `_operating_state`. | Add spatial information to token values explicitly; introduce a small learned scene summary only inside preparation/reinjection. Do not call the old operating vector a learned whole-layout state. |
| `.../attention.py`, `_RouteAttention`, `TypedAttentionBlock.forward` | MM/ME/EM are simultaneous; geometry changes scores through one linear map of signed offset and distance; donor values are token-only. There is no EE update. | Change spatial expressivity and value content as well as update order. Simultaneous updates are not themselves a coding error. |
| `Case_WindFarm/src/windfarm/content_hypergraph.py`, feature contract | Initial module attributes are rotor radius and hub height, while turbine locations are supplied separately. | Identical turbine attributes can produce identical initial module values. Measure this on input-only TRAIN fixtures before interpreting attention maps. |
| `.../organization.py`, `HypergraphOrganizer.forward`, `_measure_weighted_pooling` | B normalizes each donor across edges; W renormalizes measure-weighted B over donors. Module keys use unnormalized evolving token states. Tiny edge masses already required a stable derivative repair. | Replace winner-take-most edge allocation with direct, normalized edge-to-donor gathering, rather than adding an occupancy loss to the old quotient. Preserve the repaired v1 path for old checkpoints. |
| `.../organization.py`, `SourceCoefficientBank.forward` | Signed coefficients are multiplied by raw module B, not by the normalized W that forms edge content. | Remove this particular coupling in v2. Use the actual edge gather for both source content and source-resolved response factors. |
| `.../decoding.py`, `QueryEncoder`, `EdgeQueryReader`, `DirectSourceReader` | Query coordinates have no multiscale encoding; relative values are linearly projected. A single scalar access distribution controls all edge messages and Thermal modes. A-D also multiplies each thermal coefficient by a source softmax weight. | Use a richer shared spatial representation, several feature-head reads, and distinguish information averaging from addition of independently controlled source responses. |
| `.../model.py`, prepared-state validation; `response.py`; native adapters | Scene/query ownership, factorized heat application and native extraction already exist. | Reuse and extend them. Do not rebuild a second runtime, add dense K exports to ordinary inference, or relax historical loading. |

### 2.1 Why geometry only in scores can be ineffective

For identical donor values, attention is independent of its spatial weights:

\[
 v_j=v\quad\Longrightarrow\quad a_i=\sum_j\alpha_{ij}v_j=v,
 \qquad \frac{\partial a_i}{\partial\ell_{ij}}=\alpha_{ij}(v_j-a_i)=0.
\]

This is an algebraic statement, not a newly measured property of every trained Wind layer. In the existing Wind contract, identical radius/height attributes plus shared operating conditions make it a concrete first-layer risk. First-layer MM and EM can then mix the same values regardless of layout. ME and later blocks still provide geometry-dependent paths, and the decoder still sees relative coordinates; the current complete model is **not** being called geometry-blind.

There is a second exact limitation. Within a fixed attention row, the signed-linear part of a geometric score satisfies

\[
 w^\top(x_i-x_j)=w^\top x_i-w^\top x_j.
\]

The destination-only term cancels in softmax. The existing distance term remains receiver-dependent, so the geometric score is not wholly ineffective; it is simply much less expressive than a general directional receiver–donor relation.

### 2.2 Why the Thermal incidence picture is more than a cosmetic warning

The implemented v1 factor has the form

\[
 B_{ig}=\operatorname{softmax}_g(s_{ig}),\qquad
 W_{gi}=\frac{\mu_iB_{ig}}{\sum_j\mu_jB_{jg}},\qquad
 C_{gri}=B_{ig}\,\kappa_{gri},\qquad
 K^{\rm far}_{qi}=\sum_{g,r}\Phi_{qgr}C_{gri}.
\]

In the exact limiting case where every module assigns to one edge, all other C blocks vanish. The far response then has rank at most r=4, rather than nominal capacity Hr=24. The compact local branch can add spatially local response structure; the rank statement concerns the far operator, not the entire model. Near-one floating-point incidence does not by itself establish the trained numerical rank. Measure it.

This does **not** imply that Wind's other edges are empty: W can retain a nontrivial normalized donor mixture even when the corresponding B values are tiny. Wind has no affine C bank, and its reported receiver reads of edges3/4 are compatible with B concentrating on edge0. The next report must show actual W, content and output effects—not diagnose both datasets from B alone.

### 2.3 A short diagnostic, not another campaign

Before scientific fitting, use two input-selected low/high-M TRAIN scenes per dataset and one small synthetic fixture to check: initial module-value variance; changes in attention weights versus weighted values under a coordinate change; spatial-gradient reachability; normalized score range; edge W and C concentration; and the rank/energy spectrum of the model's far response on a fixed small receiver panel. Run the same definitions at e100, e1000 and e2500 for the new organized model. These are model diagnostics, not new physical excitations.

Record which suspected mechanisms are observed and which remain hypothetical. Do not spend a day reconstructing every older experiment before coding. No complete mathematical change should be justified by a graph's visual appearance alone.

## 3. The new family: two complementary model candidates

Use new, explicit identifiers, provisionally **A2-D** and **A2-H6**. Both use the same upgraded spatial contextualizer, query representation, local-reader capacity, native interfaces and training protocol. They train independently from fresh weights. Match identically named/shaped common initial tensors and all data streams. Parameter counts need not be artificially equalized.

| Choice | A2-D | A2-H6 |
|---|---|---|
| Context width / heads / cycles | 128 / 4 / 2 | Identical |
| Exchange | Staged MM, EM, EE, ME with raw/global reinjection | Identical |
| Spatial representation | Geometry in Q/K/V, multiscale coordinate features, nonlinear low-width relative bias | Identical |
| Global decoder | Direct physical-source reads | Six content-bearing edges |
| Reader width / feature heads | 128 total / 4 × 32 | Identical total width |
| Local detail | Same compact support and 128-wide reader | Same |
| Thermal control law | Individual signed source coefficients, then physical heat sum | Four signed modes per edge, then factorized physical heat sum |
| Ordinary edge access | Not applicable | All six; no sparsity curriculum |

Do not create separate Thermal and Wind Transformer implementations. Dataset-specific input construction, frame/length conventions, normalization, output laws and native observations remain in the adapters. Weight sharing across datasets is **not** required or claimed.

This is a substantial **family-level redesign**, not a single-variable ablation of sequential scheduling. The two primary candidates isolate organized versus direct decoding under the same new encoder. Section 9 provides a small optional fusion-only screen when more localization is needed.

## 4. Geometry-carrying, staged typed attention

### 4.1 Explicit spatial descriptors

The adapter supplies physical coordinates x, frame origin o, frame extent S, masks, measures, prescribed attributes, and positive characteristic lengths. Preserve the distinction between object size and local-reader support radius. Thermal object length comes from the actual module radius/diameter convention; Wind's rotor-diameter convention stays explicit. Environment characteristic lengths come from its declared quadrature-cell geometry, not a guessed physical obstacle radius.

Construct a shared smooth coordinate descriptor, initially

\[
 u=(x-o)\oslash S,\qquad
 \gamma(u)=\left[u,\{\sin(2\pi 2^ku),\cos(2\pi 2^ku)\}_{k=0}^{2}\right].
\]

The raw coordinates remain present; periodic features are supplementary, not a replacement that identifies different points. Include explicit scale/size ratios where available. These low, fixed frequencies are an initial representation decision, not a tunable frequency portfolio. Record derivative scaling in native units. This does not assert arbitrary rotation or scale invariance: the adapters retain their prescribed frames, dimensions and boundary conditions.

For each type t, prepare an immutable-within-forward raw embedding

\[
 z^0_{t,j}=E_t\bigl([a_{t,j},\gamma(u_{t,j}),\text{declared size ratios}]\bigr)+E_{\rm op}(c).
\]

It is a **live differentiable tensor** of prescribed inputs, not a detached cache. Its spatial component reaches values as well as keys/queries. IDs remain correspondence metadata and cannot be learned ordinal embeddings.

For score biases use a small shared-by-route two-layer MLP, width32, on signed domain-relative offsets, distance, and smooth local-scale offsets such as asinh((x_i−x_j)/ell_pair). Do not build a width128 pair-value tensor in order to compute this small score. The adapter specifies ell_pair and its dimensional convention. No hard-coded downstream wake function, CFD formula, or target field enters these features.

### 4.2 Attention law and stable magnitudes

Use donor-measure-aware SDPA with masked, normalized Q/K per attention head:

\[
 \ell_{ij}^{(a)}=\tau_a\,\widehat q_i^{(a)\top}\widehat k_j^{(a)}
 +b_a(\gamma_{ij})+\log(\mu_j/\mu_{\rm ref}),
 \qquad \widehat q=q/\max(\|q\|_2,\epsilon),
\]

and the same definition for k. A positive constant measure reference cancels within a row. Use a positive learned head scale, initialized at sqrt(head_dim); monitor it and enforce a declared numerical range, initially [1,32], rather than silently letting softmax become arbitrarily sharp. This is not a uniform-attention or occupancy target. Empty rows return exactly zero; source masks and MM self-exclusion remain unchanged.

Implement the scale once: when normalized Q/K already incorporate tau, SDPA must not accidentally divide by sqrt(head_dim) again. For a learned tensor-valued tau, multiply normalized Q by that live tensor and call SDPA with scale=1.0; do not convert tau to a Python float and sever its gradient. An equivalent formulation needs its own reference check. Tests must compare the complete formula with a simple reference attention, including float geometry biases and measure masks.

QK normalization is motivated by published saturation control, not evidence that it alone improves HONF. Geometry and value content are the central repair. [P1]

### 4.3 One ordered exchange cycle

Let M^l and E^l be current tokens. Reinject their raw encodings and a small learned scene summary G^l through residual projections, with nonzero modest initialization (for example residual scale0.1). G^0 comes from separate measure-normalized raw M/E summaries plus operating conditions, the physical module count and declared physical mass/size features. Do not use the number of environment samples as a physical count feature: duplicating a quadrature atom with divided measure must not change scene meaning. Do not mix units by adding unnormalized module and environment measures.

Execute each cycle in this order:

\[
 \widetilde M=M^l+\operatorname{Attn}_{MM}(M^l,M^l),
\]
\[
 \widetilde E=E^l+\operatorname{Attn}_{EM}(E^l,\widetilde M),
 \qquad E^{l+1}=\operatorname{FFRes}_E\bigl(\widetilde E+\operatorname{Attn}_{EE}(\widetilde E,\widetilde E)\bigr),
\]
\[
 M^{l+1}=\operatorname{FFRes}_M\bigl(\widetilde M+\operatorname{Attn}_{ME}(\widetilde M,E^{l+1})\bigr).
\]

The displayed tensors are the reinjected, masked states at the cycle boundary; use pre-normalization and one ordinary tokenwise FFN per type per cycle. Recompute the next G from separately normalized final M/E summaries. Two cycles are the initial design.

In plain terms, modules first compare with modules; the environment reads that updated layout, communicates across its own modest spatial catalogue, and modules then read the updated environment. Unlike v1, the return trip is completed inside a cycle. EE supplies a broad environmental path without forcing every environmental interaction through a small M set. With E192/E64 it is a bounded **scene-preparation** operation, never Q×Q field attention.

The learned scene summary may condition these updates, edge construction and source coefficients. **It must not become a wide, direct learned-layout-to-query bypass around the edge bank in A2-H6.** Prescribed operating conditions can enter the query encoder directly. The scene's learned information reaches far decoding through the contents and reaches local decoding through explicitly retained module states.

Prepare K/V once for each unique donor state and route projection. A changed donor state in this sequential schedule needs new projections; reusing stale K/V to save time would implement the wrong model. Cache geometry features within the forward when their coordinates are unchanged. Do not retain learned states across optimizer steps.

## 5. Edge contents that gather information rather than compete for all donors

### 5.1 Spatially registered queries

Retain H=6. Initialize six distinct reference-domain anchors using the existing deterministic spatial seeds. Give each a learned content seed and smooth, bounded anchor offsets. Initialize its per-axis gathering scale at S/H^(1/D), rather than at an undifferentiated full-domain width. A learned scale can broaden or narrow smoothly within an explicit range, initially [0.25,4] times that initial scale; it must not produce zero width.

These are weak reference-domain coordinates, not ground-truth physical regions or exclusive module ownership. Keep full support; the initial Gaussian spatial preference is a score bias, not a pruning mask. An edge can attend to distant donors when the learned content/geometry warrants it.

### 5.2 Direct typed donor gathering

Replace source-normalized-across-H allocation followed by a potentially tiny quotient with an edge query normalized directly over eligible donors of each type:

\[
 s^t_{gj}=\tau\,\widehat q_g^\top\widehat k^t_j
 +b^t(\gamma(x^t_j-a_g))
 -\tfrac12\|(x^t_j-a_g)\oslash\ell_g\|_2^2,
\]
\[
 W^t_{gj}=\frac{\mu^t_j\exp(s^t_{gj})}{\sum_{k\in\mathcal V_t}\mu^t_k\exp(s^t_{gk})},
 \qquad U^t_g=\sum_j W^t_{gj}V_tz^t_j.
\]

Compute this in stable log space; handle a genuinely empty type explicitly. Every present edge can ask every eligible donor. No edge-load balancing, Sinkhorn equalization, forced binary incidence, sparsity penalty, or artificial donor mass is introduced. A weak spatial prior and explicit geometry make differentiated content available; reconstruction decides whether it is useful.

Fuse U^M, U^E, the edge seed, anchor/scale descriptors and G into an edge state. Perform **two content reads with shared gather parameters**, with one small H×H self-attention residual between them. The second read uses the updated content query and reads the still-available individual M/E tokens again. Preserve a residual from the first typed content so that edge–edge communication does not replace all contents with one homogeneous global vector.

This is a fixed-depth refinement, not iterative clustering with a convergence controller. Count both content reads and the H×H operation. Both happen once per scene and retain explicit trace ancestry.

### 5.3 What the organizer exports

Export actual typed W, geometry and identities; first/second content-read state; final edge contents; H×H attention when enabled; per-head receiver access; C for Thermal; and the local path. W has shape [B,H,M] or [B,H,E], normalized over donors. Do not call it v1's B or plot it on the wrong axis.

An optional source-to-edge visualization can normalize transposed W over H, but label it **derived display incidence**, not a new executed gate. No inference should consume that display-only normalization. Different typed measures retain separate normalization and units. Learned weights are computational gathers, not heat fractions, energy conservation, or causal source influence.

## 6. Receiver gathering and physical response

### 6.1 Several feature heads, not several independent large readers

Use one 128-wide nonlinear query–content reader whose output is split into four32-wide feature blocks. Each head computes its own six-edge softmax using cached content keys and the shared spatial representation. Concatenate the four weighted block sums, rather than allocating four independent128-wide readers. The ordinary pass evaluates each query–edge nonlinear message once, then shares its features across heads.

This lets different receiver features consult different contents; the one-scalar-distribution-for-everything bottleneck is removed without increasing H. Cache edge-side and operating-side affine projections during preparation; compute query projections once per tile. Keep explicit relative query–anchor features and the multiscale query encoding. Full six-edge access remains the default.

### 6.2 Thermal: source-preserving modes without raw-B starvation

Retain r=4 initially, using one signed spatial mode per feature head. Let alpha^(r)_qg be its access and psi_qgr its signed basis value. Define

\[
 \Phi_{qgr}=\alpha^{(r)}_{qg}\,\psi_{qgr},
 \qquad
 C_{gri}=W^M_{gi}\,\kappa_r(z_i,H_g,\gamma(c_i-a_g),G,c).
\]

Here W^M is the **actual final module-content gather**, not the old raw B. Kappa is signed and has the adapter-declared response units after output conversion. Physical donor measure enters W once; do not multiply it again outside W. Include declared total-mass/count/size information in preparation so a normalized content average is not confused with total source forcing.

The change is intentional: **one source can participate in multiple response-bearing contents without winning a source-to-edge competition**. The source coefficients and content graph now refer to the same gather. Do not implement a dense, unrelated C network behind a decorative sparse incidence image.

Evaluate

\[
 a_{gr}(h)=\sum_i C_{gri}h_i,
\]
\[
 T(q;h)=\sum_{g,r}\Phi_{qgr}a_{gr}(h)
 +\sum_{i\in\mathcal N(q)}\chi_i(q)\,k_i^{\rm loc}(q)h_i.
\]

Compute C h before expanding over queries. Ordinary prediction, precise increments and heat JVP/VJP must not materialize dense K[Q,M]. Explicit K export remains available for bounded diagnostics and is charged separately. Do not apply nonlinearities after heating to the Thermal field. Heating cannot enter contextualization, geometry features, learned G, content gathering or access.

Maintain the same native zero-offset capability: T(0)=0 and heat-independent u/v/p/omega. These are ThermalChannel assumptions supplied by its adapter, not a universal constraint imposed on Wind or future coupled datasets. Surface, outside, material and q-proxy observations continue to come from this same temperature field.

### 6.3 A2-D: a genuinely useful uncompressed control

A2-D uses the same new contextualized physical source states, query encoding, reader width and local branch. Its nonlinear flow/velocity read can use the same four feature-head normalized information gather over physical sources.

For Thermal, however, the direct response is the signed per-source coefficient itself, followed by the physical sum over h. Do **not** multiply each independent source's Thermal coefficient by a softmax across all other heaters. Information averaging in a feature attention block and superposition of separately controlled physical sources are distinct operations. This removes an unnecessary competitive parameterization from the control while preserving exact affine heating.

This does not prove v1's softmax coefficient was mathematically incapable of fitting the data; a sufficiently expressive network could compensate. It removes a plausible representational handicap and gives the compressed candidate a competent control. Report the change as part of the decoder redesign, not as an isolated fusion-only result.

### 6.4 Local detail remains explicit and bounded

Keep Thermal3r and Wind1.5D support for the primary comparison, including all overlapping eligible neighborhoods. Use the existing smooth compact window, native geometry and deduplication rules. Widen the local reader to128 in both new arms and provide it with the same richer relative spatial descriptors. This is not permission to make the Wind local radius cover an entire downstream wake.

A later radius or near/far-composition change requires the one documented revision allowance and a specific observed failure. No global dense source bypass may be added to A2-H6 and then described as six-edge decoding. Global plus local terms remain an additive representation; do not claim a unique physical decomposition when their learned contributions compensate.

## 7. Reuse boundaries and coding sequence

### 7.1 Add versioned modules; preserve old functions

Implement the new core under an explicit v2 constructor/state schema. Proposed additions inside `src/honf_forward_core/content_hypergraph/` are `spatial_features.py`, `staged_attention.py`, `content_bank_v2.py`, and `readers_v2.py`. A small `model_v2.py` orchestration class is preferable to a growing collection of incompatible flags in v1. Share truly common scene/query/response helpers through narrow interfaces. Names are proposals; avoid duplicate classes if an existing small abstraction already fits.

| Existing component | Required action |
|---|---|
| `contracts.py` | Extend/version geometry and prepared-content metadata; preserve native source IDs, masks, measures and old schema loading. |
| `attention.py` | Keep v1 replay intact. Reuse correct masked-SDPA primitives where practical, add ordered v2 blocks without changing v1 equations. |
| `organization.py` | Preserve stable legacy pooling and its tests. Implement direct edge-to-donor gathering in v2 instead of secretly changing old B/W semantics. |
| `decoding.py`, `response.py` | Reuse compact local packing and affine algebra; implement headwise v2 reading and the new C contract. Keep dense-K export optional. |
| Thermal `content_hypergraph.py`, `native_observations.py`, `flow_curl.py` | Reuse native observations and complete fresh model; pass explicit geometry descriptors/lengths to v2. Do not substitute new physical targets or a pretrained flow provider. |
| Wind `content_hypergraph.py`, `training/content_hypergraph_task.py` | Retain corrected v2 residual transform, native aligned frame, E64, five-role sampler and component scales. Provide geometry in core values without changing reference data. |
| Thermal `training/content_hypergraph_task.py` | Retain target separation, TRAIN-only scales, response addendum and exact valid denominators. New core/model dispatch only. |
| `src/honf_runtime/content_hypergraph.py` | Strict, self-contained versioned loading of new constructor and adapter state; do not refit transforms while loading. |
| `src/honf_runtime/unified_training.py` | Reuse TrainingEngine, case/epoch sampling, named moments, epoch-safe stop, checkpoint cadence and run ownership. No new training engine. |
| `tools/content_hypergraph_train.py` | Add explicit v2 family/protocol support and a v2 follow-up qualification schema. Do not weaken historical v1 qualifiers or import-source resume checks. |
| Evaluation, benchmark, inspect and response tools | Extend version dispatch and actual head/edge trace support. Reuse native panels and reference qualification rather than copy entire tools. |

### 7.2 Implementation stages

**Stage A — arrival and focused diagnosis.** Freeze current branch/source/data bindings, protect historical roots, identify any active work, inspect native value/score/response paths, and run Section2's compact diagnostics. Inventory corrected first-round baseline artifacts and the exact six-edge/source/adapter configurations. No optimizer portfolio.

**Stage B — working v2 model.** Implement shared spatial encoding, staged attention and v2 direct read first; then the content bank and factorized organized decoder. Establish one complete native forward/backward on each dataset before adding performance options. Instantiate all trainable modules before optimizer grouping; no lazy parameters appearing after the first step.

**Stage C — correctness and short learnability.** Use the focused tests in Section8 and a bounded four-case TRAIN fit per dataset. Diagnose gross errors before expensive maturation. Fix actual bugs and record them; do not turn a toy-fixture objective into the research objective.

**Stage D — primary segmented fits.** Run A2-D/A2-H6 under Section9. Capture interpretation and costs at the actual selected and literal checkpoints, with enough reserve for full1000 follow-ups.

**Stage E — one optional remedy and full-TRAIN decision.** A substantive revision is allowed only when it targets a documented bottleneck. Apply the predeclared Section10 decision independently to each dataset. Qualified organized models proceed to fresh fullTRAIN1000; retain negative results and stop otherwise.

## 8. Correctness and efficiency that protect the experiment

### 8.1 Compact required correctness suite

Retain existing family/native/loader regression tests and add focused tests in the following groups. These are software obligations, not a request for hundreds of new physical evaluations.

1. **Geometry and attention:** constant-donor-value identity; value-geometry injection changes the relevant fixture; signed-linear bias cancellation; explicit versus SDPA normalized attention; empty types/M=1, masks, self-exclusion, head scaling and measure refinement; staged routes read the intended updated tensors.
2. **Content and response:** actual W sums and typed supports; extreme score/measure finite forward and backward; all valid contents remain numerically well-defined; C matches the declared W-coupled equation; factorized versus explicit small-K reference; T(0), heat-null flow, affine increments and adjoint identity.
3. **Native boundaries:** actual HDF5/CFD input transform oracle; Wind zero residual gives its TRAIN height profile without an extra mean or rotation; Thermal one-field native interpolation/curl/q rules; source permutation/padding and source-column correspondence; moved geometry rebuilds relevant masks and interpolation metadata.
4. **Reuse and training:** scene/query/weight/buffer/normalizer staleness; query-subset/tile consistency; no learned-context persistence across steps; all intended parameters receive live task gradients; one actual accumulated macro-update and checkpoint resume preserve the declared data/objective contract.

Use the already declared v1 numerical envelopes as the starting reference: FP64 algebra atol1e-9/rtol1e-7; FP32 primary outputs in TRAIN-scaled space atol3e-6/rtol3e-5; scaled derivatives atol1e-5/rtol1e-4. Derived observation bounds propagate primary error through the actual native linear operator. Do not retrospectively enlarge a tolerance to turn a failed optimized backend into a pass; retain the reference backend. Do not demand bitwise equality between independently trained architectures.

A large negative physical temperature, incorrect wake or wrong finite response is a model-fidelity result, not something cured by a looser arithmetic tolerance. Conversely, tiny backend roundoff does not justify declaring a physically inaccurate model accurate.

### 8.2 Efficiency requirements

The rough preparation complexity per cycle is

\[
 O\bigl((M^2+2ME+E^2)d+(M+E)d^2\bigr),
\]

plus fixed two-pass H(M+E) content reads and small H² interaction. Ordinary organized query work is O(QH d_read) plus compact local pairs and tokenwise heads. Nonlinear relative score features add low-width scene-pair work; record their real cost. These formulas are not a latency prediction.

Keep Q out of scene contextualization and edge preparation. Prepare raw geometry and every donor-side projection at its correct lifetime. Do not rerun the two cycles for each output role, Thermal interpolation neighbor, surface angle, query chunk or heating action. Native receiver unioning should consolidate all requested observation roles before the neural read, as in the existing adapter.

Preserve public ownership validation, but do it once per complete prepared request before internal tile dispatch. Do not disable checks to win a benchmark, repeatedly hash full model tensors in the inner loop, call `.item()` for each edge, or introduce per-query Python loops. Local gather/scatter remains packed before nonlinear evaluation. Sequential attention must not accidentally reuse projections from an earlier state.

Use FP32 reference training initially. A single optional pre-seal engineering choice may qualify BF16 autocast for dense linear/attention work while retaining stable FP32/FP64 reductions, physical conversion and derivative-sensitive observation work. It must be common within a comparison and pass the actual native objective/backward checks, not only inference. No precision/hardware sweep is authorized. PyTorch2.6 SDPA can select different backends under different masks, dtypes and shapes; record actual dispatch rather than claiming FlashAttention from the API name. Keep geometry and measure biases even when they prevent a fused kernel. [P5]

Tune microbatch packing, not effective batch, on at most three disposable settings. Hold cases, native queries and loss denominators fixed. Prefer higher useful query/epoch throughput with a safety margin; do not spend time maximizing VRAM occupancy itself.

Measure fixed **B8** forward/backward at Q1024/4096/8192 for each selected new candidate. Record requested Q, unique native Q, Thermal union rows, local pairs, active/padded edge or source rows, allocated/reserved memory and common resident baseline. An OOM is an OOM, not permission to silently substitute B1. A target-free Q32768 stress is optional only after those three points; repeated Wind cell IDs are not additional physical evidence.

Report complete fresh-scene native inference and complete ordinary training epochs. Prepared decode and cached-control application are additional scopes, not sums that reconstruct a complete median. Include cold Wind provider/catalogue setup and CPU output conversion separately and in total campaign accounting. Keep dataset membership/precision/device/output scope matched for comparative timing; do not benchmark two contending jobs against an idle historical run.

### 8.3 Sparse execution remains a small, optional endpoint test

Only after full-access A2-H6 has useful fields, one frozen top-k versus full comparison may use k=3 and H=6. Score all edges cheaply, form the union selected by the feature heads, gather it, and only then execute the expensive message reader. Count the **union**, which can still equal six even when each head selects three. Do not evaluate all messages and mask them afterward while calling that sparse execution.

Measure both distortion from the same full model and error against the existing physical reference, plus complete latency. A full reroute is the authoritative geometry-derivative path; fixed-route derivatives do not cover support switches. No sparsity training, adaptive-K controller, repeated threshold search or mandatory sparse promotion follows. With only six edges, full batched execution may remain faster; report that and keep it as the default.

## 9. Segmented training: sufficient maturation without a model portfolio

### 9.1 Fixed protocol

| Setting | Thermal | Wind |
|---|---|---|
| Primary population | Existing fixed25 TRAIN150 / exposed DEV22 | Existing fixed24 TRAIN72 / exposed DEV24, 24/8 layouts ×3 directions |
| Environment / native frame | E192, physical native x–y | E64, already wind-aligned native x–y–z in D |
| Primary queries | Q1024 fluid plus unchanged native auxiliary sampling | Q4096 TOTAL across five roles |
| Effective batch / initial packing | 48 /24 | 24 /12 |
| Seed | 0 | 42 |
| New primary fits | A2-D and A2-H6 | A2-D and A2-H6 |
| Horizon | 2500 actual development epochs per fit | Same |

Use the identical native query namespaces and fixed TRAIN transforms across each pair, and where compatible with the first-round baseline. Preserve original TRAIN/DEV separation and the known Wind M18 extrapolation; do not silently add missing categories after seeing errors. Direction rows from one layout are correlated.

Keep the first-round physical objective unchanged to make this an architectural study: Thermal four flow roles weight1/8 each; three temperature roles weight1/6 each; q-proxy0.05; the original-TRAIN response addendum0.1 at the final macro-update once per epoch; no new all-source operator residual. Wind retains its component-balanced native five-role objective. Do not add physics penalties, teacher distillation, occupancy losses or multi-task balancing strategies in this round. Correct the bias/norm optimizer exclusion declaration in every fresh recipe without relabelling the old records.

Use the existing AdamW parameters, clipping1, 20-epoch warmup, peak3e-4, hold through1000 and cosine to3e-6 at2500; weight decay Thermal1e-4/Wind1e-5 with explicit exclusions. Case visits, updates, effective batch and query exposure—not just epochs—must be logged. Full native endpoint metrics and sampled monitor scores remain distinct.

Review at100/500/1000 and complete healthy primary pairs to2500. Save milestones/latest/best and update plots every100 epochs, plus an actual requested safe stop. Do not stop at100 because mature classics are better. Continue while fitting is healthy and the study remains within budget; a failing numerical law, target leak or severe representational collapse requires diagnosis before continuation.

Use the common field-only saved-checkpoint selector. Report literal2500 first and selected aliases separately, with actual ages. Do not choose a checkpoint from the subsequently opened saved physical-response audit.

### 9.2 Comparison and limited localization

Four primary fresh runs are mandatory when healthy and feasible. First-round A-D/A-H6 and the reader128 revision are frozen references, not refitted teachers. Reuse their exact native outputs when panel/transform/source bindings match; otherwise make a clearly labelled fresh checkpoint replay without changing their physical law.

An optional **fusion-only screen** is permitted for each dataset: attach the new contextualizer to the unchanged v1 H6 decoder/local width64 and train fresh to500 under the same schedule. Compare at literal500 with the saved v1 H6 and primary A2-H6. This separates whether the new fusion package already helps from whether the content/readout replacement is needed. It does not isolate scheduling from spatial encoding, and it is not a mature500 result comparable to2500. At most two such screens; do not extend them automatically or make a full factorial study.

One substantive revision per dataset is allowed, two total, when a specific failure is diagnosed. Prefer a shared-core remedy when both datasets exhibit it. Examples are correcting excessive token homogenization, adjusting response capacity after measuring an actual far-operator rank bottleneck, or repairing the local/global division after native error localization. Changing many unrelated losses, frequencies, widths and support rules simultaneously is not one coherent revision.

A revised architecture receives a new identity and fresh weights/moments, unless a narrowly justified shape-compatible continuation is explicitly recorded with its inherited age. Full-data qualification requires an actual2500-epoch mature candidate, not2500 obtained by adding incompatible fit ages. Preserve parents and failed evidence. A purely numerical correctness repair is separately labelled; do not use it to hide an architectural or objective change.

## 10. Conditional fresh full-TRAIN1000: execute when measured improvement is real

The user authorizes full-data potential checks after improvement, not unconditional full-data spending on another weak pair. Apply the following **research criteria**, fixed before new DEV outcomes, independently to Thermal and Wind. They are not statistical significance tests or application certification.

### 10.1 Three decision questions

**Correct and genuinely organized?** The native software contract passes; all intended fresh heads/context are trained; source identities and physical laws hold; actual edge contents and local paths are traceable and have measured output effects. No wide hidden bypass has turned A2-H6 into A2-D. Neither all-six occupancy nor a sparse latency win is required.

**Meaningfully better fields?** At literal2500 on the unchanged native DEV panel, require:

- Thermal A2-H6 fluid-T RMSE at least10% lower than first-round A-D e2500, and its broad field score no worse than that A-D. Its fluid-T and broad-score ratios to equally trained A2-D should each be at most1.15.
- Wind A2-H6's geometric mean of the six near/downstream component-RMSE ratios to first-round A-D e2500 at most0.90, its broad field score no worse, and near/downstream vector means no more than15% above equally trained A2-D.

Use the actual coherent A-D checkpoint, not a fictional per-channel oracle made from multiple models. Inspect all role means and descriptive tails: a >25% increase in a resolved nonpriority role mean or priority p90 against first-round A-D blocks the automatic follow-up unless it is an identified evaluator defect repaired before resealing. Numerical floors are inherited TRAIN-scale floors, not fitted to errors. Keep selected-age and module-count/layout distributions beside this literal-age decision; do not discard a difficult case to pass.

**Still efficient enough to explore?** Complete high-M inference is at most2 times the first-round A-H6 on a current same-device/same-output replay, fixed-B scaling remains workable, and actual training plus full-run forecast fits the resource reserve. A2-H6 may be slower than A2-D and still qualify if it purchases useful field quality and an explicit organized representation. These ceilings are engineering guardrails, not a claim of optimal efficiency.

Failure is reported with the specific next design question. Do not respond by automatically lowering a bound, training more historical baselines, or launching5000. A very promising near-miss can be recommended to the user with a prepared inert recipe, but it is not automatically authorized.

### 10.2 Full-data execution

For each qualified dataset, launch **one selected organized candidate**, not both decoder arms. Thermal uses original TRAIN600 and exposed canonical89; Wind original TRAIN420 and exposed VALID90, with Wind TEST targets locked. Refit normalizers, profile and any loss scales only on the appropriate full TRAIN inventory. Initialize weights and AdamW moments fresh; a segmented checkpoint is not a full-data resume state.

Preserve the selected architecture, native law, primary Q, objective, effective batch, precision and optimizer recipe. Print and retain the development-to-full diff. A change in initialization, feature law, H/rank, local radius, geometry encoding or loss is a new candidate and cannot silently receive the development qualification.

Use a prospective5000 schedule with20 warmup epochs and hold through2000, but **enforce stop_after=1000 before model construction/run allocation**. Save every100 epochs and a normal latest/selected state. There is no automatic continuation. Create tested, inert future5000 instructions only as a later user-decision artifact, not an active job.

The new qualification schema must bind actual v2 architecture/configuration hashes, physical output law, development checkpoint and baseline identities, numerical/field/cost evidence, data partitions, forecast and stop. Do not simply set four booleans in the old v1 schema or weaken its thresholds globally.

### 10.3 Fair reference boards

Use three separate boards:

1. **Matched development:** A2-D versus A2-H6 at literal2500; first-round same-cohort attention references have their exact historical recipe/source qualifiers stated. This is the primary internal design comparison.
2. **Full-data early-age potential:** new full1000 versus actual saved classic1000 checkpoints on common native receivers. Print updates, case/query visits, batch, loss, environment catalogue, pretraining/frozen modules and LR schedule. Equal epoch alone is not equal effort.
3. **Mature practical distance:** native Thermal1404/1804 e5000 and Wind2102 e2440/2103 e2475, plus relevant efficient retained models where loader/panel bindings are already available. This is a deployment target, not an architecture ablation or a requirement to beat every channel.

Retain native normalizers and E catalogues for every reference. Do not feed classics v2 inputs, substitute E64 for their E512, add the Wind absolute-velocity mean, or denormalize twice. Reuse the now-qualified Thermal public CPU board where exact bindings agree; there is no need to spend another round repairing the already resolved raw-input comparison error. Missing exact-age checkpoints remain missing; no interpolation or guessed continuation.

## 11. Organizer meaning, physical response and readable evidence

### 11.1 What to measure

On all exposed DEV scenes retain a compact summary of typed gather concentration, content similarity, receiver-head access and actual work. On the fixed low/high-M representatives retain actual numerical traces with all source IDs and coordinates. Use the same representative scenes as the first round unless a source is unavailable; replacement is input-defined and documented, never error-selected.

For Thermal show actual W^M, signed C, source-indexed K on a small declared panel, and the executed edge temperature terms. A C singular-value spectrum is factorization-coordinate-dependent; also report the far K spectrum or head-weighted executed response at the same inputs. Do not interpret rescaling between Phi and C as learned physical strength. Distinguish far-rank capacity, actual numerical rank, local contributions and independently validated excitation rank.

For Wind show W^M and W^E, content similarity, feature-head accesses and measured physical outputs after interventions. Latent vector norms are not additive velocity effects. Geometry-aligned donor sets need not be exclusively local; all upstream context paths remain disclosed.

Use four same-weight conditions on the fixed representatives: normal, uniform receiver access, one input-declared content removed without renormalization, and local branch zeroed. Add at most one gather-content reassignment control with fixed physical inputs and normalization semantics. Restore exactly under the same backend. These interventions establish model reliance; compare independently trained A2-D/A2-H6 for added representation value. Do not call a large ablation response a proof of useful grouping.

No occupancy or entropy threshold is a success criterion. A useful few-content solution is acceptable; several nominal edges whose contents are identical and whose coefficients die is a diagnostic warning, especially when field accuracy also suffers.

### 11.2 Keep physical response checks bounded

Use the existing saved0291/0294/0687 heat-transfer families for the selected Thermal model: three independent layout/direction families, six signed changes, not six independent forcing bases. Never synthesize the missing0277 baseline. Preserve precise factorized increments and distinguish them from FP32 endpoint subtraction. Module-peak change is max T at one endpoint minus max T at the other, not max of the temperature increment.

Run one smooth input-derivative fixture per dataset and retain stale-state rejection. No new Wind geometry labels, new thermal solve, control optimizer, inverse generator or fabricated causal source truth. Ledger stays326/326. Physical response misses are clearly reported but do not trigger a second unbounded inverse-validation campaign or rewrite the field selector.

### 11.3 Report format: six figure families at most

Open with a plain-language answer to: **Are the fields better? What information now reaches the receiver? What does a complete call cost? Should each full1000 candidate be extended later?** Put native units and literal checkpoint age beside each number.

Use these six figure families, keeping each page readable rather than packing dozens of tiny panels:

1. Physical learning curves and fixed-weight TRAIN/DEV at the same ages, with updates/queries and measured train-loop time in clearly separated axes.
2. Thermal reference/predicted T and v plus signed residuals on identical native grids; one surface/material detail with quantitative RMSE and relative L2.
3. Wind actual native hub planes and2D/4D/8D cuts, showing Ux/Uy/Uz separately; no point-cloud projection presented as a smooth CFD slice.
4. The same physical scene's M/E -> contents -> three receivers, with actual W, content similarity, head access and output interventions. Show first-round and v2 semantics honestly rather than relabelled B heatmaps.
5. Complete cost and fixed-B query-density memory curves, plus accuracy-versus-complete-cost; cached decoding/control reuse is separate from first use.
6. Full1000 reference/potential board when launched, and a compact saved-response panel. Clearly mark a dataset not launched rather than inventing a matched full-data pair.

Use shared physical and signed-error scales, masks and coordinate frames within a comparison. Show unclipped metric values even where display limits are fixed. Every graph identifies which lines are omitted for readability and which paths actually execute. Preserve exact numeric arrays and a short evidence index, but keep long failure chronology and exhaustive provenance out of the main narrative.

## 12. Delivery, storage and stopping rule

Before new caches/checkpoints, inspect free disk/RAM and declare budgets. Reuse existing read-only geometry catalogues keyed by exact native inputs. Do not duplicate them for A2-D/A2-H6, selected checkpoints or every evaluation. A RAM limit is not permission to write a new disk cache. With the previous final capacity around100 GiB on `/data` and36 GiB on `/home`, a reasonable initial round ceiling is12 GiB new `/data` and2 GiB `/home` with20 GiB minimum free headroom, adjusted downward when actual arrival space requires it. Include revisions and both full1000 checkpoint sets in the estimate before launching.

Retain scientific milestones/latest/best, native metric arrays, exact masks/receivers, compact traces, necessary failure evidence and current figure masters. Clean only verified round-owned disposable files after checking active processes, symlinks/hardlinks and report dependencies. Never delete source datasets, protected checkpoints, required physical references, unrelated jobs, or the sole copy of a reusable catalogue. Preserve pre-existing absences as absences rather than pretending to restore them.

Deliver durable v2 core/adapters/configs/tests, tested start/resume/evaluate commands, the main report and evidence index, full1000 status per dataset, and prepared but unlaunched5000 continuation notes. Report actual time, updates, errors, costs, storage removed/retained and remaining capacity. Separate code completion, scientific improvement and full-data potential; a passed test suite is not a model-fidelity result.

Follow AGENTS.md: unwrapped prose paragraphs in Markdown, local ignored numerical/visual artifacts, full outgoing-history audit and artifact hook, commit and push durable changes on the non-default branch, then verify local/remote tips. Do not create another provenance system or make cleanup receipts the scientific centerpiece.

**Stopping rule:** finish the mature primary comparisons and qualified full1000 follow-ups within the envelope; use no more than the stated screens/revisions. Close with one preferred organized model or one precise remaining limitation per dataset. Do not keep modifying a model after its final report merely to turn every line green. The intended success is a stronger, explainable and inexpensive information path that is worth another research round—not a claim that all future inverse-design requirements have been solved.

## References and evidence anchors

**Repository evidence.** These identifiers refer to the reviewed snapshot, not to files that must be fetched repeatedly during every experiment.

- **[R1]** `HONF_Proj/docs/reports/HONF_Content_Hypergraph_Oct13_Report.md`, sections on implementation, native accuracy, organization, complete costs, conditional full-TRAIN, correctness and closeout.
- **[R2]** `HONF_Proj/docs/reports/HONF_Content_Hypergraph_Oct13_Evidence_Index.md`, completed run/checkpoint identities, corrected physical adapters, native boards and saved numerical source amendments.
- **[R3]** `HONF_Proj/docs/guides/_bk/20261029_Guide_Consolidation/02_Attention_and_Unified_Routing_Lineages/Content_Hypergraph_Family.md`; shared package `attention.py`, `model.py`, `organization.py`, `decoding.py`; Thermal/Wind `content_hypergraph.py`; `tools/content_hypergraph_train.py`, all at commit `852f995403a91d8feb0e5df94f3022c7b1f3467b`.
- **[R4]** `HONF_Redesign_Oct13.md` and `HONF_Redesign_Implementation_and_Evaluation_Oct13.md`, first-round motivation and choices. This round's changed attention/pooling/response equations are explicit amendments, not silent compatibility edits.

**External method context.** The proposed combination is a HONF hypothesis, not a published architecture with guaranteed results on these datasets. These sources motivate individual design principles only.

- **[P1]** Henry, Dachapally, Pawar and Chen, “Query-Key Normalization for Transformers,” Findings of EMNLP2020, pp.4246–4253, DOI10.18653/v1/2020.findings-emnlp.379. Supports normalized query/key magnitudes and learned scaling as a way to reduce arbitrary softmax saturation; its reported experiments are language translation, not HONF.
- **[P2]** Hao et al., “GNOT: A General Neural Operator Transformer for Operator Learning,” ICML2023, PMLR202:12556–12569. Relevant to heterogeneous inputs and geometry-aware operator organization; it does not establish that every geometry-biased attention implementation is efficient or expressive enough.
- **[P3]** Wu et al., “Transolver: A Fast Transformer Solver for PDEs on General Geometries,” ICML2024, PMLR235:53681–53705. Relevant to learning reusable content tokens for physical fields; it is not evidence that six HONF edges have independently identified physical meanings.
- **[P4]** Tancik et al., “Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains,” NeurIPS2020. Motivates a small multiscale coordinate representation rather than a raw-coordinate-only query MLP; does not guarantee wake or interface accuracy.
- **[P5]** PyTorch2.6 official documentation, `torch.nn.functional.scaled_dot_product_attention`: explicit scale, additive float masks, dropout behavior, backend selection and numerical differences. Actual dispatch and native derivative support must be measured in the installed environment.
