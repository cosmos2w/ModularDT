# HONF next phase: measured sparse execution without losing the learned field

## Decision and scope

**Develop the next scientific candidate from Run 2001 (module hubs). Apply equation-preserving executor improvements to both 2001 and 2101 checkpoints, but do not open another mean-shift training track in this goal.** Leave already-authorized parent continuations untouched. The proposed new candidate is **Run 2002**, subject to the existing run allocator confirming that the ID is unused.

This is one coherent programme with three implementation sections, not three new model families:

1. Diagnose the projection and remove redundant route compilation, using frozen checkpoints and exact arithmetic references.
2. Implement and actually benchmark an optional fused GPU reader, including its backward path.
3. Run one bounded, warm-started, execution-cost-aware sparsification experiment for at most **500 additional training epochs**. This is not permission to extend either parent, launch a sweep, or run another 2,500/5,000 epochs.

The equation-preserving work can succeed even if the sparsification hypothesis fails. Conversely, a sparse-looking routing plot is not a successful outcome without measured savings and acceptable field/interface errors.

**Planning basis:** `cosmos2w/ModularDT`, branch `agent/honf-core-next`, commit `3943710c0df76c8380feb6b98d0cf7b9842c0594`. Inspect the actual working head before editing; preserve subsequent compatible work rather than resetting to this commit. The files reviewed for this plan are listed in Section 13.

**Input reports:** `HONF_Routing_Epoch2500_Comparative_Evaluation.md`, `merged_reduction_summary.md`, and the comparative routing board `Hyper-Hub-Shift.png`. Reported experiments below are evidence supplied by those reports; derivations and proposed experiments are explicitly distinguished. No GPU experiment was independently rerun in preparing this plan.

---

## 1. What the evidence establishes

### 1.1 Accuracy is promising, but checkpoint policies must remain separate

| Policy | Actual epoch | Fluid pooled relative L2 | Case p95 | Worst case |
|---|---:|---:|---:|---:|
| Legacy 1401, exact | 2500 | 0.045285 | 0.062338 | 0.069210 |
| Dense 1804, exact | 2500 | 0.048835 | 0.066788 | 0.086315 |
| Module hubs 2001, exact | 2500 | 0.044960 | 0.069128 | 0.081208 |
| Module hubs 2001, selected snapshot | 2425 | 0.036170 | 0.061395 | 0.077733 |
| Mean shift 2101, exact | 2500 | 0.046395 | 0.072798 | 0.087336 |
| Mean shift 2101, selected snapshot | **2732** | 0.035851 | 0.060861 | 0.071849 |

The last row is **not a best-through-2500 result**. Parent best-through-budget checkpoints are not supplied for a matched selected-policy comparison. The exact-2500 evidence supports a modest pooled advantage for 2001, not a large universal victory over Legacy; both routed endpoints have worse p95 than Legacy. The selected snapshots demonstrate representational potential, not equal-budget statistical superiority.

Thermal interfaces remain important: exact 2001 final-port temperature relative L2 is 0.100844, compared with 0.072683 for Legacy and 0.047728 for Dense. Its selected snapshot improves that to 0.068158. Preserve these quantities rather than optimizing pooled fluid error alone.

### 1.2 The executed graph remains nearly dense

At the reported endpoint anchors:

- 2001 module unique/dense ratio is approximately 0.996; 2101 is 1.
- Environment unique/dense and complete-QE fractions are 1 for both.
- Mean module duplicate ratios are about 1.635 versus 2.737.
- Mean environment duplicate ratios are about 2.975 versus 3.857.
- Every occupied hub is active at the audited query projection.

The source-incidence matrices contain zeros, but those zeros disappear as an execution advantage when a query reads the union of every occupied hub.

The reports contain a counter-scope issue worth resolving in the next reducer: one 2001 module entry is expressed as 219/224, which is compatible with 32 queries and 7 modules, whereas surrounding text describes full-grid re-audits. Also distinguish *individual complete receiver rows* from *complete case-by-reader-chunk dispatches* in training counters such as 930/1104. Read the underlying metadata and state the actual denominator; do not silently relabel it or extrapolate a small diagnostic to the entire grid. This does not invalidate the observed dense selected-pair panels.

### 1.3 Current speed is not competitive with Dense

The controlled two-anchor inference test uses 8,192 queries, receiver chunk 2,048, two warmups, and five repetitions on the same GPU:

| Exact-2500 model | Full forward | Prepared decode | Incremental peak allocated |
|---|---:|---:|---:|
| 1401 | 26.00 ms | 6.52 ms | 362.4 MiB |
| 1804 | 35.44 ms | 13.01 ms | 430.9 MiB |
| 2001 | 77.68 ms | 34.78 ms | 317.2 MiB |
| 2101 | 79.24 ms | 35.95 ms | 315.8 MiB |

Do not mix these warmed measurements with the separate 90-case evaluator timing at receiver chunk 128 and no explicit warmups. Lower activation allocation does not imply fewer parameters or lower latency. Total measured allocation baselines also vary between policies; report baseline, incremental peak, total peak, and reserved memory separately in fresh isolated-process checks.

### 1.4 Why focus new work on 2001

Module hubs have simpler preparation, lower path duplication, slightly better exact-2500 global error, and a real although limited partial-support regime in mixed larger-module training batches. Mean shift has not demonstrated additional source-pair sparsity. Its selected advantage is small and comes from epoch 2732 rather than 2425.

This is a resource-allocation decision, not a claim that mean shift is intrinsically inferior. Keep its code and checkpoints, and evaluate shared executor changes on it without retraining.

---

## 2. Mathematical diagnosis: why zero incidences did not reduce fine work

### 2.1 Symbols and current operator

For one physical preparation pass, let:

- `B`: number of cases in the batch.
- `M`: active module count; padded storage widths are not physical counts.
- `E`: environmental source count.
- `Q`: receiver count for this pass/read.
- `K`: candidate count; normally one candidate per active module.
- `H=256`: fine feature width; `D=32`: routing descriptor width.
- `t` in `{M,E}`: source type, with source count `N_t`.
- `omega_i^t > 0`: normalized measure over valid sources; sums to one for a nonempty source bank.
- `A_ik^t >= 0`: source-to-hub membership, each valid source row summing to one.
- `mu_k^t = sum_i omega_i^t A_ik^t`: induced hub measure.
- `z_qk^t`: query affinity before temperature division.
- `d_qk^t`: query density, **not a probability**.
- `alpha_qk^t = mu_k^t d_qk^t`: normalized query-to-hub probability.
- `Pi_qi^t`: normalized query-to-fine-source prior.

The implemented normalizers are

\[
A_{i:}^{t}=\operatorname{sparsemax}(\ell_{i:}^{t}/T),\qquad
\mu_k^{t}=\sum_i\omega_i^{t}A_{ik}^{t},
\]

\[
d_{qk}^{t}=\left[z_{qk}^{t}/T-\tau_q^{t}\right]_+,
\qquad \sum_k\mu_k^{t}d_{qk}^{t}=1,
\]

\[
\boxed{\Pi_{qi}^{t}=\omega_i^{t}\sum_k A_{ik}^{t}d_{qk}^{t}.}
\tag{1}
\]

No pooled hub value is read as a physical response. Existing Dense-like fine preparation and the separate module/environment source states are retained.

For modules, suppressing batch indices,

\[
c_M(q)=W_o\left[\frac{M}{1+M}\sum_i\Pi^M_{qi}\,\psi_\theta(q,i)\right]+b_o.
\tag{2}
\]

For one environmental head,

\[
s_{qj}=Q_q^\top K_j/\sqrt{d_h}+b_\theta(q-y_j),
\]

\[
a_{qj}=\frac{\Pi^E_{qj}\exp s_{qj}}{\sum_l\Pi^E_{ql}\exp s_{ql}},
\qquad o_q=\sum_j a_{qj}V_j.
\tag{3}
\]

The existing output projection, global/coarse context, local correction, physical port heads, local surrogate, and refinement remain unchanged.

### 2.2 An all-active query makes the induced graph dense

If every occupied hub satisfies `d_qk > 0`, each valid source has at least one positive membership into an occupied hub. Therefore

\[
\omega_i>0\quad\Longrightarrow\quad\Pi_{qi}>0.
\tag{4}
\]

No amount of nicer source-incidence clustering can make this receiver sparse until query support excludes some hubs. This is an exact consequence of nonnegative two-hop composition, not a plotting interpretation.

### 2.3 The query projection has a broad all-active regime

Set `u_qk = z_qk/T`. Since `sum_k mu_k = 1`, if all hubs are active,

\[
\tau_q=\bar u_{q,\mu}-1,\qquad
\bar u_{q,\mu}=\sum_k\mu_k u_{qk},
\]

\[
\boxed{d_{qk}=1+u_{qk}-\bar u_{q,\mu}.}
\tag{5}
\]

The exact all-active condition is

\[
\boxed{\bar u_{q,\mu}-\min_{k:\mu_k>0}u_{qk}<1.}
\tag{6}
\]

A logit range smaller than one is sufficient. In floating-point diagnostics use the actual occupied mass `S=sum(mu)` and `tau_all=(sum(mu*u)-1)/S`; do not assume its rounded sum is exactly one when classifying a boundary. For mathematically normalized unscaled affinities, all support persists when

\[
T>\bar z_{q,\mu}-z_{q,\min}.
\tag{7}
\]

With uniform hub masses, the query probability is equivalent to

\[
\alpha_{q:}=\operatorname{sparsemax}\left(z_{q:}/(KT)\right).
\tag{8}
\]

Thus the query-side projection is not simply ordinary sparsemax at the source-side temperature: uniform masses make its probability interpretation effectively `K` times softer. This follows from the intended source-measure formulation, not an implementation bug.

A toy example with `mu=(1/3,1/3,1/3)` and `z=(0.9,0.4,0)` gives:

| Query temperature | Query probability | Support |
|---|---|---:|
| 1 | (0.488889, 0.322222, 0.188889) | 3 |
| 0.5 | (0.644444, 0.311111, 0.044444) | 3 |
| 0.25 | (0.833333, 0.166667, 0) | 2 |
| 0.125 | (1, 0, 0) | 1 |

This illustrates a threshold, not a recommended physical temperature. These scalar results were checked independently; they are not checkpoint measurements.

Before changing the model, measure Eq. (6) and the content, geometric, and propensity contributions to its margin on actual checkpoints. The current score is

\[
\ell=2\,u^\top v-0.25\log\left(1+\|\Delta x/\ell_{\rm geom}\|^2\right)
+0.25\,p-R.
\tag{9}
\]

Descriptors have bounded norms. The ThermalChannel geometry scale is **four module radii per coordinate**, not the full domain extent. Do not silently replace that scale. `R` is currently an explicitly neutral resistance. The reports do not establish which score contribution dominates all-active behavior.

### 2.4 A structural limit: one hub cannot sparsify environmental reads

For `M=K=1`, each source has `A_i1=1`, `mu_1=1`, and `d_q1=1`. Hence

\[
\Pi_{qi}=\omega_i
\]

for every query and every finite positive temperature. Environmental support is necessarily complete. The same issue applies to a one-hub background case.

No competition penalty, temperature controller, or faster kernel can produce source-level sparsity in that configuration without changing the routing model. Report `M=1` separately; it remains in overall cost and accuracy tables. Do not demand a sparse-support target that the architecture cannot express. Do not add environmental anchors in this phase merely to hide this limitation.

### 2.5 Why a global active-hub penalty is the wrong primary objective

With every source assigned to one hub, every query still reads every source. A penalty on the number of active hubs would reward precisely that configuration even though it saves no fine work.

Similarly, maximizing `||d||^2` is not an acceptable source-count objective. If a query uses a hub with mass `mu`, its density can be `1/mu`; the squared norm then rewards tiny-mass concentration rather than low executed cost. The probability is `alpha=mu*d`, not `d`.

The report's row-concentration/load-balance proposal is a research suggestion, not a demonstrated solution. This plan instead uses one cost term acting on the induced fine-source support. Do not add separate hub-count, entropy, balance, orthogonality, locality, or usage losses in this goal.

---

## 3. Equation-preserving executor: eliminate duplicate-path materialization

### 3.1 What is still wasteful at the pinned commit

`compile_two_hop_pairs_batched()` still:

1. expands integer `(q,k,i)` paths;
2. sorts/coalesces them into `(q,i)` keys;
3. evaluates bounded scalar `d @ A.T` products;
4. gathers selected priors;
5. sometimes scatters those priors straight back into a dense QE layout.

Complete QE is identified **after** the join. When queries activate every occupied hub, the complete answer was knowable before that expansion.

### 3.2 Add an exact pre-join completeness certificate

For each source type, define occupied hubs from the authoritative filtered source incidence. A sufficient certificate is:

\[
\forall k:\mu_k>0,\quad d_{qk}>0.
\tag{10}
\]

For such a receiver, compute Eq. (1) directly as a bounded scalar product and use an implicit complete source list. No raw paths, pair-key sort, coalescing, or COO-to-dense scatter is needed.

Eq. (10) is sufficient, not necessary. When it fails, compute the exact support union rather than assuming the row is partial.

For a genuinely single-candidate source bank, Eq. (1) reduces to `Pi=omega` independent of route logits. Add a tested single-candidate shortcut when detailed routing maps are not requested: retain live source measures and all fine physical computation, but avoid running a general route compiler for this constant routing function. Do not confuse one candidate with one selected hub in a multi-candidate bank; the latter still requires the learned routing and can depend on source assignments.

Empty module banks and invalid/zero-measure sources preserve historical behavior. Never evaluate a zero-prior logarithm just to discard it afterward.

### 3.3 For other rows, compile Boolean support, not all paths

Let

\[
U_{qk}=\mathbf1[d_{qk}>0],\qquad S_{ik}=\mathbf1[A_{ik}>0]\mathbf1[\omega_i>0].
\]

The exact support is

\[
C_{qi}=\bigvee_k(U_{qk}\wedge S_{ik}).
\tag{11}
\]

Implement a bounded row/block Boolean union. For the current small `K`, source and query hub masks can be packed into integer bitsets; an intersection is nonempty iff a pair is present. For larger `K`, use multiple words or a tiled hub scan. This is an encoding of learned hub membership, **not a geometric spatial hash or approximate nearest-neighbor rule**.

A simple PyTorch tiled Boolean implementation is the initial reference. A Triton support/prefix-compaction implementation is appropriate if measured compiler cost warrants it. Neither may create a global `[B,Q,K,N]` tensor. A tiled Boolean/bitset scan still examines candidate query-source pairs, and scalar `d @ A.T` still has `O(BQNK)` arithmetic. This bounds memory and removes multiplicity/sorting; it does not prove sublinear index construction. Count these scalar examinations separately from expensive fine reads. At large scales a compact inverted-index implementation may become preferable, but do not add another approximate router to hide this cost.

Do not infer support using a numerical threshold on tiny floating priors. Use the authoritative Boolean support; compute positive priors in the current scalar precision afterward. Underflow or nonfinite positive priors are numerical errors to diagnose, not permission to discard a path.

### 3.4 Use CSR-style rows where support is genuinely partial

The new runtime representation should distinguish:

- **complete rows:** implicit source range plus source validity;
- **partial rows:** receiver row pointers and unique source indices;
- **empty rows:** historical zero-context/output-bias semantics.

Do not populate three integer indices per pair for complete rows. Compact partial rows directly in receiver order so a global duplicate sort is unnecessary.

Retain rectangular batched products when they are the most efficient exact operation. Small padded `A`, `d`, source features, and scalar tiles are allowed. The target is eliminating wasteful pair-feature materialization, not banning every dense tensor.

Neither `torch_scatter.scatter` nor sorted COO scatter is inherently contention-free: GPU implementations use atomics. A row-owned CSR forward reducer avoids output collisions, but backward source gradients can still require atomics or an explicit second reduction. Preserve ordinary PyTorch as the readable reference; do not introduce a mandatory `torch_scatter` dependency without a measured advantage.

### 3.5 Bound scalar workspace independently of fine feature tiles

The initial scalar block budget remains 262,144 entries, about 2 MiB for one FP64 array. This is a workspace setting, not a sparsity target. The fine MLP tile size is independent.

Counter definitions must survive the rewrite. Compute a hypothetical raw-path count from per-hub source/query counts without expanding raw paths:

\[
P_{\rm raw}=\sum_k n^Q_k n^S_k.
\]

Report it as a logical count; actual raw-path workspace should now be zero for the new compiler.

### 3.6 Preserve the nonlinear QM network but factor its first affine evaluation

The current QM MLP starts with a linear map of concatenated module state, relative Fourier features, and global state. Partition the existing first-layer weight:

\[
t_{qi}=W_z\widetilde z_i+W_r\Phi(q-x_i)+W_g g+b.
\tag{12}
\]

Prepare `W_z z_i` and `W_g g` once per valid physical state, then compute only the relative term per retained pair. Apply the **same GELU and all subsequent layers after summation**.

This is exact affine reuse, not a low-rank pair-kernel replacement or early nonlinear pooling. Use views of the existing weight matrix; keep state keys, biases, optimizer ownership, feature order, Fourier convention, and dropout behavior unchanged. Checkpoint after compact input gathering, not after a large concatenated pair tensor has been stored.

Implement this bounded QM optimization in the new executor mode. Generalizing it to dense MM/ME/EM preparation is a later, separately measured optimization; do not rewrite all preparation in this goal.

---

## 4. Fused GPU work is a priority, with an explicit scope

### 4.1 Required prototype: selected-QE gather/content/normalization/value fusion

Implement an optional Triton QE reader supporting both implicit complete rows and CSR partial rows. It must have a working training backward, not just an inference demonstration.

First fuse:

- query/key/value loading for retained sources;
- query-key dot products;
- addition of learned geometry bias and log prior;
- numerically stable normalization;
- probability-weighted value accumulation.

Initially keep the small learned geometry-bias MLP in PyTorch, evaluating it only for retained pairs. Its `[I,heads]` scalar output is allowed; do not materialize `[I,H]` gathered Q/K/V or weighted values globally. This removes the main feature-width intermediates while preserving a straightforward gradient path to geometry and MLP parameters.

Then benchmark integrating Fourier features and the existing two-layer geometry-bias MLP into the same tile. This fusion is the next bounded kernel task, not a new architecture. Retain the split implementation if full fusion spills registers, reduces occupancy, or slows the complete workload. Document the actual fusion boundary; do not label a partial fusion as a fully fused physical operator.

The required deliverable is an executed fused-reader prototype, its backward validation, and honest full-model measurements. If an installed toolchain cannot execute Triton, record the actual error and finish the exact PyTorch implementation; do not migrate PyTorch/CUDA or promise an unexecuted kernel speedup. CUDA C++ is an alternative only when the installed environment supports it without a disruptive migration.

### 4.2 Streaming weighted attention

For one receiver/head and a tile of retained sources, define

\[
l_j=s_j+\log\Pi_j.
\]

Maintain running maximum `m`, denominator `Z`, and vector numerator `U`. For a new tile,

\[
m'=\max(m,\max_{j\in\text{tile}}l_j),
\]

\[
Z'=e^{m-m'}Z+\sum_{j\in\text{tile}}e^{l_j-m'},
\]

\[
U'=e^{m-m'}U+\sum_{j\in\text{tile}}e^{l_j-m'}V_j.
\tag{13}
\]

The result is `U/Z`. Handle an empty row separately; avoid `-inf - (-inf)`. Apply the output projection once after combining complete and partial rows.

This is mathematical equivalence, not bitwise equivalence to the old reduction order. The first parity target retains FP32 neural inputs and the existing FP64 prior/log-normalizer semantics. Do not silently enable TF32, BF16, FP16, approximate GELU, or fast-math changes. Actual register use and throughput on the installed GPU decide tile shape; the entire network cannot be assumed to fit on chip merely because a kernel is fused.

### 4.3 Backward is part of the design

For upstream vector `g` and output `o`, let

\[
\delta_j=a_j\,g^\top(V_j-o).
\]

In exact arithmetic,

\[
\frac{\partial L}{\partial s_j}=\delta_j,\qquad
\frac{\partial L}{\partial\Pi_j}=\delta_j/\Pi_j,\qquad
\frac{\partial L}{\partial V_j}=a_j g.
\tag{14}
\]

For tiny positive priors, retain the scalar precision and stable computations used by the reference. Algebraically equivalent forms of the ratio may be used after tiny-prior tests; do not introduce a minimum probability floor that changes the model.

Return gradients to Q, K, V, geometric bias, and live Pi. The PyTorch graph can then propagate bias gradients through Fourier features and the geometry MLP, and Pi gradients through A, d, omega, temperatures, descriptors, and positions. Source-gradient accumulation across receiver rows must be implemented deliberately; it is not automatically lock-free.

Save compact per-row normalizers and recompute local scores as needed, rather than saving all feature-width pair activations. A reference-recompute backward is useful for debugging but must not be advertised as a faster training kernel without measurement.

The first production use requires first-order coordinate gradients. If higher-order derivatives are not implemented, report that limitation and use the existing PyTorch reader for higher-order requests. Do not silently return incomplete gradients for inverse design or physical derivative calculations.

### 4.4 Avoid two false substitutions

1. Replacing the operator with standard SDPA does not preserve the nonlinear QM MLP or automatically fuse the learned QE geometry network and live prior algebra. Kernel eligibility must be checked on the actual dtype and mask, not inferred from the API name.
2. Fusing many scalar operations is not equivalent to proving lower full-forward latency. Measure routing, preparation, decode, backward, optimizer, and end-to-end execution separately.

---

## 5. Caching: reuse within a physical state, not across changed states

The pinned code already prepares source incidences and environmental K/V before receiver chunks. Do not claim that this existing reuse is newly introduced.

| Object | Safe reuse scope | Restriction |
|---|---|---|
| Static domain metadata and valid masks | Case/layout lifetime if genuinely unchanged | Recompute on layout/geometry changes |
| Geometry-only distances/features for module-tied centres | One forward layout; possibly fixed-layout inference | Keep coordinate gradients for layout optimization |
| Source descriptors, memberships A, hub masses mu, inverted/bitset support | One physical preparation state | Rebuild after module-state feedback or weight updates |
| Environmental K/V; QM source affine terms | One physical preparation state | Preserve autograd; refresh P0/P1/P2 as required |
| Query descriptors, query density, Pi, selected pairs | One receiver set and physical state | Do not reuse across changed queries or state |
| Mean-shift candidate coordinates | One source-state preparation | They depend on the updated contextual source states |
| Resistance field | Only if case supplies a valid fixed field | ThermalChannel currently supplies neutral resistance |

P0, P1, and P2 are not interchangeable caches. Geometry may be fixed while module states, environmental contextualization, hub descriptors, and memberships change.

Reusable scratch buffers must never overwrite tensors saved for backward. Use detached scratch only for metadata with no derivative, or save/recompute the required compact metadata. Do not detach learned values or mean-shift updates to save memory.

No global cache across optimizer steps, no spatial-hash nearest-neighbor approximation, no new cache-fingerprint system, and no per-step `empty_cache()` loop are part of this plan.

---

## 6. One scientific candidate: fine-pair-cost-aware module-hub routing

### 6.1 Preserve the physical computation

Run 2002 keeps the current module-hub generator, all module candidates, Dense fine preparation, fine source values, QM/QE networks, global/local paths, P0/P1/P2 coupling, losses on existing physical targets, and local surrogate.

The scientific additions are limited to:

- separate learnable positive source/query temperatures for each source type;
- one smooth surrogate for realized unique-pair cost.

No new value branch, expert bank, dynamic global K rule, mean-shift loop, hard top-C cap, clustering label, load-balance penalty, or fabricated barrier is introduced.

### 6.2 Typed temperatures learned in the same end-to-end objective

Use four scalar parameters:

\[
T_A^M=e^{\theta_A^M},\quad T_A^E=e^{\theta_A^E},\quad
T_Q^M=e^{\theta_Q^M},\quad T_Q^E=e^{\theta_Q^E}.
\tag{15}
\]

Initialize all `theta=0`, so every temperature starts at the historical value 1 and the warm-started predictor is unchanged before optimization. Exponentials are evaluated in the scalar route precision. Use existing nonfinite detection; do not add a topology thermostat, epoch schedule, or online heuristic controller.

Use `T_A^t` in ordinary source sparsemax and `T_Q^t` in source-measure query sparsemax. All four parameters update with the original end-to-end loss plus the single cost below. They are global learned scalars by type, not arbitrary per-query temperature networks.

Separate source and query scales are necessary to study the different threshold geometry in Eqs. (5)–(8). This does not guarantee support revival: sparsemax is piecewise differentiable and singleton regimes can have zero routing-logit gradients. Monitor rather than conceal that limitation.

### 6.3 Penalize induced fine-source participation, not hub count

Define the dimensionless relative source density without dividing by tiny omega:

\[
r^t_{qi}=\sum_k A^t_{ik}d^t_{qk},\qquad \Pi^t_{qi}=\omega_i^t r^t_{qi}.
\tag{16}
\]

For valid positive-measure sources, exact executed membership is `1[r>0]`. Use the smooth surrogate

\[
f_\epsilon(r)=\frac{r}{r+\epsilon},\qquad \epsilon=0.05
\tag{17}
\]

as an initial research setting. Epsilon defines **only a training surrogate**; inference still executes every strictly positive route. Do not drop probabilities below epsilon or claim that the surrogate equals nnz.

Let `p` index the actual P0/P1/P2 reads, `c_t > 0` be frozen measured marginal fine-read cost weights, and `V_{bpt}` be the valid receiver/source pairs. Use

\[
\boxed{
L_{\rm paircost}=
\frac{\sum_{b,p,t}c_t\sum_{(q,i)\in V_{bpt}} f_\epsilon(r^t_{bqi})}
{\sum_{b,p,t}c_t |V_{bpt}|}.
}
\tag{18}
\]

Use actual source counts and quadrature-valid masks. Per-type costs are counted separately even when their hubs overlap. The measure-normalized density prevents a uniformly weighted environmental source from appearing negligible merely because `E=192`.

Measure `c_M,c_E` with a small fixed-support reader benchmark and freeze their ratio for this experiment. They are cost estimates, not learned physical weights. If marginal estimates are unstable, use equal weights and label the objective explicitly as pair-count rather than wall-time cost. Do not run an online timing controller inside training.

The objective is

\[
\boxed{L=L_{\rm existing\ physical}+\lambda L_{\rm paircost}.}
\tag{19}
\]

No teacher-distillation loss is added. Parent prediction fidelity is audited in evaluation.

A universal super-hub that contains every source produces `r_i=1` and a high cost `1/(1+epsilon)`, not a low-cost solution. This improves the incentive compared with a hub-count penalty. It **does not prove the optimizer can escape** a completely saturated one-hub active set; report such saturation honestly.

### 6.4 Choose one conservative cost coefficient, not a sweep

On two fixed training batches at initialization, compute the original task-gradient norm and the cost-gradient norm over the shared router parameters (including the new temperatures). Choose and record one lambda giving an initial cost-gradient magnitude around **2%** of the task-gradient magnitude:

\[
\lambda=0.02\,
\frac{\|\nabla_{\theta_R} L_{\rm physical}\|}
{\|\nabla_{\theta_R}L_{\rm paircost}\|}.
\tag{20}
\]

Use a stable aggregate of the two batch norms. This is a one-time scale calibration, not a safety gate, optimizer modification, or convergence claim. If the cost gradient is zero/nonfinite, do not manufacture an enormous lambda: report the actual derivative failure, fix an implementation error if present, and treat an analytically saturated router as an unsuccessful sparsification starting point. An additional long run cannot fix an identically absent gradient by assertion.

Keep lambda fixed throughout the authorized trial. Save its value in the ordinary effective configuration. No automatic second lambda, restarted seed, penalty schedule, or rescue regularizer is authorized.

### 6.5 Why this is a hypothesis rather than guaranteed sparsity

Concave cost surrogates encourage concentration but may leave many tiny positive priors. Sparsemax can produce exact zeros, but it need not do so in the useful final union. Task gradients can also oppose pruning. Sparsemax singleton regions can stop changing local logits.

Report both Eq. (18) and actual nnz. A falling surrogate with unchanged `R_env=1` is **not success**. Do not silently threshold its tail.

No capacity ceiling is imposed on physically needed interactions. If accuracy-preserving sparsity cannot be achieved, keep the exact executor improvement and report the scientific result as negative.

### 6.6 Physical manifold claims remain bounded

The current ThermalChannel provider has seven known boundary descriptors and an identically zero additional resistance. There is no validated internal barrier field in the supplied dataset.

Do not replace it with an invented hard obstruction mask. A long range is not, by itself, evidence of a nonphysical interaction; different output channels and global boundary effects need not have the same support. A line crossing a solid is not a validated rule for deleting all thermal/pressure interaction.

Retain the case-owned resistance interface. Synthetic barrier fixtures may test software semantics, but do not count as physical validation. A physical manifold routing claim requires appropriate geometry/operator definitions and independent solver-labelled evidence, which are **Evidence Missing** in this round.

---

## 7. Bounded diagnosis before the new training experiment

Use the recorded 2001 epoch-2425 selected checkpoint as the preferred warm-start reference if it is still available. Verify actual epoch through checkpoint metadata, not the mutable filename. If it is unavailable, use the actual available explicit-epoch checkpoint and report the difference; never silently substitute a later best.

Execute these inexpensive checks on the five established anchors plus one `M=1` training case and a larger-module training batch:

1. Record `mu`, query probabilities, densities, source row supports, and the all-active margin in Eq. (6), separated by P0/P1/P2 and source type.
2. Decompose the affinity into content, geometry, propensity, and resistance. Measure across-hub spreads and gradient norms; do not blame a term from coefficient magnitude alone.
3. With weights frozen, keep source temperature at 1 and evaluate query temperatures `{1, 0.5, 0.25, 0.125}` on a small deterministic query set plus ports. This is one projection diagnosis, not four training runs. No temperature from it is silently selected for the formal model; learnable temperatures start at 1.
4. Distinguish support pruning from reweighting. On teacher-prepared states, apply each candidate support to the **teacher** fine weights/values. Separately evaluate the full new-temperature physical loop. The former isolates omitted information; the latter includes changed routing and feedback.
5. Execute actual gradient tests at a normal active set and around a support transition. Do not replace them with config inspection.

When only the teacher QE support is truncated, its discarded attention mass per head is

\[
m_q=\sum_{j\notin S_q}a_{qj}^{\rm teacher}.
\]

For nonempty retained support,

\[
o_{\rm full}=(1-m_q)o_{\rm retained}+m_qo_{\rm omitted},
\]

\[
o_{\rm full}-o_{\rm retained}=m_q(o_{\rm omitted}-o_{\rm retained}).
\tag{21}
\]

For QM, measure the actual omitted vector sum under teacher Pi; retained mass alone is not a field-error bound because messages can be large or cancel. These expensive complete-source audits run only on fixed diagnostic probes, not inside every training batch.

A poor frozen pruning result does not prove retraining cannot recover. It tells us where accuracy is at risk and which channels/ports deserve close inspection.

---

## 8. Concrete code ownership and configuration

### 8.1 Case-neutral implementation

Work primarily in:

- `src/honf_forward_core/interface_fields/routing_index/pair_join.py`: preserve reference compilers; add the union-free exact compiler/certificate.
- `.../routing_index/types.py`: a small complete/CSR row representation, without forcing historical `PackedPairs` callers to change.
- `.../routing_index/sparse_projection.py`: retain old projection functions; expose typed temperatures through call sites, not a rewritten historical sparsemax.
- `.../routed_pairwise.py`: integrate new compiler, QM affine preparation, exact mixed complete/partial reader, and optional fused QE backend.
- `.../routing_index/` or a compact sibling `kernels/` package: Triton kernels and their custom-autograd wrapper. CPU/Torch imports must still work when Triton is unavailable.
- `.../interface_fields/core.py`: pass loss-connected paircost statistics separately from detached reporting diagnostics, and retain correct physical-state lifetimes.
- `src/honf_forward_core/config.py`: additive validated options and historical defaults.

Do not modify the old `legacy_honf` or `dense_pairwise_field` equations, state keys, or default evaluation arithmetic. No architecture should instantiate new temperature parameters unless explicitly enabled.

Suggested options (final spelling must follow the maintained config conventions):

```text
routing.execution = "optimized_exact"          # historical default unchanged
                    | "compiled_exact"         # new union-free executor
routing.qe_backend = "torch" | "triton"         # optional, default torch
routing.sparsification.enabled = false          # old profiles unchanged
routing.sparsification.learn_typed_temperatures = false
routing.sparsification.relative_density_epsilon = 0.05
routing.sparsification.cost_weight = 0.0
routing.sparsification.module_pair_cost = 1.0
routing.sparsification.environment_pair_cost = 1.0
```

The new science profile enables only typed temperatures and one cost term. Avoid proliferating model families for kernel variants. Record the actual executed backend separately from the requested backend, including any explicit numerical-debug fallback.

One Run-2002 profile is enough. Exact executor benchmarks use overrides on existing checkpoints and do not need new managed training runs.

### 8.2 ThermalChannel-only work

Keep these case-specific:

- `Case_ThermalChannel/src/channelthermal/routing_geometry.py`: existing boundary descriptors and neutral resistance remain unchanged.
- Case loss assembly: add the generic scalar paircost to the existing supervised loss only for the new profile.
- Physical P0/P1/P2 metric interpretation, reference checkpoint selection, case strata, target normalization, near-interface sampling, and figures.
- Training workflow integration and saved milestone configuration.

Do not encode disk radius, `[12,6]` bounds, feature-column meanings, five output channels, or specific port variables in a generic kernel. Obtain shapes/scales from tensors and adapter metadata. Include one dimension-neutral/3-D numerical fixture without claiming 3-D physical validation.

### 8.3 Compatibility

The executor-only mode adds no trainable parameters. Its old-checkpoint strict loading should remain ordinary and exact in structure.

The science mode adds exactly four log-temperatures initialized to zero. Load the parent with an explicit documented initialization of those four new entries; do not use broad `strict=False` to ignore unrelated missing or unexpected keys.

Use the existing warm-start/partial-initialization mechanism where it fits. Preserve the parent's model and normalization. Restore the parent optimizer by existing supported named parameter mapping if available, with empty state only for the four new parameters. If that mapping is not supported safely, use a fresh optimizer with the unchanged configured policy and clearly label the reset; do not write a new checkpoint migration framework merely for this trial. This initialization choice must be explicit in the report and cannot be used to claim pure optimizer-resume equivalence.

---

## 9. Three implementation sections for one Goal-mode workflow

### Section A — Diagnose and establish the exact executor

Read the two reports and the pinned source. Reproduce one small and one large real optimizer batch and two real inference anchors. Resolve counter denominators. Execute the bounded projection diagnostics in Section 7.

Implement the completeness certificate, direct Boolean union/CSR compilation, and exact QM first-affine reuse. Compare old/new source masks, Pi, field/port outputs, losses, parameter gradients, and coordinate gradients on identical weights/inputs. Do not train to establish an exact rewrite.

Run existing relevant compatibility tests; do not regenerate historical golden data or approve differences by replacing it.

### Section B — Execute the fused GPU prototype

Implement the fused QE reader and its first-order backward. Benchmark the split-geometry and, if feasible within the same task, fused-geometry versions on actual shapes. Keep the smallest faster implementation; document negative measurements rather than retaining an unhelpful fused path as default.

Use explicit CPU/CUDA profiler scopes, including the new compiler. Inspect kernel launches, synchronizations, register spills, global memory traffic, gather materialization, and backward reductions. Final latency/memory tables must be unprofiled warmed repetitions because profiling itself adds overhead.

Both frozen parent checkpoints may be tested with the new executor. No new 2102 training run is authorized.

### Section C — One cost-aware sparsification trial

Create Run 2002 from the explicit 2001 reference checkpoint. Treat it as a warm-started sparsification study, not a fresh-training baseline or a universal stage-training requirement.

Train one candidate for at most **500 new epochs**, keeping original data, seed, batch/query sampling, supervised terms, optimizer policy, and physical loop unless the documented warm-start mechanism requires an optimizer reset. Evaluate initialization without copying a baseline checkpoint. Save local epochs 10, 50, 100, 250, 500 and ordinary best/latest checkpoints. Parent epoch and additional epoch count remain distinct in provenance.

Use physical GPU 0 or 2 only when available; do not preempt running jobs. `CUDA_VISIBLE_DEVICES=2` makes that physical card logical `cuda:0`. Do not invent a second unused GPU reservation, background monitor, or automation service.

At local 50 and 100, inspect actual progress and sparsity statistics. Continue to 500 unless ordinary numerical failures occur or the user intervenes. A slower-converging candidate is not rejected by an arbitrary intermediate accuracy ratio. No automatic long continuation, second penalty value, second seed, or new mean-shift candidate is authorized.

If the custom kernel does not support accurate training gradients on the installed stack, use the exact Torch executor for the candidate and label training/kernel validation separately. Do not hide an inaccurate fused backward behind apparent training convergence.

---

## 10. Numerical checks and research decision criteria

### 10.1 What “gates” mean here

Ordinary tests establish whether the code implements the stated computation. Numerical failures are bugs to investigate, not scientific results. Research targets below guide reporting and recommendation; they are not new approval services, speculative preflight barriers, contract freezes, or automatic early-stopping machinery.

Blocking decisions remain at genuine boundaries: overwriting/deleting existing results, changing another running job, changing shared dependencies, or any separately authorized release. Use existing Git, run allocation, typed configuration, tests, and trusted loading.

### 10.2 Exact numerical checks

On CPU float64 toy cases and real CUDA weights/inputs:

- identical Boolean support, including tiny positive entries, filtered sources, empty modules, and mixed complete/partial rows;
- `sum_i Pi_qi=1` for valid nonempty banks;
- Pi agrees with the reference to FP64-scale error (start with `rtol=1e-10, atol=1e-12` on well-scaled fixtures);
- FP32 field/context normalized RMS discrepancy initially below `2e-5` on well-scaled tests;
- gradient relative-norm disagreement initially below `1e-4` away from support boundaries, with separate absolute checks when the reference norm is nearly zero;
- gradients to A, d, omega, Q/K/V, MLP weights, temperatures, module/query coordinates; no detached learned route;
- module permutation and masked-padding invariance;
- environmental quadrature duplication preserves predictions under the same routing semantics;
- no dense fine computation on omitted pairs; scalar Boolean/dot-product tiles are counted separately;
- finite-difference step sweep on both sides of representative sparsemax support transitions. Do not expect a unique classical derivative exactly at a piecewise-linear kink;
- cached and uncached physical-loop predictions/gradients agree; changing P1 state invalidates P0 learned caches;
- outputs and allocated-memory floor stabilize over repeated optimizer batches.

These are initial numerical targets, not facts already established by this plan. Scale-aware tolerances must be justified from measured rounding and reference norms; never loosen them to conceal a systematic error.

### 10.3 Scientific fidelity reference

For Run 2002, compare with the **same explicit 2001 checkpoint used to initialize it**, ideally epoch 2425. Also retain the exact-2500 four-model table and separate mature baselines where already available. Label all warm-start compute and checkpoint policy. Do not rank a selected warm-started child against exact parent endpoints as if budgets matched.

Suggested provisional fidelity band for a useful sparsification result:

- pooled fluid relative L2 within **3%** of the initialization reference;
- case p95 within **5%** and worst-case within **10%**;
- fluid temperature/vorticity and internal/surface temperature within **5%**;
- interface flux and final port T/h within **5%**, with absolute values shown;
- engineering KPIs and module-count/wall/heating strata do not hide material regressions.

These bands are declared research tolerances, not a guarantee or a new release gate. Report the entire Pareto tradeoff even when a checkpoint falls outside them. The user decides whether any accuracy/cost tradeoff is useful.

### 10.4 Scientific sparsity and speed reference

At each saved diagnostic stage, report by source type, physical phase, and module-count stratum:

- true unique/dense pair ratio;
- logical duplicate-path ratio;
- actual expanded-path workspace (zero on new compiler where implemented);
- query-hub support, source-hub support, hub occupancy and singleton fractions;
- complete receiver fraction, complete case/chunk fraction, and actual kernel dispatch fraction separately;
- cost surrogate versus true nnz, including tiny-positive-tail counts;
- logical selected pairs versus **physically evaluated** fine pairs and padded kernel lanes;
- forward, preparation, prepared decode, backward, optimizer and total step time;
- absolute and baseline-relative allocated peak, reserved peak, and post-step live floor;
- total and trainable parameters; kernel cache/build time reported outside steady state.

A motivating target is at least 50% fewer expensive fine pairs in the reducible `M>=2` strata, but it is not an automatic training gate and may be physically unattainable at the declared fidelity. All-case cost still includes `M=1`.

A sparse-science success requires improved actual latency/memory over the **exact-accelerated parent with the same backend**, not merely over the old join-heavy executor. An engineering success may be declared separately when identical predictions run faster without learning new sparsity.

Compare against Dense 1804 and Legacy 1401 under the same measurement protocol. No speedup over either baseline is promised. If dense fine support remains necessary and Dense is still faster, state that clearly.

### 10.5 Ground-truth and mechanism evaluation

Use all 90 existing development cases at the endpoint, normalized and physical channel metrics, p95/worst, near/far errors, interfaces/ports, and the maintained KPIs. This split is not untouched test evidence.

On fixed anchors, run P0/P1-only/P2 interventions with the actual phase boundaries respected. Test uniform routing, removed fine branches, and old-versus-new support while preserving the meaning of each control. Prediction differences show dependence; only accompanying target errors show benefit.

For omitted-source audits, use Eq. (21), QM omitted vector sums, and full physical-loop target errors. A zero omitted-mass value when support is complete is vacuous for pruning quality.

Visualize exact selected pairs at multiple near-module, boundary, and far-field queries. Include query probabilities/densities and actual active source counts next to source incidences. A force-directed plot or a block-diagonal-looking heatmap does not establish physical clustering.

Physical influence, barriers, and extrapolation to large 3-D fields remain **Evidence Missing** without independent data/simulations. Synthetic scaling tests measure execution only.

---

## 11. Benchmark design and bounded resource use

Keep the initial benchmark set small:

- real inference anchors 0273 and 0653; 8,192 queries; receiver chunks 128 and 2,048;
- training `B=48,Q=1024`, one small-module and one mixed larger-module batch;
- the two established scaling shapes `(M,E,Q)=(32,768,65536)` and `(128,3072,262144)` in inference, with explicit fixed supports as well as actual router supports;
- both complete and substantially partial fixtures, including an irregular support distribution;
- one warmup for short optimizer checks and at least two inference warmups, five timed inference repetitions and three optimizer repetitions initially.

Use the same device, allocator policy, precision, input IDs, weights, optimizer state, masks, receiver chunks, diagnostic-map setting, and synchronization for paired measurements. Compilation and warmup are excluded from steady-state timing but reported separately.

Do not transplant old speedup percentages to new checkpoints. Profiled wall time is attribution evidence only. Counter collection should run untimed where instrumentation materially perturbs kernels.

For source-count scaling, report dense preparation separately: it still contains `M^2` and `ME` work. Query sparsity cannot make that term disappear, and the prepared source states cannot be cached across physical changes.

---

## 12. Deliverables, output layout, and stop condition

Deliver:

1. The readable exact reference plus new union-free executor and optional fused QE backend.
2. Focused compiler/reader/gradient/compatibility tests and real physical backward/update evidence.
3. One opt-in Run-2002 profile with explicit parent checkpoint, typed temperature initialization, measured/fallback cost ratio, fixed lambda, and maximum additional epoch budget.
4. `HONF_Routing_Sparse_Execution_Next_Phase_Report.md`, separating:
   - supplied epoch-2500 evidence;
   - same-checkpoint exact execution measurements;
   - fused-kernel forward/backward measurements and limitations;
   - warm-started sparsification accuracy/cost/topology;
   - retained negative results and missing physical evidence.
5. Actual commands and managed artifact locations; an **unexecuted** continuation command only if the result warrants user consideration.

Reuse existing run/evaluation and `diagnostics/generated/...` conventions. Store source, configs, tests, and the concise report in appropriate tracked paths; large arrays, traces, images, and checkpoints stay in the established generated locations. Do not duplicate parent results or invent extra manifests when the existing layout already records them.

Do not introduce cryptographic hashes, contract freezes, baseline snapshots, or additional approval gates. Preserve existing trusted-loading/security controls and any existing provenance fields. Never delete or rewrite parent history, regenerate golden references, or remove security under a cleanup justification.

The Goal ends after the authorized implementation/evaluation and at most one 500-additional-epoch candidate. No background monitoring framework, new service, automatic rescue run, or automatic extension is part of completion.

---

## 13. Sources and implementation references

### Supplied evidence

- `HONF_Routing_Epoch2500_Comparative_Evaluation.md` — main evaluation and controlled timing protocols.
- `merged_reduction_summary.md` — exact/selected epochs, physical metrics and the separate evaluator timing.
- `Hyper-Hub-Shift.png` — geometry, source incidence, and a query's executed pair union; a diagnostic, not a causal physical graph.

### Pinned implementation

All paths below refer to commit `3943710c0df76c8380feb6b98d0cf7b9842c0594`:

- [Routing projections](https://github.com/cosmos2w/ModularDT/blob/3943710c0df76c8380feb6b98d0cf7b9842c0594/HONF_Proj/src/honf_forward_core/interface_fields/routing_index/sparse_projection.py)
- [Pair compilation](https://github.com/cosmos2w/ModularDT/blob/3943710c0df76c8380feb6b98d0cf7b9842c0594/HONF_Proj/src/honf_forward_core/interface_fields/routing_index/pair_join.py)
- [Routed preparation and readers](https://github.com/cosmos2w/ModularDT/blob/3943710c0df76c8380feb6b98d0cf7b9842c0594/HONF_Proj/src/honf_forward_core/interface_fields/routed_pairwise.py)
- [Dense fine preparation and networks](https://github.com/cosmos2w/ModularDT/blob/3943710c0df76c8380feb6b98d0cf7b9842c0594/HONF_Proj/src/honf_forward_core/interface_fields/dense_pairwise.py)
- [MLP/Fourier definitions](https://github.com/cosmos2w/ModularDT/blob/3943710c0df76c8380feb6b98d0cf7b9842c0594/HONF_Proj/src/honf_forward_core/nn.py)
- [Routing affinity](https://github.com/cosmos2w/ModularDT/blob/3943710c0df76c8380feb6b98d0cf7b9842c0594/HONF_Proj/src/honf_forward_core/interface_fields/routing_index/geometry.py)
- [ThermalChannel geometry](https://github.com/cosmos2w/ModularDT/blob/3943710c0df76c8380feb6b98d0cf7b9842c0594/HONF_Proj/Case_ThermalChannel/src/channelthermal/routing_geometry.py)

### External methodological references

These support particular ingredients, not an already-validated HONF speedup or physical sparsity theorem:

1. Martins and Astudillo, **From Softmax to Sparsemax**, ICML 2016. Sparse probabilities and active-set Jacobians. The source-measure projection and Eqs. (5)–(8) in this plan are derived for HONF, not attributed to that paper. [Paper](https://proceedings.mlr.press/v48/martins16.html)
2. Gonçalves, Treviso, and Martins, **AdaSplash: Adaptive Sparse Flash Attention**, ICML 2025. GPU kernels are needed to turn adaptive attention sparsity into execution savings; it is not a drop-in replacement for HONF's fine MLP and weighted routing. [Paper](https://proceedings.mlr.press/v267/goncalves25a.html)
3. Triton official **Fused Attention** tutorial. Streaming attention and backward implementation reference; its displayed reduced-precision benchmark results do not predict HONF's FP32/FP64 performance. [Documentation](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
4. `pytorch_scatter` official **scatter** documentation. GPU atomic reductions are not automatically contention-free or deterministic. [Documentation](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html)
5. `pytorch_scatter` official **segment_csr** documentation. Row-pointer grouped reductions have different execution properties from unordered scatter; benchmark on actual shapes. [Documentation](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html)

The smooth cost in Eq. (17), its calibration, proposed temperatures, fidelity bands, and hardware task decomposition are research design choices made here. They have not been experimentally validated on Run 2002.
