# Run 1407 — Phase-Shared, Prototype-Anchored Group Routing

**Research design, implementation tasks, execution measurements, and staged training**  
Planning date: 19 September 2026  
Repository: `cosmos2w/ModularDT`  
Branch: `agent/honf-core-next`  
Inspected reference: `3da43e607efb83c844eac1da5a692dd05a6e60f3`

## 0. Decision in one page

Develop **one new scientific architecture**, proposed Run **1407**, alongside—not instead of—Run 1406.

```text
forward_architecture = "phase_shared_group_control_honf"
```

The new hypothesis is:

> A layout-conditioned group index can be constructed once from the initial fine contextual response, reused throughout the physical refinement loop, and queried through prototype-anchored, scale-controlled sparse routing. The fine physical source values continue to update after every local response.

There are exactly two model changes:

1. **Phase-shared controller:** build source memberships, group control, and routing-only source banks once at P0. Reuse them at P1/P2; do not recompute them from later local responses.
2. **Prototype-anchored query routing:** replace query keys derived only from bounded pooled control with keys containing the learned group prototype plus the case-conditioned control correction. Normalize query/key RMS before the existing entmax15 operation. Keep source assignment arithmetic unchanged initially.

There is one execution change:

3. **Actually execute the induced unique-source support when beneficial:** use the fixed six-bit group-support signature to obtain unique source sets, without expanding `q -> k -> s` paths. Retain an exact rectangular implementation for complete/broad supports where it is faster. Count the implementation that really ran.

Keep the successful parts of Run 1406:

- K=6; D=16; H=256; fine-message width 128; four attention heads;
- individually addressable module/environment values;
- Dense simultaneous MM/ME/EM preparation at every physical state;
- one expensive interaction per unique query/source pair, not per group path;
- the same three output contexts, `C_g + C_M + C_E`;
- the same local surrogate and P0/P1/P2 physical coupling;
- the accepted complete-QE activation-checkpoint boundary;
- existing physical losses, normalization, seed, and optimizer.

Do not add a coarse bank, local correction, top-C route cap, learned edge count, load-balancing loss, entropy loss, pair-cost loss, temperature schedule, barrier mask, or custom GPU kernel in this task.

**This is a new model equation, not an exact cache optimization of 1406.** Exactness is required between 1407's dense-reference and selected-source executors, not between 1407 and a 1406 checkpoint.

One fresh Run 1407 is authorized through 50 epochs on a free GPU. Use a measured research-budget decision to continue the **same run** to 500. Perform the bounded endpoint comparison described below. A 2,500/5,000-epoch extension requires the user's subsequent instruction; supply the actual resume command but do not launch it automatically.

---

## 1. Evidence that motivates the design

### 1.1 Use these reports as evidence, not assumptions

**[R1]** `HONF_Run1406_Epoch500_Comparative_Evaluation.md`, supplied by the user and committed under `HONF_Proj/docs/reports/`.

**[R2]** `HONF_Run1406_Performance_Diagnosis_and_Exact_Optimization_Report.md`, supplied by the user and available under the maintained reports tree.

At exact epoch 500, [R1] reports:

| Quantity | 1406 | Dense 1804 | Interpretation |
|---|---:|---:|---|
| Pooled fluid relative L2 | 0.10949 | 0.09874 | 1406 is 10.9% worse |
| Equal-case p95 | 0.12875 | 0.11253 | 1406 is 14.4% worse |
| Near-interface relative L2 | 0.12530 | 0.09759 | 1406 is 28.4% worse |
| Far-fluid relative L2 | 0.09991 | 0.09940 | Nearly equal |
| Vorticity relative L2 | 0.15360 | 0.11367 | 1406 is 35.1% worse |
| Internal temperature | 0.07976 | 0.06949 | Dense is better on this metric |
| Surface temperature | 0.10362 | 0.08724 | Dense is better |
| Normal heat flux | 0.23782 | 0.22007 | Dense is better |
| Full forward | 36.84 ms | 33.35 ms | 1406 is 10.5% slower |
| Prepared P2 | 11.00 ms | 13.02 ms | 1406 is 15.5% faster |
| M12 update | 1093.48 ms | 2090.89 ms | 1406 is 47.7% faster |
| M12 allocated peak | 23686.86 MiB | 26779.99 MiB | 1406 is 11.6% lower |

Do not merge the GPU-1 profile timings in [R2] with the GPU-0 epoch-500 timings in [R1]. Their trends agree; their protocols/devices/checkpoints differ.

[R2] localizes the preparation excess to phase-dependent routing/control construction. Its instrumented P0+P1+P2 preparation is 15.477 ms for 1406 versus 8.919 ms for Dense, with 7.060 ms attributed to router/control construction. These event ranges are nested and must not be summed as a second end-to-end timing.

The accepted memory improvement was **complete-QE checkpointing**, not a new module-tail checkpoint, padding removal, or optimizer change. Preserve it. Diagnostic elision and Fourier reuse were tested and rejected as immaterial; do not repeat them as the principal optimization project.

### 1.2 What the topology statistics actually establish

Across 90 cases [R1] reports:

- exactly two nonempty module groups;
- 4.389 nonempty environmental groups on average;
- module source degree 1.567;
- environmental source degree 1.422;
- positive query degree 5.947 of six;
- module candidate support 100%; environmental support 99.9567%.

These establish selective **source assignment**, but almost nonselective **query access**. They do not establish recovered physical mechanisms or fine-work savings.

Let `A_sk` be source membership and `alpha_qk` query membership. Because these are nonnegative,

\[
\rho_{qs}=\sum_k\alpha_{qk}A_{sk}>0
\iff
\exists k:\alpha_{qk}>0\land A_{sk}>0.
\]

If every query reads every occupied group, every valid source remains reachable even when source assignments are almost one-hot. A sparsity penalty on source assignments alone does not address this condition.

Exactly two populated module groups is **not** proof that two physical mechanisms suffice. It is an observed property of this learned representation. Do not force six balanced groups merely to improve a diagram.

### 1.3 Two reporting qualifications must survive into the new report

**All-domain interpretation.** [R1] reports an all-domain relative L2 advantage for 1406, but attributes it to stronger internal-module prediction while separately reporting worse internal-temperature and surface-temperature metrics. The aggregate ordering is reported evidence; its causal attribution is not established. Trace the existing evaluator's field tensors, solid/fluid masks, channels, and SSE denominators before claiming an internal-physics advantage. Report an unresolved attribution as **Evidence Missing**. Do not silently rewrite the old report or infer CFD truth.

**Convergence.** The last-50 median of 1406 improves, but exact-500 field validation is 0.02364 versus best 0.01747 at epoch 473. This supports continued learning, not a proof that every endpoint fluctuation is harmless noise. Keep exact and best-through-budget checkpoints separate. Epoch 500 is not a mature final ranking.

---

## 2. Code reviewed and the precise design gap

At the pinned commit:

- `interface_fields/group_control_router.py` assigns sources using `group_codes`, but queries use only `query_group_projection(group_control)` as keys.
- The group control is bounded by `tanh`; query keys and source prototypes therefore follow different scale and identity paths.
- `interface_fields/group_control_pairwise.py::prepare` reconstructs memberships, group controls, control banks, and contextual K/V after every physical update.
- `interface_fields/core.py::prepare` and ThermalChannel's `forward_interface_field` call this preparation at P0, P1, and P2.
- The ordinary optimized reader is rectangular. Its optional map path can dispatch a gathered implementation. A map-derived support count must not be called an observed normal-path operation count when the executor differs.
- `routing.py::entmax15` implements ordinary 1.5-entmax, not the source-measure sparsemax density projection used in the 2000 series. Do not transfer that earlier projection's all-active formula to 1406.

The source-derived facts support two hypotheses, not two proven explanations:

1. Recomputing a layout's group index after every local correction may be unnecessary for prediction while being measurably expensive.
2. Query keys based solely on bounded pooled controls may be insufficiently separated or poorly scaled relative to source prototypes; this could explain broad query support.

Both old source and query columns have consistent group indices; different key parameterizations are not intrinsically a coding error. The suspected problem is poor score separation, not mislabeled columns. The first hypothesis is tested by deliberate phase sharing. The second is tested by prototype anchoring and RMS-controlled logits. A short diagnostic before training measures their relevance; it is not a prerequisite for inventing more variants.

---

## 3. Mathematical notation and preserved physical computation

| Symbol | Meaning |
|---|---|
| B | Batch size |
| M / M_pack | Active module count / padded storage width |
| E | Environmental source count, initially 192 |
| Q | Requested receiver count |
| K | Group capacity, fixed at 6 |
| d / P | Spatial dimension / physical ports per module |
| m_i | Binary active-module indicator |
| d_h | Per-head feature width, H/4 = 64 |
| s (in coordinate normalization) | Adapter-provided coordinate-scale vector, distinct from source index s |
| Phi | Existing Fourier encoding with the existing coordinate normalization |
| D | Routing/control width, fixed at 16 |
| H / J | Fine-state width 256 / fine-message width 128 |
| p | Physical preparation phase, 0, 1, or 2 |
| x_i, y_j, q | Module, environment, and receiver coordinates in R^d |
| z_i^(p) | Incoming module state for preparation p |
| ztilde_i^(p), etilde_j^(p) | Fine contextualized module/environment states |
| g | Case-wide encoded operating context |
| nu_j | Environmental quadrature weight |
| omega_i^M, omega_j^E | Normalized source measures |
| A^M, A^E | Source-to-group memberships |
| h_k | Bounded D-dimensional group control |
| c_k | Learned D-dimensional group prototype |
| alpha_qk | Query-to-group probability |
| rho_qs, n_qs | Scalar overlap and D-dimensional collective pair control |

Use the same encoders and simultaneous Dense MM/ME/EM messages as 1406:

\[
\begin{aligned}
p^{MM}_{i\ell}&=\phi_{MM}[z_i^{(p)},z_\ell^{(p)},\Phi(x_i-x_\ell)],\\
p^{ME}_{ij}&=\phi_{ME}[z_i^{(p)},e_j,\Phi(x_i-y_j)],\\
p^{EM}_{ji}&=\phi_{EM}[e_j,z_i^{(p)},\Phi(y_j-x_i)].
\end{aligned}
\]

Apply the existing active-count/quadrature reductions, then

\[
\widetilde z_i^{(p)}=z_i^{(p)}+\rho_M[z_i^{(p)},a_i^{MM},a_i^{ME},g],
\]

\[
\widetilde e_j^{(p)}=e_j+\rho_E[e_j,a_j^{EM},g].
\]

All three messages use the incoming module state of their phase. Do not accidentally feed the newly updated module state into EM within the same phase.

Global and geometric inputs remain as currently defined by `InterfaceFieldCore` and the case adapter. Retain the normal coordinate scaling and source masks. In a case with no active modules, module contributions are zero; do not invent a module to avoid an empty reduction.

---

## 4. Central model change I: build the controller once at P0

### 4.1 Why P0 rather than an entirely new structural encoder

Use the already available fine P0 states, `ztilde^(0)` and `etilde^(0)`, as the controller's input. Do **not** add an extra Dense preparation just for routing. This preserves the strongest available layout- and operating-condition-dependent interaction information before sharing the controller.

P0 here means the first physical preparation, **not** an initial training epoch. The controller is rebuilt for every new forward/design instance and every optimizer step.

### 4.2 Source control and membership: retain 1406 arithmetic

For source type t in {M,E}, retain

\[
u_s^t=P_t\operatorname{LN}(\widetilde z_s^{t,(0)})
       +P_{x,t}\Phi(x_s^t/s)+G_tg_c,
\qquad g_c=P_g\operatorname{LN}(g).
\]

Here `ztilde^{E,(0)}` denotes `etilde^(0)`. Define

\[
A^t_{s:}=\operatorname{entmax}_{1.5}\left(
\frac{(u_s^t)^\top c_k}{\sqrt D}\right)_{k=1}^K.
\]

Mask inactive module rows exactly as in 1406. All temperatures remain one. Source scoring is deliberately **not** renormalized in the first 1407 experiment: source incidence is already selective, and the primary routing issue is on the query side.

Normalized measures are

\[
\omega_i^M=\frac{m_i}{\max(1,\sum_lm_l)},
\qquad
\omega_j^E=\frac{\nu_j}{\sum_l\nu_l}.
\]

### 4.3 Group control: retain 1406 moment arithmetic

\[
\mu_k^t=\sum_s\omega_s^tA^t_{sk},
\qquad
b_k^t=K\sum_s\omega_s^tA^t_{sk}u_s^t.
\]

\[
h_k=\tanh\left(F_h[b_k^M,b_k^E,K\mu_k^M,K\mu_k^E,g_c,c_k]\right).
\]

No division by tiny group mass is introduced. Group centroids remain diagnostic quantities, not mandatory routing anchors.

The case-local shared object contains

\[
\mathcal I_0=(A^M,A^E,h,g_c,\omega^M,\omega^E,\text{small control banks and support metadata}).
\]

### 4.4 Reuse at subsequent physical phases

\[
\boxed{\mathcal I^{(0)}=\mathcal I^{(1)}=\mathcal I^{(2)}=\mathcal I_0.}
\]

But

\[
\boxed{\widetilde z^{(p)},\widetilde e^{(p)}\text{ are recomputed for each physical phase}.}
\]

Consequently:

- memberships, group controls, normalized query keys, module control bank, environmental value gains, and environmental head-control banks are phase-shared;
- fine module first-affine values and environmental K/V **must refresh** as fine physical source states change;
- the local operator and its incoming port conditions still respond to feedback;
- only the structural control update through later local responses is removed.

This is the intentional scientific approximation. It may harm cases where local feedback changes which sources should interact. It is not a cache-validity trick.

### 4.5 Gradient meaning

The shared tensors remain live in the autograd graph. For total physical loss

\[
L=L_0+L_1+L_2+L_{\mathrm{interface}},
\]

This is a schematic dependency decomposition of the existing loss, not an instruction to add separate P0/P1/P2 supervision. All relevant existing loss terms contribute to the same controller parameters:

\[
\nabla_\theta L\supset
\sum_p \frac{\partial L_p}{\partial\mathcal I_0}
       \frac{\partial\mathcal I_0}{\partial\theta}.
\]

Do not detach the shared index. Do not store it persistently on the model. Do not reuse it across different layouts, boundary/operating conditions, optimizer updates, independent forwards, or separate inverse-design iterations.

The deliberate omission is the path from a later fused module state back into a newly reconstructed routing topology. Gradients still flow through the updated fine physical values.

---

## 5. Central model change II: prototype-anchored, scale-controlled query access

### 5.1 Existing rule

Run 1406 uses

\[
\ell^{1406}_{qk}=v_q^\top W_hh_k/\sqrt D,
\quad
v_q=F_Q[\Phi(q/s),g_c].
\]

The source logits address the prototypes `c_k`, while query logits address only a transformed bounded control state. The report does not supply key norms or score margins, so the cause of broad query support remains a hypothesis to audit.

### 5.2 Proposed replacement

Keep the existing D-dimensional query MLP and projection dimensions. Form one P0 key per group:

\[
k_k=c_k+W_hh_k.
\]

Use parameter-free RMS scaling

\[
\mathcal R(v)=\frac{v}{\sqrt{D^{-1}\sum_{a=1}^D v_a^2+\epsilon_R}},
\qquad \epsilon_R=10^{-6}.
\]

Then

\[
\boxed{
\ell^{1407}_{qk}=\frac{\mathcal R(v_q)^\top\mathcal R(k_k)}{\sqrt D},
\qquad
\alpha_{q:}=\operatorname{entmax}_{1.5}(\ell^{1407}_{q:}).
}
\]

Do not add another temperature parameter. The normalized group keys are prepared once; only the small normalized query is calculated per receiver.

This does two things:

- retains the group identity used by source assignment instead of allowing query access to rely exclusively on pooled-control similarity;
- prevents a small common query/key magnitude from making every score almost identical in the entmax input scale.

It does **not** enforce distinct prototypes, prevent every possible group collapse, prescribe a physical cluster boundary, or guarantee a small support. Directional similarity can still produce broad scores. Measure the result.

The group key remains case dependent through `h_k`; this is not a fixed spatial partition shared blindly across all layouts.

### 5.3 Why this differs from another sparsity loss

For entmax15,

\[
\alpha_k=[\ell_k/2-\tau]_+^2,
\qquad \sum_k\alpha_k=1.
\]

Exact zeros arise from relative score separation. There is no new regularization objective to optimize a proxy for the fine-pair count.

For active probabilities, its Jacobian is

\[
J=\operatorname{diag}(s)-\frac{ss^\top}{\mathbf1^\top s},
\qquad s_k=\sqrt{\alpha_k}.
\]

Inactive entries have zero local derivatives; singleton support has zero score Jacobian. Therefore a singleton fraction and route-turnover measurement are necessary. Do not call nonzero aggregate backend gradients proof that the query router is still learning.

Do not impose hard Top-C truncation, mask empty groups with a discontinuous new rule, or silently raise/lower temperatures when degree targets are missed. Keep 1406's well-defined zero-overlap response behavior.

### 5.4 Spatial and physical interpretation

The index is conditioned on fine P0 interaction states, source coordinates, environmental descriptors, and operating context. It is therefore geometry-aware and physics-conditioned by the supervised problem.

There is no new claim that Euclidean distance identifies nonphysical long-range interactions. Do not add wall barriers, hard distance cutoffs, or inferred advection cones without validated case data. Pressure and transport can have nonlocal effects.

The proposed change may make near-interface routing more consistent across phases, but improvement in vorticity or flux is an empirical objective, not a theorem.

---

## 6. Fine response and output equations: preserve Run 1406

For source type t, calculate

\[
\rho^t_{qs}=\sum_k\alpha_{qk}A^t_{sk},
\qquad
n^t_{qs}=\sum_k\alpha_{qk}A^t_{sk}h_k.
\]

Since controls are bounded,

\[
\|n^t_{qs}\|_\infty\le\rho^t_{qs}.
\]

Keep the unnormalized moment. Do not introduce division by `rho`.

### 6.1 Module fine interaction

For phase p, prepare

\[
a_i^{(p)}=W_z\widetilde z_i^{(p)}+W_gg+b.
\]

At the receiver,

\[
t_{qi}^{(p)}=\operatorname{GELU}[a_i^{(p)}+W_r\Phi(q-x_i)],
\]

\[
\widetilde t_{qi}^{(p)}=t_{qi}^{(p)}\odot[1+\tanh(B_Mn^M_{qi})],
\quad
\psi_{qi}^{(p)}=F_{\mathrm{tail}}(\widetilde t_{qi}^{(p)}).
\]

Let

\[
S_M(q)=\sum_i\omega_i^M\rho^M_{qi}\psi_{qi}^{(p)},
\quad
\eta_M(q)=\sum_i\omega_i^M\rho^M_{qi},
\quad
\gamma_M=\frac{M_a}{1+M_a}.
\]

Preserve the existing output bias convention:

\[
\boxed{
C_M(q)=K\gamma_M W_M S_M(q)+K\eta_M(q)b_M.
}
\]

Do not move the nonlinear activation across a source sum or change the factor on the output bias.

### 6.2 Environmental fine interaction

Prepare one dynamic K/V bank from `etilde^(p)` as in 1406. Its source-side value control uses the shared index:

\[
\bar h_j^E=\sum_k A^E_{jk}h_k,
\quad
V_j^{(p),c}=V_j^{(p)}\odot[1+\tanh(B_V\bar h_j^E)].
\]

Project controls to attention-head space once:

\[
\zeta_{qj}^{(a)}=\sum_k\alpha_{qk}A^E_{jk}[B_Sh_k]_a.
\]

For positive overlap,

\[
s_{qj}^{(a)}=
\frac{Q_q^{(a)\top}K_j^{(p,a)}}{\sqrt{d_h}}
[1+\tanh(\zeta_{qj}^{(a)})]
+b_{\mathrm{geo}}^{(a)}(q-y_j)
+\log\omega_j^E+\log\rho^E_{qj}.
\]

Normalize across supported sources, form the existing multihead response `o_q`, and preserve

\[
\boxed{C_E(q)=K\eta_E(q)[W_Eo_q+b_E]},
\qquad
\eta_E(q)=\sum_j\omega_j^E\rho^E_{qj}.
\]

Zero-support rows return exact zero environmental context with finite derivatives. Preserve the established numerical convention and test tiny positive overlap separately from exact zero. Do not call an underflowed positive route a theoretical hard zero.

### 6.3 Final field and physical ports

\[
\boxed{
C(q)=C_g(q)+C_M(q)+C_E(q),
\quad C_g(q)=F_g[\Phi_Q(q),g],
\quad \widehat U(q)=F_{\mathrm{out}}[\operatorname{LN}C(q)].
}
\]

Keep `ThreeTermInterfaceContext`. Do not add a raw-source bypass, direct `h_k` value, coarse bank, or local correction.

The physical port head continues consuming this same continuous context at port coordinates, alongside its established module/global inputs. The hypergraph is not claimed to be the only path by which module information reaches physical outputs.

---

## 7. Complete data flow

```text
Modules, geometry, environmental samples, operating conditions
                             |
                         Encoders
                             |
                 Dense fine preparation P0
                             |
              +--------------+----------------+
              |                               |
   Build shared group index once       Fine P0 source values
   A_M, A_E, h, prototype keys                 |
   D=16 banks, support table                  |
              |                               |
              +-------- P0 port reader -------+
                             |
                  Port head + local operator
                             |
                  Updated module physical state
                             |
                 Dense fine preparation P1
                             |
          Reuse SAME index; refresh fine K/V and QM affine
                             |
                  P1 outside-temperature read
                             |
                  Port refinement + local operator
                             |
                 Dense fine preparation P2
                             |
          Reuse SAME index; refresh fine K/V and QM affine
                             |
         Query -> sparse group support -> UNIQUE fine sources
                             |
                 C_g + C_M + C_E -> field
```

A repeated coordinate receives the same query routing through all phases. Its predicted field can and should change because its fine source values change.

---

## 8. Execution/data-handling change: source support, not group triples

### 8.1 Separate three quantities

For each source type report

\[
P_{\mathrm{logical}}=\sum_{q,k}\mathbf1[\alpha_{qk}>0]\#\{s:A^t_{sk}>0\},
\]

\[
P_{\mathrm{support}}=\#\{(q,s):\rho^t_{qs}>0,\omega_s^t>0\},
\]

and `P_executed`: fine rows actually evaluated by the normal runtime path, including zero-weight or padded rows when rectangular execution evaluates them.

Use

\[
R_{\mathrm{support}}=P_{\mathrm{support}}/(QN_{\mathrm{valid}}),
\qquad
R_{\mathrm{executed}}=P_{\mathrm{executed}}/(QN_{\mathrm{valid}}).
\]

They are not interchangeable. A counter from an alternate debug/gathered forward is not a count of the timed rectangular path.

### 8.2 Six-bit support index

K=6 permits a small exact support representation. For each source,

\[
m_s=\sum_{k=0}^{5}2^k\mathbf1[A_{sk}>0].
\]

For a query,

\[
b_q=\sum_{k=0}^{5}2^k\mathbf1[\alpha_{qk}>0].
\]

Then

\[
\boxed{\rho_{qs}>0\iff(m_s\ \&\ b_q)\ne0}
\]

for nonnegative memberships in exact arithmetic. Respect valid-source masks and numerical underflow conventions.

Build a small table of source supports for the **64 possible query masks**, including the empty mask. For E=192 its Boolean storage is only 64×192 entries per case. This exhaustive support table is scoped to the formal K=6 profile; do not generalize it by blindly allocating 2^K entries for arbitrary large-K configurations. It is prepared once with `I_0`, not at every physical phase and not for every receiver.

The support table is integer metadata. Differentiable `rho`, moments, priors, and source values are still calculated from live `A`, `alpha`, and `h`. Never replace them with detached table weights.

### 8.3 Query signatures organize work; they do not duplicate it

Queries sharing `(case, support_mask)` share a unique source set. Gather that set once, then calculate a rectangular attention block for those receivers and sources. Evaluate each physical pair once even if it shares several hyperedges.

Do not materialize `(q,k,s,H)` or build a duplicate two-hop path list. Do not sort source coordinates repeatedly during each receiver read. Preserve original query ordering when returning results.

An initial implementation may reuse existing unique-pair helpers for partial support. A small signature-table wrapper should replace duplicate-path expansion, not introduce a new general sparse-computing framework.

### 8.4 Hybrid execution is allowed, but must be honest

Use the existing rectangular path for complete support, small source counts, or support patterns for which gathering is measured to be slower. Use selected-source blocks for sufficiently reduced support.

Choose the runtime policy using a bounded timing calibration on actual model tensor shapes, not a learned gate or a hard-coded claim that sparse is always faster. Benchmark the calibrated policy and the rectangular reference on the same weights. Do not tune predictive parameters on the holdout to meet a speed goal.

A runtime policy must not drop any positive pair or renormalize differently. The two implementations compute the same 1407 operator.

If padding is retained for batch efficiency, label its executed rows separately. There must be no multiplicative K factor in expensive work; strict active-source bounds require active compaction, while padded rectangles may exceed `Q*M_active` and must be counted honestly.

### 8.5 No hiding a prediction change behind a diagnostic flag

`return_routing_maps` must not choose a different scientific execution policy. It can request maps and counters from the actual selected backend. A separately named reference-executor option may be used for parity tests.

Count actual forward rows at their operation boundary in an untimed instrumented invocation of the **same policy**, and record checkpoint recomputation separately. Metadata-derived support counts remain a separate column.

### 8.6 Performance limits

The best-case saved work includes two router preparations and repeated static control-bank construction. Dynamic MM/ME/EM and dynamic fine source projections remain.

The measured 7.060 ms router range in [R2] provides a scale for the opportunity, not a promised end-to-end saving. Not every preparation operation disappears, and sharing a live P0 graph can affect activation lifetimes.

Sparse support can still cost more than dense kernels when fragmented. Do not require custom Triton or CUDA code to complete this research task. Profile only the residual hotspot if the implemented sparse schedule fails to translate meaningful support reduction into time savings.

---

## 9. Generic HONF implementation tasks

### 9.1 Architecture and files

Add one opt-in architecture:

```text
phase_shared_group_control_honf
```

Suggested compact source layout:

```text
interface_fields/phase_shared_group_control.py
interface_fields/group_control_support.py       # only if not cleanly covered by current helpers
```

Reuse `GroupControlPairwiseField`, its fine preparation, modulators, `ThreeTermInterfaceContext`, and the accepted QE checkpoint implementation. Do not copy the entire 1406 backend into a second thousand-line implementation.

A small subclass/strategy is appropriate:

- `PrototypeAnchoredGroupRouter`: reuse source preparation; override query key/score construction;
- `PhaseSharedGroupControlField`: separate creating the shared P0 index from refreshing dynamic source values;
- a small runtime dataclass containing shared live controls and support metadata.

Extract an ordinary helper for dynamic Dense preparation only if needed. Keep historical public defaults and state-key meanings unchanged.

### 9.2 Runtime state contract

Conceptually:

```python
@dataclass(frozen=True)
class SharedCaseGroupIndex:
    controls: PreparedGroupControl       # live A_M, A_E, h, measures, g_c
    normalized_query_keys: Tensor        # [B,K,D]
    module_control_bank: Tensor          # [B,K,M_pack*D]
    environment_value_gain: Tensor      # [B,heads,E,head_dim]
    environment_head_source_control: Tensor
    source_support_metadata: ...        # integer/Boolean only
```

Fields are runtime data, not persistent parameters. The object belongs to one physical forward. Avoid caches keyed only by tensor shape, model attribute singletons, global dictionaries, or checkpoint-serialized runtime tensors.

P0 returns this object in the ordinary prepared backend state. P1/P2 receive the same explicit object. Do not infer phase identity from a profiling label or an internal call counter.

### 9.3 Core integration

Extend `InterfaceFieldCore.prepare` with one keyword-only optional argument such as `shared_group_index=None`, handled only by the new backend. Existing architectures called without it must follow exactly their old path.

At P0, `None` means construct from the already computed P0 fine states. At P1/P2, use the supplied index and refresh only physical values.

An explicit helper to retrieve the index from a prepared state is preferable to exposing case-specific data inside the generic backend.

All three-term architecture allowlists must include the new family, including construction, field head, auxiliaries, and coarse-count metadata. Do not accidentally instantiate `SharedInterfaceContext` and then simply ignore its parameters.

### 9.4 Configuration

Create:

```text
src/config_core/forward/phase_shared_group_control_honf_context.json
```

Base it on the current Run-1406 profile. Change only architecture, profile name, and run identity; add no scientific hyperparameter beyond the fixed new query rule described here.

Use:

```text
K=6, D=16, H=256, message_hidden_dim=128, heads=4
entmax15, all temperatures=1
activation_checkpointing=true
receiver_chunk_size=128 for training
seed=0, AdamW learning_rate=0.0003, weight_decay=0.00001
AMP=false, existing clipping and physical loss
predicted ports from epoch 1
run_id=1407
run_name=phase_shared_prototype_group_control
```

Use the existing milestones sufficient for 10, 50, 100, 250, 500, 1000, 2500, and 5000; do not generate duplicate checkpoint copies.

Execution-policy selection is runtime policy, not another neural architecture. Keep `dense_reference` available for numerical comparison. No numerical top-k cap is part of the scientific config.

### 9.5 Historical compatibility

Do not reinterpret a 1406 checkpoint as 1407 merely because parameter shapes match. The equations differ. Old architectures must instantiate their old classes and preserve trusted loading.

Use ordinary existing regression/strict-load tests. Run the maintained suite once at the integrated endpoint, not after every small edit. Existing missing-data/script failures must be reproduced and documented rather than silently skipped or turned into a release-blocking infrastructure project.

---

## 10. ThermalChannel-specific tasks

Modify `Case_ThermalChannel/src/channelthermal/interface_field_coupling.py` only to carry the new runtime object across phases:

```python
prepared0 = core.prepare(encoded, base_module_state, ...)
shared = core.shared_group_index(prepared0)  # None for historical families
# existing port/local-operator logic unchanged
prepared1 = core.prepare(encoded, updated_module_state,
                         shared_group_index=shared, ...)
# existing refinement/local-operator logic unchanged
prepared2 = core.prepare(encoded, final_module_state,
                         shared_group_index=shared, ...)
```

Pass the new keyword only when relevant if old mock/adapter signatures require it. If there is no local-response update, continue reusing `prepared0` as currently implemented.

Preserve all port coordinate definitions, clamping, radius offsets, local-surrogate calls, field temperature extraction, physical masks, P2 consistency reads, and public output shapes.

No new ThermalChannel-specific router, obstacle cost, wall mask, source partition, or loss is needed. The generic algorithm consumes the adapter's existing geometry/features/weights. Do not silently change quadrature conventions while introducing 1407.

Active-port packing was not established as a low-risk win by [R2]. It is outside the required 1407 change. It can be considered later as a shared adapter optimization with matched application to all relevant baselines.

---

## 11. Three work stages; one formal model

### Stage I — bounded diagnosis and implementation

Read [R1], [R2], the pinned source, and the maintained evaluation helpers.

Use the existing 1406 epoch-500 checkpoint on four cases: **0273, 0653, 0680, 0298**. These include familiar anchors and the candidate/baseline difficult cases.

Perform one bounded diagnosis:

1. Record source/query logit spread, query-vector norms, old key norms, pairwise key cosine, exact support sizes, singleton fractions, source mass, and query probability falling on source-empty groups.
2. Record P0-to-P1/P2 changes of `A_M`, `A_E`, and `h` on the same cases.
3. Evaluate one **frozen 1406 P0-index-reuse counterfactual** while preserving later fine-state refreshes. Report full physical, near-interface, internal/surface, pressure/vorticity effects and preparation timings. This is a new inference equation, not an exact replay. It is explanatory, not a prediction of trained 1407 accuracy.
4. Audit the all-domain/internal attribution using the existing metric definitions/arrays. No new CFD simulation or full-split evaluation is needed for this check.

Do not perform a temperature sweep or try multiple architectural variants based on these four cases.

Implement the one specified 1407 model, the real support-aware executor, and compact test coverage below. Then execute a real physical-batch backward/update. Preflight and toy tests do not replace that execution.

### Stage II — one managed Run 1407 through 50, then a research decision

Use one currently free physical GPU; do not interrupt any running 1406 continuation or other job. Use an ordinary worktree when concurrent code changes require isolation.

Launch exactly one fresh Run 1407. Do not warm-start from 1406, and do not create a second quickcheck run. Train to epoch 50, saving the normal optimizer/RNG state for same-run continuation.

At 50, use the matched benchmark and learning assessment in Sections 13–14. If the candidate is reasonably competitive and stable, continue the same run to 500. Otherwise stop and report the measured failure. A frozen score or a missing optional figure is not a reason to skip actual performance measurement.

### Stage III — epoch-500 evidence and handoff

Evaluate exact epoch 500 and separately best-through-500 against existing matched 1406 and 1804 results; retain 1404 as historical latency context where useful. Use the full 90-case development population once for the requested scientific endpoint.

Deliver an independent scientific recommendation and unexecuted continuation commands. Do not automatically run to 2500/5000. The user's later authorization can continue the same run without retraining from scratch. Preserve intermediate milestones.

No second seed, alternate K, alternate controller, or combined loss sweep belongs to this goal.

---

## 12. Focused verification: small number of high-value tests

1. **Sharing versus refreshing.** Confirm one source-router/control construction per full forward; three fine physical preparations when the local loop uses all phases. For identical q, index/alpha agree across phases. Changed later module states must still change fine values and outputs.
2. **Autograd sharing.** Compare an explicitly recomputed mathematical 1407 reference with its runtime-shared implementation. Compare outputs, representative query/module-coordinate derivatives, and all connected parameter gradients on a real predicted-port batch. The reference recomputes the same P0-based function, not a dynamic 1406 controller.
3. **Score rule.** Source incidence retains 1406 arithmetic. New query logits implement the displayed prototype-plus-control RMS rule with one `1/sqrt(D)`. RMS normalization is parameter-free. Test finite zero-vector behavior and near-zero inputs.
4. **Entmax and source support.** Existing entmax row sums, exact zeros, masks, inactive gradients, and singleton Jacobian behavior. Add no fake gradient or straight-through selection.
5. **Six-bit union.** Compare the support table with a small dense Boolean/einsum reference for all 64 masks, module padding, empty groups, and both source types. Group permutation preserves the physical result after consistent parameter/index permutation.
6. **Selected versus rectangular reader.** Same weights, actual contextual values, equal outputs and first derivatives for full, partial, all-zero-row, and tiny-positive support. Keep the direct group moments and original output-bias scaling. No top-k or post-hoc threshold is allowed.
7. **One fine evaluation.** Count rows in the actual fine MLP/geometry functions. Show that semantic group overlap does not multiply them. Report any padded rows; do not assert an active-source bound for padded execution.
8. **Checkpoint behavior.** Existing 1406 and baseline checkpoints strict-load unchanged. 1407 saves/resumes through the current workflow. Shared runtime state is rebuilt, not serialized. Preserve the accepted non-reentrant complete-QE checkpoint and test its gradients with a shared P0 graph.
9. **Physical shared-reader behavior.** P0 ports, P1 outside temperature, P2 fields, and consistency probes use the same new reader. No new coarse/local/global-data bypass appears.
10. **Execution maps.** Enabling maps does not change the execution policy or predictions. Logical/support/executed/recomputed counts are distinct and accumulated correctly across uneven receiver chunks.

Use maintained tolerances appropriate to field magnitude; do not demand bitwise CUDA equality after reordering reductions. Report numerical error and small-component absolute error rather than hiding discrepancies under a single boolean.

---

## 13. Epoch-50 measurement and continuation policy

### 13.1 Protocol

Use the same GPU and source implementation for 1407 and the explicit 1406 epoch-50 reference; benchmark 1804 in the same session when comparing cost. One model resident at a time.

- Inference: cases 0273/0653, Q=8192, receiver chunk 2048, two warmups, five synchronized repetitions, maps and profiler disabled.
- Report full forward, preparation-plus-one-query, P2 decode separately; do not subtract/sum overlapping timed regions as exact phase attribution.
- Training: B48/Q1024, real M1/M12 cases, canonical predicted-port loss, one warmup and three measured updates. Use one stated optimizer-state policy for all models; prefer restored checkpoint optimizer state for trained model comparison. Never mix fresh and restored measurements in the same ratio.
- Memory: baseline, absolute and incremental allocated peak; reserved memory separate. Retain the 1406 QE-checkpoint policy.
- Read sparsity: normal-executor fine rows, support counts, padded rows, grouping/compiler time, source-bank refresh time, and controller-build count.
- Validation: ordinary logged total/field/temperature losses plus four anchor near/interface measurements. No full 90-case extra inference study is required at epoch 50 beyond the normal training validation workflow.

No profiler is needed unless support shrinks without a timing improvement or a new hot allocation appears. Then use one bounded trace, not a kernel-rewrite campaign.

### 13.2 Decision, not an infrastructure gate

The budget decision asks whether continued training is informative at reasonable cost. It does not require a decisive accuracy victory at 50.

As **review bands**, not automatic runtime/CI blockers:

- last-ten field-validation median within roughly 25% of matched 1406, with improving loss and no severe near/interface regression;
- full forward, M12 time, and M12 memory within roughly 10% of 1406, preferably better in preparation;
- no nonfinite trajectory, frozen optimizer, loss/gradient mismatch, or persistent all-global-only prediction;
- both science mechanisms actually executed: one shared controller and normalized prototype-anchored query routing.

Do not require `R_support < 1` by epoch 50 if the trajectory is healthy. Nor may dense support be presented as success of sparse routing. If a review band is narrowly missed but other benefits justify the cost, explain the tradeoff in the research note. If the model is materially worse or unstable, stop at 50 and request direction. Do not add new losses/constraints to force a pass.

Continuation is the same Run 1407 with restored optimizer/RNG state, not a second run or a reset learning schedule. Use the maintained CLI's interpretation of total epochs and record the exact command.

### 13.3 Efficiency expectations

The primary initial target is lower controller preparation count and cheaper full forward **relative to 1406**, not a mandatory simultaneous victory over both Dense and legacy 1404. Retain Dense as the fidelity reference.

If sparse support is real but the best execution remains dense, report a learned sparse index without acceleration. If preparation improves but support stays dense, report successful phase sharing and unsuccessful fine-work sparsification. These are distinct outcomes.

---

## 14. Epoch-500 and later scientific evaluation

### 14.1 Primary accuracy

Use the same 90 development cases and exact/best policy separation. Report:

- pooled and equal-case fluid relative L2/MSE;
- median, p95, worst case, paired wins;
- near-interface and far-fluid error;
- U, V, pressure, vorticity, field temperature;
- internal temperature, surface temperature, heat flux;
- final port temperature and effective heat-transfer coefficient;
- existing geometry/count/heating-heterogeneity strata.

The accuracy target is specifically to avoid trading away near-interface, vorticity, pressure, and flux for a visually cleaner route map. Keep all-domain metrics secondary until their solid-region/channel attribution is established.

Report best-through-500 separately from exact 500. Do not declare every fluctuation convergence failure. Do not use future best checkpoints in an earlier-budget table.

### 14.2 Topology and useful routing

Measure on all 90 cases:

- nonempty groups by source type;
- source/query positive and effective degrees;
- query singleton fraction and temporal support turnover from saved milestones;
- prototype cosine/separation and query-logit spread;
- group mass, queries assigned to source-empty groups, and per-type overlap mass;
- phase agreement of A/h for 1407 and phase drift for 1406;
- R_support and R_executed by phase and source type;
- support variation between near-interface, wake/downstream, wall-adjacent, and far-field receivers using existing physical region definitions.

No target histogram, balanced occupancy, or expected number of mechanisms is imposed.

### 14.3 Small, well-defined interventions

Use 0273, 0653, 0680, and 0298, with one additional disagreement case selected transparently from the final paired table if needed.

1. **Uniform query routes** with learned memberships retained, recomputing overlaps/controls. Report physical error changes; this tests reliance on query selection, not superiority over another model.
2. **Neutral group modulation** while retaining learned routes. Set module/value/head-control modulation to identity/zero as appropriate. This tests information in group-conditioned responses, not topology.
3. **Phase-recomputed controller counterfactual** using the trained 1407 parameters and same normalized prototype query rule. This tests sensitivity to the phase-sharing assumption. It is out-of-training-distribution and is not a fair retrained dynamic-model baseline.

Perform P1-only and P2-only variants when attributing interface versus field effects. P2 cannot change earlier local outputs. Do not overinterpret tiny changes in already-computed quantities.

### 14.4 Omitted-source and physical influence checks

A sparse support cannot be validated from route weights alone. For excluded sources, compare frozen-mask versus rebuilt-index perturbations and record the resulting field/KPI differences.

Dense preparation mixes all sources before the reader. Therefore a source omitted from the final read may still influence retained contextual values. Do not claim full physical disconnection from zero reader incidence.

The 16 earlier physical-reference perturbation requests remain independent validation work. Use trustworthy outputs if supplied; otherwise state **Evidence Missing**. AD/FD agreement checks implementation derivatives, not CFD sensitivity accuracy.

### 14.5 Long-run decision

At 500, summarize convergence trends, fidelity/cost tradeoffs, and whether source sparsity is meaningful. Supply exact same-run continuation commands for 2500/5000 with intermediate checkpoints. Do not execute them without the user's instruction.

A continued Run 1406 remains a valuable mature comparator. Do not interrupt or automatically extend it in this 1407 task.

---

## 15. Visualization: show the new scientific question

Upgrade the existing board, not the plotting infrastructure.

Produce a compact four-panel board for the two standard anchors and one difficult case:

1. Physical geometry and source group assignments, with diagnostic centres labelled as such.
2. Source incidence and query incidence side by side, with **the same group identities**. Distinguish positive degree from effective degree.
3. One selected query: unique retained module/environment sources with edge weight rho; optionally show the cheap logical group paths in a separate inset. Do not draw them as duplicate expensive calls.
4. Phase and cost summary: controller builds `1` versus parent `3`, A/h phase agreement, live fine-state change across phases, R_support, R_executed, measured preparation/full/P2 time.

Include a small spatial heatmap of selected source count near modules versus far field. This is more informative than a large unreadable ledger screenshot.

For the all-domain discussion, add a mask/channel provenance table rather than another attractive aggregate chart.

Use existing plot styles, figure paths, and managed evaluation outputs. Do not generate duplicate boards for every intermediate code edit.

---

## 16. Output organization and research-code discipline

Required tracked deliverables:

```text
new opt-in model/config and focused tests
docs/reports/HONF_Run1407_Phase_Shared_Routing_Report.md
```

Generated tensors, figures, timing data, and profiler traces belong in the established ignored/managed evaluation trees, not beside source files. Reuse existing manifests and paths. Do not copy baseline predictions/checkpoints merely to create a new snapshot.

The report must separate:

- **source-derived findings** from [R1]/[R2];
- **new model hypotheses/equations**;
- **implementation measurements**;
- **ground-truth dataset accuracy**;
- **physical verification still missing**.

Use ordinary Git, the existing allocator, config validation, checkpoint loading, and targeted tests. Preserve current security and trusted-loading protections. Add no hashes, freezes, provenance services, approval machinery, or monitoring daemons. No new defensive constraint is necessary: the actual risks here are scientific misattribution, wrong equations, stale runtime tensors, and invalid measurements, handled by explicit design and ordinary tests/experiments.

Blocking infrastructure gates belong only at genuine irreversible/security/production boundaries. The 50/500 decisions here are research-budget judgments, not additions to CI or model code.

No all-or-nothing ritual around a threshold should replace running the physical model and measuring it.

---

## 17. Completion criteria and possible conclusions

The task is complete after implementing one model, executing the focused verification, running the authorized 50-to-500 sequence when justified, and delivering the report with actual commands and evidence paths.

Possible scientific conclusions must be allowed:

- **Phase sharing helps; routing remains dense.** Retain the efficiency result, but do not claim sparse physical execution.
- **Routing becomes selective; execution remains slower.** Identify the actual support/compiler/kernel cost. Do not infer a speedup from sparsity alone.
- **Both cost and accuracy improve.** Proceed to longer matched training and physical-reference checks; do not call a 500-epoch result final.
- **Near/interface fidelity degrades.** The static controller may be insufficient, or source/query alignment may be wrong. Do not immediately append a local branch or another loss. Use the specified counterfactuals to formulate one next hypothesis.
- **P0 index sharing is not adequate.** This is a negative result for phase-static control, not a proof that all hypergraphs or routing indices are unsuitable.
- **Query prototypes become selective but often select empty source groups.** Report per-type zero-overlap and gradient behavior. No hidden dense fallback or forced occupancy is allowed in the scientific graph.

Because this one trained candidate combines two scientific changes, a positive result does not isolate their individual trained effect. The bounded interventions explain sensitivity; a formal two-factor training ablation can be considered later, not silently added now.

---

## 18. References and methodological boundaries

### Supplied evidence

- **[R1]** `HONF_Run1406_Epoch500_Comparative_Evaluation.md` — exact-500 accuracy, execution, and population topology.
- **[R2]** `HONF_Run1406_Performance_Diagnosis_and_Exact_Optimization_Report.md` — phase attribution, accepted checkpoint boundary, and rejected low-value execution changes.

### Pinned implementation

All paths below are in `HONF_Proj/` at commit `3da43e607efb83c844eac1da5a692dd05a6e60f3`:

- `src/honf_forward_core/interface_fields/group_control_router.py`
- `src/honf_forward_core/interface_fields/group_control_pairwise.py`
- `src/honf_forward_core/interface_fields/core.py`
- `src/honf_forward_core/interface_fields/dense_pairwise.py`
- `src/honf_forward_core/interface_fields/three_term_context.py`
- `src/honf_forward_core/routing.py`
- `Case_ThermalChannel/src/channelthermal/interface_field_coupling.py`
- `tools/diagnostics/analyze_run1406_epoch500_comparison.py`

### Primary methodological references

1. Peters, Niculae, Martins. *Sparse Sequence-to-Sequence Models* (2019). https://arxiv.org/abs/1905.05702 — entmax probabilities and gradients. Does not guarantee low source-union cost or physical fidelity.
2. Zhang, Sennrich. *Root Mean Square Layer Normalization* (NeurIPS 2019). https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html — RMS scaling motivation. Run 1407 uses a parameter-free vector normalization, not a blanket replacement of model LayerNorm or a transfer of that paper's speed claims.
3. Roy et al. *Efficient Content-Based Sparse Attention with Routing Transformers* (TACL 2021). https://aclanthology.org/2021.tacl-1.4/ — query/source cluster-address consistency and sparse attention as an implementation/methodological precedent. Its task results do not establish the HONF physical-interface claim.

The phase-shared P0 controller, prototype anchoring, and application to this physical refinement loop are **proposed model choices**. They are not results established by these references.
