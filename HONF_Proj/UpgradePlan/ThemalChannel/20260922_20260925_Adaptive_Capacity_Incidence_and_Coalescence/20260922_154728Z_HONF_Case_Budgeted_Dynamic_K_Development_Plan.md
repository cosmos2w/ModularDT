# HONF: case-budgeted dynamic group capacity

## Mathematical specification and Codex development plan

**Proposed scientific run:** 1409, `case_budgeted_group_control` — confirm that the ID is unused before creating a run; never overwrite an existing run.

**New opt-in architecture:** `budgeted_group_control_honf`.

**Inspected repository:** `cosmos2w/ModularDT`, branch `agent/honf-core-next`, remote head `5355ecb37466b35380fecd65e54f81594583e816`.

**Reference family:** Run 1406 for the physical computation and low-dimensional collective control; Run 1404 for the cheaper historical execution reference. Dense 1804 remains the strong accuracy baseline. Run 1407 supplies a useful example of sharing a case-level plan, but this experiment does **not** silently inherit its complete phase-frozen controller.

**Status:** this is a proposed model and research workflow, not an implemented repository change or a claim of measured acceleration. One independent CPU/float64 algebra check is described in Appendix A. No HONF training or GPU benchmarking was performed in preparing this specification.

---

# 0. The decision in one page

The user proposes replacing globally fixed K with a large fixed candidate capacity and case-specific live groups. Implement this as **one gate per case and candidate group**, shared by all module, environmental, and receiver incidences for that case.

The scientific question is:

> Can case-dependent group availability reduce the necessary routing/control capacity while preserving the fine physical information responsible for Run 1406's fidelity?

There is a separate computational question:

> Does eliminating those group columns, or the induced source pairs, reduce measured cost enough to offset the gate and packing work?

These questions must not be merged. Sparse activations, a smaller active-group count, lower routing rank, fewer physical pairs, and faster GPU execution are five different observations.

The first candidate is deliberately bounded:

| Item | First formal setting |
|---|---|
| Candidate capacity | `group_count = 12` |
| Fine hidden width | 256, inherited |
| Group-control width | 16, inherited |
| Optional gates | 11 independent case-conditioned hard-concrete gates |
| Always-available slot | One ordinary learned group, index 0 |
| Physical preparation | Run-1406 Dense MM/ME/EM, unchanged |
| Phase policy | Sample the gate plan once at P0; rebuild fine states, memberships and group controls at P1/P2 using that same plan |
| Source/query normalizer | Existing entmax15, temperature 1, with gate-aware logits |
| Fine response | Run-1406 unique-pair modulation and QE attention; no per-group fine duplication |
| Context | Exactly `C_g + C_M + C_E` |
| New objective | One expected optional-live-group cost, added once per case/forward |
| Training | One fresh run: 50-epoch research review, normally up to 500 total if healthy and worth continuing |
| Long continuation | User decision; no automatic 2500/5000 extension |

Why start with 12 rather than 32? It doubles the established six-group capacity without immediately paying over five times the maximum control width. Generic code must support 32 in a small synthetic shape test, without a `2**K` allocation. There is no capacity sweep in this goal. If the trained candidate demonstrably saturates 12 useful groups, a later capacity decision is justified by evidence, not by the assumption that overparameterization is always beneficial.

Do **not** add a second new model based on legacy 1404. Keep its mathematical and speed comparisons, but avoid introducing two different physical pipelines in the same experiment.

---

# 1. What the supplied ideas and previous evidence support

## 1.1 Correct mathematical foundation

For a batch case b and source type S in {M,E}, the learned nonnegative incidences are

\[
A_b^S\in\mathbb R_+^{S_b\times K_{\max}},\qquad
\alpha_b\in\mathbb R_+^{Q_b\times K_{\max}},
\]

and the unnormalized query–source overlap is

\[
R_b^S=\alpha_b(A_b^S)^T.
\]

If only `K_live(b)` columns participate, then

\[
\operatorname{rank}(R_b^S)\le\min(Q_b,S_b,K_{\mathrm{live}}(b)).
\]

This is a rank ceiling on a **routing matrix**. It is not the intrinsic physical rank of a PDE solution, a POD energy cutoff, or a rank bound on the final nonlinear field. The source preparation, fine pair functions, nonlinear attention, global term, and field decoder are additional computations.

In particular, a one-group routing matrix can be rank one while a dense, nonlinear fine response still has considerable representational capacity.

## 1.2 What needs qualification

- A large Kmax and sparse activations do not guarantee discovery of an optimal rank.
- Sparsemax/entmax zeros in separate source/query rows do not guarantee entire case-level columns disappear.
- For nonnegative row-normalized A or alpha, an L1 penalty is constant. A softmax/sparsemax-normalized gate vector has the same L1 problem.
- `ReLU(score - threshold)` can close entries, but a closed region has zero pathwise derivative. A larger positive bias in `ReLU(score + bias)` would instead open more entries.
- A case-specific number of groups is not necessarily proportional to module count. The dataset may support the same useful count for many cases.
- Fixed shapes are often convenient on GPUs, but variable shapes are not inherently unusable. Packing, maximum active width in a batch, kernel launch sizes, and source support determine actual performance.
- Inactive parameters still occupy model/optimizer storage. Column compaction reduces runtime activations and some computation; it does not shrink the stored prototype dictionary automatically.

## 1.3 The most important counterexample

If only one group remains, every valid source and query can belong to it:

\[
A_{s1}=1,\qquad\alpha_{q1}=1.
\]

Then

\[
R_{qs}=1\quad\forall(q,s).
\]

This has the smallest nonzero rank ceiling and **complete fine-pair support**. Therefore group-count minimization is not a fine-pair sparsity objective.

The mature 1406/1407 report already separates these phenomena: 1407 has mean query degree 3.340/6 and approximately 73.22% environmental logical support, but environmental execution remains rectangular. Do not repeat that reporting conflation.

## 1.4 Why this is not a repeat of old residual extraction

This plan contains no greedy group birth loop, no SVD/residual stopping procedure, and no epoch-dependent merge/split operation. It predicts a fixed-capacity gate vector in one parallel network evaluation. It also does not revisit all 63 subsets of a frozen six-edge checkpoint; the earlier subset study cannot establish what a newly trained gated model will learn.

---

# 2. Symbols and tensor roles

Suppress b where convenient.

| Symbol | Meaning |
|---|---|
| B | Batch size |
| M, E, Q | Active/padded module, environmental, and query counts as explicitly specified in a ledger |
| H | Fine feature width, 256 |
| D | Group-control width, 16 |
| Kmax | Registered prototype capacity; 12 in this experiment |
| c_k | Learned D-dimensional prototype |
| z_k | Continuous hard-concrete availability in [0,1] with possible exact zeros/ones |
| g_k | Boolean support indicator `1[z_k>0]`, not an additional learned weight |
| K_live | Number of positive availability gates |
| K_pack | Maximum live count in the current computational batch |
| a_k | Case-conditioned hard-concrete location logit; not query routing alpha |
| p_k | Probability that optional gate k is positive during stochastic training |
| A^M, A^E | Source memberships, normalized over available groups |
| alpha | Query memberships, normalized over available groups |
| omega^M, omega^E | Normalized physical source measures |
| mu_k^S | Weighted source occupancy of group k |
| h_k | Bounded D-dimensional collective control |
| rho_qs | Raw query–source overlap |
| n_qs | D-dimensional, unnormalized collective pair-control moment |
| eta | Gate-only reference distribution used solely for normalization |
| kappa | Continuous reference-overlap scale, distinct from K_live |
| p=0,1,2 | Physical preparation phase, not a training stage |

Do not overload one `group_count` variable to mean candidate capacity, packed width, live count, and response amplitude. They are different quantities in the new model.

---

# 3. Retained physical computation

## 3.1 Input and fine preparation

Reuse the maintained `InterfaceFieldCore` encoders. Let base encoded states be z_i, e_j, and global token g_global. At physical phase p, the incoming module state includes the current local-surrogate response.

Keep the existing simultaneous Dense preparation:

\[
a_i^{MM,(p)}=\frac{\sum_{\ell\ne i}\phi_{MM}(z_i^{(p)},z_\ell^{(p)},\Phi(x_i-x_\ell))}{1+M_a},
\]

\[
a_i^{ME,(p)}=\frac{\sum_j\nu_j\phi_{ME}(z_i^{(p)},e_j,\Phi(x_i-y_j))}{\sum_j\nu_j},
\]

\[
a_j^{EM,(p)}=\frac{\sum_i\phi_{EM}(e_j,z_i^{(p)},\Phi(y_j-x_i))}{1+M_a}.
\]

Masks and coordinate normalization follow the existing code. All three consume the same incoming phase state, not successively updated intermediate states.

They produce individually retained fine states

\[
\widetilde z_i^{(p)},\qquad \widetilde e_j^{(p)}.
\]

Do not replace these states with pooled group values, sparse quadrature, or a new local expert.

## 3.2 Preserve the low-dimensional source controls

Use Run 1406's existing source-state projection, position projection, and global-control contribution to form

\[
u_i^{M,(p)}\in\mathbb R^D,\qquad u_j^{E,(p)}\in\mathbb R^D,\qquad g_c\in\mathbb R^D.
\]

Here `u` is a source-control variable, not fluid velocity. This notation has no case-specific physical meaning.

---

# 4. One case-level availability plan

## 4.1 Gate context

After the existing P0 fine preparation, build one permutation-invariant gate context. Reuse those same P0 fine states and D-wide source controls for the subsequent assignments; do not run Dense P0 preparation a second time merely to predict the gates:

\[
\chi_b=\left[
\sum_i\omega_i^M u_i^{M,(0)},
\sum_j\omega_j^E u_j^{E,(0)},
 g_c,
\log(1+M_a)
\right].
\]

The projections already include geometry and physical input conditioning. This is a **control summary**, not a replacement for the fine values. Its limitations as a summary of layout complexity are a hypothesis to test; no claim is made that it measures true physical rank.

For k=1,...,Kmax-1, predict

\[
a_{bk}=F_{\mathrm{gate}}([\chi_b,c_k]).
\]

Use one shared two-layer MLP of hidden width 32, with one scalar output per prototype. No per-group expert networks or hard-coded rule `K=f(M)` is added.

## 4.2 Always-available ordinary slot

Set

\[
z_{b0}=1.
\]

This is one ordinary learned group, not a new global-value branch. It guarantees that row normalization always has an available column and avoids the discontinuous all-groups-closed case. It is **not** forced to own any source or query; entmax can assign it zero weight. There is no locality, load-balancing, or topology label associated with it.

This is an explicit architectural minimum `K_live>=1`, not a software/security gate or a claim that a universal physical mechanism exists. Without it, a realization with all groups closed has no nonempty simplex on which to normalize source/query memberships. Version control, typing, and tests can detect that realization but cannot make normalization on an empty set defined; rejecting or resampling it would add another execution policy. One available ordinary slot is the smallest explicit convention here. Optional-group permutation tests should preserve this special availability role, or permute the role metadata along with the columns.

## 4.3 Optional hard-concrete gates

For each case, draw one independent noise per optional group:

\[
U_{bk}\sim\mathrm{Uniform}(0,1),
\]

\[
s_{bk}=\sigma\left(\frac{\log U_{bk}-\log(1-U_{bk})+a_{bk}}{t_g}\right),
\]

\[
\boxed{z_{bk}=\operatorname{clip}_{[0,1]}\left[s_{bk}(\zeta-\gamma)+\gamma\right].}
\]

First settings:

\[
t_g=2/3,\qquad\gamma=-0.1,\qquad\zeta=1.1.
\]

Use stable `logit` arithmetic with floating-point-safe uniform sampling. Do not make a large numerical clamp that materially changes the distribution.

The probability of a positive optional gate is

\[
\boxed{p_{bk}=\sigma\left(a_{bk}-t_g\log(-\gamma/\zeta)\right).}
\]

These standard hard-concrete formulas come from Louizos, Welling and Kingma (2018). Their use as conditional hyperedge availability is the proposed adaptation here.

Initial optional-open probability: 0.95. Set the output bias to

\[
a_{\mathrm{init}}=\operatorname{logit}(0.95)+t_g\log(-\gamma/\zeta),
\]

with a small nonzero final-layer weight initialization. This starts permissively rather than forcing a prematurely small rank. It does not preset which six groups must survive.

## 4.4 Gate reuse and gradients

Sample the gate plan once at P0 and reuse exactly the same z, g, and compact-column mapping in P1/P2 and all receiver chunks of this forward call.

Do not resample on:

- each query chunk;
- each physical phase;
- activation-checkpoint backward recomputation.

The gate plan retains its autograd graph. Do not detach P0 gate inputs or live gate values. All physical phases contribute gradients to the same case-level decision.

At a new training forward, draw new gate noise. At a new case or optimizer step, recompute the gate context. There is no global cache of design-dependent tensors.

## 4.5 Deterministic inference policy

Use the standard deterministic hard-concrete estimate:

\[
\boxed{\bar z_{bk}=\operatorname{clip}_{[0,1]}\left[\sigma(a_{bk})(\zeta-\gamma)+\gamma\right],\quad \bar z_{b0}=1.}
\]

This is **not** the expectation of z and is not the same as substituting U=0.5 into the temperature-divided training sample. Keep these definitions separate.

The deployed active count is

\[
K_{\mathrm{live}}^{\mathrm{det}}(b)=\sum_k\mathbf1[\bar z_{bk}>0].
\]

Do not threshold at 0.5, replace remaining values by ones, or drop a small positive gate in the name of efficiency. Those change the deployed function.

There is a stochastic-training/deterministic-inference difference. It is intentional, disclosed, and must be measured from the first review. Hard-concrete does not remove this issue by definition. Validation uses the deterministic policy throughout; a small Monte Carlo audit is separate.

---

# 5. Availability-aware incidence — close a group everywhere

Let

\[
\ell^S_{sk}=\frac{\langle u_s^S,c_k\rangle}{\sqrt D}
\]

be the Run-1406-style source logits. Define b_k=log z_k for positive z_k, and mask entries with z_k=0 out of normalization.

Then

\[
\boxed{A^S_{s:}=\operatorname{entmax}_{1.5}(\ell^S_{s:}+b),\quad S\in\{M,E\}.}
\]

Module padding remains masked. Environmental source measures remain strictly positive on valid samples.

After group controls are built, query logits retain the Run-1406 form:

\[
\ell^Q_{qk}=\frac{\langle v_q,W_hh_k\rangle}{\sqrt D},
\]

\[
\boxed{\alpha_{q:}=\operatorname{entmax}_{1.5}(\ell^Q_{q:}+b).}
\]

Do not also introduce Run-1407 prototype anchoring/RMS scaling in this first experiment. That would add another independent query-geometry change.

Closed groups therefore satisfy

\[
A^M_{ik}=A^E_{jk}=\alpha_{qk}=0.
\]

No source or query wastes normalization mass on a closed column.

### Relation to the user's diagonal-gate formulation

Let g=1[z>0]. Because both incidence families already contain the same zero columns,

\[
\alpha\operatorname{diag}(g)(A^S)^T=\alpha(A^S)^T.
\]

The binary diagonal is algebraically redundant, not missing.

The continuous hard-concrete value z enters once as a shared availability prior in the incidence logits. Do **not** additionally multiply `rho` by a third continuous z factor; that would be a different relaxation and alter amplitudes unnecessarily.

The log-gate construction is a proposed choice of availability normalization, not an assertion that the original L0 paper prescribes this HONF-specific equation.

---

# 6. Separate group count from response amplitude

This is essential when converting the current code to dynamic capacity.

Run 1406 uses `group_count` in more places than tensor dimensions:

- multiplying source group moments;
- multiplying weighted occupancies entering h;
- scaling the module output, including its bias;
- scaling the environmental output by K times overlap mass.

Using Kmax or the batch's packed width in these expressions would make field amplitude depend on unused capacity or batch composition. Using the hard live count directly would introduce a discrete scale jump when a gate closes.

## 6.1 Gate-only reference distribution

Define

\[
\eta_b=\operatorname{entmax}_{1.5}(\log z_b)
\]

with the same zero-gate mask. At least one available slot ensures a nonempty simplex.

Define

\[
\boxed{\kappa_b=\left(\sum_k\eta_{bk}^{2}\right)^{-1}.}
\]

Properties:

\[
1\le\kappa_b\le K_{\mathrm{live}}(b)\le K_{\max}.
\]

- Appending exactly closed columns leaves kappa unchanged.
- Packing active columns leaves kappa unchanged.
- With n equally available groups, kappa=n.
- This is a continuous normalization scale, **not** the actual routing rank or the number to report as K_active.

Replace the relevant multiplicative K factors by per-case kappa in the **new architecture only**.

## 6.2 Uniform-routing reference

If all source and query content logits are constant across groups,

\[
A_{s:}=\alpha_{q:}=\eta,
\]

so

\[
\rho_{qs}=\sum_k\eta_k^2=1/\kappa.
\]

Hence

\[
\boxed{\kappa\rho_{qs}=1.}
\]

The uniform reference amplitude therefore does not change solely because fewer groups are available. This identity does not equate the complete nonlinear model to Dense; the control modulation can still be active.

## 6.3 Group control construction

Retain the normalized physical measures

\[
\omega_i^M=m_i/\max(M_a,1),\qquad
\omega_j^E=\nu_j/\sum_l\nu_l.
\]

For each phase p, compute

\[
\mu_k^{S,(p)}=\sum_s\omega_s^S A_{sk}^{S,(p)},
\]

\[
b_k^{S,(p)}=\kappa\sum_s\omega_s^S A_{sk}^{S,(p)}u_s^{S,(p)}.
\]

Then

\[
\boxed{h_k^{(p)}=\tanh\,F_h[b_k^{M,(p)},b_k^{E,(p)},\kappa\mu_k^{M,(p)},\kappa\mu_k^{E,(p)},g_c,c_k].}
\]

Compute/store h only for available groups in packed execution. In the full-width reference, zero out inactive h for clarity even though zero incidences already remove its paths. Do not divide group moments by tiny occupancy.

The gate plan is phase-shared; these memberships and h values remain phase-dependent. This retains Run 1406's physical adaptation instead of adding complete phase sharing as a second scientific variable.

---

# 7. Fine interaction and output equations

## 7.1 Routing and many-body control

\[
\boxed{\rho_{qs}^{S,(p)}=\sum_k\alpha_{qk}^{(p)}A_{sk}^{S,(p)},}
\]

\[
\boxed{n_{qs}^{S,(p)}=\sum_k\alpha_{qk}^{(p)}A_{sk}^{S,(p)}h_k^{(p)}.}
\]

The control remains unnormalized. Since h is bounded,

\[
\|n_{qs}\|_\infty\le\rho_{qs}.
\]

Do not replace it with `n/(rho+epsilon)`. This would change the operator and introduce a new small-overlap derivative issue.

Each fine query–source pair is evaluated at most once per forward read, irrespective of the number of group paths. Never materialize a Q-by-source-by-group-by-H tensor or instantiate one fine MLP per group.

## 7.2 Module branch

Reuse the prepared source/global first-affine split:

\[
a_i^{(p)}=W_z\widetilde z_i^{(p)}+W_gg_{\mathrm{global}}+b.
\]

\[
t_{qi}=\operatorname{GELU}(a_i^{(p)}+W_\Delta\Phi(q-x_i)),
\]

\[
\psi_{qi}=F_{\mathrm{tail}}\bigl(t_{qi}\odot[1+\tanh(B_Mn_{qi})]\bigr).
\]

Let

\[
S_q^M=\sum_i\omega_i^M\rho_{qi}^M\psi_{qi},\quad
G_q^M=\sum_i\omega_i^M\rho_{qi}^M,\quad a_M=M_a/(1+M_a).
\]

Retain Run 1406's exact bias convention with the new scale:

\[
\boxed{C_M(q)=\kappa a_M W_{\mathrm{out}}^M S_q^M+\kappa G_q^M b_{\mathrm{out}}^M.}
\]

The bias is not multiplied by a_M. Preserve this distinction in both reference and selected execution.

## 7.3 Environmental branch

Retain one projected K/V bank per environmental source and phase, with the existing source-local group-control value gain. There is no group-specific H-wide value bank.

Per-head content and scalar modulation are retained:

\[
s_{qj}^{a}=\frac{Q_q^a\cdot K_j^a}{\sqrt{d_h}}[1+\tanh(\zeta_{qj}^a)]+b_{\mathrm{geo}}^a(q-y_j),
\]

where zeta contracts group controls directly into head space rather than making a Q-by-E-by-D tensor.

\[
w_{qj}^{a}\propto\omega_j^E\rho_{qj}^E\exp(s_{qj}^{a}).
\]

Normalize only over positive valid weights. A receiver with no environmental overlap returns zero context, including output bias.

\[
o_q=\operatorname{concat}_a\sum_jw_{qj}^aV_j^a,
\quad G_q^E=\sum_j\omega_j^E\rho_{qj}^E,
\]

\[
\boxed{C_E(q)=\kappa G_q^E\operatorname{Out}_E(o_q).}
\]

Retain the measured complete-QE activation-checkpoint boundary.

## 7.4 Global term and field head

Reuse `ThreeTermInterfaceContext` unchanged:

\[
C_g(q)=F_g[\Phi(q),g_{\mathrm{global}}],
\]

\[
\boxed{\widehat U(q)=D_\theta\left(\operatorname{LN}[C_g(q)+C_M(q)+C_E(q)]\right).}
\]

No direct pooled hyperedge value, extra local path, coarse bank, or fine environmental interpolation is added.

---

# 8. What is dynamic, and what is not

```text
Physical case and base encodings
                |
                v
Existing P0 fine MM / ME / EM preparation
                |
                v
Case gate context --> one gate sample / deterministic estimate
                |                 |
                |                 v
                |        live group IDs + gate prior + kappa
                |                 |
                v                 v
          P0 memberships and collective controls
                |
                v
          Continuous P0 port read --> local physics
                |
                v
New fine P1 states + SAME gate plan
                |
                v
          P1 memberships / controls / refinement read
                |
                v
New fine P2 states + SAME gate plan
                |
                v
         One fine interaction per retained q-source pair
                |
                v
                 Cg + CM + CE
```

The gate count is case-specific and independent of which output query set is requested. Query support is still query-specific. Fine physical values and source memberships remain phase-dependent. No model parameter shape changes during training; runtime packing only changes the live column axis.

---

# 9. Objective and training semantics

## 9.1 One group-count regularizer

Use

\[
\boxed{
\mathcal L
=\mathbb E_U[\mathcal L_{\mathrm{physical}}(F_\theta(\text{case};z(U)))]
+\lambda\,\mathbb E_b\sum_{k=1}^{K_{\max}-1}p_{bk}.
}
\]

One Monte Carlo gate sample per case/forward is sufficient for the first run. The always-available slot contributes a constant one to total expected count and is omitted from the regularizer.

This penalizes expected optional group availability, not an L1 norm of a normalized membership row. It is not a FLOP or pair-count estimator.

There is no orthogonality, balance, entropy, manifold, repulsion, coverage-target, or sparsity-temperature loss. There is no warm-up schedule, staged freezing, distillation, or gradual topology change.

## 9.2 Choose one coefficient without a sweep

Before the managed run, use two real training batches and no optimizer updates to measure gate-logit gradients from the physical loss and expected-group count. Use one fixed noise realization per batch.

Choose one conservative coefficient so the median initial gate-logit gradient norm from the regularizer is about 2% of that from the physical objective:

\[
\lambda=0.02\,\operatorname{median}_{\text{two batches}}
\frac{\|\nabla_a\mathcal L_{\mathrm{physical}}\|_2}
{\|\nabla_a\mathcal L_{\mathrm{group}}\|_2}.
\]

This is a one-time scale calibration, not a search for a desired K histogram. If either gradient is numerically absent, investigate connectivity; do not create a huge coefficient by dividing by a numerical epsilon. Record the resolved scalar in the managed config and leave it fixed.

Restore the intended starting RNG state before the formal run. Do not use validation cases to tune lambda.

## 9.3 Live loss plumbing

Return one live per-case expected-count tensor from the P0 gate plan. The trainer adds its mean exactly once.

Do not:

- detach this tensor with visualization auxiliaries;
- add it once per query chunk;
- add it separately for P0/P1/P2;
- hide it inside field MSE;
- reuse the old dense induced-pair-cost evaluator.

Log physical loss, gate regularization, and their weighted sum separately. Continue selecting best-field checkpoints by the existing physical validation field metric, not by the new penalized total.

## 9.4 Validation and inverse-design implications

Normal validation uses deterministic gates. On four anchors at each review, compare deterministic predictions with a small eight-sample stochastic audit. Report disagreement and gate counts; do not call this stochastic ensemble calibrated uncertainty.

Inverse design uses the deterministic deployed operator. Positive gates, entmax routing, and clipping provide piecewise derivatives, not a smooth guarantee through all topology changes. Gradient tests use fixed noise for stochastic execution and separately test the deterministic path. Discrete module count still requires a distinct design-search treatment.

---

# 10. Honest computation model

Let Kpack be the maximum live count in a batch, and let P_M/P_E be the actual unique fine pairs evaluated.

A schematic cost is

\[
T\sim T_{\mathrm{Dense\ prep}}(M^2+ME)
+T_{\mathrm{gate}}((M+E)D+K_{\max}D)
+T_{\mathrm{control}}((M+E+Q)K_{\mathrm{pack}}D)
+T_{\mathrm{fine}}(P_M,P_E)
+T_{\mathrm{local\ physics}}.
\]

Closing group columns can reduce the control term. Reducing P_M/P_E requires actual source/query support separation. Retaining Dense preparation means that no count-only change can remove its cost.

Do not promise Run-1404 latency. Its physical preparation, environmental reader, and initial port pathway are different. The current proposal preserves Run-1406 fidelity-relevant information and therefore retains more work than 1404.

For any acceleration claim compare:

1. full-width versus compact execution of the SAME gates and SAME weights;
2. the candidate versus matched 1406 and Dense measurements;
3. group-control savings versus actual fine-pair savings.

The combined speedup cannot be attributed solely to gates unless those comparisons support it.

---

# 11. Execution plan without the six-bit table or a new sparse compiler

## 11.1 Full-width reference

Implement a simple masked Kmax reference first. This is the authoritative predictor for output/gradient tests. It keeps all columns in storage but masks closed groups before normalized assignments and contractions.

No speed claim is made for zero columns in this reference.

## 11.2 Compact group columns once per forward

Build a per-case list of positive gate IDs once after P0 gate generation. Pack each case into Kpack=max_b K_live(b), carrying:

- original prototype IDs;
- live gate values;
- valid packed-column mask;
- kappa;
- expected optional count for loss/reporting.

Use a batched gather so different cases can have different active IDs while sharing the same packed width. The physical module and environmental token axes retain their existing batching.

One group-level packing/index extraction per forward is allowed and must be timed. Do not perform host `.item()`/`.tolist()` decisions for each receiver or fine pair. If batch-max packing removes little work, report it; do not automatically add many case-count buckets or a dynamic compilation system.

Inactive prototypes remain registered parameters. Do not physically delete weights or rewrite optimizer groups.

## 11.3 No `2**Kmax` table

Do not call or generalize `SixBitSourceSupport` for this architecture. Leave it unchanged for 1407.

For any selected fine-pair reference, compute support from the current packed incidences:

\[
C_{qs}=\bigvee_{k\in\text{packed}}[(\alpha_{qk}>0)\land(A_{sk}>0)].
\]

A bounded scalar/Boolean matrix contraction or on-demand bit intersection is sufficient. Never enumerate all possible signatures. Never construct a q-group-source path list only to coalesce it afterward.

The first formal profile uses the fastest measured exact implementation: rectangular fine reads for dense support, selected unique-pair reads only where a bounded timing calibration shows they help. Record the choice independently of diagnostic requests.

Do not implement an elaborate auto-tuner. One measured crossover choice for the established shapes is enough. If selected execution remains slower, keep the rectangular implementation and explicitly report that physical sparsity was not converted to runtime savings.

## 11.4 Bounds and ledgers

For valid sources, the forward has at most one expensive fine evaluation per query–source pair. Padded rectangular work is recorded separately. Checkpoint backward recomputation is separate again.

Report at least:

- Kmax;
- per-case stochastic and deterministic K_live;
- Kpack per batch;
- actual group-state rows and query-group score rows;
- positive source/query incidence counts;
- logical q-group-source paths;
- unique supported pairs;
- actual executed valid pairs;
- actual padded rows;
- backward recomputed rows;
- gate/packing time, preparation time, reader time, full forward and training step.

No counter is allowed to substitute for a measured wall time.

## 11.5 Keep known successful execution work

Retain:

- outer receiver chunk ownership in InterfaceFieldCore;
- no redundant internal 128-query fragmentation of complete reads;
- prepared module source/global affine;
- GEMM-friendly module-control banks;
- environmental head-space control contraction;
- one K/V source bank, not one bank per group;
- complete-QE activation checkpointing;
- opt-in large diagnostics;
- ordinary dense GEMMs when support is dense.

No sampled environment bank, grid_sample, Triton/CUDA extension, or torch_scatter dependency is required by this goal.

---

# 12. Concrete code ownership

## 12.1 New reusable core files

Recommended small additions:

```text
HONF_Proj/src/honf_forward_core/interface_fields/case_group_budget.py
HONF_Proj/src/honf_forward_core/interface_fields/budgeted_group_control.py
```

`case_group_budget.py` owns:

- the shared gate MLP;
- hard-concrete sampling/deterministic estimate/open probability;
- the single always-available role;
- safe gate-aware normalization helper;
- gate-only eta/kappa;
- live-column packing metadata;
- immutable runtime `CaseGroupBudget` tensors.

`budgeted_group_control.py` owns:

- the new router using existing D-dimensional source/query computations;
- per-phase assignments and bounded collective state with kappa;
- backend integration with Dense fine preparation and Run-1406 fine readers;
- packed control banks and per-case output scaling;
- exact reference/compact execution seams.

Reuse tested tensor operations from existing readers. Small local helper extraction is acceptable; rewriting the whole interface-field family is not.

## 12.2 Important shape audit

Search every use of `self.group_count` in the reused parent methods. Classify it as:

1. registered capacity;
2. current tensor/group width;
3. historical response normalization.

Use Kmax only for prototype registration/gate generation, packed tensor shape for current contractions, and kappa for the normalization specified above.

In particular, `_prepare_from_fine`, `_module_control_bank`, `_environment_control_full`, `_module_finalize`, and QE finalization must not infer the numerical amplitude from a padded dimension.

Do not mutate `self.group_count` during a forward to impersonate the packed width. That would corrupt concurrent/prepared-state semantics and checkpoint recomputation. Carry runtime widths in the prepared state and use tensor shapes.

## 12.3 Existing core integration points

Modify only narrow opt-in branches in:

```text
src/honf_forward_core/config.py
src/honf_forward_core/interface_fields/core.py
src/honf_forward_core/interface_fields/types.py       # only if needed
src/honf_forward_core/interface_fields/__init__.py
```

Add the new architecture name. Instantiate its backend and the existing `ThreeTermInterfaceContext` only for that name.

Extend `prepare()` narrowly to accept a case-gate plan for this mode. Existing callers/defaults remain unchanged. Reuse the runtime plan channel where appropriate, but do not misrepresent a gate-only plan as Run-1407's phase-frozen source/control object.

Avoid parameterized generic mega-routers with many dormant experimental flags.

## 12.4 Generic objective

The generic core owns the expected-group-count calculation. The case trainer only combines its returned scalar with the existing physical objective.

Preferred new output is an explicit live tensor such as `case_group_budget_expected_optional_count`, returned once from the physical forward. Detached statistics use separate names.

Search the maintained trainer's actual loss assembly and auxiliary detachment paths before wiring this in. Do not assume `interaction_aux` is live: several historical backends detach it deliberately.

## 12.5 ThermalChannel-specific changes

Only the ThermalChannel wrapper should:

- construct physical ports and perform the existing local-surrogate/refinement sequence;
- thread the same `CaseGroupBudget` from prepared P0 into P1/P2;
- expose one live regularizer to the established loss assembly;
- keep all existing physical channel weights, normalization, and case feature meanings.

Relevant maintained files include:

```text
HONF_Proj/Case_ThermalChannel/src/channelthermal/interface_field_coupling.py
HONF_Proj/Case_ThermalChannel/src/channelthermal/training_tools/losses.py
```

Locate the actual caller of these loss helpers rather than assuming the helper file is the entire trainer.

No new walls, influence cones, resistance fields, region maps, or domain geometry rules are introduced.

WindFarm and other case plugins must remain unchanged. The new scalar regularizer should be reusable without embedding temperature or velocity semantics in the core.

---

# 13. Configuration and run setup

Create one maintained profile, for example:

```text
src/config_core/forward/budgeted_group_control_honf_context.json
```

Inherit the existing Run-1406 scientific settings. The following fragment is a **schema proposal**: implement it consistently with the existing loader rather than treating it as a complete standalone JSON file.

```json
{
  "forward_architecture": "budgeted_group_control_honf",
  "interface_model": {
    "group_count": 12,
    "group_control_dim": 16,
    "source_normalizer": "entmax15",
    "query_normalizer": "entmax15",
    "module_temperature": 1.0,
    "environment_temperature": 1.0,
    "query_temperature": 1.0,
    "receiver_chunk_size": 128,
    "activation_checkpointing": true,
    "case_group_budget": {
      "enabled": true,
      "gate_hidden_dim": 32,
      "hard_concrete_temperature": 0.6666666666666666,
      "stretch_lower": -0.1,
      "stretch_upper": 1.1,
      "initial_optional_open_probability": 0.95,
      "always_available_group": 0,
      "normalization": "gate_reference_overlap"
    }
  }
}
```

The single calibrated regularization coefficient is stored as a numeric value in the resolved run/loss configuration before the managed run. Do not leave an unresolved string in a formal JSON config or silently run with the objective disabled.

Retain the parent defaults:

- H=256, message width 128, four heads;
- seed 0, AdamW/shared optimizer policy;
- learning rate 3e-4, weight decay 1e-5;
- AMP disabled unless a separate user instruction changes the established policy;
- gradient clipping at 1;
- predicted ports from the beginning; no curriculum;
- unchanged 600/90 dataset split;
- ordinary checkpoint milestones 10,50,100,250,500,1000,2500,5000, without authorizing the later milestones automatically.

The numeric `group_count=12` is a storage/candidate capacity. Never print it as the case's learned active count.

---

# 14. Focused verification before a managed run

Execute ordinary tests and the following compact additions. These are numerical correctness tests, not new production-release gates or process infrastructure.

## A. Probability and gate tests

- Hard-concrete empirical nonzero frequencies match the analytical p formula on a small CPU simulation.
- Deterministic formula matches its specification; it is not replaced by a Bernoulli threshold or by an unannounced noise-free variant.
- Group 0 is available in every case but can receive zero membership.
- Source/query rows normalize over available groups; closed columns are exactly zero.
- Padded module rows remain zero.
- Positive gate log-priors and entmax gradients are finite on ordinary and small-positive-gate fixtures.
- A sampled closed group can receive nonzero support on another sampled forward; do not promise every group revives empirically.

## B. Algebra and shape tests

- With six groups forced fully open, lambda disabled, and matching parent tensor weights, the router/read arithmetic reduces to Run 1406 within floating-point tolerance.
- Adding exactly closed capacity columns does not change outputs.
- Full-width and packed execution agree on outputs and all connected first derivatives for the same gate realization.
- Cases evaluated alone and in a mixed-live-count batch agree within numerical tolerance.
- No operation allocates `2**Kmax`; test capacities 6,12,32 in small synthetic fixtures.
- No tensor with a Q-by-source-by-group-by-H payload exists.
- K=1 gives a dense rank-one routing matrix; the diagnostic must not call this sparse fine execution.
- The bound `abs(n).max(-1)<=rho` holds.
- kappa*rho=1 in the gate-only uniform-routing reference.

## C. Real physical and gradient tests

- Execute one real predicted-port P0/P1/P2 forward/backward/update.
- Verify the exact same gate plan object/noise is reused through physical phases and backward recomputation.
- Verify dynamic fine values AND source memberships/group controls can change across phases; only availability is held fixed.
- Gate expected-count loss is live and added once, independent of query chunk count.
- Verify gradients through gate logits, source/query scores, h, pair modulation, fine values, module coordinates, and ordinary field heads.
- Test compact/reference parity with fixed random gate noise rather than comparing unrelated sampled masks.
- New-mode checkpoint resume restores optimizer/RNG behavior; old modes still strict-load their own checkpoints unchanged.

Do not require a parent checkpoint to strict-load into the new architecture with extra gate parameters. For the reduction test, copy matching tensors through a documented test fixture. Trusted historical checkpoint loading is not weakened.

---

# 15. Three bounded work stages

## Stage I — implement and execute the new operator

1. Read the current code, supplied ideas, and mature reports.
2. Implement one gate/router/backend path and one profile.
3. Run focused algebra/shape tests and an actual real-batch GPU update.
4. Calibrate lambda once on training data, recording the result.
5. Measure full-width versus compact execution on identical initialized/materialized weights and fixed gates at two representative shapes.

The initial timing is a code-path check, not a learned sparsity result. Do not replace code execution with theoretical cost estimates.

## Stage II — one fresh run, first review at 50

Use one genuinely free physical GPU. Do not stop, restart, move, or compete with any ongoing user-authorized run, including 1408. Use an ordinary worktree if the active checkout belongs to a running job.

Create proposed Run 1409 once, from scratch. If the ID is occupied, do not overwrite it; report the collision and use the established run-allocation procedure with explicit labeling.

Train to 50 epochs. Use existing logs/checkpoints only. No new monitoring service, second seed, same-goal sweep, alternate gate type, or automatic capacity increase.

At 50 review:

- deterministic validation physical losses and their recent trend;
- stochastic-training versus deterministic-validation discrepancy on fixed anchors;
- expected, sampled, and deterministic group-count distributions;
- evidence of case dependence, not just Monte Carlo variation;
- packed width and actual control/fine execution;
- training and inference cost versus 1406;
- severe numerical, memory, or fidelity problems.

The target is a promising accuracy–cost tradeoff, not a predetermined K histogram. There is no requirement to beat every Dense metric at epoch 50. Roughly within 10–15% of parent cost and within about 50% of its recent early field error is useful review context, not an automatic acceptance certificate. Near/interface behavior and convergence can justify a modest early lag.

Continue the SAME run and optimizer to 500 when the model is healthy and the measured tradeoff is worth learning more about. If it is clearly unstable, much slower without benefit, or the deterministic policy is unusable, pause and report rather than changing the architecture mid-run. Ordinary numerical failures and GPU ownership remain hard safety/resource boundaries; heuristic scientific scores are not new blocking software gates.

## Stage III — epoch-500 evidence and handoff

Evaluate exact 500 and the separately saved best-field-through-500 checkpoint. Reuse maintained exact-budget baseline artifacts where available. Run the standard 90-case evaluation once per distinct candidate checkpoint policy; if the policies select the same file, do not duplicate the evaluation.

Provide a continuation recommendation and an unexecuted resume command. Do not extend beyond 500 or launch a fixed-capacity ablation/another seed without user authorization.

---

# 16. Scientific evaluation

## 16.1 Fidelity

Compare primarily against matched-budget 1406 and Dense 1804. Include 1404 as a cheaper architectural reference, explicitly noting its different initial-port and environmental computations.

Report:

- pooled fluid SSE/MSE/relative L2;
- equal-case median, p95, worst case, and paired wins;
- near-interface and far-fluid metrics;
- per-channel quantities, especially vorticity, temperature, and pressure;
- internal/surface temperature, normal heat flux, outside temperature, effective h;
- available engineering KPIs and predefined strata.

Separate training-validation metrics from the full-grid physical evaluation. Do not combine checkpoint selection policies or use a lower penalized total as evidence of better reconstruction.

## 16.2 Dynamic capacity

Report per case:

\[
K_{\mathrm{expected}}=1+\sum_{k>0}p_k,
\]

\[
K_{\mathrm{sampled}}=\sum_kg_k,
\qquad
K_{\mathrm{det}}=\sum_k\mathbf1[\bar z_k>0].
\]

Also report:

- live but source-empty groups;
- source-occupied groups for M and E separately;
- groups actually reached by the evaluation query set;
- fractional positive deterministic gates versus gates at one;
- gate support margins and mode disagreement;
- gate/prototype gradients and relative updates.

If every deterministic case chooses the same count, the model may still choose different groups, but the data have not demonstrated variable count. If every optional group closes, that is not a successful dynamic-K result merely because the count is small.

## 16.3 Routing rank without a full Q-by-S SVD

For weighted routing R, let

\[
U=\operatorname{diag}(\sqrt{w_q})\alpha,
\qquad V=\operatorname{diag}(\sqrt{\omega_s})A.
\]

Take thin QR factorizations U=Q_U T_U and V=Q_V T_V. Nonzero singular values of the routing matrix equal those of

\[
T_U T_V^T.
\]

The SVD is therefore at most Kpack-sized. Do not construct a full-grid routing matrix solely to compute rank.

Report a numerical rank with the stated relative tolerance and an entropy effective rank from explicitly specified singular-value weights. Neither is physical energy or a proof of an optimal K.

## 16.4 Actual physical work

Measure for M and E separately:

\[
P_S=\#\{(q,s):\rho_{qs}>0,\ s\text{ valid}\},
\qquad R_S=P_S/(QS_{\mathrm{valid}}).
\]

Keep this separate from actual rectangular/selected fine rows and padding. Closing entire groups without reducing R_S is adaptive control capacity, not fine-interaction sparsity.

## 16.5 Useful bounded interventions

On four existing diagnostic anchors, perform only:

1. full-width versus compact execution with fixed identical gate values — exact implementation comparison;
2. deterministic versus eight fixed stochastic gate draws — mode-mismatch analysis;
3. fully opening gates with fixed weights — frozen capacity-sensitivity test, not a retrained comparator;
4. uniform query routing over the same available gates, using eta — routing-reliance test;
5. removing collective fine modulation while keeping incidences/gates — many-body-control reliance test.

A large intervention effect proves dependence, not superiority or physical causality. Do not select lambda, a checkpoint, or a new mask by minimizing these held-out errors.

---

# 17. Performance protocol

Use one currently free GPU and remeasure candidate/parent under the same software, dtype, flags and batch tensors. Load one model at a time for headline allocation.

Inference:

- 0273 and 0653;
- Q=8192;
- outer receiver chunk 2048;
- two warmups, five synchronized repetitions;
- deterministic gates, no diagnostic maps or profiler during headline timing.

Measure full physical forward, preparation-plus-one-query, prepared P2, and allocated/reserved peaks separately. Do not sum nested timings.

Training:

- established real B48/Q1024 M1 and M12 workloads;
- one warmup and three disposable updates;
- matched fresh/restored optimizer policy stated explicitly;
- current physical loss and one new gate cost;
- gate, packing, prepare and reader timing on a separate bounded trace if needed.

For controller scaling use a small synthetic shape check at Kmax=6/12/32 with **fixed supplied gates**. This is not an additional trained capacity study and not physical generalization evidence.

A useful final result must distinguish whether it saved:

- controller column work only;
- padded batch work;
- unique fine physical pairs;
- activation memory;
- or actual full-forward/training wall time.

---

# 18. Visualization

Extend a maintained board instead of creating a new visualization package.

Display:

1. the full candidate bank with deterministic gate values; unavailable columns greyed out;
2. source M/E memberships normalized on available columns;
3. query assignment, selected query, and positive source union;
4. K_expected, K_det, Kpack, effective routing rank, support ratios and actual executed rows;
5. global physical geometry and current errors where existing tools provide them.

Do not silently renumber a different active subset as the first K groups without retaining the original-ID legend. Gate labels are learned indices, not named physical mechanisms. Annotate the always-available role without implying it owns all global physics.

Use existing high-resolution/vector output conventions. Keep figures/arrays in managed evaluation directories, not source folders. Reuse datasets/checkpoints in place.

---

# 19. Deliverables and boundaries

Deliver:

- one new opt-in backend/gate module and profile;
- focused tests and actual physical execution evidence;
- one managed training trajectory, reviewed at 50 and at most 500 total;
- `HONF_Proj/docs/reports/_bk/20260922_165525Z_HONF_Case_Budgeted_Dynamic_K_Evaluation.md`;
- one managed evidence root referenced by the report;
- actual launch/resume commands and the resolved coefficient;
- clear outcome classification and missing evidence.

Use the existing allocator and evidence directory convention; do not invent a new manifest hierarchy if one already exists. No duplicate baseline snapshots or checkpoint copies. Ordinary Git history and existing checkpoint metadata suffice.

Preserve security and trusted loading. Do not add cryptographic hashes, contract freezes, release gates, approval daemons, monitoring services, or infrastructure. New numerical tests check new math; they are not a substitute for training or measurements.

---

# 20. Outcome interpretation

**Competitive fidelity, fewer case-dependent live groups, lower control cost:** supports adaptive routing capacity. Call it a controller improvement unless fine pairs also decrease.

**Competitive fidelity and fewer actually executed fine pairs:** supports the stronger adaptive-computation claim; verify end-to-end time, because selected execution may still be slower.

**One/few groups with dense fine reads:** the model chose a low-rank but dense weighting rule. It has not discovered a sparse physical organization.

**Gate counts vary only during stochastic training:** stochastic noise is not case adaptation; deterministic deployment has not demonstrated the claimed property.

**Deterministic validation is poor but stochastic samples are good:** deployment mismatch is unresolved. Do not hide it with post hoc threshold tuning or a long continuation.

**Same count for all cases with good accuracy:** a legitimate model-selection result; do not force variation to make a desired figure.

**Lower cost but degraded interfaces/tails:** it is a tradeoff, not a successful preservation of Run-1406 fidelity.

**Poor results at 500 with improving curves:** distinguish incomplete convergence from numerical collapse. A longer run is a user decision, not an automatic rejection or an automatic rescue.

**Gate overhead exceeds savings:** do not begin a kernel project just to make the gate count look valuable. The hypothesis may not pay off at this workload scale.

A later fixed-count retrained control is necessary for a strong publication claim that *case adaptivity*, rather than generic regularization or changed normalization, caused the advantage. That additional formal run is outside this bounded first goal.

---

# Appendix A. Independent algebra check performed for this plan

A small CPU/float64 script, independent of the repository runtime, checked:

- normalized gated entmax rows;
- full versus compact routing;
- invariance to 20 appended closed columns;
- `kappa*rho=1` for uniform content logits;
- `abs(n)<=rho`;
- one-group dense-support counterexample;
- six-open-group reduction to the parent routing;
- thin-QR singular values versus full routing SVD;
- positive-gate/normalization finite-difference gradients;
- the hard-concrete positive-probability formula against 200,000 scalar draws.

Observed maximum routing/packing discrepancies were below 1.2e-16; appended-column discrepancy below 2.8e-17; the six-group scale error was below 1.8e-15. The gradient check passed. The maximum Monte Carlo probability discrepancy was approximately 6.75e-4.

These are algebra checks, not proof of neural-network convergence, stable long-run gradients, physical accuracy, GPU speed, or repository correctness. Codex must still execute the real model.

---

# Appendix B. Evidence and methodological references

## Repository files inspected at the stated commit

- `HONF_Proj/src/honf_forward_core/interface_fields/group_control_router.py`: D-dimensional source/query controls, learned prototypes, phase-shared state definitions.
- `HONF_Proj/src/honf_forward_core/interface_fields/group_control_pairwise.py`: one fine call per pair, explicit K amplitude factors, control banks, complete/partial readers.
- `HONF_Proj/src/honf_forward_core/interface_fields/group_control_support.py`: six-bit/64-entry table; intentionally not general to arbitrary K.
- `HONF_Proj/src/config_core/forward/group_control_pairwise_honf_context.json`: Run-1406 scientific profile.
- `HONF_Proj/Case_ThermalChannel/src/channelthermal/training_tools/losses.py`: existing case loss interface and older pair-cost integration; the latter must not be reused as a group-count objective.

Repository reference: https://github.com/cosmos2w/ModularDT/tree/5355ecb37466b35380fecd65e54f81594583e816/HONF_Proj

## Supplied reports

- `HONF_Run1406_Run1407_BestBefore5000_Comparative_Evaluation.md`: mature fidelity/cost tradeoffs, checkpoint sensitivity, logical versus executed support, and phase sharing.
- `HONF_Run1406_Performance_Diagnosis_and_Exact_Optimization_Report.md`: phase-dependent controller cost and accepted complete-QE checkpointing.
- `HONF_Run1408_Hypergraph_Quadrature_Epoch500_Evaluation.md`: sampled-reader results; not a basis for replacing the fine environment reader in this experiment.
- Historical Run-1701/phase-2 restart findings supplied in the conversation: numerical collapse and frozen-bank selection failure, not evidence that dynamic case capacity is impossible.

## Primary literature

1. Martins & Astudillo (2016), *From Softmax to Sparsemax*. https://proceedings.mlr.press/v48/martins16.html — sparse simplex projection and its derivative; not automatic rank/model selection.
2. Peters, Niculae & Martins (2019), *Sparse Sequence-to-Sequence Models*. https://aclanthology.org/P19-1146/ — entmax, including 1.5-entmax.
3. Correia, Niculae & Martins (2019), *Adaptively Sparse Transformers*. https://aclanthology.org/D19-1223/ — learnable attention sparsity. The arXiv identifier 1909.00015 in the supplied comment refers to this paper, not the first entmax paper.
4. Louizos, Welling & Kingma (2018), *Learning Sparse Neural Networks through L0 Regularization*. https://arxiv.org/abs/1712.01312 — hard-concrete samples, expected positive-gate count, deterministic estimator. The conditional HONF gate placement and kappa normalization above are proposed here, not empirical results of that paper.
5. Official hard-concrete reference implementation: https://github.com/AMLab-Amsterdam/L0_regularization — cross-check formulas and numerical conventions; do not import an obsolete training stack.
6. Fedus, Zoph & Shazeer (2022), *Switch Transformers*. https://www.jmlr.org/papers/v23/21-0998.html — conditional computation and the importance of dispatch/capacity behavior. This is not an architectural reproduction of Switch, and no expert balancing loss is adopted.
