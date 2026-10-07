# HONF Run 1503-v2: reversible edge coalescence with moment-preserving fine interaction

## Decision and scope

**Starting repository:** `cosmos2w/ModularDT`, `agent/honf-core-next`.

**Source reviewed for this plan:** `aaee3fa9bcc2baf4760d34aebdd57c4344e14bc7`.

**Working scientific base:** Run 1502's `sparse_incidence_group_control_honf`, including its final environmental sparsemax refinement. Preserve Run 1501's normalizer as the existing alternative; do not change historical defaults universally.

**New research identity:** `1503-v2 / reversible_edge_coalescence`. The old Run 1503 adaptive-hyperedge-opening experiment remains an archived, failed experiment. Do not overwrite its managed directory, checkpoints, configuration, or reports. Use the normal allocator with a distinguishable name. If the allocator requires a unique numeric ID, retain `1503-v2` as the experiment label and use its assigned ID rather than bypassing the allocator.

This plan supersedes the architectural and launch instructions in `HONF_Run1502_Maturation_and_Run1503_Adaptive_Hyperedge_Opening_Plan.md`. Runs 1501 and 1502 have now finished 5000 epochs. **Do not resume either completed run.** Read their explicitly identified static checkpoints and reports only as needed.

The primary new experiment addresses **case-conditioned hyperedge formation**. Exact execution optimization is a separate, mathematically equivalent implementation track that can be applied to both the mature Run 1502 and the new model. An executor change does not by itself require a new training run.

The central design is:

> Merge redundant **query-access functions**, preserve source-resolved low-dimensional control moments, and keep every fine physical source. Do not replace fine fields with pooled hyperedge values, and do not execute an expensive physical pair once per group.

No experiment in this goal may run beyond **500 new training epochs**. Review the new run at 50 and 150, finish at or before 500, and provide a recommendation—not an automatic launch—for any subsequent 5000-epoch formal maturation.

---

## 1. Evidence being carried forward

### 1.1 What the mature comparison supports

The report `HONF_1404_1804_1501_1502_Mature_Comparison.md` uses saved validation-best-field checkpoints, not exact epoch-5000 endpoints. The selected checkpoints are 1404/e4890, 1804/e4738, 1501/e4689, and 1502/e4794.

| Quantity | 1404 | Dense 1804 | 1501 | 1502 |
|---|---:|---:|---:|---:|
| Pooled normalized fluid relative L2 | 0.03460 | 0.02896 | 0.03321 | 0.03153 |
| Near-interface pooled relative L2 | 0.03337 | 0.03504 | 0.04051 | 0.03687 |
| Far-fluid pooled relative L2 | 0.03839 | 0.02537 | 0.02899 | 0.02903 |
| Complete application, ms/case, reported native execution | 29.88 | 158.87 | 231.33 | 231.09 |

Run 1502 has lower pooled error than 1501, but wins only 49/90 individual fluid-field cases. It is not universally superior, and 1501 is better on the reported heat-flux metric. Select 1502 as the **next development base**, not as a universally established winner.

The mature structural facts are:

- Both 1501 and 1502 occupy **12/12** source-group slots in every evaluated case.
- Mean exact full-grid Kq is **3.313 / 3.371** for 1501/1502, respectively. Do not repeat the earlier epoch-500 claim that 1502 has lower Kq as a mature result.
- Environmental support is approximately **0.613 / 0.408** on the Q1024 organization sample.
- Module support increases from approximately **0.798 to 0.835**.
- Both rectangular readers execute **1,572,864 QE rows and 98,304 QM rows** per full Q8192 case.
- Historical logged training plus validation is approximately 25.55 h for Dense and 12.15 h for each new run. These were distinct training environments/source snapshots, not a controlled architectural speed experiment.

The 90-case split is development data also used for sampled validation. There is one seed per historical run. Source snapshots differ and some were dirty. Preserve these limitations; do not turn observed differences into a causal normalizer estimate or an independent generalization claim.

### 1.2 What the figures do and do not establish

The source affinities and query-support maps support concentrated, overlapping, spatially organized routing. They do **not** establish that the twelve global access functions are redundant. Three active groups at one query can coexist with twelve necessary groups across the full domain.

The upper 1404 map is dominant **pairwise contribution**, whereas 1501/1502 show dominant **query-router assignment**. Do not compare these as identical quantities or align group labels across models.

High-Kq ridges near group boundaries are consistent with overlapping access functions. They are not by themselves physical interaction boundaries, a defect, or a reason to eliminate a group.

### 1.3 Retire the previous 1503 abstraction for this round

The old 1503 introduced a coarse physical-value response plus opened per-group fine responses. Its e50 error, training cost, and application latency all regressed substantially. Do not patch that architecture into this candidate.

Two mathematical distinctions matter:

1. Partitioning source mass, `sum_k nu_j A_jk = nu_j`, does not make a mixture of **separately normalized group attentions** equal to one attention over the unique source union. Their partition functions differ.
2. Counting groups does not bound the total expensive work. Group-major execution may still repeat a `(q,j)` pair in several groups and pad each group to large batch dimensions.

The new model has neither a coarse physical-value branch nor per-group fine attention. It returns to the successful Run-1502 fine reader.

---

## 2. A concrete executor defect to repair immediately

At the reviewed commit, `group_control_pairwise.py::_read_environment` has this control flow:

```python
if self.executor_policy == "support_blocks":
    context, overlap_mass = self._read_environment_support_blocks(...)
    complete = False
elif ...:
    complete = True
else:
    complete = ...

if complete:
    context, overlap_mass = self._read_environment_rectangular(...)
else:
    context, overlap_mass = self._read_environment_partial(...)
```

Therefore `support_blocks` executes the block reader **and then** the gathered partial reader, overwriting the block result. Output/gradient parity can pass because the second implementation is also intended to be equivalent. A ledger derived only from support count can miss the double execution.

**Required repair:** mutually exclusive dispatch for all three paths, followed by their shared diagnostic reduction. Do not silently alter the historical rectangular formula. Add an executed call-count regression test using real reader computations plus instrumentation; do not substitute mocked outputs for numerical parity.

Also move the unconditional rectangular-policy decision ahead of `_complete_support` in `_module_execution_plan`. Currently the support test runs even when the answer is already fixed by policy; its Python Boolean conversions can introduce unnecessary synchronization in each inner receiver chunk.

Interpretation boundaries:

- The default rectangular mature accuracy/cost table is not invalidated by this optional-path defect.
- The earlier block-executor timings are not clean measurements of that block algorithm alone and need a corrected replay.
- The failed 1503 opening implementation has its own override; do not claim this defect explains its entire failure.
- Do not predict the speed of the repaired block reader. Measure it.

Use ordinary Git and focused tests. No new provenance service, snapshot, hash, approval layer, or immutable contract is needed.

---

## 3. Mathematical constraints on a meaningful merge

### 3.1 Kq does not determine Kcase

The quantities are distinct:

- `K_registered`: maximum proposal capacity, currently 12;
- `K_proposal`: occupied provisional groups before coalescence;
- `K_case,phase`: distinct groups after coalescence in a prepared physical phase;
- `Kq_coalesced`: groups used by one query;
- unique supported QM/QE pairs;
- executed QM/QE work, including padding and recomputation;
- latency.

Do not prescribe `K_case = mean(Kq)`, force a distribution of K across cases, or hard-code a smaller target K.

### 3.2 Naive simultaneous summation is wrong

The parent has

\[
\rho_{qs}=\sum_k\alpha_{qk}A_{sk}.
\]

A naive merge of groups a and b would use

\[
\alpha'_q=\alpha_{qa}+\alpha_{qb},\qquad
A'_s=A_{sa}+A_{sb}.
\]

Their product adds the cross terms

\[
\alpha_{qa}A_{sb}+\alpha_{qb}A_{sa}.
\]

For example, `alpha=(1,0)` and `A_s=(0,1)` have zero original overlap but overlap one after naive merging. This can expand physical support rather than reduce it.

### 3.3 A single averaged control vector is not generally sufficient

The parent many-body moment is

\[
n_{qs}=\sum_k\alpha_{qk}A_{sk}h_k.
\]

In general,

\[
\sum_{k\in C}A_{sk}h_k
\ne
\left(\sum_{k\in C}A_{sk}\right)\bar h_C
\]

for a source-independent average `h_bar_C`. Different sources mix the constituent controls differently. Discarding that dependence is another early-information-loss mechanism, even though the vectors are small.

The proposed merge preserves this source-resolved moment exactly **for the fused routing model**, as specified below.

---

## 4. New architecture: reversible query-functional edge coalescence

Suggested architecture key:

```text
coalesced_sparse_incidence_honf
```

Preserve all of the following:

- fine Dense-style MM/ME/EM preparation;
- the original contextualized module and environment source states;
- Run-1502 proposal and final source normalization;
- original source measures;
- the original low-dimensional group controls and fine modulation maps;
- Kmax=12 and D=16;
- one expensive fine response per unique physical pair, never per group;
- `C(q) = C_g(q) + C_M(q) + C_E(q)`;
- the common continuous reader for physical ports, refinement, and field queries;
- the original losses, optimizer, data, and predicted-port policy for the formal from-scratch experiment.

The only new scientific mechanism is a continuous coalescence layer between the parent source-controller preparation and query routing. The weighted routing normalization and moment-bank reduction below are necessary parts of that mechanism, not separate experiments.

### 4.1 Prepare provisional source groups exactly as in 1502

For one physical phase, obtain the parent quantities

\[
A^{M,0},\ A^{E,0},\ h^0,\ g^0,\ \mu^M,\ \nu^E,\ \kappa_0,
\]

and the prototype-anchored normalized query keys `t_k`.

For query geometry, use the **same effective type-specific centres as the parent query router**, including its fallback to the available source type when a group has no mass of the other type. Denote them `c_k^M,c_k^E`.

Let `s_g` be the existing positive geometry scale, one quarter of the coordinate-scale diagonal in the current parent.

The provisional access function is

\[
\ell_k(q)=
\frac{v(q)^\top t_k}{\sqrt D}
-
\frac{\|q-c_k^M\|+\|q-c_k^E\|}{2s_g},
\]

where `v(q)` is the parent's RMS-normalized query control. This function, rather than only the centroid or prototype ID, determines whether two edges are interchangeable from a query's perspective.

### 4.2 Build a low-dimensional descriptor of each access function

Use the explicit descriptor

\[
z_k=\left[
\frac{t_k}{\sqrt D},\quad
\frac{c_k^M}{\sqrt2\,s_g},\quad
\frac{c_k^E}{\sqrt2\,s_g}
\right]\in\mathbb R^{D+2d}.
\]

For ThermalChannel, this is 20 dimensions. It contains parameters of the entire continuous query-access function, not sampled query values. The plan therefore does not depend on the user's query grid or chunk size.

Fusing these descriptors produces shared access keys and geometry anchors. The resulting anchors are **routing anchors**, not necessarily the source mass centroids of the merged incidence. Keep those two geometries distinct in diagnostics.

Do not RMS-normalize fused keys a second time. Decode the key block by multiplying by `sqrt(D)` and use it directly. A second normalization changes the specified fusion map and can amplify near-zero keys. At zero fusion strength, the keys are exactly the parent's already-normalized keys.

### 4.3 Use source-footprint similarity to discourage inappropriate mergers

Define one nonnegative source-footprint matrix from both types:

\[
F_{sk}=\sqrt{w_s}\,A^0_{sk},
\]

with `w_s = mu_s^M/2` for module rows and `w_s = nu_s^E/2` for environment rows. Retain the parent's active-module mask. The factor one-half balances the source types rather than allowing 192 environment rows to dominate five module rows by count alone.

Define normalized footprint overlap

\[
\chi_{k\ell}=
\frac{F_{:k}^{\top}F_{:\ell}}
{\|F_{:k}\|\|F_{:\ell}\|},
\]

with ordinary numerical protection for genuinely empty columns. Work from the small Gram matrix `F^T F`; do not form any query-source-group tensor.

Over the active provisional groups, define

\[
\sigma_b^2=\frac1{K_0}\sum_k\|z_k-\bar z\|^2.
\]

Use a small numerical floor only when computing ratios. Let

\[
\widetilde w_{k\ell}
=\chi_{k\ell}^{2}
\exp\left(-\frac{\|z_k-z_\ell\|^2}{2\sigma_b^2}\right),\quad k\ne\ell,
\]

and set diagonal and inactive-group weights to zero. Normalize once by the maximum row sum, using a denominator of one for an all-zero matrix:

\[
w_{k\ell}=\frac{\widetilde w_{k\ell}}
{\max_i\sum_j\widetilde w_{ij}}.
\]

This is a source-conditioned similarity, not another learned gating network. Treat weights and scale as fixed **within** each convex solve, but retain their autograd dependence on the case and model when differentiating the implemented layer. Do not detach them and then claim full design-gradient agreement.

This footprint term is intentionally conservative. Two edges with distinct source roles should not be forced together merely because their centres are close. It may also mean the data support little merging; that is an empirical possibility, not permission to manufacture a smaller K.

### 4.4 Reversible convex fusion

Solve

\[
\boxed{
U^*=\arg\min_U
\frac12\sum_{k\in\mathrm{active}}\|u_k-z_k\|^2
+
\lambda_b\sum_{k<\ell}w_{k\ell}\|u_k-u_\ell\|_2.
}
\]

This is a convex-clustering / group-fused-lasso proximal problem. The quadratic fidelity term prevents arbitrary merging without a descriptor-distortion cost. The nonsquared L2 fusion term permits exact equality between groups; a squared-distance smoothing penalty alone would not produce exact fusion at finite strength.

**Initial experimental choice, not an established optimum:**

\[
\lambda_b(e)=\eta(e)\sigma_b,\qquad
\eta(e)=0.5\min(e/150,1).
\]

Use one fixed final strength and this single continuation schedule. Do not conduct a lambda/K/temperature sweep or choose strength from the 90-case test error. The schedule changes only the fusion layer; it is not an edge-death schedule or a train/eval hardening trick. After epoch 150, the mathematical model is fixed.

At `fusion_strength=0`, provide an exact parent-identity path. This is a correctness reference, not another formal run.

No prototype parameter is deleted, frozen, or multiplied by a permanent zero gate. At the next case/phase preparation, regenerate all provisional descriptors and solve again. Previously fused groups can separate as the case, geometry, or learned controller changes. This removes irreversible death; it does **not** prove that all forms of learned collapse are impossible.

### 4.5 Plan lifetime and case-level meaning

Create a `GroupFusionPlan` once per **case and prepared physical phase**, before receiver chunking. It contains the equivalence classes of fused provisional groups, multiplicities, fused access descriptors, and packed valid entries.

Keep this plan fixed for every query batch/chunk reading that prepared state. Arbitrary query order, query subset, and resolution must not change it.

Recompute the plan at the next physical preparation, retaining 1502's phase-local controller behavior. Report `K_case,P0`, `K_case,P1`, and `K_case,P2` separately; use final P2 for the headline formation comparison. This is a genuine case-conditioned plan, but it is **not** a claim that a single immutable partition is shared across the whole physical loop. Do not introduce P0-frozen routing as an unannounced second architectural change.

The registered bank stays at twelve proposal slots. The actual prepared incidence and query-access functions use `R = K_case,phase` merged groups. Twelve cheap provisional proposals may still be computed during preparation; do not describe that as eliminating all twelve-slot computation.

---

## 5. Exact quotient routing: preserve multiplicity, not just probability sums

Let the fusion partition be `C_1,...,C_R` and

\[
m_r=|C_r|.
\]

All virtual constituent groups in class `r` now have the same fused access logit `L_r(q)`, reconstructed from `u_r` using the parent's query formula.

Applying ordinary sparsemax only to the R logits would change competition merely because duplicate access functions were packed. Instead use **multiplicity-weighted sparsemax**.

Define the probability mass `a_qr` as the solution of

\[
\boxed{
\max_{a\ge0,\,\sum_r a_r=1}
\sum_r a_r L_r
-\frac12\sum_r\frac{a_r^2}{m_r}.
}
\]

Its explicit form is

\[
a_{qr}=m_r[L_r(q)-\tau_q]_+,
\qquad
\sum_r m_r[L_r(q)-\tau_q]_+=1.
\]

Define the access **density per constituent**

\[
b_{qr}=a_{qr}/m_r=[L_r(q)-\tau_q]_+.
\]

Distinguish these tensors throughout the code:

- `assignment` or `query_mass = a`: nonnegative, sums to one;
- `query_density = b`: used in overlap and control contractions, not required to sum to one;
- exact coalesced Kq is `count(a>0)`, equivalent to `count(b>0)`;
- `sum_r m_r 1[b_qr>0]` is virtual constituent support and must never be labeled coalesced Kq.

No literal expansion to twelve logits is needed in the normal reader. The expansion is useful only as a small test oracle.

### 5.1 GPU-friendly weighted sparsemax

Sort the R valid logits descending and reorder multiplicities identically. For a sorted prefix j,

\[
M_j=\sum_{i\le j}m_{(i)},\qquad
S_j=\sum_{i\le j}m_{(i)}L_{(i)},\qquad
\tau_j=(S_j-1)/M_j.
\]

Select the last valid prefix satisfying `L_(j) > tau_j`, then evaluate the positive parts in original order. Subtract the maximum valid logit first for numerical stability. Masked/padded multiplicities are zero; do not produce `0 * -inf` during prefix sums.

On a fixed active support, the probability-mass Jacobian is

\[
\frac{\partial a_i}{\partial L_j}
=m_i\delta_{ij}-\frac{m_i m_j}{\sum_{r\in\mathcal A}m_r},
\quad i,j\in\mathcal A,
\]

and is zero outside the active set. Ordinary differentiable tensor operations are sufficient; no straight-through estimator is needed.

---

## 6. Preserve physical controls with source-resolved merged moments

For both source types, define

\[
\boxed{\bar A^S_{sr}=\sum_{k\in C_r}A^{S,0}_{sk}},
\]

and

\[
\boxed{B^S_{sr:}=\sum_{k\in C_r}A^{S,0}_{sk}h^0_k}.
\]

`A_bar` is the actual merged source incidence. Its valid source rows still sum to one. `B` is a D-dimensional **control moment for each source and merged group**, not a pooled physical field value.

The new reader uses

\[
\boxed{\rho^S_{qs}=\sum_r b_{qr}\bar A^S_{sr}},
\]

\[
\boxed{n^S_{qs}=\sum_r b_{qr}B^S_{sr:}}.
\]

Keep each original `z_i^star`, `e_j^star`, source coordinate, source measure, fine QM affine, and environmental K/V row. Do not average these into R source values.

### 6.1 What is exactly preserved, and relative to what

Consider a **virtual fused-router model** with the parent's twelve fine source/control columns, but with the fused access logit repeated `m_r` times for class r. Sparsemax gives each member the same probability `b_qr`. Therefore

\[
\sum_{k\in C_r}\alpha^{\rm fused}_{qk}A^0_{sk}
=b_{qr}\bar A_{sr},
\]

\[
\sum_{k\in C_r}\alpha^{\rm fused}_{qk}A^0_{sk}h^0_k
=b_{qr}B_{sr}.
\]

The compact model and the virtual fused-router model have the same rho, n, source value control, and hence the same nonlinear physical response, up to floating-point arithmetic.

**This does not prove that nonzero fusion leaves the original Run-1502 predictions unchanged.** Fusion changes the access functions. The exact statement applies to packing after fusion. Ground-truth fidelity of the fusion itself is a research question to train and measure.

This distinction must appear in the implementation report.

### 6.2 Why B is necessary

Do not replace `B_sr` by `A_bar_sr * mean(h_k, k in C_r)`. That replacement generally loses information and invalidates the equivalence above.

The new hyperedge is a shared access function with a source-resolved control moment. It is not necessarily representable by one source-independent pooled control vector. This is a deliberate mathematical choice to preserve the many-body mechanism without forcing a lossy single-vector summary.

The parent already prepares an `A * h` module control bank and an `A * score_projection(h)` environment head bank. The proposed representation **reduces their group axis**; it does not introduce a new high-width physical branch.

### 6.3 Module branch

Prepare `B^M` as a GEMM-friendly `[B,R,M*D]` bank. Contract `query_density @ bank` and reshape to `[B,Q,M,D]`, as the parent does with K.

Use the original module fine function and its bias-free control gain. Keep the source-affine caching, final output weights and bias, source measures, and active-module normalization unchanged.

Do not replace the module branch with per-group fine calls. M is small and accuracy-sensitive; use its optimized rectangular implementation in this experiment.

### 6.4 Environment branch

Preserve source-local value modulation because

\[
\sum_k A^{E,0}_{jk}h_k^0=\sum_r B^E_{jr}.
\]

For the score-control projection, use linearity:

\[
Z^E_{jr,h}
=\sum_{k\in C_r}A^{E,0}_{jk}\,W_{\rm score,h}h_k^0,
\]

\[
\zeta_{qj,h}=\sum_r b_{qr}Z^E_{jr,h}.
\]

Prepare `[B,R,E*heads]` and contract directly into head space. Do **not** materialize `[B,Q,E,D]`.

The environmental score remains

\[
s_{qj,h}
=\frac{Q_{q,h}^{\top}K_{j,h}}{\sqrt{d_h}}
\bigl(1+\tanh\zeta_{qj,h}\bigr)
+g_h(q,x_j)
+\log\nu_j+\log\rho_{qj},
\]

with the parent's exact zero-support handling and one normalization over the unique supported source union. The final environmental output projection, bias, and overlap-mass envelope remain unchanged.

### 6.5 Preserve kappa calibration

Carry the parent's provisional `kappa_0` through the quotient. Do not recompute it from merged group masses as though R were the old registered K.

Likewise, do not replace the literal parent K factors in finalization by R without algebraically accounting for their cancellation with `kappa_0/K`. The net amplitude must remain the original `kappa_0` calibration in the virtual fused model.

Why this matters: collapsing duplicate labels changes `sum(pi_k^2)` even when the physical operator is identical. Naively recomputing kappa can introduce an artificial field-amplitude jump at a merge event.

`kappa_0` is now explicitly a retained controller normalization scalar, **not** the number of actual merged hyperedges. Report the latter from R.

### 6.6 Empty and padded groups

Only source-occupied provisional groups participate. A class containing at least one occupied constituent has positive source mass of at least one type. A module-only or environment-only group is valid; the absent source type contributes exactly zero.

Preserve the parent's behavior for unsupported query/source types. Do not create epsilon physical sources, renormalize an empty row to a synthetic uniform route, or require every group to contain both types.

---

## 7. Differentiable fusion implementation

### 7.1 Initial solver: small batched ADMM in ordinary PyTorch

There are at most 12 descriptor rows and 66 pair differences. Use an explicit incidence matrix T with one row `e_k - e_l` per unordered pair.

Solve

\[
\min_{U,V}\frac12\|U-Z\|_F^2
+\lambda_b\sum_e w_e\|V_e\|_2,
\quad TU=V.
\]

With scaled dual Y and ADMM penalty `r_admm = 1/Kmax`, use

\[
U^{t+1}
=(I+r_{\rm admm}T^TT)^{-1}
\left[Z+r_{\rm admm}T^T(V^t-Y^t)\right],
\]

\[
V_e^{t+1}
=\left(1-\frac{\lambda_b w_e}
{r_{\rm admm}\|(TU^{t+1}+Y^t)_e\|_2}\right)_+
(TU^{t+1}+Y^t)_e,
\]

\[
Y^{t+1}=Y^t+TU^{t+1}-V^{t+1}.
\]

For the complete Kmax graph, `T^T T = Kmax I - 11^T`. Thus the inverse is a constant buffer; at `r_admm=1/Kmax`, it is

\[
\tfrac12 I+\tfrac1{2Kmax}\mathbf1\mathbf1^T.
\]

Zero pair weights exclude inactive proposals. Keep their descriptor targets unchanged and exclude them from numerical grouping and multiplicity.

Initialize `U=Z`, `V=TZ`, `Y=0`. Start with **64 fixed iterations**, batched across cases, with no case-by-case Python solver loop. Use standard numerical safe division in vector shrinkage. Unroll the implemented solver for gradients; do not detach the result or substitute identity/straight-through gradients.

A 512-iteration float64 solve on a few small diagnostic examples is a useful numerical reference. It is not part of the production hot path. Check both primal and dual residuals; finite-step output is not automatically the exact optimizer.

No CVXPY, external C++ solver, new database, or graph-processing dependency is required. `torch.compile` on this pure-tensor solver is an optional exact optimization on the existing environment; compilation failures must leave a working eager path, not trigger an environment-wide upgrade.

### 7.2 Numerical equality and packing

The shrinkage variables V have exact zero vectors, whereas finite-iteration U may retain small residual differences. Form candidate fusion components from zero-V edges **and** numerically small corresponding U differences.

Use an initial FP32 numerical merge tolerance of `1e-5 * max(1, sigma_b)`. This is a solver/equality tolerance, not a learned sparsity threshold. In each candidate component, verify the maximum displacement to its component mean is consistent with that tolerance; avoid transitive near-equality chains whose endpoints are materially different. Conservative singleton retention is preferable to claiming a spurious exact fusion. Record residuals and the small projection displacement.

For accepted numerical components, use one differentiable component-mean descriptor. This makes their represented access functions exactly identical for packing. Gradients flow through the solver and component mean; integer labels have no derivative. Compare tolerance-halving and solver-iteration-doubling on a small actual batch to determine whether the numerical grouping is stable. Do not increase the tolerance simply to obtain fewer groups.

Implement small connected components on the Kmax graph with bounded tensor operations. Do not call a CPU graph library or copy descriptors/assignments to the host during every prepared phase. Any unavoidable dynamic-width host read should occur once per preparation, not inside query/head loops, and be included in timing.

### 7.3 Continuity and differentiability claims

For fixed nonnegative weights the convex problem has a unique solution. Its solution varies continuously with its inputs/regularization under the usual proximal formulation. The quotient makes representing an equality as one slot or several duplicate slots mathematically equivalent.

However, the full model contains sparse support changes, finite solver iterations, centre fallback logic, and numerical grouping. Do not claim global smoothness or solver-correct design sensitivities from local autograd alone.

Required empirical checks include both ordinary coordinate perturbations and perturbations that cross a merge/split event. Keep first-order and second-order support explicit. The first implementation may use the PyTorch path whenever higher-order derivatives are required.

### 7.4 No irreversible edge death

The raw proposals and parent moment paths remain live in training. A fused query-access class does not delete its source/control constituents or their parameter state. Different cases and later optimization steps can produce different fusion outcomes.

This is not a theorem that collapse cannot occur: excessive fusion strength, uninformative input data, or learned descriptor convergence can still yield R=1. Detect these behaviors through real training, reconstruction errors, and fusion statistics—not by imposing a permanent minimum K or a separate alive/dead gate.

---

## 8. Exact execution track: exploit the support that already exists

### 8.1 Formation and physical skipping are not the same objective

Even perfect coalescence need not lower the number of unique physical pairs. At exact duplicate access functions the quotient preserves that support, by design. Approximate fusion can also broaden a source union. Do not advertise smaller R as automatic QE sparsity.

Run 1502 already provides about 0.408 environmental support at maturity. Exact execution can exploit that without waiting for the new formation model to succeed. Conversely, a scientifically successful R-adaptive model may still need a better executor.

### 8.2 Establish a fair optimized rectangular reference first

After repairing dispatch, replay static mature Run-1502 checkpoints with:

- the reported native inner chunk 128;
- one larger evaluation-only inner chunk, initially 2048 when it fits.

Keep outer evaluator batching distinct from inner reader chunking. The existing outer value 32768 does not override the saved inner tile of 128.

Apply the same evaluation-only chunk to Dense 1804 and the new model for direct comparisons. Retain the native-protocol table separately. Do not attribute a shared chunking improvement uniquely to coalescence.

Preserve original training chunking initially. Measure actual full optimizer steps before changing activation-checkpoint boundaries or training tiles.

Remove only demonstrably redundant hot-path work: e.g. the already-fixed unconditional support test, repeated fixed source projections, or diagnostic-only calculations accidentally executed with maps off. Preserve trusted loading and existing security checks. Avoid a general cleanup campaign.

### 8.3 Corrected support-block replay is a measurement, not the next architecture

Benchmark the repaired pure support-block implementation once on the existing mature parent using maps-off timing and separate executed instrumentation. Reuse it as an exact alternative if it actually wins. Do not invest in another large query-signature infrastructure if fragmentation still dominates.

### 8.4 Main sparse execution prototype: one unique-pair QE reader

The principal candidate executor should:

1. Form the small scalar overlap rho from `query_density @ A_bar^T` (or parent alpha @ A^T).
2. Discover unique supported `(q,j)` pairs once on GPU. Integer support is dispatch metadata only.
3. Evaluate the inherited nonlinear geometry MLP **only on those unique pairs**, in large contiguous GEMM batches.
4. Prepare/read Q, K, and modulated V once per source/query role; never create `[I,heads,head_dim]` gathered K/V tensors for all I pairs in global memory.
5. Form the small live score-control and log-prior tensors on the same pair list.
6. Use one fused CSR/segmented QE reduction with stable per-query softmax over the **entire supported source union**, not per group.
7. Apply the unchanged output projection, bias, overlap-mass envelope, and kappa calibration in ordinary live tensor code.

Do not enumerate `(q,j,k)` paths. Do not repeat the fine geometry network for a pair that belongs to two groups.

The GPU win must come from removing expensive geometry rows and intermediate memory traffic, not just placing a sparse mask on a fully computed dense attention matrix.

### 8.5 Optional existing Triton route and its limitations

The repository already contains `interface_fields/kernels/qe_triton.py`. Inspect and reuse ideas/tests, but do not call it a drop-in HONF replacement:

- its existing primitive uses geometry bias and a live prior, not the present pair-dependent multiplicative score control;
- its documented scalar normalization path is FP64;
- it uses a row-oriented selected-source kernel;
- its first-order backward must be extended for the new multiplier if reused;
- geometry remains outside the fused kernel and must actually be evaluated on selected rows.

For the new optional path, implement the exact intended score

\[
s_{qj,h}=t_{qj,h}g_{qj,h}+b_{qj,h}+\log p_{qj},
\]

where `g=1+tanh(zeta)` and `p=nu*rho` on positive support. Preserve the current model's FP32 arithmetic policy; do not globally turn on reduced precision, TF32, AMP, or fast-math to manufacture a speedup. A separately tested FP64 reference may be used for numerical diagnosis.

A custom autograd primitive needs gradients for query, key, value, geometry bias, live prior, and multiplicative score control. For upstream output derivative dO and softmax weights a,

\[
D_{qj,h}=a_{qj,h}\langle dO_{q,h},V_{j,h}-O_{q,h}\rangle,
\]

\[
\partial s/\partial t:\quad Dg,\qquad
\partial s/\partial g:\quad Dt,\qquad
\partial s/\partial b:\quad D,\qquad
\partial s/\partial p:\quad D/p.
\]

Handle the last expression stably as positive priors approach zero; exact unsupported rows are excluded, not assigned a probability floor. The outer overlap-mass factor has its own derivative and must remain live.

The geometry MLP and routing contractions can initially remain ordinary PyTorch so coordinate, incidence, and control derivatives propagate through their existing code. Check end-to-end derivatives through actual P0/P1/P2, not just a stand-alone attention test.

Triton is optional. CPU imports, unavailable-GPU hosts, unsupported dtypes, and higher-order derivative requests retain a correct PyTorch reference. Do not upgrade the shared CUDA/PyTorch environment to force a kernel to run. Current public FlexAttention APIs are not proof that the project's installed version supports every captured-tensor gradient or geometry operation.

### 8.6 Source reuse and blocking

Prefer large query chunks, source banks retained in prepared state, and GPU-side unique support. Spatial/source ordering or block-sparse tiling is an exact implementation option only if it improves the measured path. Sorting must be undone before returning outputs; arbitrary query order must be supported.

Do not pad every group to the largest group or store one value bank per query. For any blocked prototype, count executed block rows including padding separately from unique supported rows.

Keep a rectangular fallback when selected execution loses at the actual shape. This is an execution policy based on measured supported shapes, not a new learned physical gate or a per-case online tuning service.

### 8.7 Performance ambition and its limits

Aim for a substantial full-application improvement over the **optimized** rectangular parent, not a few-percent difference on one case. A useful research target is roughly 20% or more lower application time with unchanged outputs; this is a target for interpreting results, not an automatic blocking or release rule.

Do not promise to match Run 1404's 29.88 ms just by reducing QE. Its architecture lacks several computations deliberately retained in the new fine-source model. The reported Run-1502 preparation-plus-one-query scope is already about 64 ms at native chunking. Preparation and decoding must both be examined; their separately measured timings should not be added as an exact decomposition.

---

## 9. Concrete code organization

Prefer a small opt-in implementation rather than changing historical classes wholesale.

Suggested new files:

```text
HONF_Proj/src/honf_forward_core/interface_fields/group_fusion.py
HONF_Proj/src/honf_forward_core/interface_fields/coalesced_sparse_incidence.py
```

`group_fusion.py` owns:

- descriptor construction;
- source-footprint affinity;
- batched differentiable fusion;
- numerical component packing;
- multiplicity-weighted sparsemax;
- the runtime `GroupFusionPlan` data.

`coalesced_sparse_incidence.py` owns:

- parent preparation reuse;
- merged A and source-resolved B/head-moment banks;
- access-function reconstruction and query mass/density;
- the unchanged fine reader with explicit runtime group width;
- compact formation diagnostics.

Use a new config profile/architecture key with only the fusion settings and new run identity. Keep all historical state-dict paths/loaders valid. The new coalescer requires **no new learned large network** and need not add any learned parameters.

Mechanical integrations:

- `config.py`, schema, config loader, profile registry: add the opt-in architecture and typed fusion settings;
- `interface_fields/core.py`: register the new backend and correctly concatenate/sum new query diagnostics;
- `group_control_pairwise.py`: repair dispatch and expose a minimal way to consume `query_density`, merged source control banks, and runtime R;
- tests: identity, quotient, fusion, physical-interface integration, executed dispatch, and gradients;
- diagnostics: extend existing population/evaluation reducers rather than create another reporting framework.

Important implementation hazards:

1. `_module_control_bank` currently assumes a bank can be rebuilt as `A*h`. That is false for merged B unless constituent controls agree. Store/use the explicit merged bank.
2. `_prepare_environment_bank_with_gain` must use the conserved source control `sum_r B_jr`, not `A_bar @ h_bar`.
3. Query probability mass and density must not be interchanged in rho, n, score controls, or diagnostics.
4. Capacity validation may still require Kmax=12 while tensor contractions must accept runtime R. Do not replace every `group_count` occurrence blindly.
5. Scalar ledgers add over chunks; per-query arrays concatenate; preparation-only state appears once per phase.
6. Reusing a prepared plan across query chunks is correct; caching it across optimizer updates or unrelated cases is not.
7. Hot-path functions must not repeatedly call `.cpu()`, `.tolist()`, or `.item()` inside query/head loops.
8. Parent `group_count_per_case` and `phase_occupied` describe the provisional bank. Preserve them under explicitly provisional names and emit the actual merged R/occupancy separately. A visualization that only renames twelve live columns is not coalescence.
9. The mathematical environmental B bank need not be materialized at full D width when the conserved source-value control and directly reduced head-space bank already provide all required contractions.

Do not delete the failed 1503 code or other historical architectures during this task. Excluding them from the new profile is enough. A later cleanup can remove unused abstractions after the new scientific core is established.

---

## 10. Numerical and model tests to execute

Tests support actual execution. They do not replace full model training or ground-truth evaluation.

### 10.1 Algebraic tests

- Zero-strength fusion matches the parent from identical weights on full physical forwards.
- Weighted sparsemax agrees with ordinary sparsemax on explicitly repeated logits for unequal class sizes.
- Weighted sparsemax has the correct Jacobian on fixed support.
- `A_bar` preserves source row sums and active/padded masks.
- rho, module n, environmental zeta, source value modulation, and nonlinear outputs agree with the virtual expanded fused model.
- Source-resolved moment conservation still holds when constituent h vectors are deliberately different.
- A naive `A_bar*h_bar` implementation fails the heterogeneous-control test, so this error cannot silently recur.
- Parent kappa and output biases survive merging correctly.
- Group/source permutations produce corresponding permutations without changing outputs.
- Empty source types, one active class, all singleton classes, and mixed batch R are finite and correctly masked.
- The final value path uses only original fine sources; no direct group physical-value decoder is added.

### 10.2 Fusion solver tests

- Compare 64 iterations with a longer float64 solve on small controlled examples and real descriptors.
- Measure primal/dual residuals and numerical grouping displacement.
- Test close groups, distinct groups, variable case partitions, and deliberate merge/split perturbations.
- Verify provisional parameters retain nonzero gradients where the parent physical paths supply them.
- Check differentiated solver output against finite differences away from nondifferentiable boundaries.
- Do not equate a fixed iteration count with guaranteed convergence.

### 10.3 Physical and design-gradient tests

Use actual prepared physical cases with small and large active module counts, predicted ports, and the full coupling loop. Execute forward, loss, backward, gradient clipping, and an optimizer update.

Check first gradients with respect to query coordinates, module coordinates, selected module properties, source encoders, raw group codes, and control modulation weights.

Perform short physical-coordinate sweeps spanning ordinary motion and at least one observed merge/split transition. Compare model AD with finite differences at two reasonable step sizes and retain absolute as well as relative error; tiny derivatives should not create misleading percentage failures.

These are surrogate self-consistency tests. They do not validate sensitivities against CFD unless independent solver-reference sensitivities are actually available. Do not label local AD agreement as physical derivative accuracy.

### 10.4 Executor tests

Exercise real reader code with instrumentation showing exactly which implementation executed. Check maps-on/maps-off parity, chunked/unchunked parity, and no duplicate unique pair work.

For any custom QE path, check first gradients to all six numerical inputs and through the full model. Preserve the reference tolerances and report actual discrepancies; do not loosen tolerances merely to rename a mismatch as a pass.

---

## 11. Execution order for Goal mode

### Work A — repair and replay the existing executor

Read the current report/code, repair the known fall-through and redundant policy test, execute their regression tests, and replay mature Run 1502 on the corrected alternatives at inner chunks 128 and 2048. Use current installed hardware/software and log it plainly.

This is an implementation-only result. It can produce a useful latency improvement even if new fusion later fails. Keep its measured results separate from the formation experiment.

### Work B — implement and execute the coalesced operator

Implement the specified fusion and quotient. Run actual synthetic algebra, actual case forwards, gradients, and optimizer steps. Use a bounded frozen replay of the mature parent to describe routing/output perturbations and computational behavior.

Do not reject a trainable new architecture solely because a nonzero-fusion frozen replay changes output. Do not claim that replay establishes trained accuracy either. Repair implementation failures before drawing scientific conclusions.

### Work C — launch one fresh 1503-v2

Launch **one from-scratch run**, with the parent's seed and training settings, the single specified fusion schedule, and the normal run allocator. Do not initialize it from mature 1502 or distill from Dense in this experiment. The frozen mature replays are diagnostics, not training initialization.

Why from scratch: it preserves a meaningful same-epoch architectural comparison with the stored 1502 e50/e150/e500 trajectory. A future compression/fine-tuning experiment can use the mature parent, but must be labeled separately.

Use the rectangular fine reader for training initially, with the new coalesced control banks. A mathematically equivalent faster executor can be adopted after real forward/backward execution demonstrates correctness and benefit; it must not silently change the learned operator.

### Work D — bounded exact selected-QE implementation

Develop the unique-pair fused executor in parallel with the formation run when resources permit. Apply it first to an existing static parent checkpoint, then to the new candidate. Do not await a 5000-epoch run or start another architectural training candidate for this executor.

Keep this bounded. If the correct reference and the implemented sparse path show no benefit, report that outcome and retain the faster reference; do not spawn a kernel autotuning/research infrastructure project.

---

## 12. Research reviews at epochs 50, 150, and 500

These are empirical research reviews, **not new software approval gates**. Use trends, uncertainty, and the separate scientific objectives. Do not abort merely because an arbitrary scalar threshold is crossed at e50.

The first 1502 review looked worse than 1501; its later mature result improved. Preserve that lesson. Numerical corruption, persistent non-learning, or an actual excessive-resource failure still warrants stopping and diagnosing the run.

### Epoch 50

Execute and inspect:

- training/validation losses and real updates;
- full-grid field and physical outputs on the established fixed panel;
- Kproposal, Kcase by phase, coalesced Kq, multiplicities, and fusion displacement;
- support and moment-bank behavior;
- current fusion strength and solver residuals;
- actual preparation/forward/optimizer timing.

At this point the fusion continuation is only one-third of its final strength. No-fusion behavior at e50 is evidence to understand, not a reason to force a smaller K. Small early accuracy regressions alone are not a stop criterion.

### Epoch 150

The fusion strength now reaches its fixed endpoint. Inspect the same quantities and compare the static same-epoch 1502 reference where available.

Ask:

- Has merging become genuine runtime coalescence rather than a diagnostic relabeling?
- Does R vary meaningfully across cases, or are all cases still 12/all cases identically collapsed?
- Are merged groups similar in their access functions and source footprint?
- Does source-resolved control preservation work in the actual physical loop?
- Is reconstruction learning normally, especially near interfaces and for heat flux?
- Has pair support unexpectedly expanded even though group count fell?
- Is the coalescer's preparation cost acceptable relative to the unchanged reader?

Continue the same candidate toward 500 when the observed mechanism and learning trajectory justify it. Do not conduct a second lambda/seed/normalizer run during this goal.

### Epoch 500, or an earlier documented scientific stop

Stop training no later than 500. Evaluate the exact endpoint and the saved validation-best-field checkpoint separately. For each selected checkpoint, obtain its own formation and cost measurements.

Run the full 90-case Q8192 development comparison using the maintained ground-truth evaluator, with the same normalization, physical masks, and predicted ports. Include:

- pooled and equal-case fluid relative L2;
- near/far fields;
- physical u, v, pressure, vorticity, temperature;
- internal/surface temperatures, heat flux, final ports and effective h;
- difficult-case tails and paired case differences;
- the prespecified checkpoint policy and its sensitivity.

Compare the new e500 trajectory to stored 1502/1501/Dense same-stage evidence under matched scopes. Mature parent values are useful ultimate targets and frozen implementation references, not equal-training-budget accuracy comparators to a fresh e500 run.

### Formation conclusions

Report separate conclusions, not a single score:

1. **Real adaptive formation:** runtime R is below capacity in cases where fusion occurs, and its case/phase distribution is measured. All R=12 means goal (1) has not been achieved. One identical R across all cases does not demonstrate count diversity, although topology/weights may still adapt.
2. **Fidelity:** quantify every change and uncertainty. No-loss accuracy is an objective to test, not a mathematical promise of nonzero fusion.
3. **Physical execution:** report which expensive rows were skipped and whether full application latency improved.
4. **Design suitability:** report derivative and merge/split behavior, with their limitations.

Do not introduce an arbitrary minimum K, a forced K histogram, or permanent prototype deletion to make the formation plot look successful.

### Recommendation on 5000-epoch maturation

At closeout, provide a justified recommendation to extend **this same new candidate** to a formal 5000-epoch endpoint, or to revise it first. Do not launch that continuation automatically.

A worthwhile recommendation would combine meaningful case-conditioned coalescence, competitive same-stage physical accuracy, acceptable derivative behavior, and a credible measured execution route. Goal (1) may succeed before a major latency win; say so explicitly rather than declaring it failed solely because goal (2) remains incomplete.

Conversely, lower R with degraded field/control behavior is not a success. Nor is a faster kernel with no actual coalescence evidence a success for goal (1).

---

## 13. Required evidence and timing discipline

Use existing artifact/report conventions. The minimum output is one coherent report with references to existing/static comparators and the new run's ordinary artifacts.

For formation, report at least:

- K_registered, K_proposal, K_case,phase and its across-case distribution;
- merged class multiplicities and source occupancy;
- full-grid coalesced Kq; virtual support separately;
- source degree, unique QM/QE support, logical path counts;
- query-logit distortion, rho/n distortion relative to the unfused proposal, and quotient parity relative to the virtual fused model;
- numerical solver residuals and merge projection displacement;
- representative physical-coordinate maps with active modules only;
- case-complexity and matched same-geometry/different-operating-condition examples where data permit.

For execution, report:

- preparation, prepared P2 decode, complete GPU forward, and application evaluator separately;
- inner receiver chunk and outer evaluator batch explicitly;
- warmups, repeated synchronized wall time, CUDA events, and per-case samples;
- incremental and total peak allocation; reserved memory separately if measured;
- actual fine geometry/MLP/dot-product rows, padding, and backward recomputation;
- selected topology/packing time and fusion solve time;
- compilation/setup cost separately from steady-state execution;
- at least small/large module cases, the full development population for final application results, and bounded existing large-shape execution tests.

Large synthetic shapes are execution evidence only; they do not establish extrapolated physical accuracy.

Keep maps and detailed diagnostic ledgers out of timed forwards. Use instrumentation in separate untimed runs, including actual function/kernel invocations rather than support-derived assertions alone.

---

## 14. What remains core, and what is explicitly not being added

Preserve:

1. Fine physical source contextualization before aggregation.
2. Shared source organization and continuous query access.
3. Source-resolved many-body control in a low-dimensional bank.
4. One physical response per source pair.
5. The three-term context and common physical-interface reader.

Add only:

1. A small reversible fusion layer and packed case/phase plan.
2. Multiplicity-correct query normalization and conserved moment-bank reduction.
3. A corrected, optionally fused exact executor.

Do not add:

- a coarse physical-value head;
- a local correction/bypass to recover lost accuracy;
- per-group fine attention;
- a second query router;
- source top-k, query top-k, new sparsity losses, or capacity-death gates;
- learned K targets or load-balancing losses that force occupancy;
- a new framework for runs, registries, approvals, hashes, snapshots, or monitoring;
- environment-wide dependency upgrades;
- additional seeds, normalizer sweeps, or competing architectural branches in this goal.

Use normal Git history, existing run management, strict existing checkpoint loading, and focused tests. Do not delete or weaken any existing security mechanism. New blocking process gates are unnecessary for this research task; preserve only the existing confirmations/security around irreversible mutations or production boundaries.

---

## 15. Literature and implementation rationale

These sources motivate individual tools; none establishes that the proposed HONF combination preserves ground-truth accuracy.

- Chi and Lange, **Splitting Methods for Convex Clustering**, Journal of Computational and Graphical Statistics (2015); preprint arXiv:1304.0499. The quadratic tether plus nonsquared pairwise norm produces fusion, with ADMM/AMA algorithms and continuity results. It does not guarantee that an arbitrary learned neural descriptor yields the desired physical clusters.
  - https://arxiv.org/abs/1304.0499
- Sun, Toh, and Yuan, **Convex Clustering: Model, Theoretical Guarantee and Efficient Algorithm**, JMLR 22 (2021). Supports weighted convex clustering and numerical treatment; no general-purpose solver dependency is required here.
  - https://jmlr.org/papers/v22/18-694.html
- Martins and Astudillo, **From Softmax to Sparsemax**, ICML (2016). Sparsemax supplies exact support with a tractable piecewise derivative. The multiplicity-weighted quotient in this plan is derived explicitly rather than assumed from ordinary sparsemax.
  - https://proceedings.mlr.press/v48/martins16.html
- Agrawal et al., **Differentiable Convex Optimization Layers**, NeurIPS (2019). Establishes the general idea of differentiating optimization layers. Here a small directly implemented tensor solver is preferable to introducing a broad optimization stack.
  - https://arxiv.org/abs/1910.12430
- Bolya et al., **Token Merging: Your ViT But Faster**, ICLR (2023). A useful analogy for combining redundant representations, not permission to average physical source fields or ignore multiplicity/normalization.
  - https://arxiv.org/abs/2210.09461
- Dao et al., **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**, NeurIPS (2022), and Gale et al., **MegaBlocks: Efficient Sparse Training with Mixture-of-Experts** (2022 preprint / 2023 publication). Motivate data-reuse-aware and sparse execution. Their reported speedups are not predictions for this small-E, geometry-MLP HONF reader.
  - https://arxiv.org/abs/2205.14135
  - https://arxiv.org/abs/2211.15841
- PyTorch FlexAttention documentation/blog: relevant only after checking the project's actually installed version and gradient support. It is not a mandatory dependency or a drop-in proof of compatibility.
  - https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
  - https://pytorch.org/blog/flexattention/

### Algebra-only checks performed while preparing this plan

A small independent CPU/PyTorch float64 calculation tested unequal multiplicities `[2,1,3]`, heterogeneous source memberships and group controls, nonlinear score modulation, softmax aggregation, source-value modulation, and first gradients. Compact-versus-virtual-expanded maximum differences were approximately:

- rho: `2.8e-17`;
- many-body moments: `5.6e-17`;
- nonlinear response: `2.1e-17`;
- tested first gradients: `1.9e-17`.

Three deliberately separated synthetic fusion examples recovered 3, 4, and 6 numerical clusters using the stated ADMM form. These are algebra/solver sanity checks only. They are **not** runs of ModularDT, not CFD evidence, not training results, and not GPU benchmarks. Reimplement the repository tests and execute the real model as specified above.

---

## 16. Definition of done

Complete the goal by delivering:

- the repaired dispatch and executed regression evidence;
- a measured corrected parent executor comparison;
- the reversible fusion and mathematically correct quotient implementation;
- actual physical forward/backward/optimizer execution;
- one fresh, clearly labeled 1503-v2 run, reviewed at 50 and 150 and stopped by 500;
- matched physical/formation/cost results with unambiguous scopes;
- a separate result for the exact selected-QE prototype, including negative results if it does not win;
- one report explaining what goal (1) achieved, what goal (2) achieved, remaining limitations, and the recommendation about later formal maturation;
- ordinary source/tests/report commits, without altering historical checkpoints or inventing new governance infrastructure.

The final scientific test is:

\[
\boxed{
\text{Can distinct case-dependent access functions coalesce reversibly,}
\quad
\text{while their source-resolved physical controls remain intact?}
}
\]

The execution test is separate:

\[
\boxed{
\text{Can exact zero support skip expensive unique physical-pair work}
\quad
\text{with enough GPU reuse to reduce complete inference time?}
}
\]
