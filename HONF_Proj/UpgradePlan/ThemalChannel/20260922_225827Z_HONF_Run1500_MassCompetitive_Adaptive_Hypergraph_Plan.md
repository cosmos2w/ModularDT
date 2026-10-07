# Run 1500 — Mass-Competitive Adaptive Hypergraph HONF

## 0. Purpose

Run 1500 starts a clean experimental line after the three Run-1409 attempts.

The latest occupancy-adaptive Run 1409 established two important facts:

1. **Meaningful hypergraph organization can form.** Environment groups became spatially compact and query routing respected learned group geometry.
2. **Exact positive occupancy is not a useful definition of active K.** With many environment tokens, tiny positive entmax tails populate almost every candidate column, so `Kplan=12` remains full even though each source uses only a few effective groups.

Run 1500 changes only this part of the mathematics:

> Hyperedge existence is determined by competition between the **total learned physical masses of the hyperedges**, not by whether their mass is exactly nonzero and not by a separate learned gate network.

The intended hierarchy is

\[
\boxed{
K_{\max}\text{ candidate groups}
\rightarrow
\gamma_b\text{ case hypergraph}
\rightarrow
\alpha_q\text{ query subgraph}
\rightarrow
\rho_{qs},n_{qs}\text{ fine physical interaction}
}
\]

The model should remain simple: no hard-concrete gate, no group-count loss, no sparsity schedule, no sampled quadrature, no sequential residual extraction, and no extra coarse/local branch.

The main scientific question is:

> **Can the hypergraph's own learned source mass decide how many groups are significant for a case, while preserving spatial organization and fine physical fidelity?**

---

## 1. Run identity

Use a new run family rather than another Run 1409.

Run 1409 has already tested three distinct dynamic-K definitions: explicit hard-concrete availability, delayed dense-to-sparse hard-concrete availability, and exact-positive source occupancy. Run 1500 introduces a different mathematical object: **mass competition**.

Proposed architecture:

```text
forward_architecture = "mass_competitive_group_control_honf"
```

Managed run:

```text
run-id = 1500
run-name = mass_competitive_adaptive_hypergraph
```

---

## 2. Preserve the successful model components

Retain from Run 1406 / occupancy Run 1409:

- `InterfaceFieldCore`;
- Dense MM/ME/EM source preparation;
- fine hidden width \(H=256\);
- control width \(D=16\);
- registered capacity \(K_{\max}=12\);
- learned group prototypes;
- one geometry-refinement step;
- Run-1406 group-conditioned fine module interaction;
- Run-1406 environmental K/V attention;
- one expensive fine evaluation per physical query-source pair;
- three-term context \(C(q)=C_g(q)+C_M(q)+C_E(q)\);
- normal P0/P1/P2 physical coupling;
- predicted ports;
- existing physical losses, optimizer and dataset;
- activation checkpointing.

Do not introduce explicit group-count penalties, pair-cost losses, entropy/balance/repulsion losses, hard top-k budgets, hard distance cutoffs, dynamic sequential group generation, sampled environmental interpolation, or custom kernels during the formation experiment.

---

## 3. Registered candidate groups

Let

\[
K_{\max}=12.
\]

Let

\[
c_k\in\mathbb R^D,\qquad k=1,\dots,K_{\max}
\]

be learned group prototypes.

Initialize the 12 prototype rows with approximately orthogonal normalized vectors in \(D=16\). No prototype is special or permanently active.

---

## 4. Source control states

Retain the Run-1406 low-dimensional source controls after Dense physical preparation:

\[
u_i^M\in\mathbb R^D,\qquad u_j^E\in\mathbb R^D.
\]

They continue to include source state, position encoding and global operating context.

Use normalized physical source measures

\[
\sum_i\omega_i^M=1,\qquad \sum_j\omega_j^E=1.
\]

Inactive padded modules have \(\omega_i^M=0\).

---

## 5. Stage I — content proposal

Compute content logits over all registered candidates:

\[
\ell^{M,0}_{ik}=\frac{(u_i^M)^\top c_k}{\sqrt D},
\qquad
\ell^{E,0}_{jk}=\frac{(u_j^E)^\top c_k}{\sqrt D}.
\]

Use unit-temperature 1.5-entmax:

\[
P^M_{i:}=\operatorname{entmax}_{1.5}(\ell^{M,0}_{i:}),
\]

\[
P^E_{j:}=\operatorname{entmax}_{1.5}(\ell^{E,0}_{j:}).
\]

Inactive module rows remain zero.

This is a proposal organization only. It does **not** define active K.

---

## 6. Stage II — self-derived geometry

Compute proposal masses:

\[
\mu^{M,0}_k=\sum_i\omega_i^M P^M_{ik},
\qquad
\mu^{E,0}_k=\sum_j\omega_j^E P^E_{jk}.
\]

For every group with positive source-type mass,

\[
r^{M,0}_k=\frac{\sum_i\omega_i^M P^M_{ik}x_i}{\mu^{M,0}_k},
\qquad
r^{E,0}_k=\frac{\sum_j\omega_j^E P^E_{jk}y_j}{\mu^{E,0}_k}.
\]

Define the joint proposal centre

\[
\boxed{
r^0_k=
\frac{\mu^{M,0}_k r^{M,0}_k+\mu^{E,0}_k r^{E,0}_k}
{\mu^{M,0}_k+\mu^{E,0}_k}
}.
\]

When only one source type has positive mass, that source type supplies the joint centre. Groups with no proposal mass are ignored by geometry refinement.

---

## 7. Stage III — one geometry refinement

Keep the geometry principle that produced coherent clusters in the latest Run 1409.

Let the physical coordinate scale be \(L\). Define

\[
s_{\rm geo}=0.25\sqrt{\sum_dL_d^2}.
\]

Geometry biases are

\[
b^M_{ik}=-\frac{\|x_i-r^0_k\|}{s_{\rm geo}},
\qquad
b^E_{jk}=-\frac{\|y_j-r^0_k\|}{s_{\rm geo}}.
\]

The pre-competition source assignments are

\[
\boxed{
\widetilde A^M_{i:}
=\operatorname{entmax}_{1.5}(\ell^{M,0}_{i:}+b^M_{i:})
},
\]

\[
\boxed{
\widetilde A^E_{j:}
=\operatorname{entmax}_{1.5}(\ell^{E,0}_{j:}+b^E_{j:})
}.
\]

Use only one refinement step. Do not add iterative clustering or another geometric loss.

---

## 8. Hyperedge physical mass

Compute pre-competition masses

\[
\widetilde\mu^M_k=\sum_i\omega_i^M\widetilde A^M_{ik},
\qquad
\widetilde\mu^E_k=\sum_j\omega_j^E\widetilde A^E_{jk}.
\]

Define the joint case mass

\[
\boxed{
\pi_k=\frac12\left(\widetilde\mu^M_k+\widetilde\mu^E_k\right)
}.
\]

Because source measures and assignment rows are normalized,

\[
\pi_k\ge0,\qquad \sum_k\pi_k=1.
\]

Interpret \(\pi_k\) as the fraction of the case's source organization represented by group \(k\).

---

## 9. Mass competition

Do not use exact positivity and do not use a separate gate network.

Define

\[
m_k=\log(\pi_k+\epsilon_m),\qquad \epsilon_m=10^{-8}.
\]

Apply parameter-free sparsemax:

\[
\boxed{
\gamma=\operatorname{sparsemax}(m)
}.
\]

Thus

\[
\gamma_k\ge0,\qquad \sum_k\gamma_k=1,
\]

and some groups may satisfy \(\gamma_k=0\).

Define

\[
\boxed{
\mathcal G_b=\{k:\gamma_{bk}>0\},
\qquad
K_{\rm case}(b)=|\mathcal G_b|
}.
\]

This is Run 1500's dynamic K.

There is no explicit target count, no L0/L1 group loss, no temperature schedule and no special group. A group survives only because its learned source mass is competitive within that case.

---

## 10. Why log mass is used

The current occupancy run showed that tiny positive tails make exact-positive occupancy useless.

Log mass converts multiplicative mass differences into additive score differences:

\[
\pi_a/\pi_b=100
\Rightarrow
\log\pi_a-\log\pi_b=\log100.
\]

This distinguishes tiny tail groups from important groups without an arbitrary hard mass threshold.

The first experiment fixes the transform exactly as above. Do not sweep a mass temperature in the managed run.

---

## 11. Final source assignments

Mass competition must affect the same matrices that constitute the hypergraph.

Let

\[
g_k=\mathbf1[\gamma_k>0].
\]

For positive \(\gamma_k\), define the case prior \(p_k^\gamma=\log\gamma_k\).

Final source memberships are

\[
\boxed{
A^M_{i:}
=\operatorname{entmax}_{1.5}
(\ell^{M,0}_{i:}+b^M_{i:}+\log\gamma;\,g)
},
\]

\[
\boxed{
A^E_{j:}
=\operatorname{entmax}_{1.5}
(\ell^{E,0}_{j:}+b^E_{j:}+\log\gamma;\,g)
}.
\]

Implementation must mask zero-\(\gamma\) columns before evaluating logarithms.

A closed case group therefore cannot retain source membership.

---

## 12. Final group state

Recompute source masses from final assignments:

\[
\mu^M_k=\sum_i\omega_i^M A^M_{ik},
\qquad
\mu^E_k=\sum_j\omega_j^E A^E_{jk}.
\]

Recompute module/environment centres from the final assignments.

Build the low-dimensional group control \(h_k\) with the same Run-1406 many-body inputs: module moment, environment moment, source masses, global control and learned prototype.

Do not add the group state directly to the field.

---

## 13. Capacity-invariant normalization

Use the case competition distribution itself:

\[
\boxed{
\kappa_b=\frac1{\sum_k\gamma_{bk}^2}
}.
\]

Then

\[
1\le\kappa_b\le K_{\rm case}(b).
\]

If \(n\) case groups have equal importance, \(\gamma_k=1/n\) and \(\kappa=n\).

Use \(\kappa\) in the historical group-moment/output-amplitude positions where the parent uses fixed group count.

Keep separate in all diagnostics:

\[
K_{\max},\quad K_{\rm case},\quad \kappa,\quad \operatorname{rank}(R),\quad P_{\rm support},\quad P_{\rm executed}.
\]

---

## 14. Phase sharing

The P0 case competition defines a structural plan containing active original prototype IDs and P0 \(\gamma\).

Reuse this case-level plan through P1/P2.

At later phases:

1. fine MM/ME/EM source states refresh;
2. source controls refresh;
3. source memberships are recomputed only over the P0 active prototype IDs;
4. P0 \(\log\gamma\) remains the case-level importance prior;
5. phase-local group centres and controls refresh;
6. fine module/environment values refresh;
7. query routing refreshes.

Do **not** recompute case-level sparsemax competition at P1/P2 in the first experiment.

This keeps dynamic K stable during one physical solve while allowing physical content to evolve.

---

## 15. Query routing

Compute the ordinary Run-1406 query control \(v_q\).

Use current phase group control and source centres:

\[
\ell^Q_{qk}
=
\frac{v_q^\top W_hh_k}{\sqrt D}
-
\frac12
\frac{\|q-r^M_k\|+\|q-r^E_k\|}{s_{\rm geo}}.
\]

Then apply the case prior:

\[
\boxed{
\alpha_{q:}
=\operatorname{entmax}_{1.5}
(\ell^Q_{q:}+\log\gamma;\,g)
}.
\]

There is no query sparsity loss.

Case-level mass competition and query routing have distinct roles:

- \(\gamma\): which hyperedges exist for the case;
- \(\alpha_q\): which existing hyperedges this query uses.

---

## 16. Fine physical interaction

Keep the Run-1406 many-body factorization exactly.

For source type \(S\),

\[
\boxed{
\rho^S_{qs}=\sum_{k\in\mathcal G_b}\alpha_{qk}A^S_{sk}
},
\]

\[
\boxed{
n^S_{qs}=\sum_{k\in\mathcal G_b}\alpha_{qk}A^S_{sk}h_k
}.
\]

Evaluate the expensive fine physical function only once per supported query-source pair.

Do not reintroduce a separate \((q,s,k)\) expensive path.

The final context remains

\[
\boxed{C(q)=C_g(q)+C_M(q)+C_E(q)}.
\]

---

## 17. Scientific interpretation

Run 1500 must not claim that sparsemax support is an optimal physical rank.

It estimates:

\[
\boxed{\text{case-level significant hyperedges under learned source-mass competition}}.
\]

The nonlinear field can have much higher functional complexity.

Likewise \(K_{\rm case}<K_{\max}\) does not guarantee fewer physical query-source pairs. Support and execution must be measured separately.

---

## 18. Frozen prelaunch diagnostic

Before launching managed training, use the latest occupancy-Run-1409 epoch-50 checkpoint.

From its existing geometry-refined source assignments, compute

\[
\pi_k
\]

and apply exactly

\[
\gamma=\operatorname{sparsemax}(\log(\pi+10^{-8})).
\]

This is one fixed structural diagnostic, not a parameter sweep.

Across all 90 cases report:

- resulting \(K_{\rm case}\) histogram;
- relation to module count;
- prototype selection frequencies;
- \(\kappa\) distribution;
- retained pre-competition mass on surviving groups.

The purpose is only to confirm that the proposed transform is nondegenerate on already learned source organizations.

If every case maps to \(K=1\) or every case remains \(K=12\), stop before the managed run and report. Do not automatically tune a temperature.

---

## 19. Implementation structure

Add a new opt-in backend rather than modifying historical occupancy/gate modes.

Recommended files:

```text
src/honf_forward_core/interface_fields/mass_competitive_router.py
src/honf_forward_core/interface_fields/mass_competitive_group_control.py
```

Reuse components from:

```text
group_control_router.py
group_control_pairwise.py
occupancy_group_router.py
```

Prefer moving only genuinely generic geometry helpers into shared functions. Do not refactor unrelated architectures.

---

## 20. Runtime plan object

Create an immutable runtime object such as

```text
MassCompetitiveGroupPlan
```

containing:

- registered `Kmax`;
- active original prototype IDs;
- packed IDs and packed-valid mask;
- P0 `gamma`;
- `K_case`;
- `kappa`;
- P0 source masses;
- P0 source centres for diagnostics.

There are no gate logits, gate probabilities, stochastic noise, schedules or count-loss tensors.

---

## 21. Support representation

Kmax=12 fits in a 16-bit support word.

For positive final source/query incidence,

\[
m_s=\sum_k2^k\mathbf1[A_{sk}>0],
\qquad
m_q=\sum_k2^k\mathbf1[\alpha_{qk}>0].
\]

Then

\[
(q,s)\text{ supported}\iff m_q\;\&\;m_s\ne0.
\]

Do not build a \(2^{12}\) table. Only query support masks actually present in a receiver chunk may be grouped for diagnostics or selected execution.

---

## 22. Executor policy during formation

The first managed run should use the exact rectangular fine readers.

The purpose is model formation, not sparse-kernel optimization.

Always record

\[
R_M^{\rm support},\qquad R_E^{\rm support}.
\]

If a trained checkpoint develops material unique-pair sparsity, benchmark exact selected execution against rectangular afterward.

Only enable selected/hybrid execution if it is measurably faster on real shapes.

No custom kernel is part of Run 1500 formation.

---

## 23. Prelaunch verification

Use focused tests plus two real predicted-port batches.

Verify:

1. proposal assignments normalize correctly;
2. geometry-refined assignments normalize correctly;
3. \(\pi\) is nonnegative and sums to one;
4. sparsemax \(\gamma\) is nonnegative and sums to one;
5. zero-\(\gamma\) groups are absent from final module/environment/query assignments;
6. every active source/query row retains support;
7. `K_case >= 1`;
8. \(\kappa=1/\sum\gamma^2\);
9. prototype permutation equivariance holds under a consistent permutation;
10. full/packed control arithmetic agrees;
11. one real predicted-port forward/backward/update has finite nonzero router gradients;
12. historical Run-1406 and Run-1409 modes strict-load unchanged.

Do not add a new test framework.

---

## 24. Fresh Run 1500 training

Launch:

```text
run-id = 1500
run-name = mass_competitive_adaptive_hypergraph
```

Train from scratch with seed 0, learning rate 3e-4, established AdamW settings, predicted ports and existing physical losses. No extra sparsity loss.

Train to epoch 50 first.

---

## 25. Epoch-50 review

The epoch-50 decision should prioritize **hypergraph formation**.

Evaluate all 90 cases.

### Case-level hypergraph

Report:

- \(K_{\rm case}\) mean/median/range/histogram;
- \(\kappa\) distribution;
- prototype selection frequencies;
- relation to active module count;
- \(\pi\) and \(\gamma\) concentration;
- numerical routing ranks.

A meaningful result should not be universal \(K=1\) or universal \(K=12\).

### Source organization

For module and environment assignments report:

- positive degree;
- entropy/effective groups;
- occupied groups;
- within-group spatial RMS radius;
- between-group centre separation;
- mass distribution per active group.

Compare spatial compactness to the existing mass-preserving shuffled diagnostic.

### Query organization

Report:

- positive query degree;
- effective groups;
- dominant-group maps;
- query-to-group-centre distance;
- support transition regions.

### Physical support

Report

\[
R_M^{\rm support},\qquad R_E^{\rm support},
\]

logical path multiplicity and unique query-source pair counts.

### Accuracy

Compare exact epoch 50 with:

- Run 1404;
- Run 1406;
- Dense 1804;
- latest occupancy Run 1409 as development context.

The candidate does not need to beat Dense at epoch 50. It should remain physically usable and preferably lie between Run 1404 and Run 1406 on major global/interface metrics while forming a substantially clearer dynamic hypergraph.

### Cost

Report latency and memory as secondary evidence. Do not require acceleration for continuation unless the model is unexpectedly expensive or unstable.

---

## 26. Continuation

If epoch 50 shows nontrivial \(K_{\rm case}\), coherent source geometry, healthy learning and no severe interface/thermal collapse, continue the SAME run to epoch 150 or 500.

At the later checkpoint repeat the 90-case organization and accuracy audit.

Interpretations:

- `K_case` varies but is almost exactly module count: report module-hub behavior;
- `K_case` varies independently of module count: evidence for broader case-adaptive grouping;
- `K_case` varies but fine support remains near dense: adaptive representation without adaptive computation;
- support becomes substantially sparse: benchmark selected execution;
- \(\gamma\) becomes nearly continuous with few/no exact zeros: mass organization is useful but discrete dynamic K is not strongly supported.

Do not automatically extend to 5000 epochs.

---

## 27. Visualization

For representative cases show:

1. modules colored by final hyperedge;
2. environment tokens colored by final hyperedge;
3. final group centres;
4. pre-competition mass \(\pi_k\);
5. competitive case weights \(\gamma_k\);
6. query dominant-group field;
7. query support degree;
8. one query → active groups → source-support diagram.

Also create:

- 90-case histogram of \(K_{\rm case}\);
- \(K_{\rm case}\) versus active module count;
- \(\kappa\) versus \(K_{\rm case}\);
- environment compactness versus shuffled reference.

Clearly distinguish:

```text
Kmax        = registered candidates
Kcase       = positive gamma groups
kappa       = continuous effective case complexity
query degree
logical q-source support
actual executed rows
```

Do not label groups as discovered causal mechanisms.

---

## 28. Required report

Create:

```text
docs/reports/HONF_Run1500_MassCompetitive_Evaluation.md
```

The report must answer:

1. Did mass competition produce a nontrivial case-dependent K?
2. Is K related only to module count or also to environment/operating complexity?
3. Are surviving hyperedges spatially coherent?
4. Does mass competition improve or damage source clustering?
5. Does query routing use the case hypergraph selectively?
6. Does query-source support become materially sparse?
7. Does support reduction translate into measured execution savings?
8. How does physical fidelity compare with 1404, 1406 and Dense?
9. Does the evidence support discrete dynamic K, or only a continuous effective complexity \(\kappa\)?

Use ordinary Git, focused tests and real execution. Preserve existing security and trusted checkpoint loading. Add no defensive hashes, freezes, monitoring services, approval infrastructure or unrelated refactors.
