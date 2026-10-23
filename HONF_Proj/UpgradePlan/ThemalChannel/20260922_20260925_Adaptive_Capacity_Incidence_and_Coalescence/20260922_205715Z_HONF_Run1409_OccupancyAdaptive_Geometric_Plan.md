# Run 1409 v3 — Occupancy-Adaptive Geometric Hypergraph

## 1. Goal

The next Run 1409 should test a simpler interpretation of dynamic hypergraph capacity:

> A group exists because learned module/environment assignments actually occupy it.

The previous Run 1409 v2 kept all 12 explicit hard-concrete gates open, yet its source assignments already used only 1–2 module groups and 6–9 environment groups depending on the case. This means the separate gate was not describing the organization the model actually used.

The new experiment therefore removes the gate network and group-count loss entirely. Dynamic \(K\) is derived directly from exact-zero entmax source occupancy.

Use a fresh timestamped Run 1409. Do not resume either previous Run-1409 checkpoint.

---

## 2. Keep the useful Run-1406 physics

Retain:

- `InterfaceFieldCore`;
- Dense MM/ME/EM preparation;
- hidden width \(H=256\);
- low-dimensional group control \(D=16\);
- \(K_{\max}=12\) registered candidate groups;
- Run-1406 fine module and environment readers;
- one expensive fine evaluation per unique physical query-source pair;
- `C_g + C_M + C_E`;
- P0/P1/P2 physical coupling and predicted ports;
- existing losses, optimizer, data, seed and activation checkpointing.

Do not add:

- hard-concrete gates;
- group-count or pair-count penalties;
- entropy/balance/repulsion losses;
- top-k routing;
- sampled QE;
- sequential residual group generation;
- new direct hyperedge-value branches;
- new coarse/local branches.

The only new model component is an occupancy-adaptive geometry-aware router.

---

## 3. Mathematical structure

For source type \(S\in\{M,E\}\),

\[
R_b^S=\alpha_b(A_b^S)^\top,
\qquad
\rho^S_{qs}=\sum_{k=1}^{K_{\max}}\alpha_{qk}A^S_{sk}.
\]

The registered capacity remains

\[
K_{\max}=12.
\]

The deployed active count is determined from learned source occupancy.

---

## 4. P0 proposal assignments

Let \(u_i^M,u_j^E\in\mathbb R^D\) be the ordinary Run-1406 P0 controls and \(c_k\in\mathbb R^D\) the learned group prototypes.

Compute content logits

\[
\ell^{M,0}_{ik}
=
\frac{(u_i^M)^\top c_k}{\sqrt D},
\qquad
\ell^{E,0}_{jk}
=
\frac{(u_j^E)^\top c_k}{\sqrt D}.
\]

Apply unit-temperature 1.5-entmax:

\[
P^M_{i:}=\operatorname{entmax}_{1.5}(\ell^{M,0}_{i:}),
\]

\[
P^E_{j:}=\operatorname{entmax}_{1.5}(\ell^{E,0}_{j:}).
\]

Inactive module rows remain zero.

With normalized physical source measures \(\omega^M,\omega^E\),

\[
\mu^{M,0}_k=\sum_i\omega_i^M P^M_{ik},
\qquad
\mu^{E,0}_k=\sum_j\omega_j^E P^E_{jk}.
\]

Define proposal occupancy

\[
\boxed{
g_k^0=
\mathbf1[\mu^{M,0}_k+\mu^{E,0}_k>0].
}
\]

No extra gate network is used.

---

## 5. Self-derived group geometry

For occupied groups,

\[
r^{M,0}_k
=
\frac{\sum_i\omega_i^M P^M_{ik}x_i}
{\mu^{M,0}_k},
\]

\[
r^{E,0}_k
=
\frac{\sum_j\omega_j^E P^E_{jk}y_j}
{\mu^{E,0}_k}.
\]

Define a joint physical centre

\[
\boxed{
r^0_k=
\frac{
\mu^{M,0}_k r^{M,0}_k+
\mu^{E,0}_k r^{E,0}_k
}{
\mu^{M,0}_k+\mu^{E,0}_k
}.
}
\]

If one source type has zero mass, the other source type defines the centre.

No learned fallback centre is needed for an inactive group.

---

## 6. One geometry-refinement step

Reuse the successful geometry principle from the 1401–1404 organizer.

Let

\[
s_{\rm geo}
=
0.25\sqrt{L_1^2+\cdots+L_d^2}.
\]

Define

\[
b^M_{ik}
=
-\frac{\|x_i-r^0_k\|}{s_{\rm geo}},
\]

\[
b^E_{jk}
=
-\frac{\|y_j-r^0_k\|}{s_{\rm geo}}.
\]

Then refine assignments only over proposal-occupied groups:

\[
\boxed{
A^M_{i:}
=
\operatorname{entmax}_{1.5}
(\ell^{M,0}_{i:}+b^M_{i:};g^0),
}
\]

\[
\boxed{
A^E_{j:}
=
\operatorname{entmax}_{1.5}
(\ell^{E,0}_{j:}+b^E_{j:};g^0).
}
\]

This is exactly one refinement step. Do not add iterative clustering.

Compute final masses

\[
\mu^M_k=\sum_i\omega_i^M A^M_{ik},
\qquad
\mu^E_k=\sum_j\omega_j^E A^E_{jk}.
\]

Final P0 occupancy is

\[
\boxed{
g_k=
\mathbf1[\mu^M_k+\mu^E_k>0].
}
\]

Store the original prototype IDs with \(g_k=1\). Their count is

\[
\boxed{
K_{\rm plan}=\sum_k g_k.
}
\]

This is the case-specific dynamic \(K\).

---

## 7. Capacity-invariant normalization

Do not use registered \(K_{\max}\) as a field-amplitude multiplier.

Define

\[
\pi_k=
\frac12(\mu^M_k+\mu^E_k).
\]

Because both source measures and assignment rows are normalized,

\[
\sum_k\pi_k=1.
\]

Define

\[
\boxed{
\kappa=
\frac1{\sum_k\pi_k^2}.
}
\]

Use \(\kappa\) in the historical group-moment and output-amplitude positions where Run 1406 uses fixed \(K\).

Report separately:

- \(K_{\max}\);
- \(K_{\rm plan}\);
- \(\kappa\);
- numerical routing rank;
- unique source support;
- actual executed rows.

---

## 8. Phase behavior

The P0 prototype-ID plan is shared through P1/P2.

At P1/P2:

1. Dense MM/ME/EM states refresh normally.
2. Module/environment controls refresh normally.
3. Assignments are recomputed only over the P0 planned prototype IDs.
4. Group centres refresh from current source states/assignments.
5. Group controls and fine K/V refresh normally.
6. Query routing refreshes for the current receiver coordinates.

Thus group identity/count is case-stable while physical values remain phase-responsive.

This is intentionally closer to Run 1406 than to a fully phase-static Run 1407 controller.

---

## 9. Query routing

For current phase group centres \(r_k^M,r_k^E\), use the ordinary learned low-D query/group content score plus a finite geometry bias:

\[
\ell^Q_{qk}
=
\frac{v_q^\top W_hh_k}{\sqrt D}
-
\frac12
\frac{
\|q-r_k^M\|+\|q-r_k^E\|
}{
s_{\rm geo}
}.
\]

If one source type has zero current phase mass in a planned group, use the available source centre for both distance terms.

Then

\[
\boxed{
\alpha_{q:}
=
\operatorname{entmax}_{1.5}
(\ell^Q_{q:};g_{\rm phase}).
}
\]

The geometry term is only a finite bias. It is not a hard locality cutoff.

No query sparsity loss is used.

---

## 10. Fine many-body interaction remains unchanged

For either source type,

\[
\rho^S_{qs}
=
\sum_{k\in\mathcal G_b}
\alpha_{qk}A^S_{sk},
\]

\[
n^S_{qs}
=
\sum_{k\in\mathcal G_b}
\alpha_{qk}A^S_{sk}h_k.
\]

The expensive fine function is evaluated once per physical pair:

\[
\psi(q,s,n^S_{qs}).
\]

No \((q,s,k)\) fine duplication is reintroduced.

The final context remains

\[
\boxed{
C(q)=C_g(q)+C_M(q)+C_E(q).
}
\]

---

## 11. Why this experiment is supported by existing evidence

Run 1409 v2 at epoch 150 had all 12 explicit gates open, but:

- module source organization used only 1–2 nonempty groups;
- environment source organization used 6–9 nonempty groups;
- environment routing rank varied across cases.

Therefore a case-dependent occupancy signal already exists inside \(A^M,A^E\).

The new model simply makes that learned occupancy the definition of group existence.

This is cleaner than continuing to tune a separate capacity network.

---

## 12. Implementation plan

Prefer a new opt-in architecture:

```text
forward_architecture = "occupancy_adaptive_group_control_honf"
```

The managed run ID remains **1409** because this is the next bounded attempt at the same dynamic-K research question.

Recommended new files:

```text
src/honf_forward_core/interface_fields/occupancy_group_router.py
src/honf_forward_core/interface_fields/occupancy_group_control.py
```

Reuse:

```text
group_control_pairwise.py
group_control_router.py
```

Do not duplicate Dense preparation or the fine readers.

### Runtime plan object

Create a small `OccupancyGroupPlan` containing:

- registered `Kmax`;
- original active prototype IDs;
- packed-valid mask;
- `K_plan`;
- P0 final module/environment masses;
- `kappa`;
- P0 centres for diagnostics.

There are no gate logits, gate noise, or gate probabilities.

---

## 13. Support handling for Kmax=12

Do not extend the historical \(2^6\) table.

Kmax=12 fits in a 16-bit mask.

For source and query positive memberships, build bit masks

\[
m_s=\sum_k2^k\mathbf1[A_{sk}>0],
\]

\[
m_q=\sum_k2^k\mathbf1[\alpha_{qk}>0].
\]

Then

\[
(q,s)\text{ supported}
\iff
m_q\;\&\;m_s\ne0.
\]

For diagnostics/execution, process only query masks actually observed in the receiver chunk. Never enumerate all \(2^{12}\) possible subsets.

---

## 14. Executor policy

For the first 50 epochs, keep the exact rectangular fine readers.

The initial scientific goal is hypergraph formation, not an executor race.

Always report logical support:

\[
R_M^{\rm support},
\qquad
R_E^{\rm support}.
\]

At epoch 50:

- if support is still broad, retain rectangular execution;
- if a branch shows substantial unique-pair sparsity, benchmark the existing exact selected/partial path against rectangular on the same checkpoint and real shapes;
- enable hybrid selected execution only when measured latency improves.

Do not call logical support reduction a speedup.

---

## 15. Initialization

No gate or dense-to-sparse schedule is required.

Use entmax from epoch 1.

Initialize the 12 group prototypes with diverse approximately orthogonal normalized rows in \(D=16\). Do not initialize them identically.

The geometry centre is derived from actual proposal assignments, so no fixed spatial anchor grid is required.

No prototype repulsion loss is added.

---

## 16. Prelaunch checks

Before managed training, run two real predicted-port batches and verify:

- proposal assignments are normalized;
- final assignments are normalized;
- no active source row loses all support;
- `K_plan >= 1`;
- occupied group controls receive finite gradients;
- different group rows do not all receive identical gradients;
- module/environment permutation behavior is preserved;
- one real optimizer update is finite.

Do not require sparse K at initialization.

---

## 17. Fresh managed Run 1409

Launch from scratch:

```text
run-id: 1409
run-name: occupancy_adaptive_geometric_group_control
```

Use seed 0 and the established optimizer/loss/data settings.

Train to epoch 50 first.

Do not resume either gate-based Run 1409.

---

## 18. Epoch-50 review: prioritize hypergraph formation

Evaluate all 90 cases.

### Dynamic K

Report the full distribution of

\[
K_{\rm plan}.
\]

Include:

- mean/median/range;
- histogram;
- relation to active module count;
- prototype occupancy frequency.

### Source organization

For modules and environment:

- positive source degree;
- effective groups;
- occupied groups;
- assignment entropy;
- within-group spatial RMS radius;
- between-group centre separation.

For diagnostics only, compare spatial compactness with a source-mass-preserving shuffled-assignment baseline. Do not train against this metric.

### Query organization

Report:

- positive query degree;
- effective query groups;
- dominant-group spatial maps;
- support-transition contours;
- query-to-centre distance.

### Interaction support

Report:

\[
R_M^{\rm support},
\qquad
R_E^{\rm support},
\]

logical path multiplicity, and unique physical pairs.

### Accuracy

Compare exact epoch-50 with Run 1404, Run 1406 and Dense 1804.

The primary review question is whether a coherent, nontrivial hypergraph formed.

Do not require it to beat Run 1404 latency or Dense accuracy.

A usable candidate should at least avoid universal one-group/all-group collapse and remain physically competitive enough to continue.

---

## 19. Continuation to 150/500

If epoch 50 shows healthy training and meaningful organization, continue the SAME run to epoch 150 or 500.

Repeat the population audit.

Important interpretations:

- If \(K_{\rm plan}\) equals active module count for every case, report a module-hub organization, not general adaptive rank.
- If \(K_{\rm plan}\) varies but support remains dense, report adaptive representation without adaptive computation.
- If support falls but selected execution is slower, report sparse mathematics without GPU acceleration.
- If grouping becomes coherent and case-variable while accuracy remains between the established 1404/1804 references, the central experiment is scientifically useful even before executor optimization.

Do not automatically continue to 5000 epochs.

---

## 20. Visualization

Create a clean evidence board with:

1. module coordinates colored by group;
2. environment coordinates colored by group;
3. joint group centres;
4. group module/environment masses;
5. query dominant-group map;
6. query support-degree contours;
7. one query → group → source support diagram;
8. annotations for `Kmax`, `Kplan`, `kappa`, support ratios and executed rows.

Also create a 90-case histogram of `Kplan`.

Do not label learned groups as recovered physical causal mechanisms.

---

## 21. Focused tests

Add only tests required by this new mode:

1. proposal/final normalization;
2. exact empty-column detection;
3. occupancy packing;
4. registered-capacity padding invariance;
5. geometry refinement arithmetic;
6. P0 plan reuse with P1/P2 assignment refresh;
7. query masking of source-empty groups;
8. occupancy-based kappa normalization;
9. full/packed control parity;
10. source permutation behavior;
11. real predicted-port forward/backward update;
12. historical Run-1406 and gate-based Run-1409 loading remains unchanged.

No new testing framework or defensive infrastructure.

---

## 22. Required report

Create:

```text
docs/reports/HONF_Run1409_OccupancyAdaptive_Evaluation.md
```

It must answer:

1. Did a nontrivial hypergraph form?
2. Is `Kplan` genuinely case-dependent?
3. Are environment groups spatially coherent?
4. Are module/environment groups mutually organized?
5. Does query routing respect that organization?
6. Does induced query-source support become sparse?
7. Does actual execution exploit any sparsity?
8. How does accuracy compare with 1404, 1406 and Dense?
9. Is the remaining limitation representation, routing, or executor behavior?

Preserve existing trusted loading/security and ordinary Git workflow.

Do not add hashes, freezes, monitoring services, or speculative optimization machinery.
