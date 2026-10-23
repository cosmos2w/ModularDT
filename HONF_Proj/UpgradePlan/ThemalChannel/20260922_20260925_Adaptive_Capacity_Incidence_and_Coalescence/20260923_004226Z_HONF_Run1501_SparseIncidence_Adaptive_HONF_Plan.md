# Run 1501 — Sparse-Incidence Adaptive HONF

## Detailed mathematical design, implementation plan, training/evaluation workflow

## 0. Research objective

Run 1501 starts from the cumulative lesson of Runs 1404–1500:

- fixed/grouped HONF can form useful spatial organization;
- Run 1406 showed that many-body group control can condition one fine query-source interaction without per-group duplication;
- Run 1407 showed that prototype-anchored query keys sharpen query organization;
- occupancy/mass-competition runs showed that forcing one discrete **case-level K** is not the quantity that controls physical computation;
- actual fine work is controlled by whether a query and source share at least one active hyperedge.

Therefore the adaptive object in Run 1501 is the **query-local hypergraph subgraph**, not a global integer case rank.

The central structure remains:

\[
A^M\in\mathbb R^{B\times M\times K},
\qquad
A^E\in\mathbb R^{B\times E\times K},
\qquad
\alpha\in\mathbb R^{B\times Q\times K}.
\]

For source type \(S\in\{M,E\}\),

\[
\boxed{
\rho^S_{qs}
=
\sum_{k=1}^{K}
\alpha_{qk}A^S_{sk}.
}
\]

A fine physical query-source interaction is needed only when

\[
\boxed{
\operatorname{supp}(\alpha_q)
\cap
\operatorname{supp}(A^S_s)
\neq\varnothing.
}
\]

Run 1501 asks:

> Can physically coherent source groups plus strongly selective query routing produce a genuinely sparse interaction graph while retaining the fine-state fidelity of Run 1406?

---

# 1. Run identity

Use a new opt-in architecture:

```text
forward_architecture = "sparse_incidence_group_control_honf"
```

Managed experiment:

```text
run-id = 1501
run-name = sparse_incidence_adaptive_honf
```

This is a new scientific model, not another Run-1500 continuation.

---

# 2. Keep the successful physical core

Retain:

- `InterfaceFieldCore`;
- Dense MM/ME/EM source preparation;
- hidden width \(H=256\);
- low-dimensional control width \(D=16\);
- fixed registered group capacity \(K=12\);
- learned group prototypes;
- one-step geometry-aware source organization;
- Run-1406 module fine interaction;
- Run-1406 environmental K/V attention;
- one expensive fine evaluation per unique physical query-source pair;
- final field
  \[
  C(q)=C_g(q)+C_M(q)+C_E(q);
  \]
- ordinary P0/P1/P2 physical coupling;
- predicted ports;
- existing physical loss and AdamW settings;
- existing activation-checkpointing improvements.

Do **not** add:

- case-level gates;
- hard-concrete;
- mass competition;
- count/sparsity/entropy/balance losses;
- training schedules for sparsity;
- sampled QE;
- sequential residual group generation;
- direct pooled hyperedge-value field;
- coarse/local auxiliary paths;
- custom GPU kernels during model formation.

The experiment should be mathematically small.

---

# 3. Registered group capacity

Use

\[
K=12.
\]

This is representational capacity, not an inferred physical rank.

All 12 learned prototypes remain available to every case.

No group is permanently privileged.

Initialize prototypes as approximately orthogonal normalized rows in \(D=16\), matching the successful diversity initialization already used in the occupancy experiment.

---

# 4. Source control states

After Dense physical preparation, retain the ordinary low-dimensional controls

\[
u_i^M\in\mathbb R^D,
\qquad
u_j^E\in\mathbb R^D,
\]

and global control \(g_c\).

Use normalized physical source measures

\[
\sum_i\omega_i^M=1,
\qquad
\sum_j\omega_j^E=1.
\]

Inactive padded module slots have zero measure.

---

# 5. Source proposal assignments

Compute content logits

\[
\ell^M_{ik}
=
\frac{(u_i^M)^\top c_k}{\sqrt D},
\]

\[
\ell^E_{jk}
=
\frac{(u_j^E)^\top c_k}{\sqrt D}.
\]

Use **entmax15**, not sparsemax, for source organization:

\[
P^M_{i:}
=
\operatorname{entmax}_{1.5}(\ell^M_{i:}),
\]

\[
P^E_{j:}
=
\operatorname{entmax}_{1.5}(\ell^E_{j:}).
\]

Reason:

- source assignments already become row-sparse under entmax;
- the latest occupancy Run 1409 produced spatially coherent source clusters;
- Run 1501 should isolate the effect of stronger **query** selectivity first.

Inactive module rows remain zero.

---

# 6. Self-derived source geometry

Compute proposal masses:

\[
\mu^{M,0}_k
=
\sum_i\omega_i^M P^M_{ik},
\]

\[
\mu^{E,0}_k
=
\sum_j\omega_j^E P^E_{jk}.
\]

Compute module/environment centres when the corresponding mass is positive:

\[
r^{M,0}_k
=
\frac{
\sum_i\omega_i^M P^M_{ik}x_i
}{
\mu^{M,0}_k
},
\]

\[
r^{E,0}_k
=
\frac{
\sum_j\omega_j^E P^E_{jk}y_j
}{
\mu^{E,0}_k
}.
\]

Define a joint centre:

\[
r^0_k
=
\frac{
\mu^{M,0}_k r^{M,0}_k
+
\mu^{E,0}_k r^{E,0}_k
}{
\mu^{M,0}_k+\mu^{E,0}_k
}.
\]

If only one source type has positive mass, use that source type's centre.

A group with zero total proposal mass is phase-empty and is not used in the geometry-refined assignments/query route for that phase. It is **not permanently deleted** and can reappear in a later physical phase.

There is no case-level K plan.

---

# 7. One geometry refinement for sources

Use the physical scale

\[
s_{\rm geo}
=
0.25
\sqrt{\sum_dL_d^2}.
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

Final source assignments are

\[
\boxed{
A^M_{i:}
=
\operatorname{entmax}_{1.5}
(
\ell^M_{i:}+b^M_{i:}
)
}
\]

and

\[
\boxed{
A^E_{j:}
=
\operatorname{entmax}_{1.5}
(
\ell^E_{j:}+b^E_{j:}
).
}
\]

Mask phase-empty proposal groups and inactive module rows.

Perform exactly one geometry refinement; no iterative clustering.

This part deliberately follows the source-organization mechanism that produced coherent clusters in the latest occupancy Run 1409.

---

# 8. Final source group state

Recompute source masses:

\[
\mu^M_k
=
\sum_i\omega_i^M A^M_{ik},
\qquad
\mu^E_k
=
\sum_j\omega_j^E A^E_{jk}.
\]

Recompute module/environment group centres \(r_k^M,r_k^E\).

Define joint mass

\[
\pi_k
=
\frac12(\mu_k^M+\mu_k^E).
\]

Do not threshold \(\pi\).

Do not use sparsemax on \(\pi\).

The continuous case complexity is retained only as a diagnostic/normalization quantity:

\[
\boxed{
\kappa_{\rm case}
=
\frac1{\sum_k\pi_k^2}.
}
\]

This is not called a discrete K.

Use \(\kappa_{\rm case}\) in the historical group-moment/output scaling locations so registered K=12 is not confused with physical amplitude.

Build the D-wide many-body group control \(h_k\) from:

- module moment;
- environment moment;
- module/environment mass;
- global control;
- learned prototype.

The group state remains a control state only, never a direct field-value branch.

---

# 9. Prototype-anchored query keys

Use the successful Run-1407 query-key idea.

For each phase/group define

\[
k_k
=
c_k+W_hh_k.
\]

Apply parameter-free RMS normalization

\[
\mathcal R(x)
=
\frac{x}{
\sqrt{D^{-1}\sum_a x_a^2+10^{-6}}
}.
\]

For query control \(v_q\),

\[
\bar v_q=\mathcal R(v_q),
\qquad
\bar k_k=\mathcal R(k_k).
\]

The learned content score is

\[
s^{\rm content}_{qk}
=
\frac{
\bar v_q^\top \bar k_k
}{
\sqrt D
}.
\]

This preserves prototype identity while retaining case-dependent group control.

---

# 10. Query geometry

Use current phase source centres.

Define

\[
d^M_{qk}
=
\frac{\|q-r_k^M\|}{s_{\rm geo}},
\qquad
d^E_{qk}
=
\frac{\|q-r_k^E\|}{s_{\rm geo}}.
\]

When one source type has zero phase mass, use the available centre for both terms.

Geometry bias:

\[
\boxed{
b^Q_{qk}
=
-\frac12
(d^M_{qk}+d^E_{qk}).
}
\]

This is a finite bias, not a cutoff.

Remote groups remain reachable when learned content requires them.

---

# 11. Primary Run-1501 change: sparsemax query routing

Final query logits are

\[
z_{qk}
=
s^{\rm content}_{qk}
+
b^Q_{qk}.
\]

Use **sparsemax**:

\[
\boxed{
\alpha_{q:}
=
\operatorname{sparsemax}(z_{q:})
}
\]

over groups with current phase source mass.

This is the main scientific change.

Source routing remains entmax15.

There is:

- no query sparsity penalty;
- no learned temperature;
- no top-k;
- no schedule.

The query support itself determines the query-local hypergraph:

\[
\boxed{
K_q
=
|\operatorname{supp}(\alpha_q)|.
}
\]

---

# 12. Query-source relation and many-body control

For source type \(S\),

\[
\boxed{
\rho^S_{qs}
=
\sum_k\alpha_{qk}A^S_{sk}.
}
\]

The collective group moment is

\[
\boxed{
n^S_{qs}
=
\sum_k
\alpha_{qk}A^S_{sk}h_k.
}
\]

The expensive fine physical interaction remains one per physical pair:

\[
\psi(q,s,n^S_{qs}).
\]

No \((q,s,k)\) fine-interaction duplication is introduced.

The final field remains

\[
\boxed{
C(q)=C_g(q)+C_M(q)+C_E(q).
}
\]

---

# 13. Phase behavior

Run 1501 should use **phase-local organization**, closer to Run 1406.

At P0, P1 and P2 independently:

1. Dense MM/ME/EM physical states are prepared/refreshed.
2. low-D module/environment controls are computed;
3. source proposal assignments are computed;
4. source group centres are formed;
5. one geometry refinement produces \(A^M,A^E\);
6. group controls \(h_k\) refresh;
7. prototype-anchored query keys refresh;
8. query sparsemax routes refresh.

No discrete case plan is shared across phases.

Only learned parameters/prototypes are shared.

This avoids imposing another approximation while testing sparse incidence.

---

# 14. The adaptive object

Run 1501 must distinguish three quantities:

### Continuous case organization

\[
\kappa_{\rm case}
=
1/\sum_k\pi_k^2.
\]

### Query-local discrete complexity

\[
\boxed{
K_q
=
|\operatorname{supp}(\alpha_q)|.
}
\]

### Fine physical support

\[
\boxed{
M^S_{qs}
=
\mathbf1[
\operatorname{supp}(\alpha_q)
\cap
\operatorname{supp}(A^S_s)
\ne\varnothing
].
}
\]

The third quantity controls potential physical computation.

Do not interpret \(\kappa_{\rm case}\) or \(K_q\) alone as execution savings.

---

# 15. Exact support representation

K=12 fits in a 16-bit word.

For source and query rows:

\[
m_s
=
\sum_k2^k\mathbf1[A^S_{sk}>0],
\]

\[
m_q
=
\sum_k2^k\mathbf1[\alpha_{qk}>0].
\]

Then

\[
\boxed{
M^S_{qs}
=
\mathbf1[
m_q\;\&\;m_s\ne0
].
}
\]

Do not create a \(2^{12}\) lookup table.

Do not enumerate q-group-source paths.

Use direct bit intersections and only query masks actually present in a receiver chunk when diagnostics/selected execution are needed.

---

# 16. Frozen diagnostic before training

Before launching the managed Run 1501, use a mature/relevant **K=12 geometry-organized checkpoint** from the latest occupancy Run 1409.

Keep its trained source assignments/group states fixed.

Recompute only query routing with the proposed Run-1501 rule:

- prototype-anchored RMS query/group keys;
- current geometry bias;
- sparsemax instead of entmax15.

On all 90 cases or at minimum the established population routing audit, report:

- mean positive query degree;
- mean effective query groups;
- module/environment support ratios;
- fraction of queries with 1,2,3,... positive groups;
- prediction change when only the query normalizer/key rule is altered.

This is a diagnostic, not a retrained accuracy claim.

Do **not** tune sparsemax temperature; there is no temperature.

The diagnostic does not block training solely because frozen predictions change. Its main purpose is to verify that the proposed query rule can expose materially narrower support than the parent route.

If support remains essentially dense (\(R_E\gtrsim0.9\)) even with substantially smaller query degree, explicitly record that source overlap is likely the limiting factor. Still keep the formal Run 1501 source normalizer unchanged so the experiment remains interpretable.

---

# 17. Code structure

Create a new opt-in backend rather than changing Run 1409/1500 semantics.

Suggested files:

```text
src/honf_forward_core/interface_fields/sparse_incidence_router.py
src/honf_forward_core/interface_fields/sparse_incidence_group_control.py
```

Reuse:

```text
group_control_pairwise.py
group_control_router.py
occupancy_group_router.py
```

Where practical, reuse the one-step geometry helper from the occupancy router without modifying its historical behavior.

Recommended router:

```text
SparseIncidenceGroupRouter
```

Recommended backend:

```text
SparseIncidenceGroupControlPairwiseField
```

Register:

```text
forward_architecture = "sparse_incidence_group_control_honf"
```

Add:

```text
src/config_core/forward/sparse_incidence_group_control_honf_context.json
```

No case-specific sparsity config is required.

---

# 18. Normalization implementation

Source assignments use existing entmax15.

Query assignments need a masked sparsemax implementation.

Reuse the repository's maintained masked sparsemax helper if its arithmetic and mask contract are appropriate.

Do not create a second sparsemax implementation unless necessary.

Tests must cover:

- row normalization;
- exact zeros;
- all-but-one masked group;
- gradient behavior on active query routes;
- no NaNs for phase-empty groups.

---

# 19. Executor policy during training

Use the exact rectangular fine readers for initial training.

This is deliberate.

Do not let sparse executor overhead determine whether the model learns the desired graph.

Always compute/report logical unique support in diagnostic passes:

\[
R_M^{\rm support},
\qquad
R_E^{\rm support}.
\]

Do not claim computational sparsity from query degree alone.

---

# 20. When selected execution may be evaluated

After a trained checkpoint exists, benchmark exact selected execution only when a branch shows material unique-pair reduction.

Recommended trigger for benchmarking—not a model-training gate:

\[
R_S^{\rm support}\le0.8
\]

on the population/anchor evidence, or another clearly comparable reduction supported by the data.

Benchmark:

```text
rectangular exact
vs
selected exact
```

on the same checkpoint, GPU and shapes.

Use selected/hybrid execution only if wall time actually improves.

Do not write a new custom kernel in this experiment.

---

# 21. Prelaunch verification

Before training:

1. run focused unit/integration tests;
2. run the frozen query-routing diagnostic;
3. execute two real predicted-port batches;
4. execute one AdamW update.

Verify:

- source proposal/final rows normalize;
- source geometry centres are finite;
- query sparsemax rows normalize;
- query exact zeros occur on at least some real rows, or report if not;
- phase-empty groups are masked safely;
- all major router parameters receive finite gradients;
- prototype gradients are not identical;
- one physical optimizer step is finite;
- historical Run 1406/1409/1500 checkpoint loading remains intact.

Do not require a particular Kq before launch.

---

# 22. Managed training workflow

Launch one fresh managed run:

```text
run-id = 1501
run-name = sparse_incidence_adaptive_honf
```

Use seed 0 and established training settings.

### Epoch 10

Health-only review.

Stop only for:

- numerical failure;
- disconnected router;
- obvious universal route collapse;
- non-learning loss.

### Epoch 50

Primary **formation review**, not a mature-accuracy gate.

Evaluate all 90 cases.

Continue the same run to epoch 150 when:

- training/validation are still improving or plausibly converging;
- source groups remain spatially organized;
- query routes are nontrivial and spatially structured;
- there is no universal pathological collapse.

Do not stop solely because exact-50 accuracy is below Run 1406/Dense.

### Epoch 150

First stronger physical review.

If topology remains meaningful and physical errors continue improving, continue the same run to epoch 500.

Stop if:

- query sparsemax has effectively reverted to all-group support everywhere;
- the model is structurally collapsed;
- accuracy is clearly diverging rather than simply immature.

### Epoch 500

Perform full comparative evaluation and decide whether longer training is scientifically warranted.

Do not automatically continue to 2500/5000.

---

# 23. Epoch-50 formation evaluation

Across all 90 cases report:

## Source hypergraph

For modules/environment:

- positive source degree;
- effective groups;
- row entropy;
- group source masses;
- within-group spatial RMS radius;
- mass-preserving shuffled radius;
- between-group centre separation;
- continuous case complexity \(\kappa_{\rm case}\).

## Query-local subgraph

Report:

\[
K_q=|\operatorname{supp}\alpha_q|.
\]

Provide:

- mean/median/p95/range;
- histogram of Kq;
- effective query groups;
- query entropy;
- dominant-group spatial maps;
- query support-degree maps;
- query-to-group-centre distances.

## Fine physical support

Report:

\[
R_M^{\rm support},
\qquad
R_E^{\rm support},
\]

plus:

- logical path multiplicity;
- unique query-source pair counts;
- dense denominators.

This is the primary computational-structure evidence.

## Accuracy

Compare exact epoch-50 with:

- Run 1404;
- Run 1406;
- Dense 1804;
- Run 1500 as development context.

Accuracy is secondary to formation at this checkpoint but must remain finite and physically meaningful.

## Cost

Report rectangular latency/memory, but do not require an acceleration win at epoch 50.

---

# 24. Epoch-150/500 physical evaluation

At later checkpoints include:

- pooled fluid relative L2;
- equal-case mean/median/p95/worst;
- near-interface and far-fluid errors;
- per-field channel errors;
- internal/surface temperature;
- normal heat flux;
- final port environmental temperature and effective h;
- engineering KPIs;
- module-count / geometry strata;
- Kq/support distributions;
- source clustering statistics.

At epoch 500 compare matched checkpoint policies explicitly.

If saved-best and exact endpoint differ materially, report both rather than silently selecting one.

---

# 25. Visualization

Create a publishable evidence board for representative cases.

Each board should show:

1. module/environment source memberships and group centres;
2. source-group spatial organization;
3. dominant query group;
4. query positive-degree map \(K_q(x,y)\);
5. selected query → active groups → supported sources;
6. support ratios and actual execution rows.

Population figures:

- histogram of Kq;
- Kq versus query location/region class;
- environment compactness versus shuffled;
- \(R_E^{support}\) distribution across cases;
- \(\kappa_{\rm case}\) distribution.

Clearly distinguish:

```text
K=12                  registered source-group capacity
kappa_case            continuous whole-case organization
Kq                    discrete query-local subgraph size
R_M/R_E support       unique physical pair support
executed rows         actual GPU fine work
```

Do not label group identities as physical causal mechanisms.

---

# 26. Scientific interpretation rules

A useful result can take several forms.

### Desired

- coherent source grouping;
- query Kq substantially below K;
- source/query intersections materially below dense;
- competitive accuracy.

### Adaptive representation but not computation

If Kq is small but \(R_E\approx1\), source overlap is too broad.

### Sparse mathematics but not acceleration

If support falls but selected execution loses to rectangular, report GPU-execution failure separately.

### Accurate but dense

If accuracy is strong but Kq remains broad, the many-body model may be useful without sparse computation.

### Sparse but inaccurate

If sparsemax routes become narrow but interface/thermal fidelity collapses, query information has been removed too aggressively.

Do not hide these distinctions behind one scalar score.

---

# 27. Focused tests only

Add tests for:

1. source content proposal and geometry refinement;
2. phase-local source organization;
3. prototype-anchored RMS query keys;
4. masked query sparsemax normalization/exact zeros;
5. phase-empty group safety;
6. source/query bit-mask support equivalence to direct incidence;
7. module/environment permutation behavior;
8. one fine evaluation per unique physical pair;
9. rectangular/reference output path;
10. one real predicted-port forward/backward/update;
11. strict loading of historical architectures/checkpoints.

Do not add a new testing framework, provenance system, monitoring process or release gate.

---

# 28. Required report

Create:

```text
docs/reports/HONF_Run1501_SparseIncidence_Evaluation.md
```

The report must answer:

1. Did coherent source groups form?
2. Did sparsemax produce a genuinely query-local subgraph?
3. What is the Kq distribution?
4. Did query sparsity translate into lower unique QM/QE support?
5. If not, is source overlap the remaining bottleneck?
6. How does fidelity evolve from epoch 50 to 150/500?
7. Does selected exact execution become worthwhile?
8. Does the model achieve a useful accuracy–organization–cost tradeoff?
9. What evidence would justify a future source-sparse variant?

Preserve existing security/trusted checkpoint loading.

Use ordinary Git, focused tests and actual GPU execution.

Do not add defensive hashes, freezes, approval infrastructure, monitoring services or unrelated refactors.
