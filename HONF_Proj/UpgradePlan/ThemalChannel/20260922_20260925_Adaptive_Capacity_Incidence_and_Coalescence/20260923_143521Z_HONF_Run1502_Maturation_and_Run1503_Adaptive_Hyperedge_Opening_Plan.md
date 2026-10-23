# HONF Run 1502 Maturation and Run 1503 Adaptive Hyperedge-Opening Plan

## Purpose

This document is the authoritative Goal-mode work plan following commit `6a0fff0c96da7785c202c014fd0792ee2c3a1571`.

The project should **not** treat Run 1502 as a failed experiment and should **not** create a second experiment under the same Run-1502 identity. Run 1502 is a clean structural experiment whose sole scientific change was the final environmental source refinement from entmax-1.5 to sparsemax. It should be preserved and matured unchanged.

The next architectural experiment should be a new Run 1503 based on one central idea:

> **Use the learned hypergraph as an adaptive multiscale environmental operator: every query receives a cheap global group-level environmental representation, while only the sparse groups selected by the query are opened to fine environmental sources.**

This is intended to address three limitations simultaneously without architecture pile-up:

1. Run 1502's far-field regression after aggressively sparsifying environmental fine support.
2. The lack of actual execution benefit from logical source sparsity.
3. The need to retain the successful near-interface and thermal behavior associated with query-local fine information.

This plan is deliberately not a normalizer sweep, sparsity-loss study, top-k study, or collection of incremental ablations.

---

# 1. Current scientific interpretation

## 1.1 Run 1502 is a structural success but not yet a dominant endpoint

At exact epoch 500 relative to Run 1501 exact epoch 500, Run 1502:

- reduced pooled fluid relative L2 by about 2.53%;
- reduced near-interface relative L2 by about 11.40%;
- improved several thermal/interface quantities substantially;
- reduced environmental support ratio by about 38.74%;
- reduced mean environment source degree by about 49%;
- reduced environmental logical paths by about 57%;
- retained nonempty query/source support;
- retained query-local routing with mean Kq around three;
- did not improve measured application latency because the production reader remains rectangular.

The main negative signal is not support collapse. It is the **near/far tradeoff**:

- near-interface behavior improves;
- far-field relative L2 worsens by about 8%;
- overall field accuracy still trails Dense 1804 at the same epoch;
- sparse logical structure is still not translated into faster execution.

The learned organization is therefore useful. The remaining problem is how the environmental information represented by those groups is consumed.

## 1.2 Interpretation of the Run-1502 organization figures

The representative Run-1502 figures show:

- spatially coherent environmental group territories;
- nontrivial module/group organization;
- query-local Kq mostly concentrated around a few groups rather than K=12;
- environmental incidence that is much sharper than Run 1501 but is not globally collapsed;
- query routing that varies spatially and remains multi-group.

This is exactly the prerequisite needed for adaptive aggregation.

The figure does **not** suggest that the source organizer itself is the immediate failure. The stronger interpretation is:

> The learned groups are coherent enough to be used as physical aggregation units, but the present fine-source reader treats non-overlap as complete loss of environmental access.

That is the central issue Run 1503 should address.

---

# 2. Decision on Run 1502

## 2.1 Continue the existing Run 1502 to epoch 5000

Run 1502 has earned formal maturation.

Resume the **same managed run** from its exact epoch-500 model and optimizer state to total epoch 5000. If this has already been done, do not do duplicate actions.

Do not:

- change architecture;
- change sparsemax/entmax rules;
- change losses;
- change optimizer;
- warm-start from another run;
- reset optimizer or RNG state;
- assign a new Run-1502 identity;
- launch a second Run-1502 seed during this task.

Run 1501 must not be monitored, interrupted, restarted, or diagnosed as part of this goal.

Run 1502 should similarly be allowed to mature after resume. Do not wait for its completion before performing the Run-1503 development below.

---

# 3. Why not make another Run 1502

Do not introduce:

- entmax alpha sweeps;
- sparsemax temperature;
- sparsity penalties;
- source top-k;
- query top-k;
- a learned case-level K gate;
- another occupancy rule;
- a second sparsemax variant;
- extra correction branches added only to recover metrics.

Run 1502 has already answered the narrow question:

> Can final environmental membership be made much sharper without structural collapse?

The answer is yes.

The next question is qualitatively different:

> Can the hypergraph use those coherent groups to preserve global environmental physics while opening fine detail only where needed?

That deserves a new architectural identity: **Run 1503**.

---

# 4. Run 1503 central architecture

Suggested name:

```text
adaptive_hyperedge_opening_honf
```

Suggested run name:

```text
Run_1503_adaptive_hyperedge_opening
```

The architecture must retain the clean three-term final form:

\[
C(q)=C_g(q)+C_M(q)+C_E(q).
\]

There must be no new independent coarse/local/global bypass outside these three terms.

The module term remains the Run-1502 fine group-controlled module reader.

Only the internal structure of the environmental term changes.

---

# 5. Preserve the successful Run-1502 organization

Keep unchanged unless required mechanically by the new reader:

- Dense-style fine MM/ME/EM physical preparation;
- K = 12;
- D = 16;
- phase-local source organization;
- content proposal normalization;
- one geometry refinement;
- final environmental sparsemax;
- module entmax-1.5 assignment;
- prototype-anchored RMS query/group keys;
- query geometry bias;
- masked query sparsemax;
- the existing query logits;
- the existing group-control state h_k;
- P0/P1/P2 use of the common continuous field interface;
- predicted-port training;
- losses and optimizer;
- no sparsity regularizer.

The source organizer should not be redesigned in Run 1503.

---

# 6. Core mathematical change: adaptive hyperedge opening

## 6.1 Environmental measure partition

Let

\[
A^E_{jk}\ge 0,\qquad \sum_k A^E_{jk}=1,
\]

be the existing Run-1502 environmental membership, and let \(\nu_j\) be the normalized environmental quadrature measure.

Define the hyperedge environmental mass

\[
\mu_k
=
\sum_j \nu_j A^E_{jk}.
\]

Because each environmental row sums to one,

\[
\sum_k \mu_k = 1.
\]

Define the mass-weighted centroid

\[
c_k
=
\frac{
\sum_j \nu_j A^E_{jk} x_j
}{
\mu_k+\epsilon
}.
\]

Optionally retain the second spatial moment for diagnostics:

\[
r_k^2
=
\frac{
\sum_j \nu_j A^E_{jk}\|x_j-c_k\|^2
}{
\mu_k+\epsilon
}.
\]

This is descriptive geometry; do not add a radius loss.

## 6.2 Aggregate after fine physical preparation

The project has repeatedly shown that fine nonlinear physical preparation should happen before destructive aggregation.

Therefore use the already contextualized fine environmental states

\[
e_j^\star
\]

after the existing MM/ME/EM preparation and environmental update.

Form one physically aggregated environmental state per hyperedge:

\[
\bar e_k
=
\frac{
\sum_j \nu_j A^E_{jk} e_j^\star
}{
\mu_k+\epsilon
}.
\]

Do **not** use \(h_k\) itself as a field value.

The distinction is essential:

- \(h_k\): group/control state;
- \(\bar e_k\): physically aggregated environmental source state.

The group control may modulate the interaction with \(\bar e_k\), but it must not replace \(\bar e_k\).

## 6.3 One query logit bank, two roles

Reuse the existing Run-1502 query/group logits

\[
\ell_{qk}.
\]

Do not learn a second independent router.

From the same logits form:

\[
\alpha_{qk}
=
\operatorname{masked\ sparsemax}(\ell_q)_k
\]

for the **fine opening support**, exactly as in Run 1502.

Also form

\[
p_{qk}
=
\operatorname{masked\ softmax}
\left(
\ell_{qk}+\log(\mu_k+\epsilon)
\right)
\]

for the **cheap group-level environmental mixture**.

The mass term prevents a tiny group and a large group from being treated as identical amounts of environmental measure.

The important interpretation is:

- \(p_{qk}\) preserves low-cost global environmental access;
- \(\alpha_{qk}\) decides which groups deserve fine source resolution.

There is still only one learned query/group score system.

## 6.4 Coarse group response

Project each aggregated physical environment state \(\bar e_k\) through the existing environmental K/V/value-control machinery where possible.

Construct a group-level response

\[
G_{qk}
\]

using:

- the query representation;
- the aggregate state \(\bar e_k\);
- centroid \(c_k\);
- existing group control \(h_k\);
- existing environmental output projection.

Avoid introducing a new large MLP stack.

The group response must be a response of an aggregated **physical source**, not a direct decoder of \(h_k\).

## 6.5 Fine opened-group response

For groups with positive sparse query support, compute a fine environmental response using the original sources belonging to that group.

For group \(k\), use source measure

\[
\nu_{jk}
=
\nu_j A^E_{jk}.
\]

Within the opened group, use the existing fine environmental query/source machinery, including:

- source K/V from \(e_j^\star\);
- relative geometry;
- group-conditioned score/value modulation;
- normalized attention over the group's positive-mass sources.

Call this response

\[
F_{qk}.
\]

A source may belong to multiple groups. This is acceptable because its physical measure is partitioned by \(A^E_{jk}\), and

\[
\sum_k \nu_{jk}=\nu_j.
\]

Do not duplicate the full source measure in every group.

## 6.6 Continuous opening blend

The sparse support should control expensive fine execution, but support transitions should not create an avoidable output jump.

Use the same soft and sparse query distributions to define

\[
o_{qk}
=
\operatorname{clamp}
\left(
\frac{\alpha_{qk}}
{p_{qk}+\epsilon},
0,
1
\right).
\]

Then define

\[
H_{qk}
=
(1-o_{qk})G_{qk}
+
o_{qk}F_{qk}.
\]

When \(\alpha_{qk}=0\), the group remains available through its cheap aggregate response and no fine group execution is required.

When sparsemax selects the group strongly, the response moves toward fine resolution.

Finally,

\[
\boxed{
C_E(q)
=
\sum_k p_{qk} H_{qk}.
}
\]

This is the central Run-1503 equation.

It gives the hypergraph an explicit multiscale physical meaning:

> every hyperedge is always available as a coarse environmental carrier; sparse query routing decides which hyperedges are opened to fine physical sources.

---

# 7. Why this is preferable to another sparse-normalizer tweak

The current Run-1502 reader uses sparse overlap both to determine support and to influence the magnitude/normalization of the environmental response.

That means sharper source incidence can simultaneously:

- remove fine sources;
- modify relative source weights;
- change environmental overlap mass.

Run 1503 changes the abstraction instead of tuning that coupling.

Global environmental information is no longer lost when a group falls outside sparse fine support. It is represented through \(\bar e_k\).

At the same time, fine nonlinear source information remains available in the few selected groups.

This directly addresses the observed near/far tradeoff.

---

# 8. Execution design: group-major, not query-signature buckets

Do not reuse the Run-1501 support-signature executor as the primary Run-1503 executor.

Its scientific exactness was useful, but it created many small irregular blocks and was much slower than the rectangular reader.

Run 1503 is deliberately designed so execution follows the group axis.

There are at most K=12 learned groups.

For each group \(k\):

- gather all queries with \(\alpha_{qk}>0\);
- gather all sources with \(A^E_{jk}>0\);
- evaluate one dense fine group rectangle;
- use \(\nu_jA^E_{jk}\) as source mass;
- scatter one \(F_{qk}\) vector per selected query/group pair.

The coarse group path is a regular dense Q x K operation.

This produces at most K fine group blocks instead of a potentially large number of query-signature blocks.

The expected fine block area is approximately controlled by

\[
\sum_k Q_k E_k.
\]

For Run-1502-like degrees,

\[
\frac{\sum_k Q_kE_k}{QE}
\approx
\frac{
\bar K_q\,\bar d_E
}{
K
},
\]

which is roughly 0.6 using the observed epoch-500 mean degrees before accounting for spatial alignment.

The implementation should report the actual ratio.

## 8.1 GPU implementation priority

First implement the K group blocks using GPU-regular tensor operations.

Prefer:

- batched/padded group GEMMs;
- source projections prepared once per physical phase;
- query projections prepared once per receiver chunk;
- one scatter/weighted group reduction at the end.

Avoid:

- per-pair Python loops;
- one kernel launch per pair;
- materializing Q x E x K tensors;
- rebuilding group source lists inside every attention head;
- query-specific K/V banks;
- `grid_sample`;
- source copies for every query.

A custom Triton kernel is allowed only if the regular group-major PyTorch implementation proves that launch/packing overhead still prevents a meaningful speed gain. Do not begin with a custom kernel.

---

# 9. Module path

Keep the Run-1502 module reader unchanged for the formal Run-1503 candidate.

Reasons:

- M is small;
- module interactions are central to modular design;
- Run 1502's main structural change and tradeoff are environmental;
- changing module and environment mathematics simultaneously would obscure the result.

Module sparsity can be revisited only after the environmental operator is settled.

---

# 10. Required prelaunch tests

Before any formal training:

## 10.1 Algebraic and shape tests

Verify:

- environmental memberships remain nonnegative and row-normalized;
- \(\sum_k \mu_k = 1\);
- group centroid/mass pooling is permutation invariant;
- zero-mass groups are masked safely;
- split source weights preserve pooled mass/state;
- source membership overlap partitions rather than duplicates mass;
- query softmax and sparsemax use the same logits;
- \(\sum_k p_{qk}=1\);
- \(o_{qk}=0\) whenever \(\alpha_{qk}=0\);
- no NaN at empty/near-empty group boundaries.

## 10.2 Limiting cases

Test:

1. One environment source per group:
   group aggregation should reproduce that source state.
2. One active group:
   environmental read remains finite and normalized.
3. All groups opened:
   execution uses all group fine readers without duplicating source mass.
4. No fine group opened for a query:
   prediction still receives the complete coarse group mixture.
5. Identical states inside a group:
   fine and coarse group responses should be closely aligned.
6. Support transition:
   output and design gradients should be checked around alpha crossing zero.

## 10.3 Real optimizer smoke

Run canonical predicted-port optimizer steps on at least:

- one small-module case;
- one high-module case.

Require finite:

- loss;
- parameters;
- gradients;
- optimizer state;
- group aggregate path gradients;
- fine opened-group path gradients;
- module reader gradients.

Do not save or reuse smoke-model state.

---

# 11. Frozen diagnostic before formal Run 1503

Use fixed Run-1502 epoch-500 checkpoint weights to construct a bounded diagnostic where possible.

The diagnostic is not expected to be an accuracy forecast because the new grouped reader changes learned computation.

Its purpose is to verify:

- actual group masses and radii;
- number of active fine groups per query;
- group block sizes Q_k and E_k;
- \(\sum Q_k E_k / QE\);
- absence of pathological giant groups;
- coarse group path remains globally available;
- fine opening remains spatially local.

Use the representative cases from Run 1502 plus a small fixed panel spanning module counts and geometry.

Do not reject the architecture merely because untrained grouped output differs from Run 1502.

---

# 12. Formal Run 1503 training policy

Launch **one** Run 1503 from scratch.

Use:

- the same seed as Run 1501/1502;
- the same data split;
- the same optimizer and learning rate;
- the same loss terms;
- the same predicted-port policy;
- the same K=12 and D=16;
- the same training receiver chunk unless memory requires a mechanically equivalent setting.

Do not:

- warm start from 1502;
- distill from Dense or 1502;
- launch a second seed;
- sweep opening formulas;
- sweep K;
- sweep normalizers;
- add a sparsity loss.

The first formal candidate must isolate the new environmental abstraction.

---

# 13. Run-1503 review gates

## Epoch 50

Inspect:

### Learning health
- finite losses/parameters/optimizer;
- real nonzero updates in aggregate and fine-open paths;
- no dead environmental branch.

### Organization
- Kq distribution;
- environmental source degree;
- active group mass;
- p distribution;
- alpha distribution;
- opening blend distribution;
- fine group block work ratio;
- empty group/query/source counts.

### Accuracy
Use the existing matched development protocol.

At epoch 50 the model does not need to beat mature references, but it must not show obvious collapse or a qualitatively worse near/far tradeoff than the same-epoch Run 1502.

If unhealthy, stop.

If healthy, continue to epoch 150.

## Epoch 150

Require evidence that the central mechanism is learning.

Inspect exact epoch-150 matched comparisons against the static Run-1501 and Run-1502 epoch-150 checkpoints if available.

The important question is:

> Is the coarse-all-groups + fine-open-groups representation reducing the Run-1502 far-field penalty without giving back its near-interface/thermal advantages?

Also benchmark actual application execution with maps off.

If both field behavior and execution are moving in the wrong direction, stop.

Otherwise continue to epoch 500 unchanged.

## Epoch 500

This is the formal decision point.

Evaluate:

- exact epoch 500;
- saved-best-total;
- saved-best-field if different;
- same 90-case development holdout;
- full Q8192 field;
- near/far;
- internal/surface/fluid temperatures;
- interface flux;
- environmental port quantities;
- per-case paired comparisons;
- Q1024 organization;
- matched GPU application timing and peak memory.

Do not silently combine accuracy from one checkpoint with routing/execution from another.

### Required evidence to justify extension to epoch 5000

Run 1503 should be extended only if all of the following are defensible:

1. **No structural failure**
   - no support collapse;
   - no group-mass collapse;
   - query-local opening remains adaptive;
   - fine block ratio remains materially below the full QE rectangle.

2. **Balanced field reconstruction**
   - the near-interface improvement is not purchased by a material far-field regression;
   - pooled field accuracy is at least competitive with Run 1502 at the same epoch;
   - difficult-case tails are not obviously worse.

3. **Physical quantities remain useful**
   - internal/surface temperature gains should not disappear;
   - flux/outlet/port quantities should not show a new broad failure.

4. **Real execution benefit**
   - group-major execution must produce a meaningful measured end-to-end improvement over Run 1502 under a matched inference chunk;
   - target at least ~15% lower application latency than Run 1502 at Q8192, or a comparably strong memory/scaling gain;
   - large-shape tests should show the expected asymptotic benefit.

5. **No evidence that the aggregate path is merely an unused bypass**
   - intervention/removal evidence must show both coarse group access and fine opening contribute to useful predictions.

If these conditions are not met, stop at 500 and redesign rather than extending training.

No new Run-1503 process may automatically continue beyond epoch 500.

---

# 14. Inference chunk and timing policy

Training can retain receiver chunk 128.

Inference benchmarking must separately test a larger evaluation-only receiver chunk, including 2048 where memory permits, because earlier project evidence shows that small receiver chunks can dominate latency through repeated launch/reader overhead.

Compare every model at the same evaluation-only chunk before attributing timing differences to architecture.

At minimum report:

- full physical forward;
- prepared P2 decode;
- application evaluator;
- wall timing with CUDA synchronization;
- CUDA event time;
- incremental and total peak allocation;
- actual fine group block rows;
- coarse Q x K rows;
- source projection rows.

Do not claim speed from support counts alone.

---

# 15. What not to do in this goal

Do not:

- diagnose the still-running Run 1501;
- stop or modify Run 1501;
- relaunch Run 1502 under a new seed;
- rename Run 1502;
- add extra independent field branches;
- use direct h_k field values;
- use query-specific learned sampling;
- add a second router;
- add top-k;
- add a sparsity loss;
- add a dynamic K gate;
- perform a broad hyperparameter sweep;
- begin with a custom CUDA/Triton kernel;
- claim causal physical groups from learned incidence;
- claim execution sparsity before measured speed/memory evidence.

---

# 16. Definition of done

This Goal-mode task is complete when:

1. the existing Run 1502 has been safely resumed from exact epoch 500 toward epoch 5000 without changing its scientific configuration;
2. Run 1501 has not been monitored or modified;
3. Run 1503 adaptive hyperedge opening is implemented with focused tests;
4. a real optimizer smoke passes;
5. one formal Run 1503 is launched from scratch;
6. Run 1503 is reviewed at epoch 50;
7. if healthy, reviewed at epoch 150;
8. if still justified, reviewed at epoch 500;
9. the process stops at epoch 500 unless the epoch-500 evidence explicitly justifies a later formal continuation to 5000;
10. a concise final report records the mathematical implementation, organization, exact checkpoint identities, reconstruction metrics, physical metrics, and measured execution.

The central criterion is not whether Run 1503 is "sparser."

It is whether the new model makes the hypergraph do useful physical work:

\[
\boxed{
\text{global environment through shared hyperedge aggregates}
+
\text{fine local physics through sparse hyperedge opening}
}
\]

while preserving the clean HONF interpretation and producing a real accuracy/computation advantage.
