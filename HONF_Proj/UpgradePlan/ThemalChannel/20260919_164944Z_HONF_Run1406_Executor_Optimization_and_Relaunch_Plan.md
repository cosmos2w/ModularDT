# Run 1406 — Exact Executor Optimization and Same-Model Relaunch Plan

## Decision

Keep the **Run 1406 mathematical model unchanged**. The current evidence is much more consistent with an inference-executor problem than with a failed model core.

Run 1406 already demonstrates:

- stable and improving training through epoch 50;
- M12 optimizer-step time about **0.472×** Dense 1804;
- M12 allocated memory about **1.07×** Dense, within the declared tolerance;
- much lower inference allocated memory than Dense;
- exactly one expensive fine evaluation per unique physical query-source pair;
- `R_M=1.0` and `R_E=1.0`, so the Run-1405 group-multiplicity failure has been removed.

The failure is concentrated in large-query inference: full forward is about **2.284×** Dense and prepared P2 is about **3.2×** Dense.

Therefore do **not** create a new model ID such as Run 1407. Optimize the exact Run-1406 executor, validate same-checkpoint numerical/gradient equivalence, then relaunch the same scientific Run 1406 from scratch under the optimized implementation.

---

## 1. Mathematical model that must not change

Run 1406 remains

\[
C(q)=C_g(q)+C_M(q)+C_E(q).
\]

For source `s`:

\[
\rho_{qs}=\sum_{k=1}^{6}\alpha_{qk}A_{sk},
\]

\[
n_{qs}=\sum_{k=1}^{6}\alpha_{qk}A_{sk}h_k,
\]

with `h_k` in the 16-dimensional control space.

The expensive physical function is evaluated once per unique query-source pair:

\[
\psi(q,s,n_{qs}).
\]

Keep unchanged:

- `forward_architecture = group_control_pairwise_honf`;
- K=6 and D=16;
- entmax15 source/query routing;
- Dense-1804 MM/ME/EM preparation;
- module modulation equation;
- environment value and score modulation;
- `C_g + C_M + C_E`;
- P0/P1/P2 physical coupling;
- all trainable parameter shapes and meanings;
- dataset, losses, optimizer, seed and port policy.

The existing epoch-50 checkpoint must strict-load without state migration.

---

## 2. Primary diagnosis: nested inference tiling

`InterfaceFieldCore` already chunks receivers. The controlled inference benchmark uses an outer receiver chunk of 2048.

`GroupControlPairwiseField` also has an internal `query_tile_size=128` and both complete-support readers loop over that tile.

Thus each 2048-query outer chunk is split into

\[
2048/128=16
\]

backend subtiles. The 8192-point P2 read has four outer chunks, so each complete branch executes roughly 64 internal query tiles.

The module and environment branches both pay this fragmentation.

Training does not suffer the same way: the configured training receiver chunk is already 128, so one outer chunk maps to one backend tile. This directly explains why Run 1406 can be much faster than Dense in training but much slower in large-query inference.

### Required change

For the **complete/rectangular execution path**, process the entire current `InterfaceFieldCore` receiver chunk. The outer core already owns the memory bound.

Do not introduce a new learned or scientific setting for this. It is an exact execution change.

---

## 3. Reuse overlap instead of recomputing it

Current reads calculate

\[
\rho=\alpha A^T
\]

once for support/accounting and then calculate the same contraction again in the complete branch.

Change both source types so `rho` is computed once per current receiver chunk and passed down.

### Module

Reuse

\[
\rho^M=\alpha(A^M)^T.
\]

Only the D-wide control moment requires another contraction.

### Environment

Reuse

\[
\rho^E=\alpha(A^E)^T.
\]

Do not recompute it inside the environment-control tile.

---

## 4. Replace the generic module-control einsum with prepared GEMM data

The module moment is

\[
n_{qid}=\sum_k\alpha_{qk}A^M_{ik}h_{kd}.
\]

Prepare once per physical state:

\[
S^M_{k,(i,d)}=A^M_{ik}h_{kd}.
\]

Store it as

```text
[B, K, M*D]
```

and compute

```text
alpha [B,Q,K] @ S_M [B,K,M*D]
```

with batched GEMM, then reshape to `[B,Q,M,D]`.

This is algebraically identical to the current generic `einsum`, but better aligned with GPU matrix kernels.

Do not create any new H-wide source/group bank.

---

## 5. Vectorize the environmental control contraction over the whole receiver chunk

The source-side environment head control is already prepared in a compact form equivalent to

\[
A^E_{jk} B_h(h_k).
\]

Retain that representation.

For a complete receiver chunk, perform one contraction

\[
[B,Q,K]\times[B,K,E\,H_{heads}]
\rightarrow[B,Q,E,H_{heads}]
\]

for the entire current outer receiver chunk.

Do not execute it in 128-query Python loops.

---

## 6. Reuse the core receiver Fourier features

`InterfaceFieldCore` already computes receiver Fourier features.

The current low-dimensional router ignores those features and recomputes its own Fourier encoding from the same coordinates.

Modify `LowDimensionalGroupRouter.route_queries()` so:

- when compatible `receiver_features` are supplied, use them directly;
- retain the internal Fourier fallback only for standalone callers/tests.

Add a numerical test showing that explicit supplied features and fallback feature generation produce equivalent query controls/logits/assignments.

---

## 7. Move evidence-only bookkeeping out of the timed path

Ordinary `GroupControlPairwiseField.read()` currently calculates execution-evidence quantities even when routing maps are disabled, including logical-path matrices and per-query work ledgers.

These are useful for the untimed evidence pass, not for prediction.

### Normal train/inference

Do not compute:

- `_logical_paths()`;
- per-query logical-path arrays;
- detailed semantic-pair ledgers;
- pair provenance arrays;
- visualization-only maps.

Return only the context and lightweight summaries actually needed by maintained training diagnostics.

### Untimed evidence pass

Use an explicit runtime-only debug request to calculate the detailed work ledger and map arrays.

Do not serialize this request as model state.

---

## 8. Remove Python CUDA synchronization from support dispatch

`_complete_support()` currently turns CUDA reductions into Python booleans. This creates device-to-host synchronization.

Run 1406 already structurally guarantees at most one fine physical evaluation per q-source pair, and the measured epoch-50 support is complete (`R_M=R_E=1`). For the ordinary executor, prefer a regular rectangular exact path that can safely handle zero overlap.

### Environment

Compute all E source rows for the current receiver chunk. Apply

\[
\rho_{qj}=0 \Rightarrow s_{qj}=-\infty
\]

before softmax. Supported entries retain

\[
s_{qj}=s^{content}_{qj}+s^{geometry}_{qj}+\log\omega_j+\log\rho_{qj}.
\]

If one receiver has no supported environment source, return zero environment context for that receiver.

### Module

Prefer a regular rectangular module path. Either compact active modules once during preparation or keep the small padded M axis and multiply zero-mass rows by zero. Record padded rows separately in diagnostics.

The historical partial/gathered code can remain as a reference/evidence path, but it should not be the default Run-1406 inference path.

This is an execution policy, not a new model.

---

## 9. Full-chunk environmental reader

For one current receiver chunk, perform as few regular operations as possible:

1. query projection;
2. `rho_E` contraction;
3. head-space group-control contraction;
4. query-key content GEMM;
5. geometry-bias evaluation;
6. score modulation and masked/log-weight addition;
7. softmax;
8. value GEMM;
9. output projection.

Avoid Python loops proportional to Q.

The target execution should look like Dense 1804's regular QE reader plus small K=6/D=16 control contractions.

---

## 10. Full-chunk module reader

For one receiver chunk:

1. compute `rho_M [B,Q,M]`;
2. compute control moment `[B,Q,M,D]` via prepared GEMM;
3. compute relative Fourier features `[B,Q,M,R]`;
4. use the prepared source/global first-affine term;
5. add the relative-coordinate affine term;
6. apply GELU;
7. apply D-to-message-width group modulation;
8. execute the remaining Dense QM tail once per rectangular q-module row;
9. multiply by source measure and rho;
10. reduce over M.

Do not use `nonzero`/gather/index-add in the ordinary complete inference path.

---

## 11. Same-checkpoint parity before retraining

Use the existing Run-1406 epoch-50 checkpoint and compare old versus optimized executors.

Check:

- cases 0273 and 0653;
- P0 port context;
- P1 refinement temperatures;
- P2 global field;
- P2 port/global consistency;
- internal/interface/port outputs;
- one real predicted-port backward batch.

Suggested output criterion for non-negligible field tensors:

\[
\|U_{new}-U_{old}\|/\|U_{old}\| \le 2\times10^{-6}.
\]

Use the repository's established first-derivative relative/absolute conventions for gradients.

Do not require bitwise equality because contraction association changes.

No managed training starts until the function is numerically equivalent.

---

## 12. Same-checkpoint benchmark before relaunch

Benchmark the existing epoch-50 checkpoint under the optimized executor using the same controlled protocol:

- cases 0273 and 0653;
- Q=8192;
- outer receiver chunk=2048;
- two warmups;
- five synchronized repetitions;
- no profiler in headline timings;
- detailed execution maps disabled.

Measure:

- full physical forward;
- prepared P2 decode;
- peak allocated/reserved memory;
- fixed M1/M12 optimizer steps.

### Bounded profiler condition

If the direct exact fixes still leave full-forward latency above roughly 1.25× Dense, take one bounded CPU/CUDA profiler trace for:

- prepared P2 decode on 0273;
- one full physical forward.

Fix only the measured execution hotspot and repeat unprofiled timings.

This profiler is justified by an already measured latency failure and does not replace real execution.

---

## 13. Relaunch the same scientific Run 1406

Once the optimized executor is equivalent and shows a credible acceleration path, relaunch **Run 1406**, not a new architecture/model number.

Use:

- `group_control_pairwise_honf`;
- same K=6/D=16;
- same config, seed, dataset, optimizer and losses;
- training from scratch;
- a new timestamped managed directory identified as an executor-optimized Run-1406 rerun.

Do not call it Run 1407.

Do not warm-start from the first Run-1406 checkpoint.

---

## 14. Rerun to epoch 50

Train the fresh Run-1406 rerun to epoch 50 and repeat the same Dense-1804 comparison.

Training health:

- finite losses, outputs, parameters and gradients;
- meaningful updates in router, modulation, fine preparation, module/environment paths, global term, field head and local coupling;
- improving validation trajectory.

Fidelity sanity:

- no requirement for epoch-50 parity;
- keep the existing severe-regression review threshold around 2× matched Dense validation field MSE.

Execution targets:

\[
\text{mean full-forward ratio}\le0.95
\]

\[
\text{M12 optimizer-step ratio}\le0.95
\]

\[
\text{M12 allocated-peak ratio}\le1.10.
\]

Report prepared-P2 ratio separately.

`R_M` and `R_E` are reported but are **not required below one**. If they remain 1.0 while execution is faster, describe the result as bounded-pair group-controlled execution, not learned source sparsity.

---

## 15. Continue the same rerun to epoch 500 only if epoch-50 evidence passes

If the rerun passes the declared execution/training-health evidence:

- resume the same rerun with optimizer/RNG state;
- continue to total epoch 500;
- make no scientific/model changes.

After epoch 500:

- verify completion and checkpoint existence;
- summarize ordinary training health;
- stop.

Do not start a full 90-case endpoint evaluation and do not extend toward 5k until the user explicitly asks.

---

## 16. Visualization / evidence update

Retain the group-control interaction board but show three different notions explicitly:

### Logical group organization

- A_m / A_e;
- query alpha;
- six D=16 group controls.

### Collapsed control on physical pairs

For selected q-source pairs show:

\[
\rho_{qs}
\]

and optionally

\[
\|n_{qs}\|.
\]

### Actual execution

Annotate:

- logical q→group→source path count;
- unique physical q→source pair count;
- actual fine calls;
- padded rectangular rows, if any;
- measured prepared-P2/full-forward latency.

The figure should make the central Run-1405→1406 lesson visible:

```text
many cheap logical group paths
        ↓ collapse in D=16
one expensive physical q-source interaction
```

Do not depict route links as physical causality.

---

## 17. Code scope

Expected primary changes:

```text
src/honf_forward_core/interface_fields/group_control_pairwise.py
src/honf_forward_core/interface_fields/group_control_router.py
src/honf_forward_core/interface_fields/core.py
tools/diagnostics/run_run1406_epoch50_comparison.py
```

Small focused tests/evidence tools may be updated.

Do not refactor unrelated historical architectures.

---

## 18. Research-code constraints

Use ordinary Git, current config/checkpoint handling, standard tests, and actual GPU execution.

Do not add cryptographic hashes, contract freezes, baseline snapshots, approval services, monitoring daemons, database/process infrastructure, or other defensive over-engineering.

Preserve existing security and trusted checkpoint loading.

The epoch-50 continuation decision is a research-budget decision, not a new runtime or repository blocking gate.

---

## 19. Definition of done

This optimization/relaunch task is complete when:

1. Run-1406 math and parameterization are unchanged.
2. Existing epoch-50 checkpoint strict-loads unchanged.
3. Redundant 128-query complete-path tiling is removed.
4. Overlap is computed once and reused.
5. Module control moment uses prepared GEMM-friendly contraction.
6. Receiver Fourier features are reused.
7. Evidence-only work is opt-in.
8. Normal complete execution avoids Python CUDA synchronization and irregular support dispatch.
9. Old/new same-checkpoint output and gradient parity is documented.
10. Same-checkpoint unprofiled execution is measured.
11. If necessary, one bounded profiler-driven correction is performed.
12. One fresh same-model Run-1406 rerun reaches epoch 50.
13. The same Dense comparison is repeated.
14. Only if that evidence passes, the same rerun continues to epoch 500.
15. No 90-case endpoint evaluation or 5k continuation is launched without user instruction.
