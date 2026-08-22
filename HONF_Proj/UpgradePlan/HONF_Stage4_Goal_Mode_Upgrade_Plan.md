# Adaptive HONF Stage-4 Upgrade Plan

## Decoupled optimization, lightweight residual background, and exact-endpoint efficiency

### Authoritative source

- Repository: `cosmos2w/ModularDT`
- Branch: `agent/honf-adaptive-correctness`
- Current PR head at preparation: `ccf888d3f94f67069c89949f2269978619963d0c`
- Current model implementation is unchanged from the Run-1102 hardening commit:
  `590aaa3971e23d1d08a8e73a7e564756750b483f`

## 1. Goal-mode operating instruction

Implement the mandatory stages in this document in order. Preserve every established legacy behavior as an explicit configuration choice. Do not launch a long formal training run. Stop after focused unit/integration tests, very short smoke training, and bounded microbenchmarks.

The purpose of this round is deliberately narrow:

1. retain the fast field optimization demonstrated by Run 1103 without allowing the organizer to train at the same aggressive rate;
2. retain the accuracy-critical background correction while replacing its query-to-all-environment attention with an optional cheaper formulation;
3. remove prediction-preserving endpoint overhead in scheduled normalization;
4. leave all topology semantics, sparsity schedules, losses, capacities, and inverse code unchanged.

This is **not** a new broad HONF redesign.

## 2. Evidence that motivates this upgrade

### 2.1 Optimization is the main cause of the early convergence gap

Run 1102 and Run 1103 use the same adaptive additive architecture, data, seed, loss, Stage-A checkpoint, schedules, and from-scratch initialization. Their scientific difference is learning rate:

- Run 1102: `1e-4`;
- Run 1103: `3e-4`.

Run 1103 reaches centered validation field MSE `0.01` at epoch 1,171, versus epoch 2,200 for Run 1102. It reaches approximately the classic Run-1000 curve through much of the early and middle budget and beats Run 1102's completed best field MSE in less than half the epochs.

Therefore, lack of field capacity is not the main explanation for Run 1102's slow convergence.

### 2.2 A global `3e-4` rate is unsafe for topology formation

Run 1103's organizer becomes strongly concentrated even during the protected soft phase. When module/environment sparsification begins, the run stays near one selected edge for hundreds of epochs and suffers sustained loss regressions near the hard/exact schedule endpoints.

Support conservation remains correct. The problem is scientific concentration, not NaNs, empty support, or a masking bug.

The simplest high-probability intervention is therefore:

\[
\eta_{\text{organizer}} < \eta_{\text{prediction path}}.
\]

### 2.3 The background is small in magnitude but critical in function

The full-split decomposition shows:

- summed edges carry most RMS/energy in every physical field;
- removing the learned background from frozen checkpoints increases MSE in every channel;
- degradation is especially severe for `u`, `p`, and `omega`;
- the background behaves as a broad, low-energy correction field, especially in the far field;
- current background computation accounts for only about 7–8% of measured dense decoder time in the 2-D benchmark.

Therefore:

- do not make edge-only HONF the primary next model;
- do not keep the current dense `Q × E` background as the only option;
- add a cheap background that preserves query-dependent residual correction.

### 2.4 Hyperedge specialization is still incomplete

One broad edge often dominates all five variables and both near/far regions. This round must **not** add entropy/count/diversity/load-balancing regularization. First remove the optimizer-rate coupling and obtain a topology-qualified checkpoint under unchanged scientific semantics.

## 3. Non-goals and prohibited scope expansion

Do not change any of the following in this round:

- edge capacity, initial edge count, or minimum edge count;
- selection start/transition epochs;
- module/environment/query sparsity schedules;
- viability floors, coverage target, novelty criterion, or fallback semantics;
- entmax alpha;
- environment or query locality settings;
- descriptor-first mechanism formulation or content residual scale;
- pairwise-kernel architecture;
- field losses, local/interface/port losses, or organizer regularization;
- gathered-routing limits or retained-mass floors;
- Stage-A local surrogate behavior;
- inverse model code;
- dataset processing;
- formal checkpoint selection policy;
- any long training experiment.

Do not add edge-count, entropy, diversity, orthogonality, or load-balancing losses.

Do not promote a new experiment into the canonical formal profile based only on short tests.

## 4. Target design

The round has three mandatory changes.

---

## 4.1 Mandatory change A — optional organizer-specific learning rate

### Configuration

Add one optional training field:

```json
"organizer_learning_rate": null
```

Semantics:

- `null` or absent: preserve the current one-group AdamW behavior exactly;
- positive float: use two AdamW parameter groups:
  - `organizer`: every trainable parameter whose name starts with `core.organizer.`;
  - `prediction`: every other trainable parameter.

`training.learning_rate` remains the prediction/default learning rate.

The first split-rate scientific candidate is:

```json
"learning_rate": 0.0003,
"organizer_learning_rate": 0.0001
```

Do not add an LR scheduler in this round.

### Implementation requirements

Create a small reusable optimizer builder in the forward-training workflow rather than embedding parameter classification inline.

Required behavior:

1. Exclude every `requires_grad=False` parameter.
2. Assign every trainable parameter to exactly one group.
3. Reject duplicates or omissions.
4. Reject an empty organizer group when split mode is requested for an exchangeable organizer.
5. Preserve the current single-list AdamW constructor when `organizer_learning_rate` is unset, so historical optimizer-state resume remains unchanged.
6. Use the same weight decay for both groups.
7. Save a deterministic optimizer-group inventory in the run directory, including:
   - group name;
   - learning rate;
   - number of parameter tensors;
   - total trainable scalar count;
   - sorted parameter names;
   - a digest of the ordered names.
8. Print a concise group summary at launch.

### Resume semantics

Do not silently load an optimizer state whose parameter-group structure differs.

- Existing one-group checkpoints may resume only into one-group mode.
- New split-group checkpoints may resume only with the same group structure.
- A mismatch must raise a clear error that says to use `--initialize-checkpoint` for a new optimizer grouping.
- Model-weight initialization across grouping modes remains allowed because model tensors are unchanged.

### Why only `core.organizer.*` is slowed in this round

This is the smallest controlled intervention that directly targets the Run-1103 failure. Do not also create special encoder, decoder-head, or channel-specific group rates in the same revision.

If a future formal split-rate run still concentrates early, then extending the structural group to upstream token encoders can be considered separately.

---

## 4.2 Mandatory change B — optional lightweight additive background

### Configuration

Add one core-model field:

```json
"additive_background_mode": "dense_query_attention"
```

Supported values for this round:

- `"dense_query_attention"` — exact current behavior;
- `"global_pooled_attention"` — new cheap background.

Do not add an edge-only/`none` mode in this implementation round. The existing decomposition evaluator already supports background bypass as an ablation.

The dataclass default and the value added to the current formal profile must be `"dense_query_attention"` so historical configs/checkpoints preserve their behavior.

### Current dense background

The current branch computes query-specific attention over every environment token:

\[
a_{qe}
=
\operatorname{softmax}_e
\left(
Q(q)^\top K(e)/\sqrt{H}
\right),
\]

\[
c_q
=
\sum_e a_{qe}V(e),
\]

\[
F_{\mathrm{bg}}(q)
=
f_{\mathrm{bg}}
\left[
z_q,\;z_g,\;c_q
\right].
\]

This materializes/scales with `B × Q × E`.

### New pooled background

Use the existing background modules and parameter shapes. Do not add new parameter tensors unless absolutely necessary.

Recommended formulation:

\[
q_{\mathrm{case}}
=
Q_{\mathrm{bg}}(z_g),
\]

or, only when no global token exists,

\[
q_{\mathrm{case}}
=
Q_{\mathrm{bg}}
\left(
\operatorname{mean}_q z_q
\right).
\]

Then compute one attention distribution per case:

\[
a_e
=
\operatorname{softmax}_e
\left(
q_{\mathrm{case}}^\top K_{\mathrm{bg}}(e)/\sqrt{H}
\right),
\]

\[
z_E
=
\sum_e a_eV_{\mathrm{bg}}(e).
\]

Broadcast the pooled environment state over queries:

\[
F_{\mathrm{bg}}(q)
=
f_{\mathrm{bg}}
\left[
z_q,\;z_g,\;z_E
\right].
\]

This changes environment-attention scaling from approximately

\[
O(BQEH)
\]

to

\[
O(BEH),
\]

while retaining a query-dependent correction through `z_q` and the existing background MLP.

### Compatibility requirements

1. `"dense_query_attention"` must execute the current code path without arithmetic reordering.
2. Historical checkpoints that do not contain the new config field must default to dense mode.
3. Dense and pooled modes must instantiate the same parameter keys and shapes.
4. A Run-1102/1103 state dict must load strictly into either mode after lazy parameters are materialized.
5. Exact additive closure must remain:

\[
F_{\mathrm{pred}}
=
F_{\mathrm{background}}
+
\sum_kF_k.
\]

6. Prepared/chunked decoding must match one-shot decoding in both modes.
7. `pred_field_background` and `pred_field_by_edge` export semantics must remain unchanged.
8. Add a lightweight output diagnostic identifying the active background mode, but do not add expensive per-query logging.

### Scientific interpretation

The new mode is not expected to match the dense mode before training. It is a separately trained architecture candidate.

The purpose is to preserve the broad residual-correction role shown by the full-split decomposition while removing direct query-to-all-environment attention.

---

## 4.3 Mandatory change C — exact-endpoint scheduled-normalization short circuit

The current scheduled normalizers compute the softmax branch before returning exact entmax at blend `mu=1`.

Change both relevant paths:

- `honf_forward_core.routing.normalize_assignment`;
- `honf_forward_core.organizer._scheduled_stabilized_assignment`.

Required endpoint control flow:

```python
blend = clamp(...)
if blend >= 1.0:
    return exact_entmax(...)
# only now compute soft/stabilized-soft
if blend <= 0.0:
    return soft_endpoint
# compute both only for 0 < blend < 1
```

Requirements:

1. Exact outputs at `mu=0`, intermediate `mu`, and `mu=1` remain unchanged.
2. At `mu=1`, the softmax/stabilization branch must not execute.
3. No parameters or checkpoint keys change.
4. Existing schedule semantics remain unchanged.

## 5. Required experiment overlays

The strict overlay loader permits changes only to keys already declared in the base profile. Therefore add the new keys to `adaptive_sparse_additive.json` using legacy-preserving values:

```json
"additive_background_mode": "dense_query_attention"
```

and

```json
"organizer_learning_rate": null
```

Create these small overlays under the existing forward experiment directory.

### 5.1 Uniform middle-rate control

Suggested filename:

```text
stage4_uniform_lr2e4_dense_background.json
```

Changes only:

```json
{
  "core": {
    "training": {
      "learning_rate": 0.0002,
      "organizer_learning_rate": null
    }
  }
}
```

### 5.2 Split-rate dense-background candidate

Suggested filename:

```text
stage4_split_lr_dense_background.json
```

Changes only:

```json
{
  "core": {
    "model": {
      "core_honf": {
        "additive_background_mode": "dense_query_attention"
      }
    },
    "training": {
      "learning_rate": 0.0003,
      "organizer_learning_rate": 0.0001
    }
  }
}
```

### 5.3 Split-rate pooled-background candidate

Suggested filename:

```text
stage4_split_lr_pooled_background.json
```

Changes only:

```json
{
  "core": {
    "model": {
      "core_honf": {
        "additive_background_mode": "global_pooled_attention"
      }
    },
    "training": {
      "learning_rate": 0.0003,
      "organizer_learning_rate": 0.0001
    }
  }
}
```

Update the experiment README with the scientific purpose of each overlay.

Do not alter the canonical learning rate or background mode beyond adding the declarative compatibility fields.

## 6. Staged implementation procedure and gates

---

## Stage 0 — baseline capture and scope guard

### Work

1. Confirm the checked-out branch and record `git rev-parse HEAD`.
2. Confirm that the only tracked change after `590aaa3` is documentation unless the local branch has advanced.
3. Run the current focused tests before editing.
4. Save a deterministic synthetic dense-additive forward result and state-dict key digest for parity testing.
5. Inspect the exact local run/checkpoint APIs before editing optimizer resume behavior.

### Pass gate

- Baseline focused tests pass.
- No unexplained local model-code changes exist.
- Dense reference output and key digest are recorded.
- No training run is launched.

---

## Stage 1 — optimizer grouping

### Work

1. Add configuration-loader support for `organizer_learning_rate`.
2. Add validation:
   - `null` accepted;
   - positive float accepted;
   - zero/negative rejected.
3. Implement the optimizer builder and inventory.
4. Add explicit resume-group compatibility validation.
5. Add the uniform and split-LR overlays.
6. Add tests.

### Required tests

1. Legacy config without `organizer_learning_rate` creates exactly one optimizer group.
2. Explicit `null` creates the same one-group structure.
3. One-group mode uses the existing base LR and weight decay.
4. Split mode creates exactly two groups with exact requested LRs.
5. Every `core.organizer.*` trainable parameter is in the organizer group.
6. No non-organizer parameter is in the organizer group.
7. Every trainable parameter appears exactly once.
8. Frozen local-surrogate parameters appear in neither group.
9. One forward/backward/optimizer step is finite and updates parameters in both groups.
10. A split-group optimizer checkpoint resumes into the same split configuration.
11. One-group ↔ split-group resume mismatch raises a clear error.
12. Model-only initialization across grouping modes remains valid.

### Pass gate

All tests pass. The old one-group path is unchanged. The split path is fully inventoried and resume-safe.

---

## Stage 2 — lightweight background

### Work

1. Add `additive_background_mode` to strict core configuration.
2. Implement a private background helper to keep the main additive method readable.
3. Move the existing dense arithmetic into the dense branch without changing operation order.
4. Implement the global pooled branch using existing modules.
5. Add mode diagnostics.
6. Add the pooled-background overlay and tests.

### Required tests

1. Missing background-mode field defaults to dense mode.
2. Invalid mode is rejected.
3. Dense mode produces bitwise-identical output to the Stage-0 reference.
4. Dense mode state-dict keys match the pre-change digest.
5. Dense and pooled modes have identical parameter keys and shapes.
6. Existing dense checkpoint weights load strictly into pooled mode.
7. Both modes satisfy exact additive closure.
8. Both modes support `return_edge_fields=True`.
9. Both modes have finite forward and backward passes.
10. Gradients reach background query/key/value/global/head parameters in pooled mode.
11. Prepared/chunked decoding matches one-shot decoding in both modes.
12. Context-fusion behavior and keys are unchanged.
13. In pooled mode, the environment-attention tensor is case-level `[B,E]`, not `[B,Q,E]`.

### Bounded microbenchmark

Use one fixed materialized model and synthetic or mapped ThermalChannel batch. Benchmark only after warmup.

Keep the benchmark short:

- no more than 10 warmups;
- no more than 50 measured iterations per mode;
- synchronize CUDA around timing;
- report median decoder time and peak allocated memory.

The benchmark is diagnostic, not a formal performance result.

### Pass gate

- Dense parity is exact.
- Pooled mode is finite, trainable, and closure-preserving.
- Pooled mode does not materialize query-by-environment attention.
- Pooled decoder time and allocated memory are no worse than dense in the fixed benchmark; record the observed difference without overclaiming.
- No long training is run.

---

## Stage 3 — exact-endpoint efficiency

### Work

Implement the `mu=1` short circuit in both scheduled-normalization paths.

### Required tests

1. `mu=0` equals the existing stabilized-soft/soft endpoint exactly.
2. Intermediate `mu` equals the previous convex blend.
3. `mu=1` equals exact entmax exactly.
4. A monkeypatch/call counter proves the softmax branch is not called at `mu=1`.
5. Masked and unmasked endpoints remain normalized and finite.
6. Existing scheduled-adaptive tests still pass.

### Pass gate

All exactness tests pass, and no model parameter/state key changes.

---

## Stage 4 — bounded integration validation and closeout

### Short tests only

Run:

1. Python compilation on changed modules.
2. JSON parsing/strict config composition for all three new overlays.
3. Focused core tests:
   - forward additive;
   - scheduled adaptive;
   - config and registry;
   - forward upgrade config;
   - sparse/gathered routing tests affected by normalization.
4. Focused ThermalChannel tests:
   - checkpoint/resume;
   - forward partial initialization;
   - resource/config integration.
5. The complete CPU test suites only if they remain ordinary unit-test duration.
6. CUDA-sensitive focused tests only if an idle GPU is available. Use GPU 0 by default. 
7. Very short smoke training for each new optimizer/background combination:
   - at most 2 epochs;
   - at most 4 train batches per epoch;
   - at most 2 validation batches;
   - no scientific convergence claim.

Do not launch any command resembling a formal 650-, 1,000-, 5,000-, or 10,000-epoch run.

### Smoke pass gate

For every smoke configuration:

- loss and diagnostics are finite;
- checkpoints save;
- same-config resume works for one additional bounded step;
- selected/viable/support tensors are finite;
- empty-selected and post-fallback support diagnostics remain zero on the smoke batches;
- exact additive closure holds on an evaluation batch.

### Closeout report

Write a concise implementation report containing:

- branch and final commit;
- changed files;
- exact config fields and defaults;
- optimizer-group inventories;
- dense-parity result;
- pooled-background benchmark;
- tests run and pass/skip counts;
- any unresolved limitation;
- explicit statement that no long formal run was launched.

Commit and push only after every mandatory gate passes.

## 7. Scientific run sequence after Codex stops

The following runs are **not** to be launched by Codex. They are future user-controlled experiments.

---

## Future Run A — uniform `2e-4`, dense background

Purpose: complete the missing controlled LR point.

Use:

- unchanged model;
- `learning_rate=2e-4`;
- `organizer_learning_rate=null`;
- dense background.

### Scientific acceptance gates

By epoch 350:

- all eight candidates remain viable;
- no empty selected edge;
- no post-fallback zero-support row.

Across the schedule:

- no selected/viable count at or below two for more than 100 consecutive epochs;
- 25-epoch post/pre field, temperature, and total-loss ratios at epochs 400 and 650 are each no greater than 1.10.

Convergence:

- centered field MSE `0.01` by no later than approximately epoch 1,500;
- materially faster threshold crossing than Run 1102.

Late topology window:

- median selected/functional edges at least 3;
- normalized module and environment mass entropy above 0.3;
- module and environment maximum mass below 0.8;
- no universal one-edge contribution monopoly.

If this run satisfies both convergence and topology gates, it may be preferred over split LR because it is simpler.

---

## Future Run B — split LR, dense background

Purpose: retain Run-1103 field speed while stabilizing organization.

Use:

- prediction LR `3e-4`;
- organizer LR `1e-4`;
- dense background.

### Scientific acceptance gates

All safety/topology gates from Future Run A apply.

Additional target:

- centered field MSE threshold crossing should be much closer to Run 1103 than Run 1102;
- no Run-1103-style 300–600 epoch near-one-edge interval;
- no sustained epoch-650 loss multiplication.

If the split run remains concentrated during epochs 1–350, do not add regularization. The next diagnosis is whether upstream token encoders need the lower structural LR.

---

## Future Run C — split LR, pooled background

Launch only after a dense-background optimizer configuration passes.

Purpose: test whether the broad residual correction can be represented without `Q × E` background attention.

### Scientific acceptance gates

Against the topology-qualified dense-background control:

- exact additive closure;
- aggregate full-split fluid field MSE degradation no greater than 5%;
- no individual channel MSE degradation greater than 10%;
- far-field `u`, `p`, and `omega` MSE degradation no greater than 10%;
- topology safety and concentration no worse than the control;
- measured decoder time improves by at least approximately 3%, or peak allocated memory falls materially, on the same benchmark protocol.

The 3% timing target is modest because the current dense background is only about 7–8% of measured decoder time in the 2-D case.

If pooled background fails accuracy gates, retain dense background for 2-D and defer a richer low-rank background to the later 3-D architecture round.

## 8. Interpretation rules

1. Do not select a checkpoint from field MSE alone.
2. Always compare best-field and latest checkpoints.
3. Treat one-edge concentration as a scientific failure even when support diagnostics are valid.
4. Do not infer background redundancy from energy fraction.
5. Do not infer mechanism specialization from selected-edge count alone.
6. Do not claim gathered speed until route counts and retained mass support it.
7. Do not claim the pooled background is accurate from smoke tests; only a future controlled training/evaluation can establish that.
8. Do not modify the canonical formal profile until a future run passes both predictive and topology gates.

## 9. Expected changed files

The exact organization may vary, but expected files include:

- `HONF_Proj/src/honf_forward_core/config.py`
- `HONF_Proj/src/honf_forward_core/decoder.py`
- `HONF_Proj/src/honf_forward_core/routing.py`
- `HONF_Proj/src/honf_forward_core/organizer.py`
- `HONF_Proj/src/honf_runtime/config_loader.py`
- `HONF_Proj/src/config_core/schemas/core_config.schema.json`
- `HONF_Proj/src/config_core/forward/adaptive_sparse_additive.json`
- new Stage-4 experiment overlays
- experiment README
- `HONF_Proj/Case_ThermalChannel/src/channelthermal/workflows/train_forward.py`
- focused core and ThermalChannel tests
- optional bounded benchmark tool
- implementation closeout report

Do not touch inverse code.

## 10. Stop conditions for Codex

Stop and report rather than broadening the design if:

- exact dense-background parity cannot be preserved;
- the pooled mode requires changing existing parameter shapes;
- optimizer grouping cannot resume safely without silent state loss;
- implementation requires changing topology schedules or adding regularization;
- a short benchmark is inconclusive;
- mapped external data/checkpoints are unavailable.

Unavailable external artifacts may cause explicit skips. They must not trigger a long replacement training run.

## 11. Completion definition

The task is complete when:

- old behavior is preserved and tested;
- split LR is available, inventoried, and resume-safe;
- pooled background is available with the same parameter-key structure;
- exact-endpoint redundant softmax work is removed;
- three focused overlays resolve strictly;
- short smoke and unit/integration gates pass;
- a closeout report is committed;
- no long formal run has been launched.

---

## 12. Implementation closeout — 2026-08-20

### Repository state and scope

- Branch: `agent/honf-adaptive-correctness`.
- Starting commit: `ccf888d3f94f67069c89949f2269978619963d0c`.
- Final implementation: the commit containing this closeout report; its exact
  hash is recorded in the pushed branch/PR history because a Git commit cannot
  contain its own hash.
- No inverse source or configuration was modified.
- No topology rule, schedule boundary, locality rule, loss, regularizer,
  gathered-routing behavior, Stage-A model, or checkpoint parameter tensor was
  changed.
- No long formal training was launched.

### Changed files

- `src/honf_forward_core/config.py`
- `src/honf_forward_core/decoder.py`
- `src/honf_forward_core/routing.py`
- `src/honf_forward_core/organizer.py`
- `src/honf_runtime/config_loader.py`
- `src/config_core/schemas/core_config.schema.json`
- `src/config_core/forward/adaptive_sparse_additive.json`
- `src/config_core/forward/experiments/README.md`
- `src/config_core/forward/experiments/stage4_uniform_lr2e4_dense_background.json`
- `src/config_core/forward/experiments/stage4_split_lr_dense_background.json`
- `src/config_core/forward/experiments/stage4_split_lr_pooled_background.json`
- `Case_ThermalChannel/src/channelthermal/workflows/train_forward.py`
- `tests/test_config_and_registry.py`
- `tests/test_forward_additive.py`
- `tests/test_forward_upgrade_config.py`
- `tests/test_scheduled_adaptive.py`
- `Case_ThermalChannel/tests/test_stage4_optimizer.py`
- `tools/benchmark_stage4_background.py`
- `tools/smoke_stage4.py`
- this plan/closeout document.

### Public configuration additions and defaults

- `training.organizer_learning_rate`: optional positive float or `null`;
  absent/`null` preserves the literal historical one-group AdamW path. The
  canonical adaptive profile declares `null` and retains `learning_rate=1e-4`.
- `model.core_honf.additive_background_mode`:
  `dense_query_attention` (default and canonical formal behavior) or
  `global_pooled_attention` (explicit experiment only).
- The three Stage-4 overlays resolve strictly and retain full overlay payload,
  source path, and hash provenance through the existing config bundle.

### Optimizer inventories from mapped-data smoke runs

The persisted inventories include the complete sorted parameter-name lists.

| Overlay/group | LR | tensors | trainable scalars | ordered-name SHA-256 |
|---|---:|---:|---:|---|
| uniform/all | 2e-4 | 150 | 3,795,763 | `55161cf2605b894a767a3c7845c86dc067f0481d5b2f7f47114d3a86487162bd` |
| split/organizer | 1e-4 | 28 | 1,315,584 | `579575beafc9f929daab79ffeed0bf6ac0684253c66a6d406d318e8e59d5439e` |
| split/prediction | 3e-4 | 122 | 2,480,179 | `a7c373b06160eb5c493ef4d399aa1632ba652a356db263b269645d525dd69acb` |

All trainable parameters occur exactly once, all organizer names begin with
`core.organizer.`, and frozen Stage-A parameters are excluded. Checkpoints save
the inventory beside optimizer state. One-group/split mismatch and changed
split membership/LR are rejected with an explicit instruction to use
`--initialize-checkpoint`; historical one-group checkpoints remain resumable.

### Dense parity and checkpoint compatibility

The deterministic Stage-0 CPU reference remains bitwise identical:

- state-dict key count: 84, with unchanged keys/shapes;
- `pred_field` SHA-256:
  `24d4a0e39e56f4afcc9c457e92b2ceedb9ef1ffc73547fb6d6bc0b1e185e1a0b`;
- `pred_field_background` SHA-256:
  `3fc4339b12a3089cb40b11ad4cfbaac5f12f7b34a01ce4629919b29e3b59d6ce`;
- `pred_field_by_edge` SHA-256:
  `e9c14b54947667f032d491262e2b298119af07a48b06b71206cffb5f2ec3b2a8`;
- exact exported additive-closure maximum error:
  `5.820766091346741e-11` under the Stage-0 subtraction ordering, while direct
  reconstruction comparison remains bitwise equal.

Dense and pooled models have identical parameter names/shapes. The frozen
Run-1102 `best_by_field_mse_model.pt` was also loaded strictly into pooled mode:
261 state keys, zero missing keys, and zero unexpected keys. Context-fusion
construction and state keys remain unchanged.

### Bounded background benchmark

Exact command:

```bash
python tools/benchmark_stage4_background.py \
  --device cpu --queries 512 --warmups 3 --iterations 10
```

Fixed synthetic decoder state, 192 environment tokens, PyTorch 2.6.0:

| mode | median decoder time | minimum decoder time |
|---|---:|---:|
| dense query attention | 15.495 ms | 12.604 ms |
| global pooled attention | 14.091 ms | 11.237 ms |

The pooled/dense median ratio was 0.909. All available GPUs were at 99–100%
utilization, so CUDA timing and allocated/reserved-memory measurements were
explicitly skipped rather than interfering with active work. CPU PyTorch does
not expose the corresponding CUDA allocator metrics. This benchmark establishes
only bounded execution behavior, not pooled-mode scientific accuracy.

### Bounded mapped-data smoke and resume gate

Exact command:

```bash
python tools/smoke_stage4.py \
  --device cpu \
  --output-root /tmp/honf_stage4_smoke_20260820_a
```

Each of the three overlays used 32 sampled field points, batch size 1, one
training batch and one validation batch for epoch 1, then resumed the same
checkpoint/config for the single bounded epoch-2 step. All losses and
diagnostics were finite; checkpoints saved; same-config optimizer resume
worked; empty-selected and post-fallback zero-support diagnostics were zero;
and evaluation-batch additive closure had maximum error 0.0 for every overlay.
Total wall time was 22.71 seconds.

### Tests and exact results

```bash
python -m py_compile \
  src/honf_forward_core/config.py \
  src/honf_forward_core/decoder.py \
  src/honf_forward_core/routing.py \
  src/honf_forward_core/organizer.py \
  src/honf_runtime/config_loader.py \
  Case_ThermalChannel/src/channelthermal/workflows/train_forward.py \
  tools/benchmark_stage4_background.py \
  tools/smoke_stage4.py
```

Result: pass.

```bash
python -m pytest -q \
  tests/test_forward_additive.py \
  tests/test_scheduled_adaptive.py \
  tests/test_config_and_registry.py \
  tests/test_forward_upgrade_config.py \
  tests/test_sparse_routing.py \
  tests/test_gathered_routing.py \
  Case_ThermalChannel/tests/test_stage4_optimizer.py \
  Case_ThermalChannel/tests/test_forward_partial_initialization.py \
  Case_ThermalChannel/tests/test_resources_and_checkpoint.py
```

Result: 104 passed in 4.07 seconds.

```bash
python -m pytest -q tests
```

Result: 190 passed in 5.93 seconds.

```bash
python -m pytest -q Case_ThermalChannel/tests
```

Result: 48 passed, 1 skipped in 7.50 seconds. The skip is the pre-existing
inverse joint-training integration check whose local inverse artifacts are
unavailable; inverse code was outside this task.

### Remaining limitations

- CUDA time/peak-memory comparison remains to be run when an idle GPU is
  available.
- The pooled background has only correctness, gradient, closure, strict-load,
  and bounded smoke evidence. Its accuracy must be evaluated by a future
  controlled experiment; this closeout makes no accuracy claim.
