# Run 1404 — Routing-Only Pairwise HONF

## Minimal Stage-7 revision of Run 1401

## 0. Purpose and design thesis

Run 1404 is a deliberately minimal experiment built directly from the mature Run-1401 Stage-7 model.

The purpose is to test one narrow architectural hypothesis:

> **Can the Run-1401 hypergraph stop acting as a direct latent value carrier and instead act primarily as a routing index that determines which fine query–module interactions should contribute to the field?**

The experiment therefore does **not** redesign the organizer, change the number of hyperedges, introduce dynamic topology, add sparsemax, replace the pair kernel, change the local surrogate, or modify the physical losses.

Run 1404 keeps the accepted Run-1401 organizer and query-to-hyperedge routing, but removes the direct hyperedge-value context from the predicted field and strengthens the routed fine pair path using the organizer's already-computed per-module environment context.

The central computation becomes

\[
\boxed{
\text{hypergraph routing}
\;\longrightarrow\;
\beta_{qi}
\;\longrightarrow\;
\text{fine }(q,i)\text{ interaction}
\;\longrightarrow\;
\text{field}
}
\]

rather than

\[
\text{hypergraph routing}
+\text{pooled hyperedge value}
+\text{fine pair interaction}.
\]

This is intended as the **minimum-change bridge** between classical Run 1401 and the later routing-first Runs 2000/2100/2200.

Run 1404 should answer whether the routing-only interpretation is already viable inside the proven Run-1401 topology before introducing a new sparse topology generator.

---

# 1. Repository and compatibility basis

Planning reference:

- repository: `cosmos2w/ModularDT`;
- branch: `agent/honf-core-next`;
- inspected head: `9ae17f1eed259976336e4ad2359d95014440783f`;
- base profile: `src/config_core/forward/stage7_structured_context.json`;
- historical scientific reference: Run 1401, fixed-projection organizer with \(K=6\).

Recent decoder-context ablation work on this branch must remain intact. Run 1404 is a new experimental overlay and must not change the behavior or checkpoint loading of Runs 1401, 1402, 1403, the interface-field families, or the later routing-first work.

The intended implementation should retain the **same checkpoint-visible layer shapes as Run 1401**. In particular, removing direct hyperedge values should be a runtime use decision, not a destructive removal of historical layers.

---

# 2. Why Run 1404 is feasible with a very small change

## 2.1 Run 1401 already contains the required routing factorization

Let:

- \(M\): active module count;
- \(E\): environment-token count;
- \(Q\): query count;
- \(K=6\): Run-1401 hyperedge count;
- \(H=256\): hidden width;
- \(m_i\in\mathbb R^H\): encoded module token;
- \(e_j\in\mathbb R^H\): encoded environment token;
- \(h_k\in\mathbb R^H\): organized hyperedge state;
- \(A^{MH}_{ik}\): module-to-hyperedge assignment;
- \(\alpha_{qk}\): query-to-hyperedge routing;
- \(\psi(q,i)\in\mathbb R^H\): fine query–module response.

Run 1401 already computes

\[
\bar A^{MH}_{ik}
=
\frac{A^{MH}_{ik}}
{\sum_m A^{MH}_{mk}+\epsilon},
\]

and

\[
\boxed{
\beta_{qi}
=
\sum_{k=1}^{K}
\alpha_{qk}\bar A^{MH}_{ik}.
}
\]

Because

\[
\sum_i\bar A^{MH}_{ik}=1,
\qquad
\sum_k\alpha_{qk}=1,
\]

we have

\[
\sum_i\beta_{qi}=1
\]

for ordinary full-support routing.

The routed pair term is therefore already a normalized fine-source mixture:

\[
c_{\mathrm{pair}}(q)
=
 g_p
\sum_i\beta_{qi}\psi(q,i).
\]

The existing fused executor computes exactly this factorization before pair aggregation.

## 2.2 What Run 1401 additionally does

Run 1401 also computes the direct hyperedge-value context

\[
\boxed{c_H(q)=\sum_k\alpha_{qk}V_Hh_k.}
\]

Its context-fusion decoder is

\[
c_{1401}(q)
=
c_H(q)
+
c_{\mathrm{pair}}(q)
+
c_{\mathrm{global}}(q)
+
c_{\mathrm{near}}(q).
\]

The field is

\[
\widehat U(q)
=
D_\theta\!\left(
\operatorname{LN}[c_{1401}(q)]
\right).
\]

The new routing-first hypothesis says that \(h_k\) may be useful as a **routing descriptor** without being injected as a pooled physical value.

---

# 3. Run-1404 mathematical model

## 3.1 Preserve the Run-1401 organizer exactly

Run 1404 retains:

\[
A^{ME},\quad
A^{MH},\quad
A^{EH},\quad
h_k,\quad
s_k,\quad
r_k,
\]

with the same fixed-projection organizer, six hyperedges, geometry bias, softmax assignments, module–environment auxiliary attention, and hyperedge-state construction.

There is **no new topology generator** in Run 1404.

The pooled hyperedge state remains useful for query routing:

\[
\ell_{qk}
=
\frac{
(W_Qz_q)^\top(W_Kh_k)
}{\sqrt H}
+b_{\mathrm{geo}}(q,k),
\]

\[
\boxed{
\alpha_{qk}
=
\operatorname{softmax}_k(\ell_{qk}).
}
\]

Thus \(h_k\) is retained as a routing key.

It is no longer a field value.

## 3.2 Remove only the direct hyperedge-value term

Run 1404 sets

\[
\boxed{c_H(q)=0.}
\]

The hyperedge value projection may remain registered for historical state-dict compatibility, but its output is not used by the Run-1404 field computation.

This distinction is important because the current decoder's internal variable named `hyper_context` later also contains the pair context. Run 1404 removes the **direct hyperedge-value context \(c_H\)**; it does not disable hyperedge routing or the routed pair term.

## 3.3 Reuse the organizer's per-module environment context in the fine pair kernel

Run 1401 already computes module-to-environment attention

\[
A^{ME}_{ij}
=
\operatorname{softmax}_j
\left(
\frac{Q_M(m_i)^\top K_E(e_j)}{\sqrt H}
\right),
\]

and

\[
c^{ME}_i
=
\sum_j A^{ME}_{ij}e_j.
\]

Before hyperedge construction it forms the environment-contextualized module token

\[
\boxed{
\widetilde m_i
=
\left(
 m_i + 0.25\,W_{ME}c^{ME}_i
\right)P_i,
}
\]

where \(P_i\) is the active-module mask.

Run 1401's pair kernel currently consumes the original \(m_i\), even though \(\widetilde m_i\) already exists in the organizer output.

Run 1404 changes only the pair-kernel source token:

\[
\boxed{m_i^{\mathrm{pair}}=\widetilde m_i.}
\]

No new feature width or new trainable layer is required. The tensor has the same shape \([B,M,H]\).

The fine pair response becomes

\[
\boxed{
\psi_{1404}(q,i)
=
P_\theta
\left[
\Phi(q-x_i),
\widetilde m_i,
 s_i,
 P_i
\right].
}
\]

Thus the micro-interaction remains explicitly query–module, but the module state now contains the environment information that Run 1401 had already computed before hyperedge pooling.

This is the main **pair-term enrichment** in Run 1404.

## 3.4 Route the fine pair response through the hypergraph

Define

\[
\gamma_{qik}
=
\alpha_{qk}\bar A^{MH}_{ik}.
\]

The total query–module routing weight is

\[
\boxed{
\beta_{qi}=\sum_k\gamma_{qik}.
}
\]

The Run-1404 main context is

\[
\boxed{
 c_R(q)
=
 g_p\sum_i\beta_{qi}\psi_{1404}(q,i).
}
\]

The fine interaction is therefore instantiated at the module level; the hypergraph supplies its routing weight rather than a substitute pooled value.

## 3.5 Preserve the established global and near paths

The final context is

\[
\boxed{
 c_{1404}(q)
=
 c_R(q)
+
 c_{\mathrm{global}}(q)
+
 c_{\mathrm{near}}(q).
}
\]

and

\[
\boxed{
\widehat U_{1404}(q)
=
D_\theta
\left[
\operatorname{LN}(c_{1404}(q))
\right].
}
\]

Do **not** simultaneously remove the global or near paths in Run 1404. The current branch already contains separate Stage-7 context-ablation profiles for those questions. Combining those ablations with the routing-only change would make the result harder to interpret.

The Run-1404 scientific comparison is deliberately:

\[
\boxed{
\text{same organizer}
+
\text{same global/near context}
+
\text{same pair MLP}
+
\text{no }c_H
+
\text{contextualized fine module token}.
}
\]

---

# 4. Why this is a stronger routing interpretation than Run 1401

In Run 1401, the hypergraph affects the field through two separate channels:

\[
\alpha,h_k
\rightarrow c_H,
\]

and

\[
\alpha,A^{MH}
\rightarrow\beta
\rightarrow c_{\mathrm{pair}}.
\]

Run 1404 deletes the first field-value arrow.

The remaining principal hypergraph path is

```text
module + environment organization
              ↓
hyperedge routing descriptors h_k
              ↓
query-to-edge alpha(q,k)
              ↓
edge-to-module incidence A(i,k)
              ↓
query-to-module beta(q,i)
              ↓
fine query-module response psi(q,i)
              ↓
field
```

The hyperedge state still contains pooled information because it must decide routing. However, that pooled vector is not injected into the physical field as a value representation.

This is exactly the distinction required for a routing-first interpretation:

\[
\boxed{
\text{pooled hyperedge state determines where information travels,}
}
\]

not

\[
\boxed{
\text{pooled hyperedge state replaces the fine information being transported.}
}
\]

Run 1404 remains much closer to the proven Run-1401 optimization problem than Runs 2000/2100/2200. It therefore serves as a useful bridge experiment.

---

# 5. Computational interpretation

## 5.1 Exact fused execution during training

Set

```text
pairwise_aggregation_mode = "fused_query_module"
```

and compute

\[
\beta_{qi}
=
\sum_k\alpha_{qk}\bar A^{MH}_{ik}
\]

before pair aggregation.

The dense training path evaluates all active/padded candidate pairs required by the current implementation, but avoids using the hyperedge-value field term and avoids materializing the historical full \([B,Q,K,H]\) edge-pair context in ordinary fused execution.

This is the correct training reference for Run 1404.

## 5.2 Gathered deployment is a separate execution mode

After training, use the same learned \(\beta\) to select query–module pairs before evaluating the expensive pair MLP.

For retained-mass threshold \(\tau_\beta\), select the smallest ranked module set \(S_q\) such that

\[
\sum_{i\in S_q}\beta_{qi}
\ge
\tau_\beta.
\]

Then evaluate

\[
\psi_{1404}(q,i)
\]

only for

\[
i\in S_q.
\]

The sparse approximation is

\[
\widehat c_R(q)
=
 g_p
\sum_{i\in S_q}
\beta_{qi}\psi_{1404}(q,i),
\]

with **no renormalization of retained \(\beta\)**.

This is already supported by the maintained fused/gathered machinery and should be reused rather than creating another sparse executor.

## 5.3 Expected efficiency limits

Run 1404 should not be advertised as automatically sparse.

The current ThermalChannel layouts contain relatively few active modules, and the mature Run-1401 routing was diffuse enough that retained-mass thresholds often kept every active module.

Therefore:

- fused execution can remove unnecessary edge-context tensors;
- gathered execution can avoid padded/full pair tensors and may lower memory/time;
- true asymptotic savings require the learned \(\beta\) to become concentrated as \(M\) grows.

Run 1404 tests whether routing-only training naturally makes \(\beta\) more useful as an execution mask. It does not force it with Sparsemax or a sparsity penalty.

Exact sparse topology generation belongs to the separate 2000/2100/2200 research line.

---

# 6. Run-1404 vertical flow

```text
MODULES + ENVIRONMENT + GLOBAL CONTEXT
                  ↓
          RUN-1401 ENCODER
                  ↓
       module-to-environment A_ME
                  ↓
 contextualized module token m_i~
                  ↓
      RUN-1401 FIXED-K ORGANIZER
      A_MH, A_EH, h_k, geometry
                  ↓
QUERY ───────► alpha(q,k)
                  ↓
      beta(q,i)=sum_k alpha(q,k) Abar(i,k)
                  ↓
      ┌──────────────────────────────────┐
      │ fine pair interaction           │
      │ psi(q,i ; m_i~, raw features)   │
      └──────────────────────────────────┘
                  ↓
        sum_i beta(q,i) psi(q,i)
                  ↓
      + global context + near context
                  ↓
              field head
                  ↓
          continuous physical field

DIRECT h_k VALUE PATH:
    alpha(q,k) V h_k  ──X──  removed
```

---

# 7. Minimal code changes

## 7.1 Add one opt-in pair-token source

Add a generic legacy-HONF configuration field:

```text
pairwise_module_token_source
```

Allowed values:

```text
"base"
"organizer_contextualized"
```

Historical default:

```text
"base"
```

Run 1404:

```text
"organizer_contextualized"
```

In `HypergraphGatedPairwiseKernel`, centralize module-token resolution in one helper, conceptually:

```python
if config.pairwise_module_token_source == "base":
    pair_module_tokens = organizer_output["module_tokens"]
elif config.pairwise_module_token_source == "organizer_contextualized":
    pair_module_tokens = organizer_output["module_tokens_for_hyper"]
```

Then use `pair_module_tokens` consistently in:

- dense pair embeddings;
- selected pair embeddings;
- edge-explicit compatibility path when exercised;
- diagnostic reconstruction.

Do not modify the organizer arithmetic.

The contextualized token has the same hidden width, so the pair MLP parameter shapes remain unchanged.

## 7.2 Reuse existing decoder semantics

Run 1404 should use the existing pairwise-only semantics rather than adding another large decoder implementation.

The experimental overlay should explicitly set:

```text
decoder_mode = "enhanced_honf_pairwise_only"
use_hyper_value_context = false
```

Hyperedge routing remains active because the decoder still needs \(h_k\), \(\alpha_{qk}\), and \(A^{MH}\) to build \(\beta\).

The global and near terms remain enabled.

Do not change the historical meaning of any existing decoder mode for old profiles.

## 7.3 Use fused pair aggregation

Set

```text
pairwise_aggregation_mode = "fused_query_module"
routing_execution = "dense"
query_module_retained_mass_floor = 1.0
```

for training.

The deployment/evaluation overlay may change only:

```text
routing_execution = "gathered"
query_module_retained_mass_floor = <evaluation threshold>
```

without changing model parameters.

## 7.4 Keep the pair kernel itself unchanged

Retain:

```text
pairwise_kernel_mode = "legacy_mlp"
pairwise_kernel_hidden_dim = 256
pairwise_kernel_num_layers = 4
pairwise_kernel_use_fourier = true
pairwise_kernel_fourier_frequencies = 4
pairwise_kernel_include_module_token = true
pairwise_kernel_include_module_features = true
pairwise_kernel_normalize_by_edge_mass = true
```

Do not introduce another factorized kernel or pair expert in Run 1404.

## 7.5 Configuration overlay

Create one experimental overlay, for example:

```text
src/config_core/forward/experiments/stage7_routing_only_pairwise_1404.json
```

It should inherit `stage7_structured_context` and change only:

```json
{
  "model": {
    "core_honf": {
      "decoder_mode": "enhanced_honf_pairwise_only",
      "use_hyper_value_context": false,
      "pairwise_aggregation_mode": "fused_query_module",
      "pairwise_module_token_source": "organizer_contextualized",
      "routing_execution": "dense",
      "query_module_retained_mass_floor": 1.0
    }
  },
  "checkpointing": {
    "save_epoch_milestones": [10, 50, 100, 250, 500, 1000, 2500, 5000]
  }
}
```

Do not change:

- \(K=6\);
- organizer mode;
- assignment normalizers;
- environment/query locality;
- hidden width;
- pair-kernel width/depth;
- data and losses;
- Stage-A model;
- optimizer;
- port curriculum;
- global/near context.

The proposed managed run identity is:

```text
Run 1404
stage7_routing_only_contextual_pair
```

---

# 8. Pre-training frozen-checkpoint diagnosis

Because the proposed changes do not alter checkpoint-visible parameter shapes, first use the existing Run-1401 checkpoint for a controlled evaluation-only diagnostic.

Use the same Run-1401 weights under three arithmetic variants:

### Reference

```text
A. Run 1401 normal
```

\[
c=c_H+c_{\mathrm{pair}}+c_G+c_N.
\]

### Routing-only without pair enrichment

```text
B. no c_H + base pair module token
```

\[
c=c_{\mathrm{pair}}(m_i)+c_G+c_N.
\]

### Routing-only with contextualized pair source

```text
C. no c_H + organizer-contextualized pair module token
```

\[
c=c_{\mathrm{pair}}(\widetilde m_i)+c_G+c_N.
\]

Evaluate a bounded matched subset, including ordinary and difficult layouts.

Measure:

- prediction RMS difference from normal Run 1401;
- ground-truth fluid, near/far, and per-channel errors;
- physical port/internal/interface quantities;
- pair-context norm and routing statistics.

Interpretation:

- This is **not** a prediction of final trained Run-1404 accuracy.
- It determines how much of the pretrained Run-1401 function currently relies on \(c_H\).
- It also tests whether the already-computed module–environment context contains useful information that can be exposed to the fine pair path without changing weights.

Do not tune the architecture from many frozen variants.

---

# 9. Correctness and gradient tests

Keep tests focused on the scientific change.

## 9.1 Historical compatibility

1. Historical Run-1401 config remains bitwise/arithmetic unchanged.
2. Run-1401 checkpoint strict loading remains valid.
3. Existing Stage-7 golden replay remains unchanged.
4. Existing Runs 1402/1403 overlays remain unchanged.

## 9.2 Run-1404 route semantics

For the Run-1404 profile:

- `uses_hyper_context` remains true in the sense that hyperedges are needed for routing;
- `uses_hyper_value_context == 0`;
- `pairwise_kernel_enabled == 1`;
- `uses_global_context == 1`;
- `uses_near_module_context == 1`.

Explicitly verify:

\[
c_H(q)=0.
\]

Do not infer this merely from a configuration string.

## 9.3 Contextualized pair-token use

Construct a case where

\[
\widetilde m_i\ne m_i.
\]

Verify that:

- the base-token profile uses \(m_i\);
- Run 1404 uses \(\widetilde m_i\);
- changing `module_env_context` changes the Run-1404 pair response;
- the tensor width and pair-MLP parameter shapes do not change.

## 9.4 Gradient-path audit

For a real forward/backward batch, report gradients separately for:

- `organizer.module_score`;
- `organizer.env_score`;
- `organizer.me_query` / `me_key` / `me_context_proj`;
- decoder `query_to_hyper` / `hyper_key`;
- pairwise `pair_mlp`;
- `hyper_value`.

The expected structural result is:

\[
\nabla_{\text{hyper value}}\mathcal L = 0
\]

because its output is unused, while useful gradients can still reach the organizer and query routing through

\[
\beta_{qi}.
\]

Also verify a gradient path from the pair response back through

\[
\widetilde m_i
\rightarrow
A^{ME}
\rightarrow
\text{environment tokens}.
\]

The objective is to establish that removing \(c_H\) does not detach the organizer from learning.

## 9.5 Fused execution checks

For identical Run-1404 weights and full support, compare edge-explicit and fused query-module evaluation where both paths are supported.

Then verify that gathered execution constructs its selected pair index before the pair MLP is evaluated.

Report actual evaluated-pair counts; do not use a sparse mask created after dense pair computation as evidence of acceleration.

---

# 10. Formal Run-1404 training

## 10.1 Initialization

Train Run 1404 **from scratch**.

Do not initialize it from Run 1401. The frozen Run-1401 diagnostic and the trained Run-1404 result answer different questions.

## 10.2 Initial budget

Launch at most one fresh managed Run 1404 to:

\[
500\text{ epochs}.
\]

Use an available authorized GPU without interfering with the ongoing Runs 2000/2100/2200. If the intended device is occupied by one of those jobs, finish code/tests and provide the exact launch command rather than preempting another run.

Epoch 500 is an early assessment, not an automatic final verdict. The mature Run-1401 family often continues improving materially beyond this budget.

## 10.3 No simultaneous architecture sweep

Do not launch separate formal variants for:

- different \(K\);
- different pair widths;
- different global/near removals;
- sparsemax/entmax;
- different contextual-token scaling;
- different retained-mass thresholds.

Those would obscure the minimum-change question.

---

# 11. Epoch-500 scientific evaluation

Use exact epoch 500 and the separately saved best-by-field checkpoint through epoch 500.

Compare at minimum:

- Run 1401 @500;
- Run 1404 @500;
- Run 1804 @500 as the strong dense reference where existing evidence is already available.

Do not substitute mature Run-1401 or Run-1804 results into the matched-budget table. Mature references may be shown separately.

## 11.1 Reconstruction

On the complete existing 90-case development holdout report:

- pooled normalized fluid MSE and relative L2;
- equal-case mean, median, p95 and maximum;
- normalized and physical per-channel errors;
- near-interface and far-fluid errors;
- internal temperature;
- surface temperature;
- normal heat flux;
- final port outside temperature and heat-transfer coefficient;
- maintained engineering KPIs.

The question is not only whether the aggregate MSE is close to Run 1401. Removing a pooled value path must not be hidden by deterioration in temperature, vorticity, interface response, or difficult cases.

## 11.2 Does the hypergraph actually route the result?

Report

\[
\alpha_{qk},
\qquad
\beta_{qi},
\]

through:

- query-edge effective count;
- query-module effective count;
- beta maximum and entropy/effective-support statistics;
- beta variation across query space;
- module-incidence diversity;
- environment-incidence diversity.

Run 1404 need not produce a desired sparsity level. The purpose is to determine whether routing becomes more informative when it is the only direct hypergraph contribution to the field.

## 11.3 Controlled reliance interventions

On fixed diagnostic anchors, evaluate:

1. **Uniform query-to-edge routing** while keeping learned module incidence.
2. **Uniform module-to-edge incidence** while keeping learned query routing.
3. **Base module token instead of contextualized module token** in the pair kernel.
4. **Suppress the routed pair context** while retaining global/near paths.

Report both prediction differences and ground-truth error changes.

These are frozen-checkpoint interventions, not retrained model comparisons.

A large ablation effect is evidence of reliance, not evidence of superiority.

## 11.4 Gradient interpretation

At the trained checkpoint, repeat the component-gradient audit.

The desired qualitative structure is:

```text
field loss
   ↓
fine pair interaction
   ↓
beta(q,i)
   ↓
alpha(q,k) + A_mh(i,k)
   ↓
query routing + organizer
```

while

```text
hyper_value projection
```

remains unused by the field.

---

# 12. Sparse execution evaluation on the same checkpoint

Do not train separate sparse versions.

Use the trained Run-1404 checkpoint and evaluate gathered execution with

\[
\tau_\beta\in
\{1.0,0.999,0.995,0.99,0.98\}.
\]

For each threshold report:

- mean/p05/min retained beta mass;
- mean and p95 selected module count;
- actual pair-MLP evaluations;
- dense-pair count;
- full-field discrepancy relative to dense fused Run 1404;
- ground-truth error change;
- full-forward latency;
- prepared decoder latency;
- peak allocated CUDA memory.

Use the same real anchors and at least the established larger synthetic shapes for execution scaling.

The synthetic shapes measure execution only, not physical accuracy.

If all active modules remain selected, report that result directly. Do not claim learned physical sparsity from executor-level memory improvements.

---

# 13. Interpretation of possible outcomes

## Outcome A — accuracy remains competitive and beta is more concentrated

This is the strongest result.

It would support the claim that the Run-1401 hypergraph can be reinterpreted primarily as a routing index, while fine query–module computation carries the physical detail.

It would also justify making sparse routing/execution the next focus.

## Outcome B — accuracy is competitive but beta remains diffuse

The routing-only interpretation is still viable, but Run 1404 has not yet demonstrated major sparse-compute potential on this dataset.

The later Runs 2000/2100/2200 remain relevant because they explicitly redesign the routing topology for exact sparsity.

## Outcome C — Run 1404 is substantially worse

This would mean the current direct hyperedge value contains information that the existing fine pair path and background paths do not recover, even after exposing \(\widetilde m_i\).

Do **not** immediately add more auxiliary branches.

The next analysis should determine what information \(c_H\) carries that is missing from the fine route—for example, environmental regional context not representable through a per-module contextual token.

## Outcome D — pair routing becomes unused and global/near dominate

This would falsify the intended Run-1404 decomposition in its current form.

The appropriate response is not a branch-usage penalty. It is to examine whether the fine routed pair computation is sufficiently expressive and correctly conditioned.

---

# 14. Separation from Runs 2000 / 2100 / 2200

Run 1404 and the 2000-series answer different questions.

### Run 1404

Keeps

\[
K=6,
\]

Run-1401 organizer,

\[
A^{MH}, A^{EH}, h_k,
\]

and asks:

> Can the proven topology act primarily as a routing index if the physical value path is moved to fine query–module interactions?

### Runs 2000 / 2100 / 2200

Redesign the topology generator and sparse execution mechanism itself.

They ask:

> Can the routing topology become dynamically sparse and computationally cheaper while preserving fine physical interactions?

Run 1404 is therefore an important low-risk control. If it succeeds, it strengthens the premise of the new routing-first models. If it fails, it reveals that removing pooled hyperedge content requires more than a new sparse router.

---

# 15. Required coding outputs

Codex should finish with:

1. generic config support for `pairwise_module_token_source`;
2. one centralized pair-token resolver in `HypergraphGatedPairwiseKernel`;
3. no historical arithmetic changes;
4. `stage7_routing_only_pairwise_1404.json` experiment overlay;
5. focused compatibility, route, gradient, and gathered-execution tests;
6. one bounded frozen Run-1401 three-variant diagnostic;
7. one fresh Run 1404 through at most 500 epochs if a GPU is available without disrupting ongoing work;
8. a concise Run-1404 early report;
9. an unexecuted continuation command if later training is scientifically warranted.

Generated arrays, figures, and checkpoints should use the existing managed run/evaluation trees and remain outside tracked source according to current repository policy.

---

# 16. Definition of done

Run-1404 development is complete for this goal when:

- old Run-1401/1402/1403 configs and checkpoints remain compatible;
- the new contextualized pair-token source is opt-in;
- direct hyperedge value is demonstrably zero in Run 1404;
- routing and pairwise gradients remain live;
- frozen Run-1401 routing-only diagnostics are documented;
- the formal profile is ready and validated;
- at most one Run 1404 reaches epoch 500 if resources permit;
- exact/best-through-500 accuracy and routing evidence are reported if training completes;
- dense versus gathered execution is measured on the same checkpoint;
- no automatic continuation beyond 500 is launched.

The key scientific result must be stated in terms of **field fidelity, routing dependence, and actual pair execution**, not simply the fact that a hypergraph tensor still exists.
