# HONF Case-Adaptive Forward Revision Plan — Phase 2 Restart

## Predictive-rank active-set selection over a stable mechanism bank

## 0. Decision

Run 1701 should remain stopped at epoch 500.

The Phase-2 tensor-residual implementation passed its software and numerical-correctness contracts, but its scientific formulation failed. The learned interaction tensor collapsed to approximately rank one, every tested case selected \(K_b=1\), the field trajectory stopped improving after the first few epochs, and complete preparation remained too slow.

The next Phase-2 attempt should **not** revise the tensor residual by changing its threshold, width, minimum K, or auxiliary losses.

The central redesign is:

\[
\boxed{
\text{Do not let a freely learned latent tensor decide its own rank.}
}
\]

Instead, retain a stable, sufficiently expressive candidate mechanism bank and choose the case-specific active subset according to **measured predictive fidelity** on a small deterministic probe set.

This produces a dynamic case-specific active count

\[
K_b = |S_b|
\]

without a learned count head, a residual-extraction loop, a topology regularizer, or a different training function.

---

# 1. What Run 1701 established

## 1.1 Correctness that should be preserved

Run 1701 demonstrated that the repository can support:

- a case-dependent packed mechanism axis;
- hard case-specific masks;
- exact train/evaluation forward agreement;
- strict checkpoint loading and optimizer resume;
- permutation and padding invariance;
- optional tensor diagnostics;
- isolated organizer modes that coexist with historical modes.

These software contracts are useful and should remain.

## 1.2 Scientific failure

At epoch 500:

\[
\mathrm{MSE}^{\mathrm{val}}_{\mathrm{field}}
=
1.9437,
\]

\[
\mathrm{MSE}^{\mathrm{val}}_T
=
0.8520,
\]

which are respectively \(84.56\times\) and \(56.46\times\) the matched Run-1401 values.

On 20 held-out cases:

\[
K_b=1
\qquad
\text{for every case}.
\]

The module, environment, and content unfoldings of the learned interaction tensor all have effective rank approximately one, while source and region separation are zero.

The complete forward is \(2.10\times\) Run 1401 even though prepared decoding and memory remain controlled. The organizer, not the established decoder, remains the bottleneck.

---

# 2. Root-cause interpretation

## 2.1 Self-referential rank selection

Run 1701 learns an interaction tensor

\[
C_\theta(x)
\]

and then defines active capacity from the residual rank of that same learned object:

\[
K_\theta(x)
=
\min
\left\{
r:
\rho_r(C_\theta(x))
\le
\varepsilon
\right\}.
\]

There is no independent target that requires \(C_\theta\) to preserve the true complexity of the physical response. The field optimizer can therefore simplify \(C_\theta\) itself.

A rank-one or nearly zero tensor is an easy solution for the stopping mechanism even when it is a bad representation for the field.

Increasing tensor width from one to 32 does not remove this shortcut because all 32 channels can align.

## 2.2 Squared nonnegative interaction destroys sign and favors a common mode

The implemented tensor is

\[
C_{ijc}
=
\left(
L_{ijc}
-
\overline L_{ic}
\right)^2.
\]

This has three undesirable properties.

First, positive and negative interactions become indistinguishable.

Second, if

\[
L_{ijc}\approx a_{ic}b_{jc},
\]

then

\[
C_{ijc}\approx a_{ic}^2b_{jc}^2,
\]

which is rank one in the module-environment plane for each channel.

Third,

\[
\frac{\partial C}{\partial L}=2L,
\]

so a channel driven close to zero receives a vanishing interaction gradient.

## 2.3 Marginal nonnegative factorization extracts the broad Perron component

The factor loop starts from residual marginals and alternates positive normalized reductions. For a broadly positive tensor, this naturally identifies the largest common nonnegative component first.

Once training makes the tensor nearly separable, the first component explains more than 99% of its normalized mass. The stopping rule is then functioning correctly on an uninformative object.

## 2.4 Hard-forward gating creates a self-reinforcing dead-edge state

Run 1701 began the real-data smoke with \(K\) at the case cap, but trained checkpoints selected only one mechanism. Once the first residual falls below the threshold:

- only H0 participates in the numerical forward pass;
- later mechanisms receive no direct forward utility;
- later mechanisms receive only a surrogate gradient;
- the currently active H0 receives the useful forward gradient;
- the tensor is encouraged to encode still more content in H0.

This is a winner-take-all feedback loop.

## 2.5 The straight-through gradient is not the gradient of the hard model

The forward function uses hard support while the backward path uses soft support:

\[
z_{\mathrm{ST}}
=
s+\operatorname{stopgrad}(z_{\mathrm{hard}}-s).
\]

With

\[
\tau=0.002,
\]

the maximum sigmoid derivative is

\[
\max\frac{s(1-s)}{\tau}
=
\frac{1}{4\tau}
=
125.
\]

The training loop clips the global model gradient norm to 1.0. A large gate-surrogate gradient can therefore scale down useful decoder and encoder gradients for the whole model.

This is a high-probability contributor but must be verified by a read-only gradient audit because Run 1701 did not log pre-clip gradient norms.

## 2.6 Training computes ghost candidates after stopping

During training, all packed candidate steps are evaluated. After the residual is almost zero, the factor normalizers fall back to uniform factors. Those candidates remain absent from the hard forward but enter the soft surrogate path.

The optimizer is therefore partly following gradients through uniform ghost mechanisms that do not exist in the actual prediction.

## 2.7 K=1 removes the defining HONF routing structure

For one active mechanism:

\[
\alpha_{q1}=1,
\]

\[
A^{MH}_{i1}=1,
\]

and the edge-normalized incidence becomes

\[
\bar A^{MH}_{i1}
=
\frac{1}{M_b}.
\]

Therefore,

\[
\beta_{qi}
=
\frac{1}{M_b}.
\]

The query-conditioned module-routing structure disappears. The model reduces to a constant hyperedge summary plus an unweighted mean of query-module pair responses, together with the global and near-module bypasses.

This is not a small loss of capacity; it removes the core structured routing behavior.

---

# 3. Revised principle

A dynamic active count should answer:

> How many mechanisms are required to preserve the prediction of a trusted full-capacity HONF for this case?

It should not answer:

> How many factors are required to reconstruct a freely learned latent tensor that the same optimizer can simplify?

The revised model separates:

\[
K_{\max}
=
\text{candidate capacity},
\]

from

\[
K_b
=
\text{case-specific active predictive rank}.
\]

A finite \(K_{\max}\) remains necessary as a neural capacity envelope. It is no longer interpreted as the physical mechanism count used by every case.

For the ThermalChannel proof, use:

\[
K_{\max}=6,
\]

because the Run-1401 bank is accurate and structurally differentiated.

---

# 4. Model formulation

## 4.1 Stable candidate bank

Use the established fixed-projection organizer to generate

\[
\mathcal H_b^{\max}
=
\left\{
h_{bk},A^{MH}_{b:k},A^{EH}_{b:k}
\right\}_{k=1}^{K_{\max}}.
\]

The full model prediction is

\[
\widehat U_b^{\mathrm{full}}(q)
=
F_\theta(q;\mathbf 1),
\]

where \(\mathbf 1\) means all candidate mechanisms are active.

No candidate is removed during training.

## 4.2 Deterministic probe queries

For each prepared case, construct a small probe set

\[
\mathcal P_b
=
\mathcal P_b^{\mathrm{global}}
\cup
\mathcal P_b^{\mathrm{local}}.
\]

The default global probes are environment-token coordinates.

Local probes are a small deterministic set around active modules, generated from the module radius and coordinate scale. They protect near-interface behavior without changing the field model.

The first implementation should use at most 256 probe coordinates.

## 4.3 Masked prediction

For an edge mask

\[
z\in\{0,1\}^{K_{\max}},
\qquad
\|z\|_0\ge1,
\]

define

\[
\widehat U_b(q;z)
=
F_\theta(q;z),
\]

where the mask is applied to query-to-edge logits before normalization. The existing hyperedge states and incidence columns remain unchanged; only active routing support changes.

The fused module routing becomes

\[
\beta_{bi}(q;z)
=
\sum_{k:z_k=1}
\alpha_{bk}(q;z)
\bar A^{MH}_{bik}.
\]

## 4.4 Predictive-fidelity discrepancy

Compute the full-bank probe prediction once:

\[
U_b^{\mathrm{ref}}
=
\widehat U_b(\mathcal P_b;\mathbf 1).
\]

For a candidate mask \(z\), define weighted relative RMS discrepancy:

\[
D_b(z)
=
\sqrt{
\frac{
\sum_{p\in\mathcal P_b}
\sum_{f=1}^{F}
w_f
\left[
\widehat U_{bf}(p;z)
-
U^{\mathrm{ref}}_{bf}(p)
\right]^2
}{
\sum_{p\in\mathcal P_b}
\sum_{f=1}^{F}
w_f
\left[
U^{\mathrm{ref}}_{bf}(p)
\right]^2
+\epsilon
}
}.
\]

Also define a per-channel discrepancy

\[
D_{bf}(z)
=
\frac{
\operatorname{RMS}_{p}
\left[
\widehat U_{bf}(p;z)-U^{\mathrm{ref}}_{bf}(p)
\right]
}{
\operatorname{RMS}_{p}
\left[
U^{\mathrm{ref}}_{bf}(p)
\right]
+\epsilon
}.
\]

## 4.5 Case-specific active set

Choose

\[
\boxed{
z_b^\star
=
\arg\min_z
\|z\|_0
}
\]

subject to

\[
D_b(z)\le\delta_{\mathrm{all}},
\]

and

\[
\max_f D_{bf}(z)\le\delta_{\mathrm{channel}}.
\]

Then

\[
\boxed{
K_b=\|z_b^\star\|_0.
}
\]

For \(K_{\max}=6\), all 63 nonempty masks can be evaluated in a vectorized probe batch. This gives the exact smallest subset rather than relying on a learned count or a greedy residual.

For larger future banks, use backward elimination or contribution-ranked prefix evaluation.

## 4.6 Cached deployment

The selected mask is calculated once when a case is prepared:

```text
encode case
    ↓
organize full candidate bank
    ↓
evaluate small probe set
    ↓
select z_b*
    ↓
cache active mask
    ↓
decode arbitrary large query chunks
```

The active count is case-specific, but no residual-extraction neural loop runs for every query.

For inverse-design gradients, use the full bank by default. The discrete compressed mask is an evaluation/deployment mode.

---

# 5. Why this directly addresses the failed modes

| Run-1701 failure | Predictive-rank response |
|---|---|
| Learned tensor makes itself rank one | Rank is judged from actual field fidelity |
| Squaring removes signed response content | No auxiliary tensor is constructed |
| Hard K changes training function | Training uses the unchanged full bank |
| Soft surrogate gives biased gradients | No straight-through count gradient |
| One edge monopolizes training | All candidate edges remain trainable |
| Sequential organizer is slow | Selection is a small vectorized probe calculation |
| Environment organization collapses | Reuse the differentiated Run-1401 organization |
| Threshold can be gamed internally | Threshold measures output discrepancy |
| Unknown true K | No K labels are required |

---

# 6. Step-by-step Phase-2 restart plan

## Step 0 — one forensic Run-1701 audit, no training

Use stored checkpoints at epochs 7/8, 100, 250, and 500.

Measure:

1. hard K and tensor ranks over the same 20 cases;
2. raw interaction-tensor total mass and channel concentration;
3. first-component explained fraction;
4. soft-gate derivative statistics;
5. pre-clip total gradient norm and organizer/decoder gradient norms on one fixed batch;
6. accuracy with hard support, forced full support, and H0-only support.

This audit answers whether collapse preceded the accuracy plateau and whether the straight-through gradients dominated clipping.

Do not modify or resume Run 1701.

## Step 1 — add a maskable predictive selector

Add a separate selection mode, not another organizer implementation:

```text
case_edge_selection_mode = "probe_fidelity"
```

Recommended ownership:

```text
honf_forward_core/selection/predictive_rank.py
```

The fixed organizer and decoder remain unchanged when the mode is disabled.

Required API:

```text
prepare_case(..., case_edge_selection_mode="none")
prepare_case(..., case_edge_selection_mode="probe_fidelity")
```

The prepared state stores:

- full candidate edge count;
- selected mask;
- selected K;
- probe discrepancy;
- probe-set metadata;
- search method and tolerances.

## Step 2 — exact compatibility tests

Prove:

1. selection disabled gives exact historical Run-1401 output;
2. all-edge mask gives exact historical output;
3. a mask is applied only to routing support;
4. prepared/chunked decoding is invariant;
5. module order and padding remain invariant;
6. mask search is deterministic;
7. selected K is independent of output query chunking;
8. old checkpoints and organizer modes remain unchanged.

## Step 3 — frozen Run-1401 feasibility evaluation

No training run is permitted at this step.

On 20 matched held-out cases, evaluate:

\[
\delta_{\mathrm{all}}
\in
\{0.0025,0.005,0.01\},
\]

with

\[
\delta_{\mathrm{channel}}=2\delta_{\mathrm{all}}.
\]

For each tolerance, report:

- K histogram and K by module count;
- probe discrepancy;
- full-grid discrepancy versus full Run 1401;
- ground-truth field/channel/near/far accuracy;
- selection overhead;
- prepared-decoder time after selection;
- total time for 8,192, 65,536, and 262,144 queries.

Choose at most one tolerance for the complete split.

## Step 4 — complete 90-case decision

Run one complete 90-case evaluation for the chosen tolerance.

Promotion gates:

- full-grid normalized RMS difference from full Run 1401: mean \(\le0.005\), p95 \(\le0.01\);
- pooled fluid MSE degradation versus full Run 1401: \(\le1\%\);
- each channel degradation: \(\le2\%\);
- near-interface and far-field pooled degradation: \(\le2\%\);
- p95 case-MSE degradation: \(\le3\%\);
- exact full-support parity;
- no invalid or empty mask;
- selection deterministic;
- large-query total latency is lower after amortizing probe selection.

A variable K histogram is evidence only, not a gate. If every case selects the same K under a predictive criterion, report that result rather than forcing variation.

## Step 5 — model decision

Possible outcomes:

1. **Predictive selector succeeds and K varies.**  
   Promote it as the case-adaptive evaluation/deployment path over Run 1401.

2. **Predictive selector succeeds but K is nearly constant.**  
   Conclude that this dataset supports compression but not strong case-to-case rank variation.

3. **No subset preserves fidelity.**  
   Retain all six edges and conclude that dynamic K is not currently justified.

4. **Probe fidelity does not predict full-grid fidelity.**  
   Improve the deterministic probe coverage once; do not add a learned count model.

No new formal training run is needed to answer these questions.

---

# 7. Configuration

Add selection settings outside organizer-mode semantics:

```json
{
  "case_edge_selection_mode": "none",
  "case_edge_probe_source": "environment_plus_module_local",
  "case_edge_probe_limit": 256,
  "case_edge_probe_relative_rms_tolerance": 0.005,
  "case_edge_probe_channel_tolerance": 0.01,
  "case_edge_probe_search": "exhaustive_small_bank"
}
```

Historical profiles default to:

```text
case_edge_selection_mode = "none"
```

The selected mode is an evaluation/preparation override until the frozen audit is accepted.

Do not change the recommended profile or Run-1401 checkpoint configuration.

---

# 8. Efficiency expectations

The selector adds a one-time cost proportional to a small probe set and candidate masks.

For \(K_{\max}=6\) and \(Q_p\le256\), exact subset evaluation remains small compared with future large fields. It may not help an 8,192-query job because selection overhead can dominate.

The efficiency claim must therefore be conditioned on query volume:

\[
Q\gg Q_p.
\]

Dynamic K mainly improves:

- interpretation;
- case-specific active representation;
- query-edge processing;
- possibly downstream beta sparsity.

The dominant query-module pair kernel remains governed by \(M_{\mathrm{eff}}(q)\), so beta-gathered execution remains the primary large-\(Q\) acceleration mechanism.

---

# 9. What not to launch

Do not launch:

- Run-1701 continuation;
- a new tensor width;
- a lower residual threshold;
- a minimum-K constrained tensor model;
- another seed of Run 1701;
- a signed-tensor residual model;
- an entropy/diversity-regularized model;
- a learned K classifier;
- another 500-epoch model before frozen predictive selection is evaluated.

The next decisive experiment is evaluation-only.

---

# 10. Definition of done

The Phase-2 restart is complete when:

- the Run-1701 collapse timeline and gradient-scale audit are documented;
- a predictive-fidelity active-set selector coexists with all established modes;
- full-support Run-1401 parity is exact;
- one 20-case tolerance audit is complete;
- at most one chosen tolerance is evaluated on all 90 held-out cases;
- K distribution, full-grid fidelity, physical accuracy, and amortized runtime are reported;
- the result states whether case-specific active K is supported by the current dataset;
- no formal training run has been launched without evidence that training is necessary.
