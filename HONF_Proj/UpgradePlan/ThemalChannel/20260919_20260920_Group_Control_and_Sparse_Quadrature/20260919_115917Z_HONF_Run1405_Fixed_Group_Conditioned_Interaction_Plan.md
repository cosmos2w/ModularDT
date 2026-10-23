# Run 1405 — Fixed-K Group-Conditioned Interaction HONF

## Detailed mathematical design, implementation plan, 50-epoch execution check, and formal-training handoff

### Purpose

Run 1405 is a controlled next-generation HONF experiment built from the strongest lessons of Runs 1401/1404, 1804/1806, and 2001/2002.

The hypothesis is:

> A fixed six-edge hypergraph can own reusable group-conditioned interaction computation, route continuous physical reads at ports and arbitrary field queries, and reduce receiver-side work without replacing fine module/environment information by pooled hyperedge values.

The final field context has exactly three terms:

\[
\boxed{C(q)=C_g(q)+C_{\mathrm{pair},M}(q)+C_{\mathrm{pair},E}(q)}
\]

There is no direct hyperedge-value field term, no independent coarse latent bank, no separate near/local correction branch, no direct dense source bypass in the final reader, no dynamic-K generator, and no auxiliary sparsity/balance/entropy/pair-cost regularizer in the first experiment.

The fixed hypergraph is a control and interaction structure, not a compressed field-value representation.

---

# 1. Repository basis and compatibility

Development target:

- repository: `cosmos2w/ModularDT`
- branch: `agent/honf-core-next`
- proposed run: **1405**
- proposed architecture:

```text
fixed_group_pairwise_honf
```

Run 1405 must be implemented through `InterfaceFieldCore`, not by extending the legacy `HONFNeuralField` path.

Historical behavior must remain unchanged for legacy HONF, Run 1404, Dense 1804, latent, sparse-interface, regional, hierarchical-regional, and routed-pairwise families.

All new behavior is opt-in through the new architecture/profile.

---

# 2. Scientific scope

Relative to Dense 1804, Run 1405:

1. keeps the same generic encoder family;
2. keeps Dense's fine MM/ME/EM preparation in the first implementation;
3. organizes the prepared module/environment states into exactly K=6 differentiable groups;
4. replaces Dense receiver-side QM and QE reads with group-conditioned reads;
5. replaces both historical global+near and newer coarse+local structures with one minimal query-conditioned global background term;
6. uses the same continuous interaction reader at P0 ports, P1 refinement coordinates, P2 field coordinates, and optional P2 port/global consistency probes.

Do not simultaneously change the local surrogate, P0/P1/P2 physics, loss, optimizer, data, output channels, Dense preparation, hidden width, or dynamic edge count.

---

# 3. Notation

Let B be batch size, M padded module count, M_a active modules, E environment-token count, Q receiver count, K=6 hyperedges, H hidden width, x_i module coordinate, y_j environment coordinate, q receiver coordinate, nu_j quadrature mass, and m_i module presence.

Encoded states:

\[
z_i,e_j,g\in\mathbb R^H.
\]

Dense-contextualized states:

\[
z_i^\star,e_j^\star.
\]

Hypergraph variables:

\[
A^M_{ik},\quad A^E_{jk},\quad h_k,\quad \alpha_{qk}.
\]

Final terms:

\[
C_g,\quad C_{\mathrm{pair},M},\quad C_{\mathrm{pair},E}.
\]

---

# 4. Input encoding

Use `InterfaceFieldCore` semantics.

\[
g=E_G(c)
\]

\[
z_i=E_M(f_i)+E_X(\Phi_X(x_i/s))
\]

\[
e_j=E_E[\Phi_X(y_j/s),a_j].
\]

Do not broadcast the global token into every environment token. Global information enters interaction/update networks explicitly.

ThermalChannel feature meaning remains case-owned.

---

# 5. Preserve Dense 1804 fine preparation

Reuse `DensePairwiseField.prepare_fine_messages()`.

Module-module:

\[
p^{MM}_{i\ell}=\phi_{MM}[z_i,z_\ell,\Phi_R((x_i-x_\ell)/s)]
\]

\[
a_i^{MM}=\frac{\sum_{\ell\neq i}m_\ell p^{MM}_{i\ell}}{1+M_a}.
\]

Module-environment:

\[
p^{ME}_{ij}=\phi_{ME}[z_i,e_j,\Phi_R((x_i-y_j)/s)]
\]

\[
a_i^{ME}=\frac{\sum_j\nu_jp^{ME}_{ij}}{\sum_j\nu_j}.
\]

Environment-module:

\[
p^{EM}_{ji}=\phi_{EM}[e_j,z_i,\Phi_R((y_j-x_i)/s)]
\]

\[
a_j^{EM}=\frac{\sum_i m_i p^{EM}_{ji}}{1+M_a}.
\]

Prepared source states:

\[
\boxed{z_i^\star=z_i+\rho_M[z_i,a_i^{MM},a_i^{ME},g]}
\]

\[
\boxed{e_j^\star=e_j+\rho_E[e_j,a_j^{EM},g]}.
\]

Fine source identities remain individually addressable.

---

# 6. Fixed-K group construction

Implement a new generic `FixedGroupRouter`.

Maintain six learned group-code vectors c_k and six learned normalized fallback anchors r_k^0.

## 6.1 Module incidence

\[
\ell^M_{ik}=s_M(z_i^\star,g,c_k)
\]

\[
\boxed{A^M_{i:}=\operatorname{Entmax}_{1.5}(\ell^M_{i:}/T_M)}
\]

with fixed T_M=1 initially.

Inactive module rows are zero.

## 6.2 Module group centre

\[
m_k^M=\sum_iA^M_{ik}
\]

and, for positive mass,

\[
r_k^M=\frac{\sum_iA^M_{ik}x_i}{m_k^M}.
\]

If m_k^M=0, use the learned fallback anchor mapped to the physical domain.

## 6.3 Environment incidence

\[
\ell^E_{jk}=s_E(e_j^\star,g,c_k)+b_E((y_j-r_k)/s)
\]

\[
\boxed{A^E_{j:}=\operatorname{Entmax}_{1.5}(\ell^E_{j:}/T_E)}
\]

with T_E=1.

Quadrature stays separate as nu_j A^E_jk.

## 6.4 Environment region centre

\[
m_k^E=\sum_j\nu_jA^E_{jk}
\]

\[
r_k^E=\frac{\sum_j\nu_jA^E_{jk}y_j}{m_k^E}
\]

for positive mass.

---

# 7. Group control state

Normalized summaries:

\[
\bar z_k=\frac{\sum_iA^M_{ik}z_i^\star}{m_k^M+\epsilon}
\]

\[
\bar e_k=\frac{\sum_j\nu_jA^E_{jk}e_j^\star}{m_k^E+\epsilon}.
\]

Construct

\[
\boxed{
h_k=\rho_H[\bar z_k,\bar e_k,g,\Phi_R(r_k/s),\Phi_R(r_k^E/s),m_k^M,m_k^E,c_k].
}
\]

`h_k` is forbidden from being directly added to final field context.

It may be used only for routing and fine interaction conditioning.

---

# 8. Shared group-conditioned source computation

The hyperedge owns reusable prepared computation.

For every positive module-group incidence:

\[
\boxed{
u^M_{ik}=F_M^{src}[z_i^\star,h_k,g,\Phi_R((x_i-r_k)/s)].
}
\]

For every positive environment-group incidence:

\[
u^E_{jk}=F_E^{src}[e_j^\star,h_k,g,\Phi_R((y_j-r_k^E)/s)]
\]

followed by

\[
K^E_{jk}=W_Ku^E_{jk},\qquad V^E_{jk}=W_Vu^E_{jk}.
\]

These are prepared once per P0/P1/P2 state and reused across receiver chunks.

---

# 9. Query-to-group routing

Receiver features:

\[
r_q=\Phi_Q(q/s).
\]

Query/group descriptor:

\[
u^Q_{qk}=F_Q[r_q,h_k,g,\Phi_R((q-r_k)/s),\Phi_R((q-r_k^E)/s)].
\]

Logits:

\[
\ell^Q_{qk}=w_Q^\top u^Q_{qk}.
\]

Mask only physically empty groups.

\[
\boxed{
\alpha_{q:}=\operatorname{Entmax}_{1.5}(\ell^Q_{q:}/T_Q)
}
\]

with T_Q=1.

The same alpha routes both module and environment group responses.

---

# 10. Group-conditioned module term

For every semantic triple with alpha_qk>0 and A^M_ik>0:

\[
\boxed{
\psi_M(q,i,k)=F_M^{pair}[u^Q_{qk},u^M_{ik},\Phi_R((q-x_i)/s)].
}
\]

Column-normalize module membership:

\[
\bar A^M_{ik}=\frac{A^M_{ik}}{m_k^M+\epsilon}.
\]

Group response:

\[
\boxed{
R^M_{qk}=\sum_{i:A^M_{ik}>0}\bar A^M_{ik}\psi_M(q,i,k).
}
\]

Final module term:

\[
\boxed{
C_{\mathrm{pair},M}(q)=\sum_k\alpha_{qk}R^M_{qk}.
}
\]

The same q-i pair may produce different responses through different groups.

---

# 11. Group-conditioned environment term

Use grouped attention rather than a wide MLP on every q-j-k triple.

For each active q-k:

\[
Q^E_{qk}=W_Q^E u^Q_{qk}.
\]

For positive environment-group incidences:

\[
s_{qjk}^{(h)}
=
\frac{Q^{E,(h)}_{qk}\cdot K^{E,(h)}_{jk}}{\sqrt{d_h}}
+b_h(\Phi_R((q-y_j)/s))
+\log\nu_j
+\log A^E_{jk}.
\]

Within group k:

\[
a_{qjk}^{(h)}
=
\operatorname{softmax}_{j:A^E_{jk}>0}s_{qjk}^{(h)}.
\]

Then

\[
\boxed{
R^E_{qk}
=
W_O\operatorname{concat}_h
\sum_{j:A^E_{jk}>0}a_{qjk}^{(h)}V^{E,(h)}_{jk}.
}
\]

Final environment term:

\[
\boxed{
C_{\mathrm{pair},E}(q)=\sum_k\alpha_{qk}R^E_{qk}.
}
\]

Fine environment sources remain individually represented.

---

# 12. Minimal global background

Use only

\[
\boxed{
C_g(q)=F_g[\Phi_Q(q/s),g].
}
\]

A two-layer MLP is sufficient.

It must not receive module tokens, environment tokens, group states, or local-neighbour features.

---

# 13. Final fusion

\[
\boxed{
C(q)=C_g(q)+C_{\mathrm{pair},M}(q)+C_{\mathrm{pair},E}(q)
}
\]

\[
\boxed{
\widehat U(q)=D_\theta[\operatorname{LN}(C(q))].
}
\]

No additional raw global/query/source bypass should enter the field head.

---

# 14. Shared physical reader

Use the existing `forward_interface_field()` path.

The same continuous operator must serve:

- P0 port coordinates;
- P1 outside-temperature coordinates;
- P2 global field coordinates;
- optional P2 port/global consistency coordinates.

Prepared group states/source codes are refreshed whenever module state changes.

---

# 15. Execution strategy

Do not reuse the Runs-2001/2002 two-hop compiler.

Because K=6 is fixed, use group-major execution.

During prepare cache:

- A_m [B,M,K]
- A_e [B,E,K]
- group masses/centres/states
- positive module members per group
- positive environment members per group
- module/group source codes
- environment/group K/V

For each receiver chunk compute alpha [B,Q,K].

For each k:

- gather active query rows;
- gather module members;
- gather environment members;
- evaluate group-conditioned module interactions;
- evaluate group-conditioned environmental attention.

Prefer a small number of rectangular GPU blocks/BMMs over large irregular scatter graphs.

Do not add Triton initially.

---

# 16. Compute ledger

Dense read cost:

\[
P_M^{dense}=QM_a,\qquad P_E^{dense}=QE.
\]

Run-1405 semantic grouped rows:

\[
P_M^{1405}=\sum_{q,k:\alpha_{qk}>0}|\mathcal M_k|
\]

\[
P_E^{1405}=\sum_{q,k:\alpha_{qk}>0}|\mathcal E_k|.
\]

Report

\[
\boxed{R_M=P_M^{1405}/(QM_a)}
\]

\[
\boxed{R_E=P_E^{1405}/(QE)}.
\]

Also report mean active group counts s_Q, s_M, and s_E.

These are descriptive. Measured GPU latency and memory are authoritative.

---

# 17. General code changes

## 17.1 Configuration

Add `fixed_group_pairwise_honf` to `FORWARD_ARCHITECTURES`.

Add a compact fixed-group config with:

```text
group_count = 6
source_normalizer = "entmax15"
query_normalizer = "entmax15"
module_temperature = 1.0
environment_temperature = 1.0
query_temperature = 1.0
group_code_dim = 32
```

Do not add balance/sparsity/support-target schedules or dynamic K.

## 17.2 New generic files

Create:

```text
src/honf_forward_core/interface_fields/fixed_group_router.py
src/honf_forward_core/interface_fields/fixed_group_pairwise.py
src/honf_forward_core/interface_fields/three_term_context.py
```

`fixed_group_router.py` owns group identities, memberships, geometry, group state, query routing, and support summaries.

`fixed_group_pairwise.py` should preferably subclass `DensePairwiseField`, reuse Dense preparation, and replace only the receiver-side reader.

`three_term_context.py` should provide the small API surface needed by `InterfaceFieldCore`:

- no-op/tiny prepare state;
- `read_coarse` returns C_g;
- `read_local` returns exact zeros without constructing distances;
- `predict_field` applies LayerNorm(total context) and the field head only.

## 17.3 InterfaceFieldCore

For 1405 instantiate `FixedGroupPairwiseField` plus `ThreeTermInterfaceContext`.

Do not globally refactor historical architecture behavior.

---

# 18. ThermalChannel-specific changes

The existing non-legacy `forward_interface_field()` path should be reused.

No new ThermalChannel-specific model branch should be added.

Use the ordinary 24x8 environment tokens and existing features.

Do not request regional IDs, environment hierarchy, or routed resistance metadata.

Use the existing physical loss unchanged.

---

# 19. Focused tests before training

Run ordinary tests plus focused scientific tests:

1. historical representative checkpoints/configs still load and execute;
2. entmax assignment rows are nonnegative and normalized;
3. exact zeros are preserved;
4. empty groups are safe;
5. no direct h_k value contribution exists;
6. fixing q and source but changing h_k changes psi_M(q,i,k);
7. group-conditioned environment K/V or query features change with h_k;
8. P0/P1/P2 use the same backend reader;
9. module permutation invariance;
10. environment-token permutation invariance;
11. one real predicted-port forward/backward/update batch has finite gradients through all major 1405 components.

Pre-flight checks never substitute for actual execution.

---

# 20. Visualization upgrade

Add one concise fixed-group visualization board.

Panel A: physical geometry, group centres, module/env dominant membership.

Panel B: A_m and A_e incidence maps with group masses.

Panel C: selected query and active alpha_qk group routes.

Panel D: actual semantic q->group->module/environment triples, colored by group.

Annotate:

```text
sQ, sM, sE
R_M, R_E
prepared-decode latency
```

Label all lines as learned interaction routes, not physical causality.

---

# 21. One essential later ablation

After user-authorized endpoint evaluation, preserve learned topology A_m, A_e, alpha but remove group-state conditioning from the fine kernels.

This asks whether the benefit comes from grouped routing only or from genuinely group-conditioned computation.

Do not train a second formal model for this ablation.

---

# 22. Managed execution sequence

Codex may use one currently free physical GPU.

Inspect current GPU/process occupancy and choose a free GPU without interrupting existing jobs. With `CUDA_VISIBLE_DEVICES=<physical>`, use logical `cuda:0`.

## Phase A — implementation validation

Before managed training:

- focused tests;
- real GPU predicted-port forward/backward/update;
- finite gradient audit;
- one smoke execution.

Do not call the smoke measurement an acceleration result.

---

# 23. Run 1405 through epoch 50

Create one managed Run 1405 from scratch:

```text
run-id: 1405
run-name: fixed_group_conditioned_interaction
```

Train epochs 1-50 first.

Do not create a separate quickcheck run. If continuation is justified, resume this same run and optimizer.

Use existing metrics/log/checkpoint mechanisms only.

---

# 24. Epoch-50 controlled execution comparison with 1804

Use the epoch-50 1405 checkpoint and an existing 1804 checkpoint on the same free GPU.

Inference protocol:

- cases 0273 and 0653
- Q=8192
- receiver chunk 2048
- 2 warmups
- 5 synchronized repetitions
- routing maps disabled

Measure:

- full physical forward
- prepared P2 decode
- peak allocated memory
- peak reserved memory

Training-step protocol:

- fixed real M1 and M12 buckets
- B=48
- Q=1024
- canonical predicted-port loss
- fresh optimizers for both models using matched hyperparameters
- 1 warmup
- 3 measured updates

Report median step time, allocated/reserved peak, and actual P_M/P_E/R_M/R_E.

---

# 25. Epoch-50 continuation criteria

These are empirical continuation criteria for GPU-budget use, not model/runtime software gates.

Training health:

- all relevant losses, gradients, parameters, and outputs finite;
- no OOM/CUDA failure;
- all major 1405 parameter groups have finite scheduled gradients;
- last-10 validation median lower than first-10 median;
- no repeated catastrophic instability in the last 10 epochs.

Fidelity sanity:

- validation trajectory still improving;
- epoch-50 validation field MSE not more than 2x matched Run-1804 epoch-50 value.

Acceleration:

1. mean full-forward latency across 0273/0653 at least 5% lower than 1804;
2. M12 optimizer-step median at least 5% lower than 1804;
3. peak allocated memory no more than 10% worse on either controlled workload;
4. actual grouped interaction counts satisfy R_M<1 and R_E<1 on measured multi-module workloads.

If these are not met, stop at epoch 50 and report.

Do not weaken criteria after observing results.

M1 is measured but is not an acceleration continuation criterion.

---

# 26. Formal continuation to epoch 500

If all continuation criteria pass:

- resume the same Run 1405 from epoch 50;
- restore optimizer/RNG state with the existing resume workflow;
- continue to total epoch 500;
- change no architecture/loss/optimizer setting;
- launch no sweep or second seed.

After epoch 500:

- verify completion/checkpoint existence;
- summarize ordinary training health only;
- do not run the full 90-case scientific endpoint evaluation;
- do not extend toward 2,500 or 5,000 epochs.

Wait for the user.

---

# 27. Later evaluation after user instruction

When explicitly requested, compare matched policies for Runs 1401, 1404, 1804, and 1405.

Report field accuracy, tails, physical channels/interfaces, KPIs, strata, R_M/R_E/s_Q/s_M/s_E, runtime/memory, and bounded routing/group-conditioning/branch interventions.

Keep checkpoint selection policy explicit.

---

# 28. Research-code constraints

Use ordinary Git, current config validation, standard tests, and existing managed run/checkpoint handling.

Do not add cryptographic hashes, contract freezes, baseline snapshots, approval services, monitoring daemons, database/process infrastructure, or other defensive over-engineering.

The identified risks are sufficiently handled by Git history, tests, actual GPU execution, and empirical measurement.

Preserve all existing security and trusted checkpoint-loading behavior.

---

# 29. Interpretation of outcomes

If 1405 is faster and later accurate: this is strong evidence for shared group-conditioned computation plus fine-entity fidelity plus real acceleration.

If grouped work is smaller but GPU time is not: profile the measured group-major kernel before changing the math.

If supports remain broad: stop after the 50-epoch continuation decision and inspect the learned factorization rather than adding immediate regularizers.

If fast but inaccurate: use branch/topology/group-conditioning interventions to locate the missing information.

If removing group conditioning does not hurt: the benefit is grouped routing rather than genuinely higher-order grouped computation.

---

# 30. Definition of done for this Goal-mode task

Codex should finish with:

1. opt-in `fixed_group_pairwise_honf`;
2. fixed K=6 group router/control states;
3. reused Dense MM/ME/EM preparation;
4. triadic module interaction;
5. group-conditioned environment attention;
6. minimal C_g;
7. exactly three final context terms;
8. shared P0/P1/P2 continuous reader;
9. focused compatibility/numerical/gradient tests;
10. upgraded hypergraph visualization;
11. one managed Run 1405 through epoch 50;
12. controlled speed/memory/stability comparison with 1804;
13. continuation of the same run to epoch 500 only if the empirical criteria pass;
14. no full endpoint evaluation or 5,000-epoch extension without new user instruction.

The final development note must distinguish mathematical support sparsity, semantic grouped interaction count, measured GPU latency, memory, fidelity, and evidence that group conditioning itself is useful.
