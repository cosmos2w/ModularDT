# HONF Case-Adaptive Forward Revision Plan

## 0. Purpose and working state

This document turns the conceptual design in `HONF_Case_Adaptive_Forward_Model.md` into an implementation and validation plan for Codex.

Repository and branch:

- repository: `cosmos2w/ModularDT`;
- working branch: `agent/honf-core-next`;
- required starting commit: `f59ec191a7956752ec6c62b6f0fcf5dc0684c036`;
- project root: `HONF_Proj/`.

The branch is intentionally a portable HONF code tree. Generated diagnostics, checkpoints, and formal run outputs remain local and ignored by Git.

The scientific objective is narrow:

> Add exactly one new organizer mode that replaces a globally fixed hyperedge count with a case-specific residual stopping rule, while retaining the established encoders, fused query-module decoder, pairwise kernel, ThermalChannel coupling, training objective, and all historical organizer modes.

The new mode is named:

```text
organizer_mode = "case_adaptive_residual"
```

Its central behavior is

\[
\text{extract one mechanism}
\;\longrightarrow\;
\text{subtract the interaction it explains}
\;\longrightarrow\;
\text{stop independently for each case when the residual is small}.
\]

The case-specific mechanism count is

\[
K_b
=
\min\left\{
r:
\frac{\|R_b^{(r)}\|_F^2}
{\|R_b^{(0)}\|_F^2+\epsilon}
\leq
\varepsilon_{\mathrm{edge}}
\right\}.
\]

Here, \(b\) identifies a physical case. There is no globally prescribed scientific \(K\) for this organizer. A padded per-batch width remains necessary for tensor storage, but it is derived from the active module counts in that batch rather than from a fixed hyperedge parameter.

---

# 1. Non-negotiable scope and boundaries

## 1.1 Preserve the established model

The following behavior must remain unchanged for all existing profiles and checkpoints:

- `fixed_projection` organizer;
- `exchangeable_slots` organizer;
- context-fusion and historical additive research paths;
- legacy and factorized pair-kernel modes;
- fused query-module execution;
- gathered retained-beta evaluation;
- Run-1000 and Run-1401 strict checkpoint loading and golden replay;
- ThermalChannel Stage-A local surrogate and one-pass interaction refinement;
- existing inverse-core code and public inverse contracts.

Do not rename or move existing checkpoint-visible parameters. The new organizer must register its parameters only when `organizer_mode="case_adaptive_residual"` is selected.

## 1.2 Change only what the new organizer requires

The major redesign belongs in the organizer. The only permitted surrounding changes are those required to support:

- optional global-token conditioning in the new organizer;
- a per-case soft/hard mechanism-support mask in query routing;
- new scalar diagnostics;
- variable-active-edge visualization and export;
- configuration, tests, and one new generic evaluator.

Do not simultaneously replace:

- the module, environment, global, or query encoders;
- the legacy pair MLP;
- the final field head;
- field/local/interface/port losses;
- optimizer policy;
- environment-token resolution;
- data splits or normalization;
- the inverse model.

The scientific comparison must remain

\[
\boxed{
\text{same encoder}
+
\text{same decoder}
+
\text{same physical training procedure}
+
\text{new organizer only}.
}
\]

## 1.3 Do not revive failed dynamic-K machinery

The new mode must not use:

- simultaneous anonymous slot creation followed by pruning;
- a learned discrete count head;
- one global candidate-edge capacity as the scientific K;
- epoch-based selection or sparsity schedules;
- detached CPU greedy selection;
- split/merge heuristics;
- count labels;
- entropy, load-balancing, diversity, orthogonality, or anti-collapse penalties;
- hard advection cones or ThermalChannel-specific topology priors.

The extraction process itself must provide the mechanism ordering and anti-redundancy pressure.

## 1.4 Training limits

Codex may use `cuda:2` for all GPU checks.

Before the formal run, Codex may perform only:

- unit and integration tests;
- one-batch forward/backward checks;
- synthetic residual-decomposition tests;
- a small real-data overfit or workflow smoke test with tightly limited batches/steps;
- dry-run launch validation;
- no-training checkpoint diagnostics.

Codex must not launch multiple formal architecture runs or any run beyond 2500 epochs.

After all implementation gates pass, Codex is authorized to launch exactly one formal candidate from scratch to epoch 2500 on `cuda:2`. No automatic continuation to 5000 or 10000 is allowed in this task.

---

# 2. Current code structure and exact revision boundary

The current core has stable ownership boundaries. Retain them.

| File or package | Current role | Required change |
|---|---|---|
| `src/honf_forward_core/config.py` | Strict shared configuration | Add the new organizer mode and a minimal set of residual-organizer fields with mode-specific validation. |
| `src/honf_forward_core/model.py` | Encode, organize, and decode orchestration | Pass the encoded global token to the organizer through a backward-compatible optional argument. Do not alter established token construction. |
| `src/honf_forward_core/organizer.py` | Stable facade plus fixed organizer | Add one delegation branch for the new mode. Keep the fixed organizer body and parameter names unchanged. |
| `src/honf_forward_core/organization/residual_adaptive.py` | New file | Implement the complete case-adaptive residual organizer. |
| `src/honf_forward_core/organization/helpers.py` | Shared geometry/statistics helpers | Add only generic helpers needed by the new organizer when this avoids duplicated, error-prone tensor code. Do not refactor legacy paths merely for style. |
| `src/honf_forward_core/decoder.py` | Query-to-edge routing and context fusion | Apply the new organizer's per-case support weights to query routing. Legacy all-edge behavior must remain numerically unchanged. |
| `src/honf_forward_core/decoding/pairwise.py` | Fused beta routing and pair evaluation | Prefer no architectural change. Verify variable packed K and active-edge masking. Change only if a correctness issue is demonstrated. |
| `src/honf_forward_core/training/diagnostics.py` | Generic scalar diagnostics | Add residual/count diagnostics without adding a new organizer penalty for the first formal candidate. |
| `Case_ThermalChannel/src/channelthermal/model.py` | Coupled ThermalChannel wrapper | Pass `global_token` in every direct organizer call, including provisional and final refinement passes. Preserve local coupling. |
| `Case_ThermalChannel/src/channelthermal/model_support.py` | Organizer compatibility view | Carry the new diagnostic keys and use the true effective-edge mask for the new mode. |
| `Case_ThermalChannel/src/channelthermal/training/epoch.py` | Loss and metric aggregation | Consume the new generic diagnostics; do not change physical loss weights. |
| `Case_ThermalChannel/src/channelthermal/training/reporting.py` | Training figures | Add one compact organizer-health plot for the new metrics. Do not duplicate the main loss figures. |
| `Case_ThermalChannel/src/channelthermal/evaluation/results.py` | Organizer arrays and result summaries | Handle a variable active mechanism set and include residual/count summaries. |
| `Case_ThermalChannel/src/channelthermal/evaluation_tools/organizer_visualization.py` | Physical organization plots | Filter or visibly mark inactive padded mechanisms and add residual-extraction summaries. |
| `Case_ThermalChannel/src/channelthermal/evaluation_tools/routing_visualization.py` | Query routing plots | Use the active mechanism mask and support variable K per case. |
| `tools/diagnostics/` | Reusable evaluation tools | Add at most one generic case-adaptive residual evaluator; extend existing accuracy/topology/benchmark tools rather than cloning them. |
| `src/config_core/forward/` | Maintained launch profiles | Add one complete candidate profile and update the registry/documentation without changing the recommended profile. |
| `tests/` | Core/runtime regression suite | Add focused tests for the new mode and preserve all old tests. |

The new organizer should be self-contained. Avoid spreading its mathematical operations across the ThermalChannel wrapper or training workflow.

---

# 3. Mathematical contract of the new organizer

To avoid overloaded symbols in implementation and diagnostics, use:

- \(\mathbf{x}^{\mathrm{src}}_{b,r}\): module-weighted source centroid of mechanism \(r\);
- \(\mathbf{x}^{\mathrm{reg}}_{b,r}\): environment-weighted response-region centroid;
- \(z^{\mathrm{hard}}_{b,r}\in\{0,1\}\): hard evaluation support;
- \(\gamma_{b,r}\in[0,1]\): differentiable training survival weight;
- \(h_{b,r}\in\mathbb R^H\): learned mechanism state.

The case index \(b\) may be omitted inside single-case equations, except where it is needed to distinguish support or count semantics.

## 3.1 Inputs and batch packing

For one batch:

- \(B\): number of cases;
- \(M_{\mathrm{batch}}\): largest active module count after dynamic batch collation;
- \(E\): environment-token count;
- \(H\): hidden width;
- \(d\): spatial dimension, currently two in the maintained core.

The organizer receives

\[
M_{\mathrm{tok}}
\in
\mathbb R^{B\times M_{\mathrm{batch}}\times H},
\]

\[
E_{\mathrm{tok}}
\in
\mathbb R^{B\times E\times H},
\]

\[
X_M
\in
\mathbb R^{B\times M_{\mathrm{batch}}\times d},
\]

\[
X_E
\in
\mathbb R^{B\times E\times d},
\]

\[
P_M
\in
\{0,1\}^{B\times M_{\mathrm{batch}}},
\]

and the encoded global token

\[
g
\in
\mathbb R^{B\times H}.
\]

The existing `ChannelThermalBatchCollator` already compacts active modules and pads only to the largest active count in the current batch. Therefore, the natural extraction ceiling is

\[
K_{\mathrm{pack}}
=
M_{\mathrm{batch}},
\]

while each case has a cap

\[
K_{\mathrm{cap},b}=M_b=\sum_i P_{M,bi}.
\]

`K_pack` is a tensor-packing width, not a globally fixed scientific K and not a learned parameter dimension.

The first implementation may require at least one active module per case. If the dataset later needs zero-module cases, handle them through an explicit background-only contract in a separate change; do not add a hidden zero-module special case now.

## 3.2 Global-conditioned nonnegative module-environment coupling

Condition module and environment tokens with the global state:

\[
\widetilde m_i
=
 m_i+W_g^M g,
\]

\[
\widetilde e_j
=
 e_j+W_g^E g.
\]

Define normalized relative geometry

\[
\Delta_{ij}
=
\operatorname{relative}(y_j-x_i),
\]

using the core's existing coordinate scale and periodic minimum-image convention.

Construct shared coupling logits

\[
L_{ij}
=
\frac{
\left(W_C^M\widetilde m_i\right)^\top
\left(W_C^E\widetilde e_j\right)
}{\sqrt H}
+
f_\Delta\!\left(\Phi(\Delta_{ij})\right),
\]

where `f_delta` is a small shared scalar network and has no module-, edge-, or case-index parameters.

The nonnegative coupling is

\[
C_{ij}
=
P_i\,\operatorname{softplus}(L_{ij}).
\]

Normalize once per case:

\[
\widetilde C
=
\frac{C}{\|C\|_F+\epsilon}.
\]

This normalization is mandatory. It prevents the network from satisfying the stopping rule merely by shrinking the absolute magnitude of the coupling matrix.

The existing module-to-environment context can be derived from the same coupling:

\[
A^{ME}_{ij}
=
\frac{C_{ij}}
{\sum_{j'}C_{ij'}+\epsilon},
\]

\[
c^{ME}_i
=
\sum_j A^{ME}_{ij}e_j,
\]

\[
m_i^{\mathrm{org}}
=
\left(m_i+0.25W_{ME}c_i^{ME}\right)P_i.
\]

This retains the current `A_me` and `module_env_context` contracts without computing an unrelated second module-environment attention map.

The matrix \(\widetilde C\) is a learned interaction measure. It must not be described as physical energy or a PDE residual.

## 3.3 Sequential residual extraction

Initialize

\[
R^{(0)}=\widetilde C.
\]

For extraction step \(r=1,\ldots,K_{\mathrm{pack}}\), first compute residual row mass

\[
\mu^{M,(r-1)}_i
=
\sum_jR^{(r-1)}_{ij}.
\]

Initialize the module factor

\[
a^{(0)}_{ir}
=
\operatorname{masked\_softmax}_i
\left[
 f_M^{\mathrm{anchor}}(m_i^{\mathrm{org}},g)
+
\log\left(\mu^{M,(r-1)}_i+\epsilon\right)
\right].
\]

Inactive padded modules must receive exactly zero.

For a small fixed number \(T_R\) of shared refinement iterations:

1. Compute the source center

   \[
   \mathbf{x}^{\mathrm{src}}_r
   =
   \sum_i a_{ir}x_i.
   \]

2. Compute residual support on environment tokens

   \[
   \nu^{E}_j
   =
   \sum_iR^{(r-1)}_{ij}a_{ir}.
   \]

3. Form an environment factor

   \[
   b_{jr}
   =
   \operatorname{softmax}_j
   \left[
   \frac{
   \left(W_Q^E\bar m_r\right)^\top
   \left(W_K^E e_j\right)
   }{\sqrt H}
   +
   \log(\nu^E_j+\epsilon)
   +
   f_{\mathrm{geo}}\!\left(\Phi(y_j-\mathbf{x}^{\mathrm{src}}_r)\right)
   \right],
   \]

   where

   \[
   \bar m_r=\sum_i a_{ir}W_Mm_i^{\mathrm{org}}.
   \]

4. Compute residual support on modules

   \[
   \nu^M_i
   =
   \sum_jR^{(r-1)}_{ij}b_{jr}.
   \]

5. Refine the module factor

   \[
   a_{ir}
   =
   \operatorname{masked\_softmax}_i
   \left[
   \frac{
   \left(W_Q^M\bar e_r\right)^\top
   \left(W_K^Mm_i^{\mathrm{org}}\right)
   }{\sqrt H}
   +
   \log(\nu^M_i+\epsilon)
   \right],
   \]

   where

   \[
   \bar e_r=\sum_jb_{jr}W_Ee_j.
   \]

Use shared networks at every extraction step. Do not add a step-index embedding. The extraction order must arise from the changing residual.

The initial profile should use

\[
T_R=2.
\]

## 3.4 Mechanism strength and residual update

The rank-one mechanism pattern is

\[
P_r=a_rb_r^\top.
\]

Its analytic nonnegative strength is

\[
\lambda_r
=
\frac{
\langle R^{(r-1)},P_r\rangle
}{
\|P_r\|_F^2+\epsilon
}.
\]

Update the unexplained interaction:

\[
\boxed{
R^{(r)}
=
\operatorname{ReLU}
\left(
R^{(r-1)}-\lambda_rP_r
\right).
}
\]

Because \(R^{(r-1)}\), \(a_r\), \(b_r\), and \(\lambda_r\) are nonnegative,

\[
0\leq R^{(r)}_{ij}\leq R^{(r-1)}_{ij},
\]

and therefore

\[
\|R^{(r)}\|_F
\leq
\|R^{(r-1)}\|_F.
\]

This monotonicity is a hard correctness invariant and must be tested.

Define

\[
\rho_r
=
\frac{
\|R^{(r)}\|_F^2
}{
\|R^{(0)}\|_F^2+\epsilon
},
\qquad
\rho_0=1.
\]

Define the marginal explained fraction

\[
\Delta\rho_r
=
\rho_{r-1}-\rho_r.
\]

Extraction order now has a direct meaning: mechanism \(r\) is the \(r\)-th residual-explaining component, rather than an arbitrary persistent edge identity.

## 3.5 Case-specific stopping and soft training support

Define

\[
K_{\min,b}
=
\min(M_b,K_{\min}).
\]

The hard case-specific count is

\[
K_b
=
\min\left\{
r\in\{K_{\min,b},\ldots,M_b\}:
\rho_{b,r}\leq\varepsilon_{\mathrm{edge}}
\right\},
\]

when this set is nonempty. If the residual tolerance is not reached by the case-specific cap, define

\[
K_b=M_b,
\qquad
\texttt{case\_adaptive\_cap\_hit}=1.
\]

Thus \(K_{\min,b}\leq K_b\leq M_b\) is well-defined for every valid case.

For evaluation, mechanism \(r\) is active when

\[
z_{b,r}^{\mathrm{hard}}
=
\mathbf 1[r\leq K_{\min,b}]
\lor
\mathbf 1[\rho_{b,r-1}>\varepsilon_{\mathrm{edge}}],
\]

and \(r\le M_b\).

For training, use the soft survival value

\[
\gamma_{b,r}
=
\mathbf 1[r\leq K_{\min,b}]
+
\mathbf 1[r>K_{\min,b}]
\sigma\left(
\frac{
\rho_{b,r-1}-\varepsilon_{\mathrm{edge}}
}{
\tau
}
\right),
\]

again masked by \(r\le M_b\).

Training must not use detached count selection. Gradients must flow through \(\rho\), the soft survival values, the factors, and the coupling network.

The first profile should use

\[
\varepsilon_{\mathrm{edge}}=0.02,
\qquad
\tau=0.05,
\qquad
K_{\min}=1.
\]

A no-training proxy audit may justify one change to \(\varepsilon_{\mathrm{edge}}\) before the formal launch. Do not train a sweep of tolerances.

## 3.6 Mechanism state

For each extraction step, calculate

\[
\mathbf{x}^{\mathrm{reg}}_r
=
\sum_jb_{jr}y_j,
\]

and corresponding weighted source/region scales using existing generic helper functions.

Form a shared mechanism state

\[
h_r
=
H_{\mathrm{mix}}
\left[
\bar m_r,
\bar e_r,
\Phi(\mathbf{x}^{\mathrm{src}}_r),
\Phi(\mathbf{x}^{\mathrm{reg}}_r),
\lambda_r,
\rho_{r-1},
\Delta\rho_r,
 g
\right].
\]

An equivalent clean residual formulation is acceptable, but all extraction steps must use the same parameters and no edge-index embedding.

## 3.7 Incidence assembly

Retain the source design's soft incidence convention. Define

\[
w_r=\gamma_r\lambda_r.
\]

Then

\[
A^{MH}_{ir}
=
\frac{
w_ra_{ir}
}{
\sum_{\ell}w_\ell a_{i\ell}+\epsilon
},
\]

and

\[
A^{EH}_{jr}
=
\frac{
w_rb_{jr}
}{
\sum_{\ell}w_\ell b_{j\ell}+\epsilon
}.
\]

During evaluation, replace \(\gamma_{b,r}\) with the hard support mask \(z_{b,r}^{\mathrm{hard}}\).

For every active module row and environment row,

\[
\sum_rA^{MH}_{ir}=1,
\qquad
\sum_rA^{EH}_{jr}=1,
\]

up to numerical tolerance.

The implementation must contain a deterministic fallback only for a true numerical zero-denominator case. It must not use the former greedy coverage selector.

## 3.8 Query routing

The decoder continues to construct content and geometry logits

\[
\ell_{qr}
=
\frac{
(W_Qz_q)^\top(W_Hh_r)
}{\sqrt H}
+
b_{\mathrm{query\text{-}geo}}(q,r).
\]

For the new organizer only, incorporate mechanism support as a routing prior:

\[
\widetilde\ell_{qr}
=
\ell_{qr}
+
\log(\gamma_r+\epsilon)
\]

in training, and use the hard mask in evaluation.

Then

\[
\alpha_{qr}
=
\operatorname{softmax}_r(\widetilde\ell_{qr}).
\]

At least one edge must remain active for every valid case.

Do not route padded or hard-inactive mechanisms. Their query attention must be exactly zero in evaluation.

This masking branch must be entered only when the organizer exports the new support contract. Existing fixed all-edge profiles must follow the same arithmetic path as before and retain golden replay.

## 3.9 Fused query-module execution

Keep the established fused formulation:

\[
\bar A^{MH}_{ir}
=
\frac{A^{MH}_{ir}}
{\sum_{i'}A^{MH}_{i'r}+\epsilon},
\]

\[
\beta_{qi}
=
\sum_r\alpha_{qr}\bar A^{MH}_{ir},
\]

\[
c_{\mathrm{pair}}(q)
=
g_{\mathrm{pair}}
\sum_i\beta_{qi}\psi(q,i).
\]

The new candidate must use

```text
pairwise_aggregation_mode = "fused_query_module"
pairwise_kernel_mode      = "legacy_mlp"
routing_execution         = "dense"
```

for training. Gathered beta execution remains an evaluation/deployment mode.

## 3.10 Final field

Do not change

\[
\widehat u(q)
=
D_\theta\left[
\operatorname{Norm}\left(
c_H(q)
+c_{\mathrm{pair}}(q)
+c_{\mathrm{global}}(q)
+c_{\mathrm{near}}(q)
\right)
\right].
\]

The global and near-module paths remain important. The dynamic organizer is responsible for mesoscopic interaction structure, not every local singularity or background condition.

## 3.11 Critical identifiability and shortcut safeguards

The coupling matrix and mechanism count are latent; the dataset provides no ground-truth \(K_b\). Therefore, a finite loss and a variable count histogram are not sufficient evidence that the organizer learned physical interaction structure.

The implementation and evaluation must explicitly guard against five shortcuts:

1. **Artificially low-rank coupling.** Because \(C_\theta\) is learned, it could make \(\widetilde C\) nearly rank one so that every case stops at one mechanism, while the global, near, or pair kernel carries most of the prediction. Normalize \(C\) as specified, report its evaluation-only effective singular-value rank, and report whether residual rank/count correlates with physical interaction descriptors.
2. **Decoder bypass.** Measure the field sensitivity to evaluation-only organizer ablations: replace active incidences by their row-uniform values, suppress \(c_H\), or collapse active mechanisms to one pooled mechanism. These are diagnostics only and must not be used during training. If predictions barely change, interpretability claims are unsupported even if accuracy is good.
3. **Support cancellation during row normalization.** Apply \(\gamma_r\) or \(z_r^{\mathrm{hard}}\) before incidence normalization, then assert that every hard-inactive column remains exactly zero afterward. A small survival weight must not be numerically reactivated by a zero or tiny denominator.
4. **Soft-training/hard-evaluation mismatch.** Evaluate the same checkpoint and cases in both soft-support diagnostic mode and normal hard-evaluation mode. Report field discrepancy, count discrepancy, and stopping margin. A model that trains accurately only under diffuse soft support has not solved case-adaptive execution.
5. **Cap saturation.** Report the fraction of cases for which \(\rho_{M_b}>\varepsilon_{\mathrm{edge}}\). Such cases are valid outputs but show that the extractor or coupling representation failed to meet its own tolerance within the theoretically available module-axis cap.

Do not add losses merely to make these diagnostics look favorable. They are falsification tests for the central claim.

---

# 4. Required implementation design

## 4.1 New class and parameter ownership

Create

```text
src/honf_forward_core/organization/residual_adaptive.py
```

with a class such as

```python
class CaseAdaptiveResidualOrganizer(nn.Module):
    ...
```

`HypergraphOrganizerCore.__init__` should register it only for the new mode, for example under

```python
self.case_adaptive_residual
```

Do not instantiate fixed-organizer `module_score` or `env_score` layers in the new mode. Conversely, do not register new residual-organizer parameters in the old modes.

The new state-dict path should be deterministic, for example

```text
core.organizer.case_adaptive_residual.*
```

and should not depend on the module count or the inferred number of mechanisms.

## 4.2 Shared internal components

A clean first implementation will likely need:

- global-to-module and global-to-environment projections;
- shared coupling query/key projections;
- a small generic relative-geometry scalar encoder;
- one shared module-anchor score;
- shared module-summary-to-environment and environment-summary-to-module projections;
- shared module/environment value projections;
- one shared mechanism-state mixer;
- the existing-style module-environment context projection.

Do not add per-step `ModuleList` blocks of length \(K\) and do not create edge embeddings.

## 4.3 Forward algorithm

The forward method should perform the following logical sequence:

```text
1. Validate shapes and at least one active module per case.
2. Build globally conditioned module/environment tokens.
3. Construct nonnegative coupling C and normalized residual R0.
4. Derive A_me and module_env_context from C.
5. Set K_pack to the dynamically collated module width.
6. For r = 1..K_pack using shared parameters:
   a. compute soft/hard eligibility from module count and rho_(r-1);
   b. initialize module factor from residual row mass;
   c. alternate environment and module factor refinement;
   d. compute geometry and mechanism state;
   e. compute analytic lambda_r;
   f. subtract the rank-one component with ReLU;
   g. store rho_r and delta_rho_r.
7. Assemble soft incidences in training or hard incidences in evaluation.
8. Build geometry, mass, purity, strength, and descriptor outputs.
9. Return the existing organizer contract plus residual-specific diagnostics.
```

Do not transfer tensors to CPU inside this algorithm. Do not use Python loops over batch cases. A short Python loop over extraction steps is acceptable because extraction is inherently sequential and the current dynamically padded module count is small.

## 4.4 Output contract

The new organizer must return the standard keys required by the decoder, wrapper, evaluation, and exports:

```text
A_me
A_mh
A_eh
hyper_state
hyper_source_coords
hyper_region_coords
hyper_source_variance
hyper_source_scale
hyper_region_variance
hyper_region_scale
hyper_module_mass_raw
hyper_env_mass_raw
hyper_module_mass
hyper_env_mass
hyper_module_purity
hyper_env_purity
hyper_strength
edge_quality
edge_active_mask
hard_selected_edge_mask
edge_transition_gate
effective_edge_mask
candidate_edge_count
selected_edge_count
viable_selected_edge_count
hard_selected_edge_count
active_edge_count
mechanism_geometry_features
mechanism_mass_features
mechanism_raw_features
mechanism_descriptor_features
module_tokens
module_tokens_for_hyper
env_tokens
env_coords
module_centers
module_present
module_env_context
routing_execution_gathered
```

For compatibility fields that do not naturally have a candidate/selected distinction, set the candidate values to the unmasked extracted factors and the selected values to the soft/hard supported factors. Do not fabricate old greedy-selection statistics as though they were used.

Add the following explicit residual-organizer keys:

```text
residual_stop_fraction                 scalar
residual_soft_stop_temperature         scalar
residual_fraction_trace                [B, K_pack + 1]
residual_marginal_explained_fraction   [B, K_pack]
residual_mechanism_strength            [B, K_pack]
residual_module_factor                 [B, M, K_pack]
residual_environment_factor            [B, E, K_pack]
residual_coupling_row_mass              [B, M]
edge_survival_weight                   [B, K_pack]
hard_case_edge_mask                    [B, K_pack]
case_adaptive_edge_count               [B]
case_adaptive_edge_cap                 [B]
case_adaptive_stop_reached             [B]
case_adaptive_cap_hit                   [B]
case_adaptive_stop_margin               [B]
residual_monotonic_violation_max        [B]
```

The full \([B,K,M,E]\) residual history must not be returned in normal execution.

The normalized initial coupling can be reconstructed for explicit diagnostics as

\[
\widetilde C_{ij}
=
A^{ME}_{ij}\,\mu_i,
\]

where `residual_coupling_row_mass` stores \(\mu_i=\sum_j\widetilde C_{ij}\). The stored factors and strengths are sufficient to reconstruct each rank-one explained component offline.

## 4.5 Soft versus hard public semantics

Use these meanings consistently:

- `edge_survival_weight`: actual soft training support or hard evaluation support;
- `effective_edge_mask`: same effective support used by the decoder;
- `hard_case_edge_mask`: detached residual-threshold count for reporting, even during training;
- `edge_active_mask`: hard support in evaluation; during training, retain a detached hard reporting mask rather than falsely labeling every packed step active;
- `case_adaptive_edge_count`: sum of the hard reporting mask;
- `selected_edge_count` and `active_edge_count`: aliases of the hard case count for this mode;
- `soft_functional_edge_count`: may remain derived from strength and survival in the generic diagnostics.

Keep `candidate_hyper_state` as the unmasked shared mechanism state. For `hyper_state`, zero only steps that are invalid because `r > M_b`; do not multiply the content state by survival and then also apply the log-survival routing prior. Support should act through incidences and query routing exactly once in each path.

Do not let plotting or metrics infer the count from `hyper_strength > 0.05` when the organizer already provides a direct count and mask.

## 4.6 Global-token interface

Add an optional `global_token` argument to the organizer facade and the new organizer.

Update all call sites, including:

- `HONFNeuralField.encode_and_organize`;
- the provisional organizer call in `ChannelThermalHONFModel`;
- the final organizer call in `ChannelThermalHONFModel`;
- any test or tool that calls the organizer directly.

Legacy modes may ignore the optional token. Their outputs must not change.

## 4.7 Decoder support

In `HypergraphFieldDecoder`:

1. Read `edge_survival_weight` only when present.
2. Apply the soft log-prior or hard mask before query softmax for the new mode.
3. Ensure padded/inactive queries receive exactly zero attention.
4. Continue using the actual output K dimension, not `config.num_hyperedges`, whenever the runtime tensor shape is available.
5. Return new scalar routing diagnostics such as mean active case count and mean soft support, without materializing additional dense maps by default.

Do not change the legacy query-logit path when `edge_survival_weight` is absent. Golden replay has priority over code unification.

## 4.8 Pairwise support

The fused pair kernel already obtains K from `hyper_attention` and `A_mh`. Verify that it works for the packed residual mechanism axis.

Required properties:

- inactive mechanisms have zero query routing;
- beta remains normalized over active modules at full support within tolerance;
- dense and gathered full-support outputs match;
- no use of `config.num_hyperedges` determines runtime K in the new path;
- query chunking remains identical.

Avoid changing `pairwise.py` unless one of these tests exposes a real issue.

---

# 5. Configuration and profile plan

## 5.1 Minimal new core fields

Add only these new `UnifiedForwardConfig` fields:

```python
residual_stop_fraction: float = 0.02
residual_soft_stop_temperature: float = 0.05
residual_factor_refinement_steps: int = 2
residual_coupling_fourier_frequencies: int = 2
```

Reuse the existing `minimum_active_edges` field for the minimum mechanism count.

Mode-specific validation:

- allow `organizer_mode="case_adaptive_residual"`;
- require `num_hyperedges == 0` for the new maintained profile, making the absence of global K explicit;
- require `edge_capacity == 0` for the new mode;
- require `0 < residual_stop_fraction < 1`;
- require `residual_soft_stop_temperature > 0`;
- require positive refinement steps and nonnegative Fourier frequencies;
- require context fusion and fused query-module aggregation for the first supported candidate profile;
- require at least one minimum active edge;
- treat old selection/schedule fields as inactive compatibility fields rather than applying them.

Do not add a configurable residual subtraction mode, count head, hard-stop schedule, or rank penalty.

## 5.2 New complete profile

Create one complete profile:

```text
src/config_core/forward/case_adaptive_residual_context.json
```

It should copy the accepted Stage-7 K=6 physical/training settings except for the organizer and exact fused execution settings.

The core organizer section should state clearly:

```json
{
  "num_hyperedges": 0,
  "organizer_mode": "case_adaptive_residual",
  "edge_capacity": 0,
  "minimum_active_edges": 1,
  "residual_stop_fraction": 0.02,
  "residual_soft_stop_temperature": 0.05,
  "residual_factor_refinement_steps": 2,
  "residual_coupling_fourier_frequencies": 2,
  "field_assembly_mode": "context_fusion",
  "pairwise_aggregation_mode": "fused_query_module",
  "pairwise_kernel_mode": "legacy_mlp",
  "routing_execution": "dense",
  "query_module_retained_mass_floor": 1.0,
  "query_module_limit": 0
}
```

Retain:

- hidden width 256;
- Stage-7 environment grid;
- all softmax assignments where still applicable;
- residual/raw mechanism state behavior in the decoder;
- shared AdamW learning rate `3e-4`;
- weight decay `1e-5`;
- seed 0;
- AMP disabled;
- gradient clipping 1.0;
- current ThermalChannel case config and Stage-A dependency.

Set the formal profile duration to 2500 epochs and retain milestones:

```json
"save_epoch_milestones": [100, 250, 500, 1000, 1500, 2000, 2500]
```

Keep `save_latest_every_epochs=10` and the existing best-by-total, best-by-field, best-by-temperature, and best-predicted checkpoints.

The profile note must explain:

- `num_hyperedges=0` means no global scientific K;
- packed mechanism width is derived from dynamically collated module count;
- hard K is case-specific at evaluation;
- soft survival is used during training;
- no organizer count regularization is active.

## 5.3 Registry and documentation

Update `profile_registry.json` with a new entry such as:

```text
name: case_adaptive_residual_context
status: candidate
base: null
mode: case-adaptive residual mechanisms / context fusion / fused dense / legacy kernel
checkpoint_families: [case-adaptive residual v1]
owner: forward-research
stage: next
```

Do not change `recommended_forward_profile` until a later complete evaluation promotes the model.

Update:

- `src/config_core/forward/experiments/README.md` or the nearest maintained profile guide;
- `docs/configuration.md`;
- `Model_Explain.md` only enough to document the optional candidate mode and its boundaries.

Do not rewrite the public README as though the candidate were already accepted.

---

# 6. Training diagnostics and reporting

## 6.1 New scalar diagnostics

Extend `HONF_DIAGNOSTIC_KEYS` and `compute_honf_diagnostics` with compact quantities suitable for `metrics.csv`:

```text
case_adaptive_edge_count_mean
case_adaptive_edge_cap_mean
case_adaptive_soft_edge_count_mean
case_adaptive_stop_reached_fraction
case_adaptive_cap_hit_fraction
residual_fraction_final_mean
residual_fraction_final_max
residual_first_marginal_mean
residual_last_active_marginal_mean
case_adaptive_stop_margin_mean
case_adaptive_stop_margin_min
residual_monotonic_violation_max
```

Use direct organizer values. Do not derive K from a fixed threshold on `hyper_strength`.

The metrics must be valid for all modes. For old organizers, return zeros or an explicit not-applicable default without changing prior columns unexpectedly unless the existing CSV contract is intentionally versioned in a backward-readable way.

## 6.2 No new organizer loss in v1

The first formal candidate must keep

```json
"organizer_regularization": {"enabled": false, ...}
```

and `loss_organizer` should remain zero.

Do not add a count penalty. Do not reward low K. Do not force a nonconstant K distribution.

The residual stopping rule defines the count; the field and existing physical losses determine whether the learned coupling needs richer structure.

An optional residual loss may be considered only in a later model round after observing a concrete failure. It is not part of this implementation goal.

## 6.3 Training plots

Extend the current training reporting with one compact diagnostic figure under the canonical

```text
plots/diagnostics/
```

location. It should show, when available:

1. train/validation hard mean \(K_b\);
2. train/validation soft effective count \(\sum_r\gamma_{b,r}\);
3. train/validation final residual fraction;
4. cap-hit and stop-reached fractions.

Do not create a second copy of the loss history or write plots at the run root.

---

# 7. Evaluation and visualization changes

## 7.1 Wrapper compatibility

In `ChannelThermalModelSupportMixin._legacy_organizer_aux`:

- include the residual trace, factors, survival, count, and cap keys;
- use `effective_edge_mask` as `active_hyperedge_mask` for both `exchangeable_slots` and `case_adaptive_residual`;
- preserve fixed-projection visualization behavior exactly.

## 7.2 Organizer arrays

Update `extract_organization_arrays` to include:

```text
active_hyperedge_mask
edge_survival_weight
case_adaptive_edge_count
case_adaptive_edge_cap
residual_fraction_trace
residual_marginal_explained_fraction
residual_mechanism_strength
residual_module_factor
residual_environment_factor
residual_coupling_row_mass
case_adaptive_cap_hit
```

The default presentation view should display only hard-active mechanisms. A debug view may show inactive packed columns in gray.

## 7.3 New residual-mechanism visualization

Add one reusable visualization entry point, preferably in a focused file such as

```text
Case_ThermalChannel/src/channelthermal/evaluation_tools/residual_mechanism_visualization.py
```

or as a clean extension of the existing organizer visualization module if that remains readable.

For a selected case, produce:

### A. Residual waterfall

A vertical or compact plot showing

\[
\rho_0,\rho_1,\ldots,\rho_{K_{\mathrm{cap}}}
\]

with:

- the stopping tolerance;
- hard selected K;
- marginal explained fractions;
- mechanism strengths;
- soft survival values;
- cap-hit warning when applicable.

### B. Coupling decomposition

Reconstruct

\[
\widetilde C
=
A^{ME}\odot\mu^M,
\]

and display:

- initial coupling matrix;
- each active \(\lambda_ra_rb_r^\top\) component;
- final residual matrix;
- module labels and environment-token order.

Do not save one image per mechanism. Use one summarized figure per evaluated case.

### C. Physical mechanism map

Extend the existing source-to-region physical map so labels follow extraction order and the title states

```text
K_case = ... / K_cap = ...
final residual = ...
```

Use the active mask rather than plotting every packed mechanism.

## 7.4 Evaluation CLI and artifacts

Add at most one CLI option, for example

```text
--residual-mechanism-view {none,summary,all}
```

or integrate the new summary into `--organization-view=all` when the checkpoint uses the new mode.

Write outputs only under the existing category-oriented layout:

```text
evaluations/single_case/<case_timestamp>/
├── fields/
├── organization/
├── routing/
├── metrics/
├── arrays/
├── diagnostics/
├── summary.json
└── evaluation_manifest.json
```

Residual trace JSON/NPZ belongs in `diagnostics/` or `arrays/`; figures belong in `organization/`.

## 7.5 Existing hypergraph plan and topology signature

Do not change the inverse model or silently redefine schema-v3 topology signatures.

The new organizer should still satisfy the existing plan/signature inputs through:

- `A_mh`;
- `A_eh`;
- active masks;
- source/region geometry;
- masses and strengths;
- query-routing summaries.

Keep extraction-order residual trace as a separate forward diagnostic. A future schema revision can add it after the new organizer is scientifically accepted.

## 7.6 One generic multi-case evaluator

Add at most one maintained tool, for example:

```text
tools/diagnostics/evaluate_case_adaptive_residual.py
```

It should support:

- one or more checkpoints;
- dataset split and maximum case count;
- query batch size;
- deterministic case selection;
- optional stability perturbations;
- canonical output directory;
- CSV plus JSON summary.

It should report:

- per-case module count;
- hard K and cap;
- soft count;
- residual trace and final residual;
- cap hit and stopping margin;
- evaluation-only coupling effective rank;
- soft-support versus hard-support field discrepancy;
- organizer-reliance ablation discrepancies;
- whole/fluid and near-interface errors;
- channel errors;
- query/pairwise effective rank when routing maps are requested;
- runtime and memory only when explicitly benchmarked.

Do not create several Stage-named synthesis scripts.

---

# 8. Step-by-step verification plan

No formal run may begin before Sections 8.1–8.7 pass.

## 8.1 Configuration and construction tests

Add tests proving:

1. `case_adaptive_residual` is accepted.
2. Invalid residual tolerance, temperature, or refinement count is rejected.
3. The new profile resolves through the strict config loader.
4. `num_hyperedges=0` is legal only where appropriate and is not used as a runtime tensor width in the new mode.
5. Old profiles resolve to the same values as before.
6. The new organizer's state-dict keys and parameter count are independent of input module count.

## 8.2 Mathematical unit tests

Use small deterministic tensors to prove:

1. \(C\ge0\).
2. \(\|\widetilde C\|_F=1\) for valid cases within tolerance.
3. \(a_r\) sums to one across active modules and is zero on padding.
4. \(b_r\) sums to one across environment tokens.
5. \(\lambda_r\ge0\) and is finite.
6. The residual is elementwise non-increasing.
7. \(\rho_r\) is non-increasing.
8. Rank-one or nearly rank-one synthetic coupling stops earlier than a deliberately multi-component coupling.
9. `case_adaptive_edge_count <= active_module_count` for every case.
10. At least one mechanism remains active.
11. `A_mh` and `A_eh` row sums are one for valid rows.
12. Hard-inactive incidence columns remain exactly zero after row normalization.
13. Inactive padded mechanisms contribute exactly zero.
14. If the tolerance is not reached, the count falls back to `M_b` and reports a cap hit.

## 8.3 Invariance tests

Verify:

- permuting module order and consistently permuting module features/coordinates leaves the predicted field unchanged within the established tolerance;
- adding inactive padding leaves the organizer and field unchanged;
- batch composition does not change a case's hard K or output beyond numerical tolerance;
- repeated evaluation is deterministic;
- environment coordinate batching remains correct.

For exact symmetric modules, mechanism labels may swap only if the final physical output and unordered mechanism set remain equivalent. Because extraction is residual ordered, ordinary nonsymmetric cases should retain the same order.

## 8.4 Gradient tests

On `cuda:2`, run a complete forward/backward step and prove finite nonzero gradients for:

- coupling projections;
- geometry coupling network;
- module/environment factor refinement;
- mechanism mixer;
- soft survival path;
- query routing;
- existing pair kernel and field head.

There must be no CPU transfer, detached greedy decision, or `torch.no_grad()` around the training count path.

## 8.5 Decoder and wrapper integration tests

Verify:

- soft training support and hard evaluation support both produce finite fields;
- a diagnostic soft-versus-hard prediction gap is measurable without mutating checkpoint configuration;
- hard-inactive query attention is exactly zero;
- query attention sums to one across active mechanisms;
- beta is finite and uses only active modules;
- one-shot and chunked prepared decoding match;
- full-support dense and gathered beta execution match;
- provisional and final ThermalChannel organizer passes support different case-specific counts without shape errors;
- base-versus-final organizer diagnostics remain valid;
- port/global consistency probes and Stage-A refinement still run.

## 8.6 Compatibility gates

Run the complete repository tests.

When local frozen checkpoints are available, run:

```bash
conda run --no-capture-output -n ModularDT \
  python tools/diagnostics/replay_forward_golden.py \
  --device cuda:2
```

Required:

- Run 1000 replay unchanged;
- Run 1401 replay unchanged;
- historical state-key inventories unchanged;
- old optimizer-resume tests unchanged;
- fixed and exchangeable organizer tests unchanged.

Do not weaken golden tolerances.

## 8.7 Limited real-data smoke

Before the formal run, perform one tightly bounded ThermalChannel integration smoke on `cuda:2`.

Preferred sequence:

1. `train.py --dry-run` with the new profile.
2. One real batch in training mode and one validation batch in evaluation mode.
3. A short overfit/integration smoke of no more than 20–50 optimizer steps or no more than three epochs with very small `--max-train-batches` and `--max-val-batches`.
4. Confirm loss decreases at least on the tiny overfit subset, all new metrics are finite, checkpoints strict-load, and visualization/export completes for one case.
5. On the same smoke checkpoint, record soft-support versus hard-support output discrepancy and one organizer-reliance ablation; do not require a favorable scientific conclusion at smoke scale, but verify that the diagnostics execute correctly.

This smoke is not a formal model result. Its output should be clearly named `smoke_case_adaptive_residual` and remain in ignored managed output. Do not create a tracked report or multiple smoke run directories.

## 8.8 Pre-launch implementation report

Before launching the formal run, Codex should summarize:

- files changed;
- parameter/state-key inventory for the new mode;
- all test results;
- golden replay results;
- smoke command and result;
- resolved formal profile;
- exact formal launch command;
- any deviation from this mathematical contract and why.

If a correctness gate fails, stop without launching the formal run.

---

# 9. Optional no-training feasibility proxy

This is useful but must not become a large research detour.

Using the locally available accepted Run-1401 checkpoint, obtain the existing nonnegative `A_me` maps for a deterministic subset of cases. Treat them only as a proxy coupling matrix, normalize them, and apply a simple offline greedy nonnegative rank-one residual decomposition.

For

\[
\varepsilon\in\{0.01,0.02,0.05,0.10\},
\]

record the implied count distribution and correlations with:

- active module count;
- nearest-neighbor distance;
- heat heterogeneity;
- wall proximity;
- near-interface error.

Use this audit only to check that `0.02` is not obviously pathological. At most one tolerance may be selected for the formal profile. Do not launch trained tolerance variants.

If the local checkpoint or required arrays are unavailable, record that fact and continue with the mathematically defined default `0.02`.

---

# 10. Authorized formal run

## 10.1 Launch conditions

Codex may launch the formal candidate only after:

- all new and historical tests pass;
- golden replay passes when local checkpoints are available;
- the new profile dry-run is correct;
- the limited real-data smoke passes;
- the Git diff is reviewed for one-mode isolation;
- the implementation is committed to `agent/honf-core-next`.

The formal run must:

- train from scratch;
- use `cuda:2`;
- use the current dataset, seed, normalization, losses, Stage-A checkpoint, and shared optimizer policy;
- stop at epoch 2500;
- use dense fused query-module execution during training;
- create no second formal candidate.

Do not use `--initialize-checkpoint` or `--resume-checkpoint` for the first launch.

## 10.2 Run identity

Use Run 1700 if it is free in the local managed run store. If it is occupied, select the next free numeric ID and report it. Suggested name:

```text
case_adaptive_residual_v1
```

Expected command pattern:

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj

conda run --no-capture-output -n ModularDT \
  python train.py \
  --config project://src/config_core/forward/case_adaptive_residual_context.json \
  --workflow forward \
  --device cuda:2 \
  --epochs 2500 \
  --run-id 1700 \
  --run-name case_adaptive_residual_v1 \
  --yes
```

Codex must first run the same command with `--dry-run` instead of `--yes` and inspect the resolved launch.

## 10.3 Runtime monitoring and stop conditions

The run should normally continue to 2500, but Codex must terminate it early if any of the following occurs:

- NaN or infinite loss/metric;
- repeated optimizer or checkpoint corruption;
- residual monotonicity violation above numerical tolerance;
- zero active mechanisms or attention normalization failure;
- all validation cases hit the module-count cap together with a large unresolved residual for a sustained interval;
- all cases collapse to one mechanism while field error remains grossly worse than the matched Run-1401 trajectory and shows no recovery;
- memory growth or runtime behavior indicates an implementation leak.

Do not terminate merely because K variation is modest. A nearly constant inferred count is a scientific outcome, not automatically an implementation failure.

Use Run-1401 trailing validation references as descriptive checks:

| Epoch | Run-1401 field MSE, trailing 50 | Run-1401 temperature MSE, trailing 50 |
|---:|---:|---:|
| 500 | `2.2987e-2` | `1.5089e-2` |
| 1000 | `1.1217e-2` | `8.3802e-3` |
| 2500 | `3.4105e-3` | `3.7403e-3` |

Do not alter the model mid-run to chase these values.

## 10.4 End-of-goal checks

After epoch 2500 or a justified early stop, Codex should verify:

- managed run manifest status;
- final and milestone checkpoint inventory;
- strict loading of `latest`, milestone, and best checkpoints;
- optimizer state presence;
- finite metric history;
- organizer-health plot existence;
- no generated artifacts tracked by Git;
- branch working tree clean after any final source/documentation commit.

Codex should not perform the full next-round scientific comparison unless explicitly requested. The goal ends with the implementation, bounded evidence, and one completed 2500-epoch candidate.

---

# 11. Follow-up evaluation plan for the next round

The following evaluation is documented now but should be initiated separately after the formal run.

## 11.1 Checkpoints to compare

Primary candidate checkpoints:

- epoch 500;
- epoch 1000;
- epoch 2500;
- best by field MSE;
- best by temperature MSE.

Primary fixed baseline:

- Run 1401 epoch 500;
- Run 1401 epoch 1000;
- Run 1401 epoch 2500;
- Run 1401 accepted best, epoch 4585.

Run 1601 may be retained as a fused K=6 execution control. Run 1602 is a useful K=4 research reference but should not become the main promotion baseline.

## 11.2 Accuracy evaluation

On all 90 held-out cases, report:

- whole-domain and fluid-only normalized MSE;
- median, p95, and worst-case MSE;
- channel MSE for `u`, `v`, `p`, `omega`, and temperature;
- near-interface fluid MSE;
- far-field fluid MSE;
- internal temperature, interface, and port metrics;
- matched maturity comparisons at 500, 1000, and 2500.

The key question is whether dynamic organization preserves localized physics rather than merely matching aggregate field error.

## 11.3 Case-adaptive count evaluation

For every case and every relevant organizer pass, report:

- active module count \(M_b\);
- hard \(K_b\);
- soft effective count \(\sum_r\gamma_{b,r}\);
- cap \(M_b\);
- stop reached/cap hit and stopping margin;
- evaluation-only effective rank of the normalized initial coupling;
- final residual fraction;
- complete residual trace;
- marginal explained fractions;
- mechanism strengths.

Summarize:

- histogram of K;
- conditional K distribution by module count;
- K versus nearest-neighbor spacing;
- K versus heat heterogeneity;
- K versus wall proximity;
- K versus field and near-interface error;
- base, provisional, and final organizer count changes.

Do not define success as “the histogram must be broad.” The count variation must be stable and physically defensible.

## 11.4 Stability evaluation

On at least 20 deterministic cases, repeat inference under:

- random module permutation;
- extra inactive padding;
- alternate batch composition;
- mild environment-token subsampling or deterministic jitter, when geometrically valid;
- alternate query chunking.

Measure:

- exact/near-exact field agreement where invariance is expected;
- K agreement rate;
- residual-trace deviation;
- matched mechanism-set distance;
- source/region centroid variation.

The inferred count should not change because module order or padding changed.

## 11.5 Representation and interpretability evaluation

Report:

- environment and query effective rank, both raw and normalized by active K;
- query edge-column cosine;
- pairwise-map effective rank;
- module/environment incidence purity;
- mechanism source/region separation;
- overlap between extracted rank-one factors;
- residual monotonicity and diminishing marginal gain;
- soft-support versus hard-support prediction gap;
- evaluation-only organizer-reliance ablations (uniform incidence, pooled single mechanism, and no hyper-value context), clearly labeled as diagnostics;
- physical maps and coupling decompositions for representative simple, medium, and difficult cases.

The residual sequence should give a clearer reason for each additional mechanism than the old indexed columns.

## 11.6 Efficiency evaluation

Use one controlled GPU process and identical query shapes. Measure:

- organizer-only time;
- prepared-decoder time;
- full-forward time;
- training forward/backward step time;
- incremental allocated/reserved memory;
- packed K, hard K, and soft K;
- gathered selected query-module pair count.

Benchmark real ThermalChannel cases and synthetic scaling over increasing \(M\), \(E\), and \(Q\).

The new organizer may cost more than the fixed linear projection. Promotion does not require it to be faster on five-module cases. It must avoid pathological overhead and retain the Stage-7 gathered beta scaling path.

## 11.7 Promotion gates

A later promotion decision should require all of the following:

### Correctness

- all historical tests and golden replay pass;
- strict checkpoint loading and resume pass;
- no residual monotonicity violations;
- no invalid/empty cases;
- hard-inactive incidence/routing is exactly zero;
- soft-versus-hard prediction discrepancy is acceptably small and reported.

### Matched-budget accuracy

At epoch 2500, target approximately:

- validation field and temperature metrics within `1.15x` of Run 1401 at 2500;
- complete-split whole-fluid and near-interface MSE within `1.10x` of matched Run 1401;
- no critical channel degradation beyond `1.15x` without a strong compensating benefit.

These are continuation/promotion guides, not reasons to alter the completed run after seeing results.

### Adaptivity

- count is stable under permutation, padding, and batching;
- cap-hit fraction is acceptably low and explicitly reported;
- residual stopping behaves consistently;
- any case-to-case count variation has a defensible relationship to interaction complexity or localized difficulty.

A constant K outcome may still be scientifically informative, but it would not support the claim that the new organizer solved case-adaptive topology.

### Interpretability

- marginal residual reduction is meaningful and generally diminishing;
- added mechanisms correspond to distinct source/region patterns;
- the residual trace explains why extraction stopped;
- organizer-reliance ablations show that the learned organization materially affects the prediction, rather than being bypassed by residual decoder paths.

### Efficiency

- no serious regression in full-forward or training memory;
- organizer overhead remains bounded;
- fused gathered beta execution remains available and correct;
- variable K provides measurable routing/organizer savings on at least some cases or larger synthetic shapes.

## 11.8 Next decision

After the complete evaluation, choose one of four outcomes:

1. **Promote** the case-adaptive organizer and consider continuation to 5000.
2. **Retain as research** because it is accurate but K is unstable or uninterpretable.
3. **Retain fixed K** because all cases converge to effectively the same rank.
4. **Reject** because accuracy, local physics, stability, or computation is unacceptable.

Do not launch a second case-adaptive architecture until this outcome is understood.

---

# 12. Artifact and repository hygiene

Maintained source belongs in:

- `src/`;
- `Case_ThermalChannel/src/`;
- `tests/`;
- `tools/diagnostics/`;
- maintained config/docs paths.

Generated artifacts belong only in ignored managed locations:

- `Trained_Results/...` for formal and smoke runs;
- `diagnostics/generated/case_adaptive_residual/...` for no-training synthesis;
- canonical `evaluations/<kind>/<job>/...` folders attached to managed runs.

Do not commit:

- checkpoints;
- generated CSV/JSON/NPZ/PNG evidence;
- smoke run directories;
- copied baseline outputs;
- duplicate diagnostic scripts.

Use the existing `EvaluationArtifactLayout` categories and manifests.

---

# 13. Required final Codex report

At task completion, report:

1. starting and final commit SHAs;
2. complete file-change list;
3. mathematical implementation summary;
4. new state-dict and parameter inventory;
5. configuration/profile additions;
6. full test and golden replay results;
7. smoke-test commands and results;
8. exact formal run ID, command, directory, and status;
9. checkpoint inventory through epoch 2500;
10. basic training trajectory and organizer-health observations;
11. any deviations or unresolved issues;
12. confirmation that no second formal run, no run beyond 2500, no inverse change, and no generated evidence commit occurred.

---

# 14. Definition of done

The goal is complete when:

- exactly one new `case_adaptive_residual` organizer mode exists;
- its parameters are shared and independent of inferred K;
- K is determined per case by residual stopping;
- old organizer modes and checkpoints are unchanged;
- the decoder correctly masks soft/hard case-specific mechanisms;
- fused beta routing remains correct;
- training diagnostics and plots expose count and residual behavior;
- single-case evaluation can visualize the extraction sequence;
- one generic multi-case evaluator is available for the next round;
- all tests and bounded smoke checks pass;
- one from-scratch formal run has completed or justifiably stopped no later than epoch 2500 on `cuda:2`;
- milestone and best checkpoints are valid;
- the branch remains clean and generated outputs remain untracked.

The central claim to test is not merely that the code can produce different tensor counts. It is:

\[
\boxed{
\text{A shared residual organizer can allocate a stable, physically meaningful number of interaction mechanisms to each case while preserving HONF field accuracy.}
}
\]
