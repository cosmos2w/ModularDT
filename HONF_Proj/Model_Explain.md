# HONF model mathematics and implemented settings

This document describes the model that is implemented in `src/honf_forward_core` and the ThermalChannel coupling implemented in `Case_ThermalChannel/src/channelthermal`. It distinguishes the checkpoint-compatible default profile from the opt-in `adaptive_sparse_additive` profile because the two profiles intentionally instantiate different organizers, mechanism encoders, field heads, and routing execution paths.

## 1. Scope, ownership, and notation

The reusable forward core owns generic module, environment, hyperedge, query, routing, and field tensors. ThermalChannel owns the physical feature definitions, packed datasets, Stage-A local thermal-disk surrogate, local/global coupling, field names, physical losses, and visualizations. The top-level `train.py` and `evaluate.py` compose a core profile with `Case_ThermalChannel/configs/case_default.json` through the case plugin.

For one batch, let (B) be the number of cases, (M) the batch-local padded module width, (E) the number of environment tokens, (K) the fixed edge count or runtime candidate-edge capacity, (Q) the number of global queries, (P) the number of angular ports, (H) the hidden width, and (F) the output-field width. The principal tensors are module centers (X\in\mathbb{R}^{B\times M\times2}), module features (S\in\mathbb{R}^{B\times M\times d_m}), module mask (P_M\in\{0,1\}^{B\times M}), environment coordinates (Y\in\mathbb{R}^{B\times E\times2}), environment features (R\in\mathbb{R}^{B\times E\times d_e}), global context (c\in\mathbb{R}^{B\times d_c}), and query coordinates (q\in\mathbb{R}^{B\times Q\times2}).

The ThermalChannel field order is fixed by the packed dataset and normally resolves to

$$U(q)=[u(q),v(q),p(q),\omega(q),T(q)]\in\mathbb{R}^{5}.$$

Inactive module rows are masked throughout organization, Stage-A execution, losses, and metrics. No forward-core parameter shape depends on (M), and `dynamic_module_padding=true` makes (M) the maximum active count in the current batch rather than a model capacity.

## 2. ThermalChannel inputs

`ChannelThermalInputAdapter` constructs ten module features in this exact order: dataset-scaled heat, absolute dataset-scaled heat, case-relative heat, absolute case-relative heat, active flag, solid diffusivity, fluid diffusivity, solid conductivity, fluid conductivity, and module radius.

The current `padding_invariant_v2` global context has eighteen entries in this exact order: Reynolds number, inlet velocity, active count, `log1p` active count, module number density, occupied-area fraction, total scaled heat, total scaled heat per domain area, mean active scaled heat, maximum absolute scaled heat, domain length (L_x), domain length (L_y), viscosity, solid diffusivity, fluid diffusivity, solid conductivity, fluid conductivity, and module radius. Historical checkpoints that used `legacy_v1` retain the original fourteen-entry schema with a saved fixed reference-slot denominator; runtime padding never changes that historical feature.

`ChannelThermalEnvironmentBuilder` creates a cell-centered (24\times8) grid in both maintained forward profiles, hence (E=192). Each token contains normalized (x), normalized (y), normalized distances to the bottom wall, top wall, inlet, and outlet, plus centerline proximity. Query-side ThermalChannel features use the first six of those geometric quantities.

## 3. Shared encoders

The core maps physical tensors into hidden tokens with shared encoders and Fourier coordinate features:

$$m_i=E_m([S_i,\Phi_x(X_i/s_x)])P_{M,i},\qquad e_j=E_e([R_j,\Phi_y(Y_j/s_y)]),\qquad g=E_g(c),\qquad z_q=E_q([q/s_q,\Phi_q(q/s_q),f_{\mathrm{case}}(q)]).$$

The forward profiles use (H=256), four Fourier frequencies for module, environment, query, and pairwise relative coordinates, `dropout=0.0`, `use_layer_norm=true`, `coordinate_scale=[12,6]`, `geometry_mode="nonperiodic"`, `periodic_axes=[]`, and `query_time_mode="none"`. The domain length and module radius fields marked `auto` are resolved from the packed dataset before construction.

When `use_A_me_auxiliary=true`, modules first attend to environment tokens:

$$A^{ME}_{ij}=\operatorname{softmax}_{j}\left(\frac{(W_qm_i)^\top(W_ke_j)}{\sqrt H}\right)P_{M,i},\qquad \widetilde m_i=m_i+0.25W_c\sum_jA^{ME}_{ij}e_j.$$

This Stage-A/local coupling and its case ownership are unchanged by either forward architecture profile.

## 4. Organizer interface and geometry

Both organizer modes return module-to-edge incidence (A^{MH}\in\mathbb{R}^{B\times M\times K}), environment-to-edge incidence (A^{EH}\in\mathbb{R}^{B\times E\times K}), an edge state (h_k\in\mathbb{R}^{H}), an active mask (a_k\in\{0,1\}), and source/region geometry. Assignment rows are normalized over the edge axis after masking.

For an active edge, normalized column weights and weighted centroids are

$$w^M_{ik}=\frac{A^{MH}_{ik}}{\sum_iA^{MH}_{ik}+\epsilon},\qquad s_k=\sum_iw^M_{ik}X_i,$$

$$w^E_{jk}=\frac{A^{EH}_{jk}}{\sum_jA^{EH}_{jk}+\epsilon},\qquad r_k=\sum_jw^E_{jk}Y_j.$$

The diagonal source and region variances and scales are

$$v^M_k=\sum_iw^M_{ik}\Delta(X_i,s_k)^2,\quad \sigma^M_k=\sqrt{v^M_k+\epsilon},\qquad v^E_k=\sum_jw^E_{jk}\Delta(Y_j,r_k)^2,\quad \sigma^E_k=\sqrt{v^E_k+\epsilon}.$$

Here (Delta) applies the minimum-image convention only on axes listed in `periodic_axes`; current ThermalChannel profiles are nonperiodic. Normalized module and environment masses are (mu^M_k=(\sum_iA^{MH}_{ik})/(\sum_{i,k}A^{MH}_{ik}+\epsilon)) and (mu^E_k=(\sum_jA^{EH}_{jk})/(\sum_{j,k}A^{EH}_{jk}+\epsilon)). Assignment purity is the fraction of a column owned by tokens for which that edge is the row-wise winner, and the selection quality is (Q_k=\sqrt{\pi^M_k\pi^E_k}).

## 5. Default compatibility organizer

The default and saved-config fallback is `organizer_mode="fixed_projection"`. It instantiates edge-indexed output columns only in this compatibility path:

$$A^{MH}_{ik}=\mathcal N_M(W_M\widetilde m_i)_kP_{M,i},\qquad A^{EH}_{jk}=\mathcal N_E((W_Ee_j)_k+b^{\mathrm{geom}}_{jk}).$$

For the default profile, (mathcal N_M=mathcal N_E=\operatorname{softmax}), (K=6), every edge is active, and (b^{\mathrm{geom}}_{jk}) is the negative distance from environment token (j) to source centroid (s_k) divided by one quarter of the domain diagonal. The edge content state is

$$h_k=\operatorname{MLP}\left(\frac{\sum_iA^{MH}_{ik}W_M^h\widetilde m_i}{\sum_iA^{MH}_{ik}+\epsilon}+\frac{\sum_jA^{EH}_{jk}W_E^he_j}{\sum_jA^{EH}_{jk}+\epsilon}\right).$$

Saved forward configurations that omit all upgraded mode fields reconstruct this path because `UnifiedForwardConfig` supplies the strict defaults `fixed_projection`, `residual_concat`, `context_fusion`, three `softmax` normalizers, and `dense` routing. The default profile also sets `use_hyper_mechanism_encoder=false`, so its edge state remains the historical organizer state rather than passing through a descriptor encoder.

## 6. Exchangeable candidate-edge organizer

The upgraded profile selects `organizer_mode="exchangeable_slots"`, uses candidate capacity (K_{cap}=8), starts with six active candidates during selection warmup, and enforces at least one active edge. It does not instantiate learned edge-index embeddings or edge-specific projections. Changing the runtime candidate capacity changes tensor extent without changing learned parameter shapes.

Candidate (k) starts from a pooled case state and a deterministic sinusoidal code (d_k):

$$h_k^{(0)}=W_b\bar h+\operatorname{softplus}(W_s\bar h)\odot d_k,\qquad \bar h=\operatorname{mean}_{i:P_{M,i}=1}\widetilde m_i+\operatorname{mean}_je_j.$$

All candidates use the same query, key, value, GRU, and normalization maps. At refinement step (ell), competitive module and environment assignments are computed from shared dot products, weighted module and environment summaries update the slot with one shared GRU cell, and this repeats for `slot_refinement_steps=2`.

The upgraded module, environment, and query normalizers are `entmax15`. In general,

$$\operatorname{entmax}_{\alpha}(z)=\arg\max_{p\in\Delta}\left(p^\top z+H_{\alpha}(p)\right),\qquad \alpha=1.5,$$

which yields exact zeros while keeping the nonzero probabilities normalized. Environment routing uses `environment_locality_mode="bounded_gaussian"`. Given anisotropic scale (sigma_k) bounded below by `minimum_region_scale=0.05` of each domain scale, the normalized squared distance is

$$\rho_{jk}^2=\sum_d\left(\frac{\Delta(Y_j,r_k)_d}{\sigma_{kd}}\right)^2,$$

and the routing-logit bias is

$$b^{loc}_{jk}=-\frac{\lambda}{2}\min\left(\rho_{jk}^2,\rho_{max}^2\right),$$

with `environment_locality_strength` (lambda) set to `1.0` and `locality_radius_cap` (rho_max) set to `3.0`. Query routing applies the same bounded Gaussian log-bias. The locality factor is finite and strictly positive at every distance before entmax; only entmax or an explicit active mask creates exact-zero routes. The accepted `compact_kernel` mode remains available to reconstruct configurations that explicitly selected it.

Active-edge selection is detached from gradient flow. During the first 200 training epochs it takes the six highest-quality candidates. After warmup it visits candidates in descending quality order, rejects candidates whose maximum module or environment cosine overlap with any selected edge exceeds `0.85` unless needed for the minimum count, and stops when at least `0.95` of active module tokens and environment tokens each receive selected assignment mass of at least `0.50`. It therefore selects by quality, coverage, and novelty without adding an edge-count objective; the case profile’s optional organizer regularizer is disabled by default.

## 7. Descriptor-first mechanism state

The upgraded mode constructs a 16-value explicit descriptor for every candidate:

$$d_k=[s_k/s,\sigma^M_k/s,r_k/s,\sigma^E_k/s,\Delta(r_k,s_k)/s,\|\Delta(r_k,s_k)\|/\|s\|,\mu^M_k,\mu^E_k,\pi^M_k,\pi^E_k,a_k].$$

The mechanism state is descriptor-first with a bounded content residual:

$$t_k=\operatorname{LayerNorm}\left(E_d(d_k)+0.35E_h(h_k)\right).$$

Both (E_d) and (E_h) are shared over candidates. The descriptor is therefore the primary representation, while learned organizer content can refine but not replace it. The compatibility mode instead retains its historical content state and optional residual-concatenation encoder behavior.

## 8. Query routing and pairwise module relevance

For query (q), edge logits combine learned state compatibility and ten source/region-relative geometry features:

$$\ell_{qk}=\frac{(W_qz_q)^\top(W_kt_k)}{\sqrt H}+W_g\gamma(q,s_k,r_k)+b^{loc}_{qk}.$$

Inactive candidates are masked before normalization. The default path uses dense softmax (alpha_{qk}=\operatorname{softmax}_k(\ell_{qk})). The upgraded path uses entmax15, preserves its exact zeros, and limits execution to at most `query_edge_limit=3` nonzero routes per query.

The pairwise kernel describes a query relative to every relevant module with normalized (dx), (dy), distance, downstream distance, upstream distance, and lateral distance, optionally Fourier-encodes these quantities, concatenates the module token and raw module features, and applies one shared four-layer MLP. If (psi(q,i)) is this embedding and (ar A^{MH}_{ik}) is module incidence normalized by edge mass, then

$$c^{pair}_{qk}=\sum_i\bar A^{MH}_{ik}\psi(q,i),\qquad c^{pair}_q=\sum_k\alpha_{qk}c^{pair}_{qk}.$$

The learned pairwise gate is initialized to `0.1`. In gathered execution, module relevance is

$$\beta_{qi}=\sum_k\alpha_{qk}\bar A^{MH}_{ik}.$$

The implementation selects at most `query_module_limit=8` active modules per query using (eta), gathers their centers, tokens, raw features, and incidences, and only then calls the expensive pair MLP. It similarly gathers nonzero/top-ranked query-edge routes before calling the edge head. This is actual sparse execution; the dense compatibility path evaluates all pairs and routes, and merely multiplying dense results by zero is not described as computational sparsity.

## 9. Field assembly

### 9.1 Default context-fusion path

`enhanced_honf_pairwise.json` uses `field_assembly_mode="context_fusion"` and `decoder_mode="enhanced_honf_pairwise"`. Its query context is

$$c_q=c^H_q+g_q+c^{near}_q+\sigma(\eta_{pair})c^{pair}_q,\qquad c^H_q=\sum_k\alpha_{qk}W_vt_k,$$

where (g_q) is the projected global token and (c^{near}_q) is distance-weighted local module context. After layer normalization, one shared field head produces

$$\widehat U(q)=H_{field}(\operatorname{LayerNorm}(c_q)).$$

This formula, parameter construction, and state-dict path are unchanged for old saved configs and existing forward checkpoints.

### 9.2 Upgraded exact background-plus-edge path

`adaptive_sparse_additive.json` uses `field_assembly_mode="edge_additive"`. A background head sees the query state, global state, and query-attended environment context but no module memory:

$$U_{bg}(q)=H_{bg}([z_q,W_gg,\sum_j\rho_{qj}W_ve_j]),\qquad \rho_{qj}=\operatorname{softmax}_j((W_q^{bg}z_q)^\top(W_k^{bg}e_j)/\sqrt H).$$

One shared edge head evaluates only selected active routes:

$$U_k(q)=a_k\alpha_{qk}H_{edge}([z_q,t_k,\gamma(q,s_k,r_k),\sigma(\eta_{pair})c^{pair}_{qk}]).$$

The output is exactly

$$\boxed{\widehat U(q)=U_{bg}(q)+\sum_{k=1}^{K_{cap}}U_k(q)}.$$

When `return_edge_fields=true`, the decoder returns `pred_field_background` and `pred_field_by_edge`, and their sum is numerically the returned `pred_field`. It also reports per-edge absolute mean, RMS, energy fraction, selected/available route counts, and background-versus-edge norms.

## 10. ThermalChannel Stage-A coupling

The Stage-A `LocalModuleSurrogate` receives seven module parameters, port tokens ([\theta,\cos\theta,\sin\theta,T_{env},h]), and optional normalized local query coordinates. It encodes module parameters and port tokens, updates learned latent queries by cross-attention to the unordered port sequence, pools a module-response latent, predicts internal solid temperature at arbitrary local coordinates, and predicts interface ([T_s,q_n]) at every port.

The provided local checkpoint `Trained_Results/ThermalChannel/Local_Module_Runs/thermal_disk/Run_0000_base/latest_model.pt` stores `module_param_dim=7`, `port_token_dim=5`, `interface_target_dim=2`, hidden and latent widths 128, 16 port latents, four attention heads, four cross-attention layers, six coordinate Fourier frequencies, and zero dropout. It is the epoch-6357 latest checkpoint and carries input/output normalization statistics. The forward CLI treats it as a trusted local Stage-A dependency, loads it strictly, copies its normalizers, and freezes it because `freeze_local_surrogate=true`.

The global wrapper first organizes the case, predicts (T_{env}) and positive (h) at each angular port, chooses predicted, teacher, or mixed port conditions, runs Stage A only on active modules, and anchors the corrected flux at the Robin relation

$$q^{Robin}_n=h(T_s-T_{env}).$$

The default `local_surrogate_flux_mode="corrected_physics"` adds a learned zero-initialized residual to this physical value. Six local response statistics and the 128-value local latent are fused into each module state. With `interaction_refinement_steps=1`, the wrapper performs one provisional organization, probes global temperature just outside each port, applies one residual update to (T_{env}) and (h), reruns Stage A, fuses the final response, and recomputes the final organizer before decoding the requested field. Core changes do not move these steps or their parameters out of the ThermalChannel wrapper.

The main outputs are `pred_field [B,Q,F]`, `pred_internal_temperature [B,M,Ql,1]`, `pred_interface [B,M,P,2]`, `pred_port_condition [B,M,P,5]`, `organizer_aux`, and `routing_aux`.

## 11. Training objective and default data settings

The case profile uses 600 training cases and 90 test cases from `thermal_channel_global_v1`, samples 1024 global points per case, uses training and validation batch size 48, four workers, input and target normalization, random training-point sampling, dynamic padding, and module-count bucketing. The default forward optimizer uses AdamW with learning rate (3\times10^{-4}), weight decay (10^{-5}), gradient clipping at 1.0, no AMP, seed 0, and a nominal 10,000 epochs.

The coupled objective is a weighted sum of global field MSE, internal-temperature MSE, interface loss, autonomous port supervision, angular port smoothness, port/global consistency, and predicted-port consistency after warmup. Concrete weights in `case_default.json` are `field_mse_weight=1.0`, `temperature_weight=1.0`, `internal_temperature_weight=1.0`, `interface_weight=0.2`, `port_condition_weight=0.3`, `port_supervised_weight=0.3`, `port_smoothness_weight=0.01`, `port_global_consistency_weight=0.2`, and `predicted_consistency_weight=0.05` after 100 warmup epochs. Organizer regularization is implemented for experiments but `enabled=false`; there is no default edge-count penalty.

## 12. Unordered topology signature

`src/honf_forward_core/evaluation/topology_signature.py` exports `honf_topology_signature` schema version 3. A signature contains candidate and active counts, active mask, edge features, module and environment incidences, pairwise edge relations, reference-query routing summaries, per-field contribution summaries, reference-query digest/measure, field names, domain/periodicity metadata, case ID, and forward-checkpoint SHA-256. The 22 edge features include geometry, scales, masses, purities, effective module count, routing statistics, and contribution statistics; the nine relation features include module overlap, environment overlap, query co-routing, and pairwise source/region displacements and distances.

Edges are semantically unordered. Canonical ordering is used only for deterministic display and NPZ serialization, and `serialization_permutation` records that operation. Comparison extracts active tokens, builds normalized feature costs, pads unequal cardinalities with an explicit unmatched cost, uses Hungarian assignment, and adds a weighted relation error after matching:

$$d_{topo}(G_1,G_2)=d_{matched\ features}+\lambda_{rel}d_{relations},\qquad \lambda_{rel}=0.25\ \text{by default}.$$

The exporter can also reconstruct module affinity and query-to-module influence without assigning persistent edge labels. Evaluation writes `topology_signature.npz`, `topology_signature_summary.json`, diagnostics, and case-owned plots when `--export-topology-signature` is requested.

## 13. Inverse topology flow after schema v3

The accepted upgraded inverse profile is `src/config_core/inverse/train_inverse_topology_set_template.json`. It sets `plan_token_mode="exchangeable_set"`, `plan_conditioning_mode="set_cross_attention"`, two set-interaction layers, four attention heads, `matching_mode="sinkhorn"`, and requires topology schema name `honf_topology_signature`, version 3, plus the exact 64-character SHA-256 of the forward checkpoint that created the targets.

For a target topology set (G_1), Gaussian noise (G_0\sim\mathcal N(0,I)), and (t\sim\mathcal U[0,1]), rectified-flow training uses

$$G_t=(1-t)G_0+tG_1,\qquad v^*(G_t,t)=G_1-G_0,\qquad \mathcal L_{RF}=\|v_\theta(G_t,t,R,c)-v^*\|_2^2.$$

The exchangeable plan velocity network uses shared token projections and permutation-equivariant self-attention, instantiates no learned edge-index embedding, supports runtime topology capacity, and uses differentiable Sinkhorn matching for set targets. The downstream layout flow cross-attends its module-slot states to active topology tokens and masks inactive topology tokens. Its module-slot embeddings represent physical layout slots, not forward edge identity. The fixed-width indexed inverse profile and its ordered-flat layout conditioner remain available for compatible earlier inverse checkpoints and are instantiated only when selected.

## 14. Concrete forward profiles

| Setting | Default `enhanced_honf_pairwise` | Upgraded `adaptive_sparse_additive` |
|---|---:|---:|
| Organizer | `fixed_projection` | `exchangeable_slots` |
| Fixed edges / candidate capacity | 6 / n.a. | n.a. / 8 |
| Initial / minimum active edges | all 6 | 6 / 1 |
| Selection | all | quality + 95% coverage + novelty |
| Candidate module/environment mass floors | n.a. | 0.01 / 0.01 |
| Module/environment/query normalizer | softmax / softmax / softmax | entmax15 / entmax15 / entmax15 |
| Environment/query locality | none | bounded Gaussian, strength 1.0, radius cap 3.0 |
| Mechanism state | `residual_concat`, encoder disabled | `descriptor_first`, residual scale 0.35 |
| Field assembly | `context_fusion` | `edge_additive` |
| Additive output stabilization | n.a. | input LayerNorms, sigmoid edge gate 0.1, final-layer std 0.001 |
| Routing execution | `dense` | `gathered` |
| Query module / edge limit | 0 / 0 | 8 / 3 |
| Topology signature flag | false | true |
| Hidden width / dropout | 256 / 0.0 | 256 / 0.0 |
| Environment grid | 24 x 8 | 24 x 8 |
| Decoder | enhanced hyper + pairwise + global + near | enhanced hyper + pairwise feeding exact additive edge fields |
| Stage A | frozen, predicted ports, one refinement | unchanged |
| Default run ID | 0002 | 0003 |

## 15. Compatibility rules

- Missing upgraded fields in a saved config resolve to the exact fixed-projection, context-fusion path; they do not silently select the adaptive architecture.
- Only mode-specific modules are instantiated, which preserves historical state-dict names and strict checkpoint loading for the default path.
- The upgraded forward and inverse modes do not contain learned edge-index embeddings.
- Runtime module width and exchangeable edge capacity are tensor extents rather than learned parameter capacities.
- Stage-A/local coupling and all physical loss semantics remain case-owned.
- Sparse probabilities and sparse execution are reported separately; only gathered pre-MLP execution is called computationally sparse.
- Adaptive warmup exposes every viable candidate. Selection progress is serialized and is independent of `train()`/`eval()` mode.
- New diagnostics distinguish candidate, selected, viable-selected, functional, soft-functional, empty-selected, and effective query-edge counts.
- Additive edge exports include the learned output gate, preserving exact closure while logging background/edge scale and cancellation.
- No active-edge-count penalty is enabled in the shipped case profile.

## 16. Code-to-equation map

| Topic | Implementation |
|---|---|
| Strict forward mode defaults and validation | `src/honf_forward_core/config.py` |
| Fixed and exchangeable organizers, descriptors, selection | `src/honf_forward_core/organizer.py` |
| Entmax15 | `src/honf_forward_core/routing.py` |
| Query routing, gathered pair execution, field assembly | `src/honf_forward_core/decoder.py` |
| Core encoding and orchestration | `src/honf_forward_core/model.py` |
| Topology schema, serialization, matching, diagnostics | `src/honf_forward_core/evaluation/topology_signature.py` |
| Thermal physical inputs and environment features | `Case_ThermalChannel/src/channelthermal/input_adapter.py`, `environment.py` |
| Stage-A model and coupling | `Case_ThermalChannel/src/channelthermal/local_surrogate/model.py`, `local_coupling.py` |
| Complete coupled forward | `Case_ThermalChannel/src/channelthermal/model.py` |
| Forward loss/training policy | `Case_ThermalChannel/src/channelthermal/training_tools/losses.py`, `workflows/train_forward.py` |
| Exchangeable inverse plan flow | `src/honf_inverse_core/models/plan_flow.py` |
| Set-cross-attention layout flow | `src/honf_inverse_core/models/layout_flow.py` |
| Forward launch profiles | `src/config_core/forward/enhanced_honf_pairwise.json`, `adaptive_sparse_additive.json` |
| Exchangeable inverse profile | `src/config_core/inverse/train_inverse_topology_set_template.json` |
