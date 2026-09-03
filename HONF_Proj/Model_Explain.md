# HONF forward model: current mathematics, configuration, and code map

This document describes the maintained HONF forward platform. The recommended
profile uses fixed six-edge softmax organization, the organizer's raw residual
hyperedge state, dense query routing, and context fusion. The fused
query-module executor is the behavior-equivalent efficient path. Exchangeable,
adaptive, entmax, additive, and gathered mechanisms remain loadable optional
research and checkpoint-compatibility modes. The registry also contains the
standalone `case_adaptive_residual_context` candidate; it is not accepted or
recommended until its bounded evaluation gates pass.

## 1. Which configuration is current?

The profile registry at `src/config_core/forward/profile_registry.json` declares `stage7_structured_context` as `recommended_forward_profile` and gives it `status="current"`. Efficient execution is represented by a strict overlay rather than rewriting the compatibility profile.

| Role | Profile | Status |
|---|---|---|
| Recommended profile | `src/config_core/forward/stage7_structured_context.json` | Fixed K=6 architecture and 5K training policy |
| Promoted efficient execution | `src/config_core/forward/experiments/stage7_fused_query_module.json` | Exact K=6 fused dense legacy-kernel path; sparse beta-0.98 is evaluation-only |
| Optional residual candidate | `src/config_core/forward/case_adaptive_residual_context.json` | Case-specific residual mechanism count; 2500-epoch research candidate |
| Compatibility profile | `src/config_core/forward/enhanced_honf_pairwise.json` | Established checkpoint architecture |
| Optional research base | `src/config_core/forward/adaptive_sparse_additive.json` | Compatibility/research only |
| ThermalChannel case policy | `Case_ThermalChannel/configs/case_default.json` | Current case, data, Stage-A, loss, and evaluation settings |

The recommended and compatibility profiles instantiate the same 237-key model architecture. Their main differences are run identity, training duration, plot/checkpoint cadence, and explicit modern configuration fields. The K=6 fused legacy-kernel overlay adds no parameters and preserves the checkpoint contract.

One operational distinction is important: the profile registry recommends
`stage7_structured_context`, but the current `train.py` and `evaluate.py` CLI
fallback still points to `enhanced_honf_pairwise`. Use the Stage-7 profile
explicitly for new scientific work:

```bash
python train.py --config src/config_core/forward/stage7_structured_context.json --experiment-overlay src/config_core/forward/experiments/stage7_fused_query_module.json --run-id 1500 --dry-run
python evaluate.py --config src/config_core/forward/stage7_structured_context.json --checkpoint /path/to/checkpoint.pt
```

Replace `1500` with an unused numeric run ID.

## 2. Problem definition and tensor contract

For each case, the reusable HONF core receives:

- module centers $X\in\mathbb{R}^{B\times M\times2}$;
- module features $S\in\mathbb{R}^{B\times M\times d_m}$;
- an active-module mask $P_M\in\{0,1\}^{B\times M}$;
- environment coordinates $Y\in\mathbb{R}^{B\times E\times2}$ and features
  $R\in\mathbb{R}^{B\times E\times d_e}$;
- case context $c\in\mathbb{R}^{B\times d_c}$;
- arbitrary query coordinates $q\in\mathbb{R}^{B\times Q\times2}$.

$B$ is batch size, $M$ is the batch-local padded module width, $E$ is the
environment-token count, $K$ is the hyperedge count, $Q$ is the query count,
and $H$ is hidden width. Dynamic padding makes $M$ the maximum active module
count in the current batch; it is not a learned model capacity. Inactive module
rows remain masked through organization, Stage-A execution, losses, and
metrics.

For ThermalChannel, the predicted global field has the fixed dataset order

$$
\widehat U(q)=[\widehat u(q),\widehat v(q),\widehat p(q),
\widehat\omega(q),\widehat T(q)]\in\mathbb{R}^{5}.
$$

The primary output contract is:

| Output | Shape | Meaning |
|---|---:|---|
| `pred_field` | `[B,Q,5]` | Global continuous field |
| `pred_internal_temperature` | `[B,M,Ql,1]` | Local solid temperature |
| `pred_interface` | `[B,M,P,2]` | Interface temperature and normal flux |
| `pred_port_condition` | `[B,M,P,5]` | Angular port tokens used by Stage A |
| `organizer_aux` | dictionary | Incidences, hyperedge geometry, masks, and statistics |
| `routing_aux` | dictionary | Query routing and pairwise diagnostics |

## 3. ThermalChannel physical inputs

`ChannelThermalInputAdapter` constructs ten module features in this order:

1. dataset-scaled heat;
2. absolute dataset-scaled heat;
3. signed case-relative heat;
4. absolute case-relative heat;
5. active flag;
6. solid diffusivity;
7. fluid diffusivity;
8. solid conductivity;
9. fluid conductivity;
10. module radius.

The current `padding_invariant_v2` global context has eighteen entries:
Reynolds number, inlet velocity, active count, `log1p` active count, module
number density, occupied-area fraction, total scaled heat, heat per domain
area, mean active heat, maximum absolute heat, domain lengths $L_x$ and $L_y$,
viscosity, solid and fluid diffusivities, solid and fluid conductivities, and
module radius.

Historical checkpoints using `legacy_v1` retain their fourteen-entry context
and saved fixed reference-slot denominator. Runtime padding does not alter that
historical transform.

`ChannelThermalEnvironmentBuilder` creates a cell-centered $24\times8$ grid,
so the current profile uses $E=192$ environment tokens. Each token contains
normalized $x$ and $y$, normalized distances to the bottom wall, top wall,
inlet, and outlet, and centerline proximity. Query-side case features use the
first six geometric quantities.

## 4. Shared encoding

The Stage-7 core uses hidden width $H=256$, zero dropout, layer normalization,
four Fourier frequencies for module, environment, query, and pairwise relative
coordinates, nonperiodic geometry, and no query-time coordinate.

The reusable encoder can be summarized as

$$
g=E_g(c),
$$

$$
m_i=\left(E_m(S_i)+E_x\!\left(\Phi(X_i/s)\right)\right)P_{M,i},
$$

$$
e_j=E_e([\Phi(Y_j/s),R_j])+g,
$$

$$
z_q=E_q([q/s,\Phi(q/s),f_{\mathrm{case}}(q)]).
$$

The global token is added to environment tokens because the selected decoder
uses global context. The coordinate scale is `[12, 6]`; physical domain lengths
and module radius marked `auto` are resolved from the dataset before model
construction.

## 5. Current fixed six-edge organizer

### 5.1 Module-to-environment context

With `use_A_me_auxiliary=true`, every active module first attends over all
environment tokens:

$$
A^{ME}_{ij}=\operatorname{softmax}_{j}
\left(\frac{(W_qm_i)^\top(W_ke_j)}{\sqrt H}\right)P_{M,i},
$$

$$
\widetilde m_i=\left(m_i+0.25W_c\sum_jA^{ME}_{ij}e_j\right)P_{M,i}.
$$

### 5.2 Fixed softmax incidences

The accepted organizer uses `organizer_mode="fixed_projection"`, $K=6$, and
learned edge-indexed projection columns:

$$
A^{MH}_{ik}=\operatorname{softmax}_{k}(W_M\widetilde m_i)_kP_{M,i}.
$$

The environment assignment includes a source-centered geometry bias:

$$
A^{EH}_{jk}=\operatorname{softmax}_{k}\left((W_Ee_j)_k-
\frac{\lVert Y_j-s_k\rVert_2}{0.25\sqrt{s_x^2+s_y^2}}\right).
$$

All six edges are active. There is no adaptive selection, entmax sparsity,
edge-capacity schedule, or gathered execution in the Stage-7 primary path.
Fields such as `initial_active_edges`, `minimum_active_edges`, and
`slot_refinement_steps` are compatibility schema fields and do not change the
fixed organizer.

### 5.3 Hyperedge geometry and state

For edge $k$, column-normalized incidence weights define source and thermal
region centers:

$$
w^M_{ik}=\frac{A^{MH}_{ik}}{\sum_iA^{MH}_{ik}+\epsilon},\qquad
s_k=\sum_iw^M_{ik}X_i,
$$

$$
w^E_{jk}=\frac{A^{EH}_{jk}}{\sum_jA^{EH}_{jk}+\epsilon},\qquad
r_k=\sum_jw^E_{jk}Y_j.
$$

The organizer also calculates diagonal source/region variances and scales,
normalized module/environment masses, assignment purities, source-to-region
displacements, edge strengths, and diagnostic descriptors.

The content state is

$$
h_k=H_{mix}\left(
\frac{\sum_iA^{MH}_{ik}W^h_M\widetilde m_i}{\sum_iA^{MH}_{ik}+\epsilon}
+\frac{\sum_jA^{EH}_{jk}W^h_Ee_j}{\sum_jA^{EH}_{jk}+\epsilon}
\right).
$$

The profile retains `mechanism_state_mode="residual_concat"` for strict
configuration compatibility, but sets `use_hyper_mechanism_encoder=false`.
Consequently, the decoder uses the organizer's raw $h_k$ directly. It does not
instantiate or apply a second descriptor/mechanism encoder on the accepted
path.

## 6. Query routing and context-fusion decoder

The current decoder mode is `enhanced_honf_pairwise`, whose enabled components
are hyperedge value context, hypergraph-gated query/module pair context, global
context, and near-module context.

### 6.1 Query-to-edge attention

Query-to-edge logits combine state compatibility with ten learned geometric
features relative to the edge source and region:

$$
\ell_{qk}=\frac{(W_qz_q)^\top(W_kh_k)}{\sqrt H}
+W_\gamma\gamma(q,s_k,r_k).
$$

The current query normalizer is dense softmax:

$$
\alpha_{qk}=\operatorname{softmax}_{k}(\ell_{qk}).
$$

`hyper_attention_topk=0`, `query_edge_limit=0`, and
`routing_execution="dense"`, so no route is pruned. The Stage-7 profile also
sets query and environment locality to `none`.

The hyperedge value context is

$$
c^H_q=\sum_k\alpha_{qk}W_vh_k.
$$

### 6.2 Hypergraph-gated pairwise context

For every query/module pair, a shared four-layer legacy MLP receives relative geometry, module presence, the encoded module token, and the raw module features. Let its output be $\psi(q,i)$. With edge-mass-normalized incidence

$$
\bar A^{MH}_{ik}=\frac{A^{MH}_{ik}}{\sum_iA^{MH}_{ik}+\epsilon},
$$

the edge-local and query-reduced pair contexts are

$$
c^{pair}_{qk}=\sum_i\bar A^{MH}_{ik}\psi(q,i),
$$

$$
c^{pair}_q=\sigma(\eta_{pair})\sum_k\alpha_{qk}c^{pair}_{qk}.
$$

The promoted executor contracts routing first,

$$
\beta_{qi}=\sum_k\alpha_{qk}\bar A^{MH}_{ik},
\qquad
c^{pair}_q=\sigma(\eta_{pair})\sum_i\beta_{qi}\psi(q,i),
$$

which is algebraically equivalent to the accepted edge-explicit computation at full support while avoiding the full query-edge-module intermediate; the frozen Run-1401 output replay is numerically exact. The learned pairwise gate remains initialized to `0.1`, the legacy MLP parameter path remains unchanged, and K=6 checkpoints retain the same 237 keys. Dense full-beta execution is used for training. Evaluation may gather the smallest module set reaching retained beta mass 0.98 without renormalizing truncated beta; this is an evaluation override, not a training curriculum.

### 6.3 Final context and field head

The near-module context is Gaussian pooling over active module tokens with
`local_context_scale=0.45`:

$$
c^{near}_q=\sum_i
\frac{P_{M,i}\exp(-\lVert q-X_i\rVert^2/(2\sigma^2))}
{\sum_jP_{M,j}\exp(-\lVert q-X_j\rVert^2/(2\sigma^2))+\epsilon}m_i.
$$

The complete accepted context is

$$
c_q=c^H_q+c^{pair}_q+W_gg+W_nc^{near}_q.
$$

The direct module/environment residual branch is not enabled by
`enhanced_honf_pairwise`, and `output_mean_residual_split=false`. Therefore

$$
\boxed{\widehat U(q)=H_{field}\!\left(\operatorname{LayerNorm}(c_q)\right)}.
$$

This is context fusion, not edge-additive output. Individual hyperedges
organize and route latent information, but the primary model does not claim
that the final physical field is an exact sum of separately predicted edge
fields.

## 7. ThermalChannel Stage-A coupling

The generic HONF core has no ThermalChannel physics. The
`ChannelThermalHONFModel` wrapper owns the frozen local disk surrogate and the
coupled execution order.

The case configuration loads a trusted local Stage-A checkpoint strictly and
freezes it. The Stage-A surrogate uses seven module parameters,
five-value angular port tokens
$[\theta,\cos\theta,\sin\theta,T_{env},h]$, hidden and latent width 128, 16 port
latents, four attention heads, four cross-attention layers, six coordinate
Fourier frequencies, and zero dropout.

For a normal coupled forward pass, the wrapper performs:

1. build physical module, environment, global, and query features;
2. encode the case and compute a base organizer;
3. predict angular $T_{env}$ and positive $h$ port conditions;
4. choose predicted, teacher, or mixed ports;
5. execute Stage A only for active modules;
6. assemble interface temperature and corrected physical flux;
7. fuse six local-response summary values and the 128-value local latent into
   the module state;
8. perform one configured outside-temperature refinement, rerun Stage A, and
   fuse the final response;
9. recompute the final organizer and decode the requested global queries.

The corrected flux is anchored to the Robin relation

$$
q_n^{Robin}=h(T_s-T_{env}),
$$

with a learned zero-initialized correction under
`local_surrogate_flux_mode="corrected_physics"`.

`PreparedChannelThermalCase` stores the final organizer tensors and global
token. `decode_prepared()` can then evaluate additional query chunks without
repeating feature encoding, Stage A, refinement, or organization. This is an
execution optimization only; it does not change the prediction formula.

## 8. Current data, training, and checkpoint policy

The case profile uses the `thermal_channel_global_v1` manifest with 600 training
and 90 test cases, 1024 sampled field points per case, train and validation
batch size 48, four workers, input and target normalization, random
training-point sampling, dynamic module padding, and module-count bucketing.

The Stage-7 training profile specifies:

| Setting | Value |
|---|---:|
| Seed | 0 |
| Epoch budget | 5000 |
| Optimizer | AdamW, one parameter group |
| Learning rate | $3\times10^{-4}$ |
| Organizer learning rate | `null` (shared optimizer group) |
| Weight decay | $10^{-5}$ |
| Gradient clipping | 1.0 |
| AMP | false |
| Port mode | predicted, no curriculum schedule |
| Plot cadence | 50 epochs |
| Latest-checkpoint cadence | 10 epochs |
| Milestones | 500, 1000, 2500, 5000, 7500, 10000 |

Milestones above the 5000-epoch budget are inert unless the budget is extended.
Checkpoint selection should follow the relevant validation criterion rather
than automatically using the latest epoch.

The coupled objective is

$$
\mathcal L=\lambda_F\mathcal L_{field}
+\lambda_I\mathcal L_{internal}
+\lambda_\Gamma\mathcal L_{interface}
+\lambda_P\mathcal L_{port}
+\lambda_S\mathcal L_{port\ smooth}
+\lambda_G\mathcal L_{port/global}
+\lambda_C\mathcal L_{predicted\ consistency}
+\mathcal L_{organizer}.
$$

Current principal weights are `1.0`, `1.0`, `0.2`, `0.3`, `0.01`, `0.2`, and
`0.05`, respectively. Predicted-port consistency warms up over 100 epochs. The
interface and port-$h$ terms use smooth L1. Organizer regularization is
implemented for research, but `enabled=false`, so it contributes zero in the
current profile. There is no active-edge-count penalty in Stage 7.

Checkpoint resume validates model identity, strict state loading, optimizer
group count, parameter order, parameter names, and optimizer state before
continuing. Evaluation also loads maintained checkpoint families with
`strict=True`.

## 9. Compatibility and research modes

`UnifiedForwardConfig.from_dict()` supplies the historical defaults
`fixed_projection`, `residual_concat`, `context_fusion`, softmax assignments,
and dense execution when upgraded fields are absent. Old saved configurations
therefore reconstruct their original path rather than silently selecting a
research architecture.

The following mechanisms remain supported but are outside the primary reading
path:

| Mechanism | Maintained implementation | Primary Stage-7 value |
|---|---|---|
| Exchangeable slot organizer | `organization/exchangeable.py` | fixed projection |
| Adaptive quality/coverage selection | exchangeable organizer | all six active |
| Scheduled softmax-to-entmax assignments | `organization/helpers.py`, `routing.py` | softmax |
| Descriptor-first mechanism state | `decoding/pairwise.py` | raw organizer state |
| Exact background-plus-edge fields | `decoding/research.py` | context fusion |
| Gathered query-edge/module execution | `decoding/research.py`, `decoding/pairwise.py` | dense |
| Topology schema v3 export | `src/honf_forward_core/evaluation/topology_signature.py` | optional evaluation feature |

`adaptive_sparse_additive.json` remains the base for optional research and
compatibility overlays. Lifecycle/status metadata lives in
`profile_registry.json` instead of being written into individual configs.

Sparse execution for the accepted context-fusion model is implemented as the
promoted fused gathered beta-0.98 evaluation/deployment path. It preserves the
Stage-7 scientific formula and full-support numerical parity without
reactivating additive field assembly.

The optional `case_adaptive_residual_context` profile changes only the
organizer. It builds a normalized nonnegative module-environment coupling,
extracts shared rank-one mechanisms sequentially, subtracts each explained
component, and stops independently per case when the residual fraction reaches
the configured tolerance. `num_hyperedges=0` is a profile declaration that no
global scientific K exists; tensor packing still uses the batch-local active
module width. Soft survival supports training and a hard residual mask supports
evaluation. The candidate retains the established encoders, context-fusion
decoder, fused beta routing, legacy pairwise kernel, ThermalChannel coupling,
losses, and disabled organizer regularization. Residual traces and counts are
evaluation diagnostics, not topology-signature schema changes.

## 10. Current code structure

The Stage-7 cleanup preserved public facades and checkpoint-visible parameter
ownership while moving optional or procedural logic into focused modules.

| Responsibility | Maintained source |
|---|---|
| Strict core config and `BatchData` | `src/honf_forward_core/config.py` |
| Generic encode/organize/decode orchestration | `src/honf_forward_core/model.py` |
| Fixed organizer and stable parameter owner | `src/honf_forward_core/organizer.py` |
| Organizer math helpers | `src/honf_forward_core/organization/helpers.py` |
| Exchangeable/adaptive organizer | `src/honf_forward_core/organization/exchangeable.py` |
| Context-fusion decoder and stable parameter owner | `src/honf_forward_core/decoder.py` |
| Pairwise and optional mechanism encoders | `src/honf_forward_core/decoding/pairwise.py` |
| Additive/gathered compatibility execution | `src/honf_forward_core/decoding/research.py` |
| Assignment normalization and entmax | `src/honf_forward_core/routing.py` |
| Thermal physical input and environment features | `Case_ThermalChannel/src/channelthermal/input_adapter.py`, `environment.py` |
| Complete coupled ThermalChannel model | `Case_ThermalChannel/src/channelthermal/model.py` |
| Non-registering ThermalChannel support methods | `Case_ThermalChannel/src/channelthermal/model_support.py` |
| Stage-A model and coupling | `Case_ThermalChannel/src/channelthermal/local_surrogate/model.py`, `local_coupling.py` |
| Forward workflow facade | `Case_ThermalChannel/src/channelthermal/workflows/train_forward.py` |
| Epoch/loss execution | `Case_ThermalChannel/src/channelthermal/training/epoch.py` |
| Optimizer inventory and resume parity | `Case_ThermalChannel/src/channelthermal/training/optimizer.py` |
| Checkpoint construction and validation | `Case_ThermalChannel/src/channelthermal/training/checkpoints.py` |
| Metrics and plots | `Case_ThermalChannel/src/channelthermal/training/reporting.py` |
| Evaluation workflow facade | `Case_ThermalChannel/src/channelthermal/workflows/evaluate_forward.py` |
| Evaluation load, prepared decode, and results | `Case_ThermalChannel/src/channelthermal/evaluation/{loading,prepared,results}.py` |
| Canonical artifact layout | `src/honf_runtime/artifact_layout.py`, `src/honf_runtime/run_store.py` |
| Maintained offline diagnostics | `tools/diagnostics/` |

The fixed organizer layers remain registered directly on
`HypergraphOrganizerCore`, and context-fusion decoder layers remain registered
directly on `HypergraphFieldDecoder`. `ResearchDecoderExecutionMixin` and
`ChannelThermalModelSupportMixin` register no `nn.Module` children. This is why
the cleanup can improve source organization without changing state-dict keys,
parameter shapes, parameter order, or optimizer-resume semantics.

## 11. Evaluation artifacts and frozen references

The canonical evaluator writes one timestamped job containing category-owned
artifacts such as:

- `fields/` for quicklooks and local/interface plots;
- `metrics/metrics_<mode>.csv`;
- `arrays/evaluation_outputs_<mode>.npz`;
- `organization/`, `routing/`, `diagnostics/`, and `plans/` when requested;
- `summary.json` and `evaluation_manifest.json`.

Generated evidence is separate from maintained diagnostic source. Historical
diagnostic entry points remain as compatibility wrappers, while maintained
implementations live under `tools/diagnostics/`.

The reusable compatibility contract is stored at:

| Evidence | Location |
|---|---|
| Golden numerical replay contract | `tests/fixtures/forward_cleanup/golden_replay.json` |
| Public schema snapshots | `tests/fixtures/forward_cleanup/public_schemas.json` |
| Replay command | `tools/diagnostics/replay_forward_golden.py` |
| Architecture contract | `docs/architecture/` |

Trusted local checkpoints can be replayed against these schemas and fixtures;
checkpoint binaries and generated replay evidence remain local-only.
