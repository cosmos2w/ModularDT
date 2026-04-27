# Multi-cylinder PhiFlow Demo

Updated: 2026-04-27

This demo covers the periodic multi-cylinder inert and active thermal wake workflow:

1. simulate wake fields with PhiFlow
2. preprocess raw cases into a packed HDF5 dataset
3. train a hypergraph-organized neural field surrogate
4. reconstruct full flow fields from saved checkpoints

## Active Thermal Cases

Updated: 2026-04-27

The active workflow reuses the existing simulator, preprocessing, deterministic
model, and generative model. It is not a separate active-only framework.

Active simulation mode adds one-way temperature advection/diffusion on top of
the flow solve. Each cylinder contributes a Gaussian shell heat source controlled
by `layout.heat_powers`; positive values heat and negative values are allowed
for cooling when `thermal.power_min` / `thermal.power_max` include them.

New active configs can delay all thermal physics until the flow warmup has
finished:

```json
"thermal": {
  "activate_after_warmup": true,
  "reset_temperature_at_activation": true,
  "save_only_after_thermal_activation": true,
  "thermal_start_after_flow_warmup": true
}
```

Before `thermal_start_step`, the flow evolves as an inert incompressible wake
and temperature remains ambient. At activation, temperature can be reset to the
ambient scalar field, then advection/diffusion/source terms begin. Saved active
`frame_index.csv` files include `thermal_time`, `thermal_active`,
`thermal_start_time`, and `thermal_start_step`.

`src/launch_inert_dataset_batch.py` remains backward compatible; its default
constants launch inert cases. To batch active cases, set `DATASET_MODE =
"active"`, use `config_active.json`, and let `layout.heat_powers = null` so
`materialize_layout()` samples powers from `thermal.power_min/max`.

Packed HDF5 datasets now carry explicit field metadata:

- inert: `channel_order = ["u", "v", "p", "omega"]`, `field_dim = 4`
- active: `channel_order = ["u", "v", "p", "omega", "temperature"]`, `field_dim = 5`

Old inert packed datasets remain valid. If `channel_order`, `field_dim`,
temperature, or `heat_powers` are missing, loaders fall back to the inert
four-channel convention and zero heat powers.

Preprocessing is still launched by `src/preprocess_inert_multicyl_dataset.py`
for compatibility, but it is mode-aware:

```bash
python src/preprocess_inert_multicyl_dataset.py \
  --input-root ./Data_Saved/Active_Raw \
  --output-root ./Data_Saved/Processed_Active_Dataset \
  --include-temperature auto \
  --phase-bins 36 \
  --save-cycles 4 \
  --active-time-mode multicycle_absolute \
  --active-save-contiguous-cycles 4 \
  --active-warmup-cycles 1
```

With `--active-time-mode multicycle_absolute`, preprocessing does not tile one
canonical cycle. It identifies contiguous post-activation wake cycles, bins each
cycle by flow phase, and concatenates the real cycles. The packed dataset stores:

- `phase_tau_centers`: periodic flow phase in `[0, 1)`
- `tau_abs_centers`: `cycle_index + phase_tau`
- `thermal_time_centers`: non-periodic active thermal age
- `cycle_index_centers`: selected active cycle index

Training batches use `query_tau` for periodic flow quantities and `query_time`
for temperature age. Thus `u, v, p, omega` can remain phase-periodic while
temperature is not forced to repeat from cycle to cycle.

Deterministic training uses the same organizer and decoder; only decoder output
width changes with `model.field_dim`. Active cases can pass normalized
per-cylinder heat powers through `extra_module` by setting:

```json
"dataset": {
  "use_heat_power_module_feature": true,
  "heat_power_scale": "auto"
},
"model": {
  "field_dim": 5,
  "future_module_feature_dim": 1,
  "use_nonperiodic_query_time": true,
  "use_temperature_time_head": true,
  "temperature_channel_index": 4
}
```

Optional combined active+inert training is controlled by `dataset.USE_INERT`.
Before mixing, the trainer checks grid/domain/channel compatibility. Compatible
inert samples are promoted to active shape by filling temperature with
`inert_temperature_value`, heat powers with zero, and residual temperature with
zero. By default validation remains active-only (`use_inert_for_val=false`).

Generative Stage 1 infers the AE input/output channel count from the packed
dataset. Stage 2 requires the deterministic conditioner checkpoint to have the
same `field_dim` as the generative target dataset, and it raises a clear error
on mismatch.

## Current Dynamic-Residual Upgrade

This revision targets the residual-dynamics plateau with four scoped changes:

1. Local top-k refinement is now wake-aware. Module and environment tokens are
   ranked by periodic proximity, hyperedge wake relevance, and learned local
   attention bias.
2. Dynamic mode capacity was increased modestly with more hyperedges, wider
   dynamic tokens, and a second local refinement layer.
3. A single dynamic-energy loss matches masked residual energy, defaulting to
   the omega channel, to discourage residual/vorticity amplitude collapse.
4. Phase-window batches replace part of each training item with the same xy
   locations sampled across multiple tau values from `canonical_cycle`, so the
   residual branch sees phase-dependent motion rather than isolated snapshots.
5. `loss_curve.png` is now a 2x2 figure: total, field, residual, and dynamic
   energy.

## Organizer Collapse Fix

Updated: 2026-04-27

The organizer previously allowed a bad but low-entropy solution: `A_mh` could
split cylinders/modules across several hyperedges while `A_eh` assigned nearly
all environment tokens to one hyperedge. That made a hyperedge look strong even
when it had environment support but almost no cylinder/module support.

The correction has four parts:

1. `A_eh` receives direct factor supervision from `A_me` and `A_mh`:

   `A_eh_target[e,k] proportional to sum_i A_me[i,e] * A_mh[i,k]`

2. normalized environment hyperedge mass is aligned with normalized module
   hyperedge mass.
3. `hyper_strength` is now joint source/environment support:

   `sqrt(module_mass * env_mass)`

4. env-to-hyperedge scoring now sees hyperedge-specific periodic relative
   geometry from each environment token to the provisional `A_mh` source center.

The old organizer sparsity/entropy regularizers are disabled in the template
because a collapsed `A_eh` can look confidently sparse. The new losses instead
encourage each interaction hyperedge to own both a cylinder/module group and a
corresponding wake/environment region.

## Optional Active-Edge Masking

Updated: 2026-04-27

Hyperedge slots are intentionally overcomplete: `num_hyperedges` is capacity,
not a promise that every slot maps to a distinct physical interaction. After the
collapse fix, some runs may still learn redundant slots. `DISABLE_EDGE` now
performs active-edge compression, not naive deletion. A duplicate edge has a
very similar `A_mh/A_eh` signature and nearby source/wake geometry to a stronger
representative; it is disabled as a decoder token, but its incidence mass is
merged into the representative through `A_mh_effective` and `A_eh_effective`.
A truly collapsed edge is weak by soft strength/module/environment mass and has
no active representative parent.

Set `model.DISABLE_EDGE = true` to compute `hyper_active_mask` and use it in the
decoder memory, dynamic hyperedge memory, phase-conditioned hyper context, and
organization visualizations. Tensor dimensions are not deleted. Raw `A_mh` and
`A_eh` remain available for losses, old checkpoints, raw diagnostics, and raw
visualization checks. Effective `A_mh_effective` and `A_eh_effective` are used
for active-edge visualization and downstream relevance scoring so active views
retain environmental coverage after duplicate compression.

Hard environment-token count is diagnostic unless soft environment mass is also
weak. An edge with zero hard-dominant tokens but meaningful soft mass can remain
active or be merged into a representative instead of disappearing from the
active organizer view.

The compatibility default is:

```json
"DISABLE_EDGE": false
```

Recommended experimental setting:

```json
"DISABLE_EDGE": true,
"disable_edge_min_active_edges": 2,
"disable_edge_prune_duplicates": true
```

Evaluation accepts `--disable-edge`, `--no-disable-edge`, and
`--show-disabled-edges`. Inactive hyperedges are greyed or hidden in physical,
matrix, sankey, and schematic organization views.

Stage-2 generative conditioning consumes deterministic organizer aux outputs.
When active masks are present, pooled hyperedge conditioning uses active-edge
masking. Stage-2 checkpoints trained with an old deterministic conditioner do
not automatically gain the new active-edge semantics; to fully benefit, train or
condition the generative model with a deterministic checkpoint trained/evaluated
using the revised organizer behavior.

## Periodic Boundary Assumption

The PhiFlow simulator in this demo uses periodic boundaries. The surrogate keeps
that assumption by design:

- all organizer and decoder proximity features use periodic minimum-image
  geometry
- wake direction uses directed periodic downstream distance in `+x`
- the model is not an open-channel approximation

Two periodic x-semantics matter throughout the code:

- minimum-image periodic distance: shortest wrapped separation, used for
  proximity and top-k local gathering
- directed periodic downstream distance: `(x_dst - x_src) mod Lx`, used for
  wake direction, wake relevance, and wake-focused sampling

## Model Summary

The forward path is:

`structure -> organizer -> behavior / dynamic memory -> mean branch + residual branch -> pred_field`

Key outputs from `src/model.py`:

- `pred_field`, `pred_mean`, `pred_residual`
- `freq_pred`
- `A_me`, `A_mh`, `A_eh`
- `hyper_source_coords`
- `hyper_wake_coords`
- `hyper_wake_axis`
- `hyper_wake_extent`
- `hyper_module_mass`
- `hyper_env_mass`
- `hyper_strength`
- `dynamic_global_token`
- `dynamic_hyper_base`
- optional `hyper_phase_offsets`, `hyper_phase_sin_coeff`, `hyper_phase_cos_coeff`

### Organizer

The organizer now returns both source-centered and wake-centered hyperedge
geometry.

- `hyper_source_coords` are circular weighted means of cylinder coordinates from
  `A_mh`
- `hyper_wake_coords` are circular weighted means of environment-token
  coordinates from `A_eh`
- `hyper_wake_axis` points from source center to wake center under periodic
  geometry
- `hyper_wake_extent` is a lightweight weighted wake spread
- `hyper_module_mass` and `hyper_env_mass` expose normalized hyperedge usage on
  both sides of the interaction
- `hyper_strength = sqrt(hyper_module_mass * hyper_env_mass)`, so one-sided
  environment or module collapse is not treated as a strong interaction

This makes hyperedges represent interaction regions rather than only
source-cylinder clusters.

### Dynamic Hyper Tokens

The residual branch no longer relies on a mostly static hyper token alone.
Dynamic hyper memory is phase-conditioned with lightweight low-rank harmonic
modulation:

- `dynamic_hyper_base`: per-hyperedge base dynamic memory
- harmonic coefficients: low-rank `sin/cos` phase responses per hyperedge
- query-time phase context: combined with hyperedge relevance weights over
  `K_h`, without building a dense `[B, Q, K_h, D]` tensor

The decoder keeps spatial and phase encoders separate:

- spatial encoder for `(x, y)`
- harmonic phase encoder for `tau`

### Decoder

Mean branch:

- spatial query encoding
- shallow global read over structured memory
- predicts smooth mean structure

Residual branch:

- spatial encoding + phase encoding + dynamic global token + phase-conditioned
  hyper context
- wake-aware global attention over all memory
- top-k local refinement over wake-relevant module/env tokens plus all hyper
  tokens

Local module/env gathering still preserves periodic minimum-image distance, but
the default ranking also uses hyperedge wake relevance and learned attention
bias. Hyperedge relevance uses wake-centered geometry.

## Training Changes

This revision improves dynamic reconstruction by changing representation and
sampling, not by adding a large loss stack.

### Cylinder-order randomization

Training now randomizes the valid cylinder order each epoch before padding to
`max_num_cylinders`. This reduces slot/index memorization and improves
generalization across layouts. Validation stays deterministic by default.

### Wake-focused point sampling

Training point subsampling now supports `dataset.point_sampling_mode =
"wake_focused"`.

The sampler still starts from a contiguous chunk, then mixes:

- uniform points
- near-cylinder points using periodic minimum-image distance
- downstream wake points using directed periodic downstream distance in `+x`
- high-`|omega|` points

Validation stays deterministic uniform by default unless explicitly configured
otherwise.

### Losses

The objective stays intentionally simple:

- `field_mse`
- `residual_mse`
- light `freq_mse`
- dynamic-energy loss on residual energy, defaulting to omega
- light direct organizer supervision

Recommended defaults:

- `field_mse_weight = 1.0`
- `residual_mse_weight = 1.25`
- `mean_mse_weight = 0.0`
- `freq_mse_weight = 0.05`
- `dynamic_energy_weight = 0.02`
- `organizer_me_weight = 0.01`
- `organizer_mm_weight = 0.05`
- `organizer_consistency_weight = 0.05`
- `organizer_eh_factor_weight = 0.05`
- `organizer_mass_align_weight = 0.02`

No spectral, gradient, adversarial, diffusion, or flow-matching loss was added
in this revision.

## Validation And Checkpoint Selection

Validation logs:

- `val_total_loss`
- `val_field_mse`
- `val_mean_mse`
- `val_residual_mse`
- `val_freq_mse`
- `val_dynamic_energy`
- `val_residual_focus`

By default:

`val_residual_focus = val_residual_mse + residual_focus_field_weight * val_field_mse`

Best-checkpoint selection now follows `validation.best_metric_name`, which
defaults to `val_residual_focus`.

## Organizer Visualization Diagnostics

`src/evaluate.py` writes several organization diagnostics for each evaluated
case. The old ambiguous physical-plus-tripartite figure has been split into
clearer outputs:

1. `organization_physical_*.png`
   - physical-domain overlay plus a hyperedge summary table
   - cylinder locations, environment tokens, source centers, wake centers, and
     learned wake-axis arrows
   - source-to-wake and cylinder-to-hyperedge links use periodic shortest-image
     geometry

2. `organization_matrices_*.png`
   - `A_mh` heatmap for cylinder/module to hyperedge assignment
   - `A_eh` heatmap for environment-token to hyperedge assignment
   - per-hyperedge spatial maps showing where each hyperedge owns environment
     tokens

3. `organization_sankey_*.png`
   - abstract tripartite graph for topology/debugging
   - `C_i` means cylinder/module
   - `H_k` means learned interaction hyperedge/group
   - `EnvGroup_k` means environment tokens dominated by `H_k`
   - hyperedge and environment-group rows use collision-avoiding vertical
     layout, so labels do not all inherit the same wake-center y coordinate

4. `organization_summary_*.csv` and `organization_summary_*.json`
   - machine-readable hyperedge summary with strength, source/wake centers,
     wake axis, extent, module mass, environment mass, top cylinders, and top
     environment tokens

Training logs also include organizer collapse diagnostics:

- `loss_org_eh_factor`
- `loss_org_mass_align`
- `org_mass_l1`
- `org_env_effective_hyperedges`
- `org_module_effective_hyperedges`
- `org_env_mass_max`
- `org_module_mass_max`

Generative stage-2 models share the same organizer semantics whenever they
condition on deterministic aux outputs from `src/model.py`. Stage-2 models
trained or conditioned with an older deterministic checkpoint will still reflect
the old organizer behavior; retrain or condition on a deterministic checkpoint
trained after this fix to benefit from the revised organizer.

Useful evaluator arguments:

- `--organization-view {all,physical,matrices,sankey}` selects which diagnostic
  figures to render; the default is `all`
- `--organization-threshold` controls which soft assignment edges are drawn
- `--topk-me-links` controls optional light cylinder-to-environment links
- `--organization-topk-cylinders` controls how many cylinder memberships appear
  in the summary table/export
- `--organization-topk-env` controls how many environment tokens appear in the
  summary table/export
- `--organization-min-gap` controls vertical spacing in the Sankey layout
- `--no-organization-table` hides the side table in the physical view

## Default Config Highlights

The default config lives in `Config_Train/train_config_template.json`.

Important defaults in this revision:

- `num_env_tokens_x = 24`
- `num_env_tokens_y = 8`
- `num_hyperedges = 6`
- `hidden_dim / behavior_dim / latent_dim / dynamic_token_dim = 80`
- `phase_fourier_frequencies = 2`
- `use_phase_conditioned_dynamic_tokens = true`
- `dynamic_phase_harmonics = 3`
- `dynamic_phase_rank = 12`
- `use_wake_centered_hyper_geometry = true`
- `perceiver_num_layers_local = 2`
- `perceiver_refine_topk_env = 24`
- `dynamic_energy_weight = 0.02`
- `randomize_cylinder_order = true`
- `point_sampling_mode = "wake_focused"`
- `use_phase_window_batches = true`
- `best_metric_name = "val_residual_focus"`
- `residual_focus_field_weight = 0.25`

## Workflow

Run commands below from `0_Demo_MultiCylinder/`.

### 1. Simulate raw inert cases

```bash
python src/simulate_multicylinder_phiflow.py --config-json config_inert.json
```

### 2. Preprocess cases into a packed dataset

```bash
python src/preprocess_inert_multicyl_dataset.py \
  --input-root ./Data_Saved \
  --output-root ./Data_Saved/Processed_Inert_Dataset \
  --device cuda:0
```

### 3. Train the surrogate

```bash
python src/train.py --config train_config_template.json --device cuda:0
```

### 4. Evaluate a trained checkpoint

```bash
python src/evaluate.py --case-id 0002 --dataset-case-id 0161 --dataset-split test
```

### 5. Evaluate the generative model

Stage-2 generative evaluation has two modes. Snapshot mode preserves the old
single-phase workflow:

```bash
python src/evaluate_gen.py --stage 2 --case-id gen001 --split test --phase-index 0 --n-samples 4
```

Cycle mode evaluates one case across selected canonical phase bins and writes a
compressed `cycle_reconstruction.npz`, `cycle_metrics.json`,
`per_phase_metrics.csv`, `cycle_omega.gif`, `cycle_omega_gt_generated.gif`,
and `cycle_montage_omega.png`:

```bash
python src/evaluate_gen.py --stage 2 --case-id gen001 --split test --cycle --n-samples 4
```

The stage-2 rectified-flow model is trained on phase snapshots, so cycle mode
generates one tau-conditioned stochastic field per phase. `--cycle-noise-mode`
controls the initial latent coupling across phases:

- `independent`: fresh latent noise for every phase
- `shared`: one latent noise field reused for all phases in a sample
- `harmonic`: sinusoidal latent interpolation over tau for smoother samples

For memory control, use `--phase-chunk-size` and `--sample-chunk-size`.

## Directory Layout

```text
0_Demo_MultiCylinder/
├── Config_Train/
├── MODEL_EXPLAIN.txt
├── README_multicyl_demo.md
└── src/
```
